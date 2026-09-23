"""Tests del gate de adquisición de YouTube en extractor_audio.py.

No tocan la red: mockean subprocess.run. Verifican:
- un error de YouTube (429 / bot-check / runtime faltante) se traduce a
  YouTubeAcquisitionError con exit code 7 y diagnóstico accionable;
- un error genérico sigue siendo ExtractionError (exit 5) — sin regresión;
- NO existe retry a nivel aplicación: yt-dlp se invoca UNA sola vez;
- el comando incluye --js-runtimes y (opt-in) --cookies-from-browser;
- el contenido de cookies nunca aparece en los argumentos.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from core import extractor_audio  # noqa: E402
from core.errors import ExtractionError, YouTubeAcquisitionError  # noqa: E402
from core.exit_codes import ExitCode  # noqa: E402


@pytest.fixture(autouse=True)
def _restore_cfg():
    snapshot = dict(config._KEEP)
    yield
    config._KEEP.clear()
    config._KEEP.update(snapshot)


class _Proc:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _patch_run(monkeypatch, proc, *, create=None):
    """Intercepta subprocess.run; opcionalmente crea el audio esperado."""
    calls = {"n": 0, "cmd": None}

    def fake_run(cmd, **kwargs):
        calls["n"] += 1
        calls["cmd"] = list(cmd)
        if create is not None and proc.returncode == 0:
            create()
        return proc

    monkeypatch.setattr(extractor_audio.subprocess, "run", fake_run)
    return calls


class TestYouTubeErrorMapping:
    def test_bot_check_maps_to_youtube_error_exit_7(self, tmp_path, monkeypatch):
        proc = _Proc(1, stderr="ERROR: [youtube] x: Sign in to confirm you're not a bot.")
        calls = _patch_run(monkeypatch, proc)
        with pytest.raises(YouTubeAcquisitionError) as ei:
            extractor_audio.extract_stream_audio("https://youtube.com/watch?v=x", tmp_path)
        assert int(ei.value.exit_code) == int(ExitCode.DOWNLOAD_FAILED)
        assert "cookies_from_browser" in ei.value.message
        # Hard stop: una sola invocación, sin loops.
        assert calls["n"] == 1

    def test_429_maps_to_youtube_error(self, tmp_path, monkeypatch):
        proc = _Proc(1, stderr="ERROR: HTTP Error 429: Too Many Requests")
        _patch_run(monkeypatch, proc)
        with pytest.raises(YouTubeAcquisitionError):
            extractor_audio.extract_stream_audio("https://youtube.com/watch?v=x", tmp_path)

    def test_missing_runtime_maps_to_youtube_error(self, tmp_path, monkeypatch):
        proc = _Proc(1, stderr="WARNING: No supported JavaScript runtime could be found.")
        _patch_run(monkeypatch, proc)
        with pytest.raises(YouTubeAcquisitionError):
            extractor_audio.extract_stream_audio("https://youtube.com/watch?v=x", tmp_path)

    def test_generic_error_stays_extraction_error(self, tmp_path, monkeypatch):
        proc = _Proc(1, stderr="ERROR: network unreachable")
        _patch_run(monkeypatch, proc)
        with pytest.raises(ExtractionError) as ei:
            extractor_audio.extract_stream_audio("https://youtube.com/watch?v=x", tmp_path)
        assert not isinstance(ei.value, YouTubeAcquisitionError)
        assert int(ei.value.exit_code) == int(ExitCode.FFMPEG_FAILED)

    def test_no_app_level_retry(self, tmp_path, monkeypatch):
        proc = _Proc(1, stderr="ERROR: HTTP Error 429: Too Many Requests")
        calls = _patch_run(monkeypatch, proc)
        with pytest.raises(YouTubeAcquisitionError):
            extractor_audio.extract_stream_audio("https://youtube.com/watch?v=x", tmp_path)
        assert calls["n"] == 1, "No debe reintentar en bucle"


class TestCommandConstruction:
    def test_includes_js_runtime_by_default(self, tmp_path, monkeypatch):
        audio = tmp_path / "audio.mp3"
        calls = _patch_run(monkeypatch, _Proc(0), create=lambda: audio.write_bytes(b"x"))
        out = extractor_audio.extract_stream_audio("https://youtube.com/watch?v=x", tmp_path)
        assert out == audio
        assert "--js-runtimes" in calls["cmd"]
        assert calls["cmd"][calls["cmd"].index("--js-runtimes") + 1] == "node"

    def test_runtime_disabled_omits_flag(self, tmp_path, monkeypatch):
        monkeypatch.setitem(config._KEEP, "youtube.js_runtime", "none")
        audio = tmp_path / "audio.mp3"
        calls = _patch_run(monkeypatch, _Proc(0), create=lambda: audio.write_bytes(b"x"))
        extractor_audio.extract_stream_audio("https://youtube.com/watch?v=x", tmp_path)
        assert "--js-runtimes" not in calls["cmd"]

    def test_cookies_opt_in_included(self, tmp_path, monkeypatch):
        monkeypatch.setitem(config._KEEP, "youtube.cookies_from_browser", "chrome")
        audio = tmp_path / "audio.mp3"
        calls = _patch_run(monkeypatch, _Proc(0), create=lambda: audio.write_bytes(b"x"))
        extractor_audio.extract_stream_audio("https://youtube.com/watch?v=x", tmp_path)
        assert calls["cmd"][calls["cmd"].index("--cookies-from-browser") + 1] == "chrome"

    def test_cookies_disabled_omits_flag(self, tmp_path, monkeypatch):
        audio = tmp_path / "audio.mp3"
        calls = _patch_run(monkeypatch, _Proc(0), create=lambda: audio.write_bytes(b"x"))
        extractor_audio.extract_stream_audio("https://youtube.com/watch?v=x", tmp_path)
        assert "--cookies-from-browser" not in calls["cmd"]

    def test_binary_missing_raises_extraction_error(self, tmp_path, monkeypatch):
        def boom(cmd, **kwargs):
            raise FileNotFoundError(cmd[0])
        monkeypatch.setattr(extractor_audio.subprocess, "run", boom)
        with pytest.raises(ExtractionError):
            extractor_audio.extract_stream_audio("https://youtube.com/watch?v=x", tmp_path)
