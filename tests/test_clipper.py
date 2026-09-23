"""Tests del recorte real (Fase 3) — core/clipper.py.

Cubre:
- formato de timestamps para yt-dlp --download-sections.
- streaming: comando correcto con la sección de la ClipWindow + keyframes, SIN
  red (mock _run).
- direct_file: ffmpeg local con re-encode, SIN red (mock _run).
- validación del output real (existencia, tamaño, duración) y errores.
- E2E real offline: corta un video sintético con ffmpeg real y valida que los
  clips existen, duran dentro de 15-90 s y el manifest relaciona source+hook.

La ventana (15-90 s) la decide core.clip_window; acá se verifica su
materialización.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config  # noqa: E402
import core.clipper as clipper  # noqa: E402
from core.errors import (  # noqa: E402
    ClipError,
    InsufficientDurationError,
    YouTubeAcquisitionError,
)
from models.schemas import ClipWindow, HookCandidate  # noqa: E402

from tests.fixtures import synthetic_video, video_synthetic  # noqa: E402


def _hook(start: float = 5.0, end: float = 10.0, score: float = 0.8) -> HookCandidate:
    return HookCandidate(start=start, end=end, score=score, title="Hook", reason="test")


def _window(
    start: float = 3.0, end: float = 18.0,
    hook_start: float = 5.0, hook_end: float = 10.0,
) -> ClipWindow:
    return ClipWindow(
        start=start, end=end, hook_start=hook_start, hook_end=hook_end
    )


@pytest.fixture(autouse=True)
def _restore_cfg():
    snapshot = dict(config._KEEP)
    yield
    config._KEEP.clear()
    config._KEEP.update(snapshot)


class TestFormatTs:
    def test_formats_hms_millis(self):
        assert clipper._format_ts(12.345) == "00:00:12.345"
        assert clipper._format_ts(0) == "00:00:00.000"
        assert clipper._format_ts(3661.5) == "01:01:01.500"


class TestClipStreaming:
    def test_builds_section_command(self, tmp_path: Path, monkeypatch):
        captured = {}

        def fake_run(cmd, *, what, ytdlp=False, tool="ffmpeg", **kwargs):
            captured["cmd"] = cmd
            captured["what"] = what
            out = tmp_path / "out" / "clip_000.mp4"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(b"fake")
            return None

        monkeypatch.setattr(clipper, "_run", fake_run)

        dest = tmp_path / "out"
        out = clipper._clip_streaming(
            "https://youtube.com/watch?v=x", _window(3.0, 18.0), 0, dest,
        )

        assert out.name == "clip_000.mp4"
        cmd = captured["cmd"]
        assert "--download-sections" in cmd
        idx = cmd.index("--download-sections")
        assert cmd[idx + 1] == "*00:00:03.000-00:00:18.000"
        assert "--force-keyframes-at-cuts" in cmd
        assert "--merge-output-format" in cmd
        assert "https://youtube.com/watch?v=x" == cmd[-1]
        # Gate YouTube: runtime JS habilitado por defecto.
        assert "--js-runtimes" in cmd
        assert cmd[cmd.index("--js-runtimes") + 1] == "node"

    def test_cookies_opt_in_included(self, tmp_path: Path, monkeypatch):
        captured = {}

        def fake_run(cmd, *, what, ytdlp=False, tool="ffmpeg", **kwargs):
            captured["cmd"] = cmd
            out = tmp_path / "out" / "clip_000.mp4"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(b"fake")

        monkeypatch.setattr(clipper, "_run", fake_run)
        monkeypatch.setitem(config._KEEP, "youtube.cookies_from_browser", "chrome")

        clipper._clip_streaming(
            "https://youtube.com/watch?v=x", _window(), 0, tmp_path / "out",
        )
        cmd = captured["cmd"]
        assert cmd[cmd.index("--cookies-from-browser") + 1] == "chrome"

    def test_youtube_error_maps_to_acquisition_error(self, tmp_path: Path, monkeypatch):
        """_run REAL + subprocess mockeado: un 429 de yt-dlp sale como
        YouTubeAcquisitionError (exit 7), no como ClipError (exit 5)."""
        class _Proc:
            returncode = 1
            stdout = ""
            stderr = "ERROR: HTTP Error 429: Too Many Requests"

        monkeypatch.setattr(clipper.subprocess, "run", lambda *a, **k: _Proc())
        with pytest.raises(YouTubeAcquisitionError):
            clipper._clip_streaming(
                "https://youtube.com/watch?v=x", _window(), 0, tmp_path / "out",
            )

    def test_generic_error_stays_clip_error(self, tmp_path: Path, monkeypatch):
        class _Proc:
            returncode = 1
            stdout = ""
            stderr = "ERROR: algo no reconocido"

        monkeypatch.setattr(clipper.subprocess, "run", lambda *a, **k: _Proc())
        with pytest.raises(ClipError) as ei:
            clipper._clip_streaming(
                "https://youtube.com/watch?v=x", _window(), 0, tmp_path / "out",
            )
        assert not isinstance(ei.value, YouTubeAcquisitionError)

    def test_fails_when_binary_missing(self, tmp_path: Path, monkeypatch):
        monkeypatch.setitem(config._KEEP, "bins.ffmpeg", "this_binary_does_not_exist")
        with pytest.raises(ClipError):
            clipper._run(
                [str(config.get("bins.ffmpeg")), "-y"], what="recorte"
            )


class TestClipDirect:
    def test_cuts_local_with_ffmpeg(self, tmp_path: Path, monkeypatch):
        cmd_captured = {}
        monkeypatch.setattr(clipper, "_probe_has_video", lambda p: True)
        monkeypatch.setitem(config._KEEP, "bins.ffmpeg", "ffmpeg")
        monkeypatch.setitem(config._KEEP, "ffmpeg.preset", "medium")
        monkeypatch.setitem(config._KEEP, "ffmpeg.crf", 23)

        def fake_run(cmd, *, what, tool="ffmpeg"):
            cmd_captured["cmd"] = cmd
            Path(cmd[-1]).write_bytes(b"fake")

        monkeypatch.setattr(clipper, "_run", fake_run)

        raw = tmp_path / "raw.mp4"
        raw.write_bytes(b"raw")
        dest = tmp_path / "out"
        out = clipper._clip_direct(raw, _window(3.0, 18.0), 0, dest)

        assert out.name == "clip_000.mp4"
        cmd = cmd_captured["cmd"]
        assert cmd[0] == "ffmpeg"
        assert "-ss" in cmd and "-i" in cmd and "-t" in cmd
        assert "-c:v" in cmd and "libx264" in cmd
        assert cmd[cmd.index("-ss") + 1] == "3.000"
        assert cmd[cmd.index("-t") + 1] == "15.000"

    def test_audio_only_source_cuts_without_video_codec(self, tmp_path: Path, monkeypatch):
        cmd_captured = {}
        monkeypatch.setattr(clipper, "_probe_has_video", lambda p: False)
        monkeypatch.setitem(config._KEEP, "bins.ffmpeg", "ffmpeg")

        def fake_run(cmd, *, what, tool="ffmpeg"):
            cmd_captured["cmd"] = cmd
            Path(cmd[-1]).write_bytes(b"fake")

        monkeypatch.setattr(clipper, "_run", fake_run)

        raw = tmp_path / "raw.mp3"
        raw.write_bytes(b"raw")
        clipper._clip_direct(raw, _window(0.0, 20.0, 2.0, 6.0), 0, tmp_path / "out")

        cmd = cmd_captured["cmd"]
        assert "-vn" in cmd
        assert "-c:v" not in cmd


class TestValidateOutput:
    def test_missing_clip_raises(self, tmp_path: Path):
        from core.errors import ClipFilesystemError

        with pytest.raises(ClipFilesystemError):
            clipper._validate_clip(tmp_path / "nope.mp4", _window(), 0)

    def test_empty_clip_raises(self, tmp_path: Path):
        from core.errors import ClipFilesystemError

        empty = tmp_path / "empty.mp4"
        empty.write_bytes(b"")
        with pytest.raises(ClipFilesystemError):
            clipper._validate_clip(empty, _window(), 0)

    def test_duration_out_of_range_raises(self, tmp_path: Path, monkeypatch):
        from core.errors import FFmpegError

        clip = tmp_path / "clip.mp4"
        clip.write_bytes(b"x")
        monkeypatch.setattr(clipper, "_probe_duration", lambda p: 3.0)
        with pytest.raises(FFmpegError):
            clipper._validate_clip(clip, _window(), 0)

    def test_valid_clip_returns_duration(self, tmp_path: Path, monkeypatch):
        clip = tmp_path / "clip.mp4"
        clip.write_bytes(b"x")
        monkeypatch.setattr(clipper, "_probe_duration", lambda p: 20.0)
        assert clipper._validate_clip(clip, _window(), 0) == pytest.approx(20.0)


class TestClipHooks:
    def test_no_hooks_noop(self):
        assert clipper.clip_hooks("file://x", "direct_file", [], "job_none") == []

    def test_direct_requires_raw_path(self):
        with pytest.raises(ClipError):
            clipper.clip_hooks(
                "https://example.com/v.mp4", "direct_file", [_hook()], "job_x"
            )

    def test_media_too_short_raises(self):
        with pytest.raises(InsufficientDurationError):
            clipper.clip_hooks(
                "file://x", "direct_file", [_hook(1.0, 3.0)], "job_short",
                media_duration=5.0,
            )

    def test_invalid_hook_yields_no_clips(self, tmp_path: Path):
        # Hook fuera del medio: se descarta; job exitoso con 0 clips + manifest.
        entries = clipper.clip_hooks(
            "file://x", "direct_file", [_hook(500.0, 505.0)], "job_bad",
            media_duration=120.0,
        )
        assert entries == []
        manifest = config.output_dir() / "job_bad" / "clips_raw" / "manifest.json"
        assert manifest.exists()
        assert json.loads(manifest.read_text(encoding="utf-8"))["clips"] == []

    def test_direct_e2e_real_ffmpeg_single_window(self, monkeypatch):
        """E2E real OFFLINE: corta el fixture de 20s con ffmpeg real.

        Ambos hooks colapsan a la ventana completa (0-20): dedup -> 1 clip.
        """
        src = video_synthetic()
        assert src.exists(), "falta tests/fixtures/video_synthetic.mp4"

        entries = clipper.clip_hooks(
            str(src), "direct_file",
            [_hook(5.0, 10.0), _hook(12.0, 17.0)],
            "job_real",
            raw_path=src,
        )

        assert len(entries) == 1
        entry = entries[0]
        assert entry.clip_id == "clip_000"
        assert entry.qc_status == "passed"
        clip_path = config.output_dir() / entry.path
        assert clip_path.exists()
        assert 15.0 <= entry.duration <= 90.0
        assert entry.source_start == pytest.approx(0.0)
        assert entry.source_end == pytest.approx(20.0)

    def test_direct_e2e_real_multiple_windows(self, tmp_path: Path):
        """E2E real OFFLINE con video largo: dos hooks separados -> dos clips
        con ventanas distintas, duración real validada y nombres estables."""
        src = synthetic_video(tmp_path / "long.mp4", duration=120.0)

        entries = clipper.clip_hooks(
            str(src), "direct_file",
            [_hook(10.0, 18.0), _hook(70.0, 82.0)],
            "job_multi",
            raw_path=src,
            media_duration=clipper.probe_duration(src),
        )

        assert [e.clip_id for e in entries] == ["clip_000", "clip_001"]
        starts = [e.source_start for e in entries]
        assert starts == sorted(starts)
        for entry in entries:
            clip_path = config.output_dir() / entry.path
            assert clip_path.exists()
            assert clip_path.stat().st_size > 0
            assert 15.0 <= entry.duration <= 90.0
            measured = clipper.probe_duration(clip_path)
            assert measured == pytest.approx(entry.duration, abs=1.0)
        # El manifest relaciona source + hook + ventana + status.
        manifest = config.output_dir() / "job_multi" / "clips_raw" / "manifest.json"
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        assert payload["source"] == str(src)
        assert payload["source_type"] == "direct_file"
        assert len(payload["clips"]) == 2
