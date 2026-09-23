"""Test E2E del pipeline (Fase 1) vía CliRunner, sin depender de red.

Cubre:
- direct_file local -> descarga/extracción -> transcript exit 0
- abort de presupuesto (config bajo) -> exit code 6 ANTES de whisper

Nota: los logs JSON no llegan a result.output porque el logger queda
bind al stdout original en import time; el contrato con el orquestador son
los exit codes, que CliRunner sí propaga.
"""

from __future__ import annotations

import json
import sys
import wave
from pathlib import Path

import pytest
from typer.testing import CliRunner

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import main  # noqa: E402
import config  # noqa: E402
import core.clipper as clipper  # noqa: E402
from core import cost_guard  # noqa: E402
from models.schemas import HookCandidate, TranscriptSegment  # noqa: E402
from tests.fixtures import video_synthetic  # noqa: E402


def _wav(path: Path, seconds: float = 2.0) -> Path:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * int(16000 * seconds))
    return path


@pytest.fixture(autouse=True)
def _clean():
    cost_guard._TRACKERS.clear()
    yield
    cost_guard._TRACKERS.clear()


class TestDirectFileSuccess:
    def test_local_file_e2e_exit_0(self, tmp_path: Path, monkeypatch):
        audio = _wav(tmp_path / "in.wav")

        # Sin whisper real en el suite: devolvemos un segmento dummy.
        monkeypatch.setattr(
            main,
            "transcribe",
            lambda audio_path, job_id, **kw: [
                TranscriptSegment(start=0.0, end=2.0, text="hola", words=[])
            ],
        )
        # Mock hooks para no depender del LLM real en tests.
        monkeypatch.setattr(
            main,
            "find_hooks",
            lambda segments, job_id, n_candidates=None: [],
        )

        result = CliRunner().invoke(main.app, ["run", "--url", str(audio), "--n-clips", "3"])

        assert result.exit_code == 0, result.output
        # Buscamos el job_id en los logs y validamos el transcript.
        jobs = list((config.output_dir()).glob("job_*"))
        assert jobs, "No se creó el directorio de salida del job"
        tx = max(jobs, key=lambda p: p.stat().st_mtime) / "transcript.json"
        assert tx.exists()
        payload = json.loads(tx.read_text(encoding="utf-8"))
        assert payload["source_type"] == "direct_file"
        assert payload["segments"][0]["text"] == "hola"


class TestClipE2EReal:
    def test_video_recorta_clips_offline(self, monkeypatch):
        """E2E offline real: video sintético -> hooks mockeados ->
        recorte quirúrgico con ffmpeg real -> clips_raw -> clips_vertical ->
        clips_captioned con subtítulos ASS quemados + QC + manifest."""
        src = video_synthetic()
        assert src.exists()

        # Words cubriendo [0, 20] con última palabra en end=20.0 exacto:
        # main.py las aplana y pasa a clip_hooks -> snapping conserva
        # source_start==0.0 y source_end≈20.0 en el video sintético de 20s.
        monkeypatch.setattr(
            main,
            "transcribe",
            lambda audio_path, job_id, **kw: [
                TranscriptSegment(
                    start=0.0,
                    end=10.0,
                    text="a",
                    words=[
                        {"start": float(i), "end": float(i + 1), "word": f"w{i}"}
                        for i in range(10)
                    ],
                ),
                TranscriptSegment(
                    start=10.0,
                    end=20.0,
                    text="b",
                    words=[
                        {"start": float(i), "end": float(i + 1), "word": f"w{i}"}
                        for i in range(10, 20)
                    ],
                ),
            ],
        )
        monkeypatch.setattr(
            main,
            "find_hooks",
            lambda segments, job_id, **kw: [
                HookCandidate(start=1.0, end=6.0, score=0.9,
                              title="t1", reason="r1"),
                HookCandidate(start=8.0, end=12.0, score=0.7,
                              title="t2", reason="r2"),
            ],
        )
        monkeypatch.setitem(config._KEEP, "clip.buffer_seconds", 1.0)

        result = CliRunner().invoke(
            main.app, ["run", "--url", str(src), "--n-clips", "2"]
        )

        assert result.exit_code == 0, result.output
        jobs = list((config.output_dir()).glob("job_*"))
        job_dir = max(jobs, key=lambda p: p.stat().st_mtime)
        clips_raw = job_dir / "clips_raw"
        # En un video de 20s ambas ClipWindow colapsan a [0, 20] -> dedup: 1 clip.
        assert (clips_raw / "clip_000.mp4").exists()
        manifest = clips_raw / "manifest.json"
        assert manifest.exists()
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        assert payload["source"] == str(src)
        assert payload["source_type"] == "direct_file"
        assert len(payload["clips"]) == 1
        # Se conserva el hook de mayor score.
        assert payload["clips"][0]["hook"]["start"] == 1.0
        assert payload["clips"][0]["source_start"] == 0.0
        assert payload["clips"][0]["source_end"] == pytest.approx(20.0, abs=0.1)
        assert 15.0 <= payload["clips"][0]["duration"] <= 90.0

        # Fase 4: composición vertical 9:16 real del clip horizontal.
        vertical_dir = job_dir / "clips_vertical"
        assert (vertical_dir / "clip_000.mp4").exists()
        v_manifest = vertical_dir / "manifest.json"
        assert v_manifest.exists()
        v_payload = json.loads(v_manifest.read_text(encoding="utf-8"))
        assert v_payload["source"] == str(src)
        assert v_payload["source_type"] == "direct_file"
        assert len(v_payload["clips"]) == 1
        vc = v_payload["clips"][0]
        assert vc["clip_id"] == "clip_000"
        assert vc["qc_status"] == "passed"
        assert vc["aspect_ratio"] == "9:16"
        assert vc["composition_mode"] == "center_crop"
        # ffprobe real del vertical: dimensiones 9:16 + audio.
        rw, rh = clipper.probe_dimensions(vertical_dir / "clip_000.mp4")
        assert rw > 0 and rh > 0
        assert rw / rh == pytest.approx(9 / 16, rel=0.02)
        assert clipper._probe_has_audio(vertical_dir / "clip_000.mp4") is True

        # Fase 5: subtitulos ASS quemados en clips_captioned/.
        captioned_dir = job_dir / "clips_captioned"
        assert (captioned_dir / "clip_000.mp4").exists()
        assert (captioned_dir / "manifest.json").exists()
        assert (captioned_dir / "subtitles" / "clip_000.ass").exists()
        cap_manifest = json.loads(
            (captioned_dir / "manifest.json").read_text(encoding="utf-8")
        )
        assert len(cap_manifest["subtitles"]) == 1
        cap_entry = cap_manifest["subtitles"][0]
        assert cap_entry["clip_id"] == "clip_000"
        assert cap_entry["qc_status"] == "passed"
        assert cap_entry["subtitle_source"] == "transcript"
        assert cap_entry["caption_count"] >= 1
        assert cap_entry["font"] != ""
        # ffprobe real del captioned: resolución 9:16 + audio + duración.
        cw, ch = clipper.probe_dimensions(captioned_dir / "clip_000.mp4")
        assert cw > 0 and ch > 0
        assert cw / ch == pytest.approx(9 / 16, rel=0.02)
        assert clipper._probe_has_audio(captioned_dir / "clip_000.mp4") is True
        cap_dur = clipper.probe_duration(captioned_dir / "clip_000.mp4")
        assert 15.0 <= cap_dur <= 90.0


class TestBudgetAbortE2E:
    def test_local_file_abort_exit_6_before_whisper(self, tmp_path: Path, monkeypatch):
        audio = _wav(tmp_path / "in.wav")
        monkeypatch.setitem(config._KEEP, "max_api_cost_per_job", 0.0000001)

        # Si whisper llegase a cargarse, que reviente: el abort DEBE pasar antes.
        import core.transcriber as tc

        def boom(*a, **k):
            raise AssertionError("Whisper se cargó pese al budget abort E2E")
        monkeypatch.setattr(tc, "_get_model", boom)

        result = CliRunner().invoke(main.app, ["run", "--url", str(audio), "--n-clips", "3"])

        assert result.exit_code == 6, result.output

    def test_streaming_url_abort_exit_6(self, monkeypatch):
        """Para streaming no hace falta red: el abort corre tras la extracción,
        si la extracción existe. Pero acá forzamos el abort ANTES de cualquier
        descarga mockeando extract_audio para no tocar red, manteniendo el
        check de presupuesto real en transcribe."""
        monkeypatch.setitem(config._KEEP, "max_api_cost_per_job", 0.0000001)

        import tempfile  # solo para fabricar un audio real de 2s
        d = Path(tempfile.mkdtemp(prefix="autoclipper_test_"))
        try:
            audio = _wav(d / "fake_audio.mp3")
            monkeypatch.setattr(
                main,
                "extract_stream_audio",
                lambda url, dest: audio,
            )

            import core.transcriber as tc

            def boom(*a, **k):
                raise AssertionError("Whisper se cargó pese al budget abort streaming")
            monkeypatch.setattr(tc, "_get_model", boom)

            result = CliRunner().invoke(
                main.app,
                ["run", "--url", "https://youtube.com/watch?v=dQw4w9WgXcQ", "--n-clips", "3"],
            )
            assert result.exit_code == 6, result.output
        finally:
            import shutil
            shutil.rmtree(d, ignore_errors=True)