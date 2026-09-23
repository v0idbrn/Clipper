"""Tests de transcriber.py.

Verificación clave de la fase: el kill-switch de presupuesto corre ANTES de
invocar Whisper. Con un max_api_cost_per_job forzado bajo, el job aborta con
BudgetAbort y el modelo jamás se carga (ni se llama model.transcribe).

Se mockea el modelo (no se descarga el modelo real en CI).
"""

from __future__ import annotations

import sys
import wave
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import transcriber, cost_guard  # noqa: E402
from core.errors import BudgetAbort, TranscriptionError  # noqa: E402
import config  # noqa: E402


def _write_silent_wav(path: Path, seconds: float = 1.0, sr: int = 16000) -> Path:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(b"\x00\x00" * int(sr * seconds))
    return path


@pytest.fixture()
def wav_file(tmp_path: Path) -> Path:
    return _write_silent_wav(tmp_path / "silence.wav", seconds=2.0)


@pytest.fixture(autouse=True)
def _clean():
    yield
    cost_guard._TRACKERS.clear()


class TestProbeAndEstimate:
    def test_probe_duration_wav(self, wav_file: Path):
        assert transcriber.probe_duration(wav_file) == pytest.approx(2.0, abs=0.05)

    def test_estimate_scales_with_hours(self):
        # 1 hora de audio a $0.06/hora = $0.06
        assert transcriber.estimate_transcription_cost(3600) == pytest.approx(0.06)


class TestBudgetAbortBeforeWhisper:
    def test_aborts_before_model_load_when_budget_tiny(
        self, wav_file: Path, tmp_path: Path, monkeypatch
    ):
        """El caso que exige la spec: aborta ANTES de la llamada real."""
        # Fuerzo un presupuesto ridículamente bajo.
        import config as cfg
        monkeypatch.setitem(cfg._KEEP, "max_api_cost_per_job", 0.0000001)

        loaded = {"n": 0}
        called = {"n": 0}

        def fake_load(*args, **kwargs):
            loaded["n"] += 1
            raise AssertionError("No debe cargarse el modelo si falla el budget")

        def fake_transcribe(*args, **kwargs):
            called["n"] += 1
            raise AssertionError("No debe invocarse Whisper si falla el budget")

        monkeypatch.setattr(transcriber, "_get_model", fake_load)
        cost_guard.reset("job_tiny")

        with pytest.raises(BudgetAbort):
            transcriber.transcribe(wav_file, "job_tiny")

        assert loaded["n"] == 0, "Se cargó el modelo pese al budget abort"
        assert called["n"] == 0, "Se llamó a Whisper pese al budget abort"

    def test_aborts_when_prev_spend_leaves_no_room(
        self, wav_file: Path, tmp_path: Path, monkeypatch
    ):
        """Aunque el estimado sea chico, si el job ya gastó casi todo, aborta."""
        # Reconstruyo el config con el límite real pero el job ya gastó 99.9%.
        limit = float(config.get("max_api_cost_per_job", 1.50))
        cost_guard.reset("job_spent")
        cost_guard.spend(limit - 0.00001, job_id="job_spent", model="llm")

        called = {"n": 0}

        def fake_load(*args, **kwargs):
            called["n"] += 1
            raise AssertionError("No debe cargarse el modelo")

        monkeypatch.setattr(transcriber, "_get_model", fake_load)

        with pytest.raises(BudgetAbort):
            transcriber.transcribe(wav_file, "job_spent")
        assert called["n"] == 0


class TestSuccessfulTranscription:
    def test_transcribes_and_registers_cost(self, wav_file: Path, tmp_path: Path, monkeypatch):
        cost_guard.reset("job_ok")

        class Seg:
            start = 0.0
            end = 1.0
            text = "hola mundo"
            words = [type("W", (), {"start": 0.0, "end": 0.3, "word": "hola"})(),
                     type("W", (), {"start": 0.4, "end": 1.0, "word": "mundo"})()]

        class FakeModel:
            def transcribe(self, path, **kwargs):
                return iter([Seg()]), None

        calls = {"n": 0}

        def fake_load(*args, **kwargs):
            calls["n"] += 1
            return FakeModel()

        monkeypatch.setattr(transcriber, "_get_model", fake_load)

        segments = transcriber.transcribe(wav_file, "job_ok")
        assert calls["n"] == 1
        assert len(segments) == 1
        assert segments[0].text == "hola mundo"
        assert segments[0].words[1]["word"] == "mundo"
        # El gasto se registró.
        assert cost_guard.spent_total("job_ok") > 0.0

    def test_whisper_exception_wrapped(self, wav_file: Path, tmp_path: Path, monkeypatch):
        cost_guard.reset("job_err")

        class Boom(Exception):
            pass

        def fake_load(*args, **kwargs):
            raise Boom("modelo roto")

        monkeypatch.setattr(transcriber, "_get_model", fake_load)

        with pytest.raises(TranscriptionError):
            transcriber.transcribe(wav_file, "job_err")

    def test_missing_audio_raises(self, tmp_path: Path):
        with pytest.raises(TranscriptionError):
            transcriber.transcribe(tmp_path / "no_existe.wav", "job_missing")