"""Transcripción con faster-whisper, con timestamps a nivel de palabra.

Regla crítica de la Fase 1: ANTES de invocar Whisper se estima el costo según
la duración del audio y se consulta cost_guard.check_budget(). Si el estimado
revienta el presupuesto del job, se aborta con BudgetAbort SIN gastar un
centavo (ni siquiera se carga el modelo).
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import config
from core import cost_guard
from core.errors import BudgetAbort, TranscriptionError
from models.schemas import TranscriptSegment
from utils.logger import get_logger

log = get_logger()

_DURATION_RE = re.compile(r"Duration:\s*(\d+):(\d+):(\d+)(?:\.(\d+))?")


def probe_duration(audio_path: Path) -> float:
    """Duración en segundos vía ffprobe/ffmpeg (sin cargar el audio en RAM)."""
    audio_path = Path(audio_path)
    ffprobe = str(config.get("bins.ffprobe", "ffprobe"))
    try:
        result = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(audio_path)],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
    except (FileNotFoundError, ValueError):
        pass

    # Fallback: parsear el header "Duration:" del stderr de ffmpeg.
    ffmpeg = str(config.get("bins.ffmpeg", "ffmpeg"))
    try:
        result = subprocess.run(
            [ffmpeg, "-i", str(audio_path)],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
    except FileNotFoundError as exc:
        raise TranscriptionError("ffmpeg no está disponible para probar el audio") from exc
    match = _DURATION_RE.search(result.stderr or "")
    if not match:
        raise TranscriptionError(f"No se pudo determinar la duración de {audio_path}")
    h, m, s, ms = (int(match.group(i) or 0) for i in (1, 2, 3, 4))
    return h * 3600 + m * 60 + s + ms / 100.0


def estimate_transcription_cost(seconds: float) -> float:
    """Costo estimado (USD) de transcribir `seconds` de audio con Whisper."""
    rate_per_hour = float(config.get("whisper.cost_per_hour_usd", 0.06))
    return (seconds / 3600.0) * rate_per_hour


def _get_model(model_name: str, device: str, compute_type: str):
    """Cargador del modelo aislado para poder mockearlo en tests."""
    from faster_whisper import WhisperModel  # import tardío: carga pesada

    return WhisperModel(model_name, device=device, compute_type=compute_type)


def _segments_from_result(segments_iter, info) -> list[TranscriptSegment]:
    """Convierte el iterador de faster-whisper a TranscriptSegment con words."""
    result: list[TranscriptSegment] = []
    for seg in segments_iter:
        words = []
        for w in (seg.words or []):
            words.append({"start": w.start, "end": w.end, "word": w.word})
        result.append(
            TranscriptSegment(
                start=seg.start,
                end=seg.end,
                text=seg.text.strip(),
                words=words,
            )
        )
    return result


def transcribe(
    audio_path: Path,
    job_id: str,
    *,
    model_name: str | None = None,
    device: str = "auto",
    compute_type: str = "auto",
) -> list[TranscriptSegment]:
    """Transcribe audio -> list[TranscriptSegment]. Aborta si el presupuesto no da."""
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise TranscriptionError(f"Audio no encontrado: {audio_path}")

    # 1. Duración + estimación de costo.
    duration = probe_duration(audio_path)
    estimated = estimate_transcription_cost(duration)

    # 2. Kill-switch ANTES de tocar Whisper (ni siquiera carga el modelo).
    if not cost_guard.check_budget(estimated, job_id):
        spent = cost_guard.spent_total(job_id)
        limit = float(config.get("max_api_cost_per_job", 1.50))
        log.error(
            "budget_abort",
            job_id=job_id,
            stage="transcribe",
            estimated_usd=round(estimated, 4),
            spent_usd=round(spent, 4),
            limit_usd=limit,
        )
        raise BudgetAbort(
            f"Transcripción estimada ${estimated:.4f} excede el presupuesto restante "
            f"de ${max(0.0, limit - spent):.4f} del job {job_id}"
        )

    # 3. Whisper real.
    try:
        model = _get_model(
            model_name or str(config.get("models.whisper", "base")),
            device=device,
            compute_type=compute_type,
        )
        segments_iter, info = model.transcribe(
            str(audio_path),
            vad_filter=True,
            word_timestamps=True,
        )
        segments = _segments_from_result(segments_iter, info)
    except Exception as exc:  # noqa: BLE001 - cualquier fallo de whisper
        raise TranscriptionError(f"Fallo de transcripción con Whisper: {exc}") from exc

    # 4. Registrar el gasto real; si pulsa el límite acumulado, igual abortamos
    #    antes de cualquier OTRA llamada paga (el próximo check_budget lo frena).
    cost_guard.spend(estimated, job_id, model="whisper")

    log.info(
        "transcribed",
        job_id=job_id,
        stage="transcribe",
        segments=len(segments),
        duration_s=round(duration, 2),
        cost_usd=round(estimated, 4),
    )
    return segments