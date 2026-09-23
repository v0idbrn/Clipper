"""AutoClipper — Entry point CLI.

Uso:
    python main.py run --url <URL> --n-clips N

Fase 3: ingesta + transcripcion + hooks + recorte quirurgico end-to-end.
Fase 4: composicion vertical 9:16.
Fase 5: subtitulos dinamicos ASS quemados en clips verticales.

Salida: outputs/<job_id>/transcript.json + hooks.json + clips_raw/*.mp4 +
clips_vertical/*.mp4 + clips_captioned/*.mp4 (con subtitulos ASS quemados,
Fase 5).
"""

from __future__ import annotations

import json
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import typer

import config
from core import cost_guard
from core.clipper import clip_hooks, probe_duration
from core.downloader_direct import download_full
from core.errors import AutoClipperError
from core.exit_codes import ExitCode
from core.extractor_audio import extract_local_audio, extract_stream_audio
from core.llm_analyzer import find_hooks
from core.source_router import detect_source_type
from core.transcriber import transcribe
from core.vertical import render_vertical_clips
from core.captions import captions_for_window
from core.subtitles import render_captioned_clips, write_captioned_manifest
from utils.logger import get_logger
from utils.temp_manager import temp_dir

app = typer.Typer(add_completion=False, no_args_is_help=True)

log = get_logger()

EXIT_OK = ExitCode.OK
EXIT_INVALID_URL = ExitCode.INVALID_URL

_URL_RE = re.compile(r"^https?://", re.IGNORECASE)


@app.callback()
def _main_callback() -> None:
    """AutoClipper — pipeline de clipping de video."""


def validate_url(url: str) -> bool:
    """Validación mínima: URL http(s), o path local existente (archivo crudo).

    El input del pipeline es "URL o archivo"; un path local de un .mp4/.wav/etc.
    es tan válido como un link de YouTube.
    """
    candidate = Path(url)
    if candidate.exists() and candidate.is_file():
        return True
    try:
        parts = urlparse(url)
    except ValueError:
        return False
    return bool(_URL_RE.match(url)) and bool(parts.netloc)


def _is_local_path(source: str) -> bool:
    p = Path(source)
    return p.exists() and p.is_file()


def _new_job_id(url: str) -> str:
    """job_id con short-hash de la URL + timestamp (idempotencia en Fase 3+)."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"job_{stamp}_{uuid.uuid4().hex[:8]}"


def _write_transcript(job_id: str, source_url: str, source_type: str,
                      segments: list) -> Path:
    out_dir = config.output_dir() / job_id
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "job_id": job_id,
        "source_url": source_url,
        "source_type": source_type,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "segments": [s.model_dump() for s in segments],
    }
    path = out_dir / "transcript.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def _write_hooks(job_id: str, source_url: str, source_type: str,
                 hooks: list) -> Path:
    out_dir = config.output_dir() / job_id
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "job_id": job_id,
        "source_url": source_url,
        "source_type": source_type,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "llm_mode": config.get("llm.mode", "mock"),
        "hooks": [h.model_dump() for h in hooks],
    }
    path = out_dir / "hooks.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


@app.command()
def run(
    url: str = typer.Option(..., "--url", help="URL del video o archivo a procesar."),
    n_clips: int = typer.Option(
        5, "--n-clips", min=1, help="Cantidad de clips a generar."
    ),
) -> None:
    """Ejecuta el pipeline completo hasta el recorte de clips (Fase 3)."""
    job_id = _new_job_id(url)
    log.info("job_started", job_id=job_id, url=url, n_clips=n_clips)

    if not validate_url(url):
        log.error("invalid_url", job_id=job_id, url=url)
        raise typer.Exit(EXIT_INVALID_URL)

    if n_clips < 1:
        log.error("invalid_n_clips", job_id=job_id, n_clips=n_clips)
        raise typer.Exit(EXIT_INVALID_URL)

    try:
        source_type = detect_source_type(url)
        log.info(
            "source_detected",
            job_id=job_id,
            source_type=source_type,
            url=url,
        )

        # El tempdir ES el ciclo de vida: al salir, se borran el audio maestro
        # y el archivo crudo descargado (Fase 1 spec: "se borra al terminar").
        raw_path = None
        with temp_dir(prefix="autoclipper_") as tmp:
            if source_type == "streaming":
                audio = extract_stream_audio(url, tmp.path)
            else:
                # direct_file: primero descarga completa a disco (con chequeo
                # de espacio ANTES), recién después ffmpeg local. Si ya es un
                # path local, no hay nada que descargar.
                if _is_local_path(url):
                    audio = extract_local_audio(Path(url), tmp.path)
                    raw_path = Path(url)
                else:
                    raw = download_full(url, tmp.path)
                    raw_path = raw
                    log.info(
                        "raw_downloaded",
                        job_id=job_id,
                        raw=str(raw),
                        size_bytes=raw.stat().st_size,
                    )
                    audio = extract_local_audio(raw, tmp.path)

            log.info(
                "audio_ready",
                job_id=job_id,
                audio=str(audio),
                size_bytes=audio.stat().st_size,
            )

            segments = transcribe(audio, job_id)
            hooks = find_hooks(segments, job_id, n_candidates=n_clips)
            transcript_path = _write_transcript(
                job_id, url, source_type, segments
            )
            hooks_path = _write_hooks(job_id, url, source_type, hooks)

            # Fase 3: ventana narrativa determinista (15-90 s) + recorte real.
            # La duración del medio se toma del audio maestro (equivale a la del
            # video para ambas rutas); evita otra descarga/consulta.
            # P1: aplanar word timestamps para alinear source_start/source_end
            # a límites reales de palabra (evita cortes a mitad de palabra).
            media_duration = probe_duration(audio)
            words = [
                word
                for segment in segments
                for word in segment.words
            ]
            clips = clip_hooks(
                url,
                source_type,
                hooks,
                job_id,
                raw_path=raw_path,
                media_duration=media_duration or None,
                words=words,
            )

            # Fase 4: composición vertical 9:16 de cada clip horizontal
            # (clips_vertical/, manifest propio trazable al source clip).
            vertical_clips = render_vertical_clips(clips, job_id) if clips else []

            # Fase 5: subtitulos dinamicos ASS quemados en clips verticales
            # cuando subtitles.enabled esta activo en config.
            captioned_clips = []
            if clips and config.get("subtitles.enabled", False):
                for clip_entry, vc in zip(clips, vertical_clips):
                    clip_caps = captions_for_window(
                        segments, clip_entry.source_start, clip_entry.source_end
                    )
                    if clip_caps:
                        captioned_clips.append((vc, clip_caps))
                if captioned_clips:
                    cap_verticals = [vc for vc, _ in captioned_clips]
                    cap_dict = {vc.clip_id: caps for vc, caps in captioned_clips}
                    results = render_captioned_clips(cap_verticals, cap_dict, job_id)
                    write_captioned_manifest(job_id, results)

        total_cost = round(cost_guard.spent_total(job_id), 6)
        cost_guard.reset(job_id)
        clips_dir = config.output_dir() / job_id / "clips_raw"
        vertical_dir = config.output_dir() / job_id / "clips_vertical"
        captioned_dir = config.output_dir() / job_id / "clips_captioned"
        log.info(
            "job_completed",
            job_id=job_id,
            stage="clips",
            transcript=str(transcript_path),
            hooks=str(hooks_path),
            hooks_count=len(hooks),
            clips_count=len(clips),
            clips_raw=str(clips_dir),
            vertical_clips_count=len(vertical_clips),
            vertical_9x16=str(vertical_dir),
            captioned_clips_count=len(captioned_clips) if captioned_clips else 0,
            clips_captioned=str(captioned_dir),
            total_cost_usd=total_cost,
            segments=len(segments),
        )
    except AutoClipperError as exc:
        log.error(exc.__class__.__name__.lower(), job_id=job_id, error=exc.message)
        cost_guard.reset(job_id)
        raise typer.Exit(int(exc.exit_code))
    except Exception as exc:  # noqa: BLE001
        log.error("unhandled_error", job_id=job_id, error=str(exc))
        cost_guard.reset(job_id)
        raise typer.Exit(int(ExitCode.GENERIC))

    raise typer.Exit(EXIT_OK)


def main() -> None:
    try:
        app()
    except AutoClipperError as exc:
        log.error(exc.__class__.__name__.lower(), error=exc.message)
        sys.exit(int(exc.exit_code))
    except Exception:  # noqa: BLE001
        log.error("unhandled_error")
        sys.exit(int(ExitCode.GENERIC))


if __name__ == "__main__":
    main()