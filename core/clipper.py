"""Recorte de clips (Fase 3): HookCandidate -> ClipWindow -> clip_000.mp4.

La ventana narrativa (15-90 s) la decide `core.clip_window` de forma
determinista; este módulo solo MATERIALIZA la ventana contra el medio:

- "streaming":  yt-dlp --download-sections "*{start}-{end}" +
  --force-keyframes-at-cuts. Descarga SOLO la sección temporal (nunca el video
  completo) y corta en keyframe exacto. Nombre determinista: clip_000.mp4, ...
- "direct_file": ffmpeg local contra el archivo crudo ya descargado en disco,
  corte con re-encode (preset/crf de config). El archivo crudo sigue la regla
  de oro de la Fase 1: vive en el tempdir del job y se borra al salir.

El LLM identifica el hook; el sistema determina la duración final del clip. No
se carga el video completo en RAM y se preservan audio y video sin tocar
resolución/aspect ratio ni aplicar filtros.

Salida: clips_raw/*.mp4 + clips_raw/manifest.json con los ClipManifestEntry.
Errores: fallo de herramienta -> FFmpegError/YouTubeAcquisitionError; ventana
imposible/hook inválido -> HookValidationError/InsufficientDurationError/
WindowError; filesystem -> ClipFilesystemError (todos con exit code existente).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import config
from core import clip_window as clip_window_mod
from core import ytdlp_opts
from core.errors import (
    ClipError,
    ClipFilesystemError,
    ExtractionError,
    FFmpegError,
    YouTubeAcquisitionError,
)
from models.schemas import ClipManifestEntry, ClipWindow, HookCandidate
from utils.logger import get_logger

log = get_logger()

_VIDEO_EXT = "mp4"

# Tolerancia de validación de la duración real del clip vs la solicitada
# (keyframes/re-encode pueden mover el corte unas décimas).
_DURATION_TOLERANCE_S = 1.0


def _run(
    cmd: list[str],
    *,
    what: str,
    ytdlp: bool = False,
    tool: str = "ffmpeg",
    cwd: str | None = None,
) -> None:
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=cwd,
        )
    except FileNotFoundError as exc:
        raise ClipFilesystemError(f"Binario no encontrado: {cmd[0]}") from exc
    if result.returncode != 0:
        raw = result.stderr or result.stdout or ""
        tail = raw.strip().splitlines()[-5:]
        if ytdlp:
            hint = ytdlp_opts.diagnostic_hint(raw)
            if hint:
                raise YouTubeAcquisitionError(
                    f"Fallo {what}\n  {hint}\n  cmd: {' '.join(cmd)}\n  "
                    + "\n  ".join(tail[-5:])
                )
        error_cls = FFmpegError if tool == "ffmpeg" else ClipError
        raise error_cls(
            f"Fallo {what}\n  cmd: {' '.join(cmd)}\n"
            + "\n".join(tail[-5:])
        )


def _format_ts(seconds: float) -> str:
    """'12.345' -> '00:00:12.345' (formato esperado por yt-dlp --download-sections)."""
    ms = round(seconds * 1000)
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def _probe_duration(path: Path) -> float:
    """Duración del archivo vía ffprobe (0.0 si no se puede determinar)."""
    ffprobe = str(config.get("bins.ffprobe", "ffprobe"))
    try:
        result = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
    except (FileNotFoundError, ValueError):
        pass
    return 0.0


def probe_duration(path: Path) -> float:
    """API pública: duración (s) del medio; 0.0 si es indeterminada."""
    return _probe_duration(path)


def probe_dimensions(path: Path) -> tuple[int, int]:
    """Ancho x alto del primer stream de video vía ffprobe; (0, 0) si no lo hay.

    API pública, usada por core.vertical para la composición (Fase 4).
    """
    ffprobe = str(config.get("bins.ffprobe", "ffprobe"))
    try:
        result = subprocess.run(
            [ffprobe, "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height",
             "-of", "csv=p=0:s=x", str(path)],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        if result.returncode == 0 and result.stdout.strip():
            raw = result.stdout.strip().split("x")
            if len(raw) == 2:
                return int(raw[0]), int(raw[1])
    except (FileNotFoundError, ValueError):
        pass
    return (0, 0)


def _probe_has_video(path: Path) -> bool:
    """Detecta si el archivo tiene stream de video (ffprobe)."""
    ffprobe = str(config.get("bins.ffprobe", "ffprobe"))
    try:
        result = subprocess.run(
            [ffprobe, "-v", "error", "-select_streams", "v",
             "-show_entries", "stream=codec_type", "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        return "video" in result.stdout
    except (FileNotFoundError, ValueError):
        return True  # optimista: si no se puede probar, asumir video


def _probe_has_audio(path: Path) -> bool:
    """Detecta si el archivo tiene stream de audio (ffprobe)."""
    ffprobe = str(config.get("bins.ffprobe", "ffprobe"))
    try:
        result = subprocess.run(
            [ffprobe, "-v", "error", "-select_streams", "a",
             "-show_entries", "stream=codec_type", "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        return "audio" in result.stdout
    except (FileNotFoundError, ValueError):
        return False


def _validate_clip(clip_path: Path, window: ClipWindow, index: int) -> float:
    """Valida el clip REAL (no basta con exit code 0 de FFmpeg).

    Comprueba existencia, tamaño > 0 y duración real vía ffprobe dentro del
    rango 15-90 s (tolerancia por keyframes). Devuelve la duración medida.
    """
    if not clip_path.exists():
        raise ClipFilesystemError(f"clip {index} no existe: {clip_path}")
    try:
        size = clip_path.stat().st_size
    except OSError as exc:
        raise ClipFilesystemError(f"no se pudo leer el clip {index}: {exc}") from exc
    if size <= 0:
        raise ClipFilesystemError(f"clip {index} vacío (0 bytes): {clip_path}")

    duration = _probe_duration(clip_path)
    if duration <= 0:
        raise FFmpegError(f"clip {index} sin duración válida (ffprobe): {clip_path}")

    min_d = float(config.get("clip.min_duration_seconds", 15))
    max_d = float(config.get("clip.max_duration_seconds", 90))
    if duration < min_d - _DURATION_TOLERANCE_S:
        raise FFmpegError(
            f"clip {index} dura {duration:.3f}s < mínimo {min_d}s "
            f"(ventana solicitada {window.start:.3f}-{window.end:.3f})"
        )
    if duration > max_d + _DURATION_TOLERANCE_S:
        raise FFmpegError(
            f"clip {index} dura {duration:.3f}s > máximo {max_d}s"
        )
    return duration


def _clip_streaming(url: str, window: ClipWindow, index: int, dest_dir: Path) -> Path:
    """Materializa una ClipWindow del streaming con yt-dlp (descarga la sección).

    Retry con backoff exponencial: si yt-dlp falla (ej: 403 por URL firmada
    expirada), reintentos hasta section_max_retries veces con delay 1s, 2s, 4s.
    Cada retry genera una URL firmada nueva (yt-dlp la resuelve de nuevo).
    """
    import time as _time

    dest_dir.mkdir(parents=True, exist_ok=True)
    out_tpl = str(dest_dir / f"clip_{index:03d}.%(ext)s")

    section = f"*{_format_ts(window.start)}-{_format_ts(window.end)}"
    max_retries = int(config.get("youtube.section_max_retries", 3))

    last_error = None
    for attempt in range(max_retries + 1):
        cmd = [
            sys.executable, "-m", "yt_dlp",
            *ytdlp_opts.common_ytdlp_args(),
            "-f", "bestvideo*+bestaudio/best",
            "--download-sections", section,
            "--force-keyframes-at-cuts",
            "--merge-output-format", _VIDEO_EXT,
            "-o", out_tpl,
            "--no-playlist",
            url,
        ]
        try:
            _run(cmd, what=f"recorte (yt-dlp) del clip {index}", ytdlp=True, tool="yt-dlp")
            out = dest_dir / f"clip_{index:03d}.{_VIDEO_EXT}"
            if not out.exists():
                found = list(dest_dir.glob(f"clip_{index:03d}.*"))
                if not found:
                    raise FFmpegError(f"yt-dlp no produjo el clip {index}")
                out = found[0]
            return out
        except (FFmpegError, ExtractionError) as exc:
            last_error = exc
            if attempt < max_retries:
                delay = 2 ** attempt  # 1s, 2s, 4s
                log.warn(
                    "section_retry",
                    clip_index=index,
                    attempt=attempt + 1,
                    max_retries=max_retries,
                    delay_s=delay,
                    error=str(exc)[:200],
                )
                _time.sleep(delay)
                # Limpiar archivo parcial si existe
                for f in dest_dir.glob(f"clip_{index:03d}.*"):
                    f.unlink(missing_ok=True)
            else:
                log.error(
                    "section_failed",
                    clip_index=index,
                    attempts=attempt + 1,
                    error=str(exc)[:200],
                )
    raise last_error


def _clip_direct(raw_path: Path, window: ClipWindow, index: int, dest_dir: Path) -> Path:
    """Materializa una ClipWindow con ffmpeg local (re-encode)."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = dest_dir / f"clip_{index:03d}.{_VIDEO_EXT}"

    start = window.start
    duration = window.duration
    if duration <= 0:
        raise FFmpegError(f"Ventana degenerada para el clip {index} (dur<=0)")

    ffmpeg = str(config.get("bins.ffmpeg", "ffmpeg"))
    preset = str(config.get("ffmpeg.preset", "medium"))
    crf = int(config.get("ffmpeg.crf", 23))

    cmd = [
        ffmpeg, "-y",
        "-ss", f"{start:.3f}",
        "-i", str(raw_path),
        "-t", f"{duration:.3f}",
    ]
    if _probe_has_video(raw_path):
        cmd += ["-c:v", "libx264", "-preset", preset, "-crf", str(crf),
                "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k"]
    else:
        # Fuente solo de audio: codec de audio válido en contenedor mp4.
        cmd += ["-vn", "-c:a", "aac", "-b:a", "128k"]
    cmd.append(str(out))

    _run(cmd, what=f"recorte (ffmpeg) del clip {index}", tool="ffmpeg")
    if not out.exists():
        raise FFmpegError(f"ffmpeg no produjo el clip {index}")
    return out


def _resolve_media_duration(
    source_type: str,
    hooks: list[HookCandidate],
    raw_path: Path | None,
    media_duration: float | None,
) -> float:
    """Obtiene la duración del medio para construir las ventanas.

    Prioridad: valor explícito -> probe del archivo crudo (direct_file) ->
    fallback conservador (suficiente para alojar la ventana máxima tras el
    último hook), documentado como limitación si se usa.
    """
    if media_duration is not None and media_duration > 0:
        return float(media_duration)
    if source_type == "direct_file" and raw_path is not None:
        probed = _probe_duration(Path(raw_path))
        if probed > 0:
            return probed
    fallback = max(h.end for h in hooks) + float(
        config.get("clip.max_duration_seconds", 90)
    )
    log.warn("media_duration_unknown", assumed_seconds=round(fallback, 3))
    return fallback


def clip_hooks(
    source: str,
    source_type: str,
    hooks: list[HookCandidate],
    job_id: str,
    *,
    raw_path: Path | None = None,
    media_duration: float | None = None,
    words: list | None = None,
) -> list[ClipManifestEntry]:
    """Convierte hooks en ventanas y las materializa en outputs/<job_id>/clips_raw/.

    Devuelve los ClipManifestEntry y escribe clips_raw/manifest.json.
    Si no hay hooks (o ninguno produce ventana válida), no genera clips.

    `words`: word timestamps aplanados del transcript para el snapping de
    boundaries (None = legacy sin snapping; [] con hooks = WindowError).
    """
    if not hooks:
        log.info("clips_skipped", job_id=job_id, reason="no_hooks")
        return []

    dest_dir = config.output_dir() / job_id / "clips_raw"
    try:
        dest_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ClipFilesystemError(f"no se pudo crear {dest_dir}: {exc}") from exc

    resolved_duration = _resolve_media_duration(
        source_type, hooks, raw_path, media_duration
    )

    # SEMANTIC DETECTION (hooks) -> WINDOW SELECTION (determinista).
    selected = clip_window_mod.build_clip_windows(
        hooks, resolved_duration, words=words
    )
    if not selected:
        log.warn("no_clip_windows", job_id=job_id, hooks=len(hooks))
        write_clip_manifest(
            job_id, [], source_url=source, source_type=source_type
        )
        return []

    entries: list[ClipManifestEntry] = []
    section_delay = float(config.get("youtube.section_delay_seconds", 3))
    for index, (hook, window) in enumerate(selected):
        if source_type == "streaming":
            clip_path = _clip_streaming(source, window, index, dest_dir)
            # Delay entre descargas de secciones para evitar 403 por URLs
            # firmadas que expiran. No se aplica al último clip.
            if section_delay > 0 and index < len(selected) - 1:
                import time as _time
                log.info(
                    "section_delay",
                    job_id=job_id,
                    clip_index=index,
                    delay_s=section_delay,
                )
                _time.sleep(section_delay)
        else:
            if raw_path is None:
                raise ClipError(
                    f"direct_file requiere el path del archivo crudo (clip {index})"
                )
            clip_path = _clip_direct(Path(raw_path), window, index, dest_dir)

        # Validación del output REAL (existe, tamaño, duración vía ffprobe).
        duration = _validate_clip(clip_path, window, index)
        has_video = _probe_has_video(clip_path)
        has_audio = _probe_has_audio(clip_path)

        entry = ClipManifestEntry(
            clip_id=f"clip_{index:03d}",
            path=str(clip_path.relative_to(config.output_dir())),
            hook=hook,
            duration=round(duration, 3),
            source_start=round(window.start, 3),
            source_end=round(window.end, 3),
            resolution="source",
            format=_VIDEO_EXT,
            qc_status="passed" if has_video else "pending",
        )
        entries.append(entry)
        log.info(
            "clip_ready",
            job_id=job_id,
            clip=entry.clip_id,
            path=str(clip_path),
            duration_s=round(duration, 2),
            window_start=round(window.start, 2),
            window_end=round(window.end, 2),
            hook_start=round(hook.start, 2),
            hook_end=round(hook.end, 2),
            hook_offset=round(window.hook_offset, 2),
            has_video=has_video,
            has_audio=has_audio,
        )

    write_clip_manifest(
        job_id, entries, source_url=source, source_type=source_type
    )
    return entries


def write_clip_manifest(
    job_id: str,
    entries: list[ClipManifestEntry],
    *,
    source_url: str | None = None,
    source_type: str | None = None,
) -> Path:
    """Persiste clips_raw/manifest.json (source + clips).

    El manifest permite reproducir qué decisión produjo cada clip: source,
    hook, start/end de la ventana, duración real, output path y status.
    """
    dest_dir = config.output_dir() / job_id / "clips_raw"
    try:
        dest_dir.mkdir(parents=True, exist_ok=True)
        path = dest_dir / "manifest.json"
        payload = {
            "job_id": job_id,
            "source": source_url,
            "source_type": source_type,
            "clips": [e.model_dump() for e in entries],
        }
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError as exc:
        raise ClipFilesystemError(f"no se pudo escribir el manifest: {exc}") from exc
    return path
