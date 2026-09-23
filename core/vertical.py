"""Composición vertical 9:16 (Fase 4): clip horizontal -> clip vertical.

Estrategia V1, determinista y CPU-first ("center_crop"):

    clip horizontal (W x H)  --clips_raw/clip_000.mp4--
        |
        v
    calculate_vertical_crop(W, H)   [función pura, testeable sin FFmpeg]
        -> CropGeometry: region central de aspecto 9:16 + resolución de salida
        |
        v
    UNA sola pasada de FFmpeg  (crop + scale + libx264 re-encode, audio copiado)
        |
        v
    clip vertical validado con ffprobe (dimensiones, audio, duración)
        -> VerticalClipEntry + clips_vertical/manifest.json

El clip horizontal se conserva en clips_raw/ como evidencia/intermedio; el
vertical vive en clips_vertical/ con el mismo índice (clip_000.mp4, ...).

Reglas de calidad:
- NADA de deformación: el scale es uniforme (el crop ya es 9:16).
- Resolución objetivo: clip.output_resolution (1080x1920 por defecto). Se
  llena SOLO si el factor de escala no supera vertical.max_upscale; si no,
  la salida queda más chica (también 9:16) para no inventar detalle.
- Audio copiado (-c:a copy): no se re-comprime lo que ya está bien.
- YUV420p + dimensiones pares: MP4 reproducible en cualquier reproductor.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import config
from core.clipper import (
    _probe_has_audio,
    _probe_has_video,
    _run,
    probe_dimensions,
    probe_duration,
)
from core.errors import ClipError, ClipFilesystemError, FFmpegError
from models.schemas import ClipManifestEntry, CropGeometry, VerticalClipEntry
from utils.logger import get_logger

log = get_logger()

_VIDEO_EXT = "mp4"

# Tolerancia de duración del vertical vs su source (el re-encode mueve unas
# décimas igual que en Fase 3).
_DURATION_TOLERANCE_S = 1.0

# Límite de upscaling para NO inventar detalle innecesariamente. Un factor
# mayor produce salida menor (siempre 9:16) en vez de enormes interpolaciones.
MAX_UPSCALE = 2.5

# Modos soportados. Solo "center_crop" existe en V1; otra cadena = error
# explícito (no aceptar silenciosamente una composición que no implementamos).
SUPPORTED_MODES = ("center_crop",)


def _parse_resolution(raw: str, default: str = "1080x1920") -> tuple[int, int]:
    """'1080x1920' -> (1080, 1920). Invalida config corrupta."""
    value = str(raw or default).strip().lower()
    try:
        w, h = value.split("x")
        return int(w), int(h)
    except (ValueError, AttributeError) as exc:
        raise ClipError(
            f"clip.output_resolution inválido: {raw!r} (esperado 'WxH')"
        ) from exc


def _to_even(value: int | float) -> int:
    """Entero par >= 2 más cercano (mínimo seguro para yuv420p)."""
    n = max(2, int(round(float(value))))
    return n if n % 2 == 0 else n + 1


def calculate_vertical_crop(
    width: int,
    height: int,
    *,
    target_width: int | None = None,
    target_height: int | None = None,
    max_upscale: float | None = None,
    mode: str = "center_crop",
) -> CropGeometry:
    """Geometría determinista del crop central 9:16 para `width x height`.

    Pura (sin I/O, sin FFmpeg), reproducible y validada:
    - recorta de la fuente una region central con relación 9:16;
    - scale uniforme (sin deformación) hasta la resolución objetivo, acotado
      por `max_upscale` para no inventar detalle;
    - devuelve CropGeometry (dimensiones pares en la salida).

    Levanta ValueError si el modo es desconocido o los inputs son inválidos
    (no finitos, <= 0, no enteros, max_upscale < 1).
    """
    if mode not in SUPPORTED_MODES:
        raise ValueError(f"modo de composición no soportado: {mode!r}")

    target_width = 1080 if target_width is None else target_width
    target_height = 1920 if target_height is None else target_height

    for name, v in (
        ("width", width), ("height", height),
        ("target_width", target_width),
        ("target_height", target_height),
    ):
        try:
            f = float(v)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} no es numérico: {v!r}") from exc
        if not math.isfinite(f) or f <= 0:
            raise ValueError(f"{name} debe ser finito y > 0: {v!r}")
        if f != int(f):
            raise ValueError(f"{name} debe ser entero: {v!r}")

    tw, th = int(target_width or 1080), int(target_height or 1920)
    src_w, src_h = int(width), int(height)
    mu = float(max_upscale if max_upscale is not None else MAX_UPSCALE)
    if not math.isfinite(mu) or mu < 1.0:
        raise ValueError(f"max_upscale debe ser >= 1.0: {max_upscale!r}")

    # El crop central de la fuente: siempre recorta de la fuente una region
    # central de aspecto 9:16. `crop_w`/`crop_h` son pares (seguro yuv420p).
    if src_w / src_h >= 9 / 16:
        # Fuente más ancha que 9:16: altura completa, ancho recortado (centrado).
        crop_h = src_h
        crop_w = min(src_w, _to_even(round(src_h * 9 / 16)))
    else:
        # Fuente más alta que 9:16: ancho completo, alto recortado (centrado).
        crop_w = src_w
        crop_h = min(src_h, _to_even(round(src_w * 16 / 9)))
    crop_x = (src_w - crop_w) // 2
    crop_y = (src_h - crop_h) // 2

    # Escala uniforme para llenar el target. `MAX_UPSCALE` acota el upscaling:
    # si el factor necesario supera el límite, la salida queda MÁS CHICA pero
    # siempre 9:16 y con la misma proporción (no deforma, no inventa detalle).
    scale = max(tw / crop_w, th / crop_h)
    out_w, out_h = tw, th
    if scale > MAX_UPSCALE:
        # Sin upscaling absurdo: salida acotada por el límite, mis dimensiones
        # siguen siendo 9:16 (el factor es el de escala uniforme).
        factor = MAX_UPSCALE / scale
        out_w = _to_even(crop_w * scale * factor)
        out_h = _to_even(crop_h * scale * factor)

    return CropGeometry(
        source_width=src_w,
        source_height=src_h,
        crop_width=crop_w,
        crop_height=crop_h,
        crop_x=crop_x,
        crop_y=crop_y,
        target_width=out_w,
        target_height=out_h,
    )


def _validate_vertical(
    clip_path: Path, geom: CropGeometry, source_duration: float
) -> float:
    """Valida el vertical REAL (no basta el exit code de FFmpeg).

    Existe, tamaño > 0, dimensión exacta (width x height == geom.target), hay
    audio, hay video y la duración es equivalente a la del source.
    """
    if not clip_path.exists():
        raise ClipFilesystemError(f"clip vertical no existe: {clip_path}")
    try:
        size = clip_path.stat().st_size
    except OSError as exc:
        raise ClipFilesystemError(
            f"no se pudo leer el clip vertical: {exc}"
        ) from exc
    if size <= 0:
        raise ClipFilesystemError(f"clip vertical vacío (0 bytes): {clip_path}")

    width, height = probe_dimensions(clip_path)
    if (width, height) != (geom.target_width, geom.target_height):
        raise FFmpegError(
            f"clip vertical {width}x{height} != esperado "
            f"{geom.target_width}x{geom.target_height}: {clip_path}"
        )
    if not _probe_has_video(clip_path):
        raise FFmpegError(f"clip vertical sin stream de video: {clip_path}")
    if not _probe_has_audio(clip_path):
        raise FFmpegError(f"clip vertical sin stream de audio: {clip_path}")

    duration = probe_duration(clip_path)
    if duration <= 0:
        raise FFmpegError(f"clip vertical sin duración válida (ffprobe): {clip_path}")
    if abs(duration - source_duration) > _DURATION_TOLERANCE_S:
        raise FFmpegError(
            f"clip vertical dura {duration:.3f}s, source {source_duration:.3f}s "
            f"(tolerancia {_DURATION_TOLERANCE_S:.1f}s)"
        )
    return duration


def render_vertical(
    entry: ClipManifestEntry,
    job_id: str,
    *,
    target_width: int | None = None,
    target_height: int | None = None,
    max_upscale: float | None = None,
) -> VerticalClipEntry:
    """Renderiza el vertical de UN clip horizontal (una sola pasada FFmpeg).

    Reads the source clip (clips_raw), calcula la geometría 9:16, corre
    FFmpeg a clips_vertical/<clip_id>.mp4 y valida el output real con ffprobe.
    """
    source_clip = config.output_dir() / entry.path
    if not source_clip.exists():
        raise ClipFilesystemError(f"source clip no existe: {source_clip}")

    width, height = probe_dimensions(source_clip)
    if (width, height) == (0, 0):
        raise FFmpegError(f"source clip sin video (imposible componer vertical): {source_clip}")

    mode = str(config.get("vertical.mode", "center_crop"))
    if mode not in SUPPORTED_MODES:
        raise ClipError(f"vertical.mode no soportado: {mode!r}")

    if target_width is None or target_height is None:
        target_width, target_height = _parse_resolution(
            config.get("clip.output_resolution", "1080x1920")
        )
    if max_upscale is None:
        max_upscale = float(config.get("vertical.max_upscale", MAX_UPSCALE))
    geom = calculate_vertical_crop(
        width, height,
        target_width=target_width,
        target_height=target_height,
        max_upscale=max_upscale,
        mode=mode,
    )

    dest_dir = config.output_dir() / job_id / "clips_vertical"
    dest_dir.mkdir(parents=True, exist_ok=True)
    out_path = dest_dir / f"{entry.clip_id}.{_VIDEO_EXT}"

    ffmpeg = str(config.get("bins.ffmpeg", "ffmpeg"))
    preset = str(config.get("ffmpeg.preset", "medium"))
    crf = int(config.get("ffmpeg.crf", 23))
    cmd = [
        ffmpeg, "-y",
        "-i", str(source_clip),
        "-vf", f"{geom.ffmpeg_crop},scale={geom.target_width}:{geom.target_height},setsar=1",
        "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        "-c:a", "copy",
        "-movflags", "+faststart",
        str(out_path),
    ]
    _run(cmd, what=f"composición vertical de {entry.clip_id}", tool="ffmpeg")

    source_duration = entry.duration or probe_duration(source_clip)
    duration = _validate_vertical(out_path, geom, source_duration)

    vertical = VerticalClipEntry(
        clip_id=entry.clip_id,
        source_clip=str(source_clip.relative_to(config.output_dir())),
        source_width=width,
        source_height=height,
        output_width=geom.target_width,
        output_height=geom.target_height,
        aspect_ratio="9:16",
        composition_mode=mode,
        crop=geom,
        duration=round(duration, 3),
        path=str(out_path.relative_to(config.output_dir())),
        qc_status="passed",
    )
    log.info(
        "clip_vertical_ready",
        job_id=job_id,
        clip=vertical.clip_id,
        path=str(out_path),
        source=f"{width}x{height}",
        output=f"{geom.target_width}x{geom.target_height}",
        duration_s=round(vertical.duration, 2),
    )
    return vertical


def render_vertical_clips(
    entries: list[ClipManifestEntry], job_id: str
) -> list[VerticalClipEntry]:
    """Renderiza el vertical de todos los clips horizontales + manifest.

    Devuelve los VerticalClipEntry y escribe clips_vertical/manifest.json
    heredando source/source_type del manifest horizontal (clips_raw) para
    mantener la trazabilidad a la fuente original. Si no hay clips, no genera
    nada.
    """
    if not entries:
        log.info("vertical_skipped", job_id=job_id, reason="no_clips")
        return []

    # La fuente original ya quedó persistida por clip_hooks (clips_raw).
    source_url = None
    source_type = None
    raw_manifest = config.output_dir() / job_id / "clips_raw" / "manifest.json"
    if raw_manifest.exists():
        try:
            payload = json.loads(raw_manifest.read_text(encoding="utf-8"))
            source_url = payload.get("source")
            source_type = payload.get("source_type")
        except (OSError, ValueError) as exc:
            raise ClipFilesystemError(
                f"no se pudo leer el manifest horizontal: {exc}"
            ) from exc

    verticals = [render_vertical(e, job_id) for e in entries]
    write_vertical_manifest(
        job_id, verticals, source_url=source_url, source_type=source_type
    )
    return verticals


def write_vertical_manifest(
    job_id: str,
    entries: list[VerticalClipEntry],
    *,
    source_url: str | None = None,
    source_type: str | None = None,
) -> Path:
    """Persiste clips_vertical/manifest.json con el envelope de Fase 3.

    Misma forma que clips_raw/manifest.json (un solo "sistema" de manifest):
    cada entrada trae la trazabilidad completa (source clip, dimensiones,
    geometría de crop, output, modo, duración, path, status).
    """
    dest_dir = config.output_dir() / job_id / "clips_vertical"
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
        raise ClipFilesystemError(
            f"no se pudo escribir el manifest vertical: {exc}"
        ) from exc
    return path