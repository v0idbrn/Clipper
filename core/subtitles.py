"""Subtitulos dinamicos (Fase 5) - render ASS quemado en el clip vertical.

Motor elegido: ASS/libass (confirmado por usuario), por robustez: el texto es
UNA linea de Dialogue como dato (UTF-8) que libass renderiza con sus propias
reglas; nunca se interpreta como shell/expresion. Filename del filtro `ass`
con ruta RELATIVA al proyecto (evita el escape problematico del parser de
filtergraph en Windows, verificado empiricamente).

Contrato:
    vertical (VerticalClipEntry) + captions (list[CaptionSegment], clip-rel)
        ->
    render_captioned() -> SubtitleRenderResult
        -> clips_captioned/<clip_id>.mp4  (+ subtitles/<clip_id>.ass)

Reglas:
- Resolucion/audio/duracion idénticos al vertical (ffprobe lo valida).
- Source: resolve_font_family() = config o autodeteccion segura con fallback
  y fail-fast (nunca se asume una fuente Windows sin verificar el archivo).
- PlayRes = resolucion real del clip => el tamano de fuente/posicion escala sola.
- Contraste: contorno + sombra, texto grande, safe-area inferior.
- Determinismo: build_ass() es pura; el mismo input produce el mismo .ass.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import config
from core.captions import normalize_text
from core.clipper import (
    _probe_has_audio,
    _probe_has_video,
    _run,
    probe_dimensions,
    probe_duration,
)
from core.errors import (
    ClipError,
    ClipFilesystemError,
    SubtitleDataError,
    SubtitleFontError,
    SubtitleRenderError,
)
from models.schemas import CaptionSegment, SubtitleRenderResult, VerticalClipEntry
from utils.logger import get_logger

log = get_logger()

_VIDEO_EXT = "mp4"
_DURATION_TOLERANCE_S = 1.0

# Familiares por orden de preferencia para la autodeteccion segura (fallback).
# Solo se usan si el archivo EXISTE en el dir de fuentes; si nada existe:
# fail-fast con SubtitleFontError (nunca asumir que "Arial" está ahi).
_PREFERRED_FAMILIES = (
    "Arial",
    "Segoe UI",
    "Calibri",
    "Carlito",
    "Verdana",
    "Tahoma",
    "Consolas",
    "DejaVu Sans",
)

# Total ASS: color en &HBBGGRR (el orden BGR de libass, no RRGGBB del config).
_ASS_WHITE_BGR = "&H00FFFFFF"
_ASS_BLACK_BGR = "&H00000000"
_ASS_DIM_BGR = "&H80000000"


# --------------------------------------------------------------------------
# Resolucion segura de fuente
# --------------------------------------------------------------------------

def _fonts_dirs() -> list[Path]:
    """Directorio de fuentes del sistema (Windows) con suposicion minimizada."""
    dirs: list[Path] = []
    win = os.environ.get("WINDIR") or "C:\\Windows"
    sysfonts = Path(win) / "Fonts"
    if sysfonts.is_dir():
        dirs.append(sysfonts)
    return dirs


def _family_matches(family: str, filename: str) -> bool:
    """True si un archivo de fuente corresponde a una familia (heuristica
    determinista: 'Arial' -> arial.ttf/arialbd.ttf, 'Carlito' ->
    carlito-regular.ttf)."""
    key = family.replace(" ", "").lower()
    stem = filename.lower()
    return stem.startswith(key) and stem.endswith((".ttf", ".ttc", ".otf"))


def _find_family(family: str, fonts_dir: Path) -> bool:
    try:
        return any(
            _family_matches(family, f.name)
            for f in fonts_dir.iterdir()
            if f.is_file()
        )
    except OSError:
        return False


def resolve_font_family(
    cfg_font: str | None = None,
    *,
    fonts_dir: Path | None = None,
) -> str:
    """Resuelve la familia de fuente a usar en el .ass.

    - Config explicita (subtitles.font): se valida contra el dir de fuentes;
      si no existe -> SubtitleFontError (fail-fast, config corrupta).
    - Config vacia: primer (en orden de preferencia) de _PREFERRED_FAMILIES
      cuyo archivo exista. Si ninguno -> SubtitleFontError.
    Determinista: mismo input, mismo resultado.
    """
    candidate = normalize_text(cfg_font or "").strip()
    dirs = [fonts_dir] if fonts_dir is not None else _fonts_dirs()

    if candidate:
        for d in dirs:
            if _find_family(candidate, d):
                return candidate
        raise SubtitleFontError(
            f"subtitles.font={candidate!r} no existe en los dirs de fuentes "
            f"({', '.join(str(d) for d in dirs)})"
        )

    for family in _PREFERRED_FAMILIES:
        for d in dirs:
            if _find_family(family, d):
                return family
    raise SubtitleFontError(
        "no se pudo resolver ninguna fuente segura (subtitles.font vacio y "
        "ninguna familia preferida detectada)"
    )


# --------------------------------------------------------------------------
# Escapado de texto (datos, nunca comandos)
# --------------------------------------------------------------------------

def esc_ass(data: str) -> str:
    """Escapa un texto SIEMPRE literal para una linea de Dialogue ASS.

    Backslash -> doble-backslash, llaves -> escape-brace (libass las
    interpreta de otro modo como override blocks). El orden importa:
    primero backslash para que nuestros propios escapes no se re-procesen.
    """
    text = normalize_text(data)
    out = text.replace("\\", "\\\\")
    out = out.replace("{", "\\{")
    out = out.replace("}", "\\}")
    return out


def _bgr(color: str) -> str:
    """'RRGGBB' -> '&H00BBGGRR' (ASS). Config corrupta -> SubtitleDataError."""
    value = normalize_text(color).upper()
    if len(value) != 6 or any(ch not in "0123456789ABCDEF" for ch in value):
        raise SubtitleDataError(f"subtitles.highlight_color invalido: {color!r}")
    return f"&H00{value[4:6]}{value[2:4]}{value[0:2]}"


# --------------------------------------------------------------------------
# Layout de texto (determinista)
# --------------------------------------------------------------------------

def _line_len(parts: list[str], candidate: str) -> int:
    return sum(len(p) for p in parts) + (len(parts) - 1) + len(candidate)


def wrap_words(
    words: list[str],
    *,
    max_chars: int = 28,
    max_lines: int = 2,
) -> list[list[str]]:
    """Agrupa palabras en lineas de <= max_chars (greedy y deterministico).

    Devuelve grupos de palabras por linea. La estructura de lineas es la MISMA
    sin importar si despues se aplica un override de color por word (la base y
    el accent comparten la misma particion => libass las posiciona igual).
    Una palabra que supera max_chars se corta por caracteres.
    """
    max_chars = max(1, int(max_chars))
    max_lines = max(1, int(max_lines))
    lines: list[list[str]] = []
    current: list[str] = []
    for word in words:
        if len(word) > max_chars:
            if current:
                lines.append(current)
                current = []
            chunk, rest = word[:max_chars], word[max_chars:]
            lines.append([chunk])
            while len(rest) > max_chars:
                lines.append([rest[:max_chars]])
                rest = rest[max_chars:]
            if rest:
                current = [rest]
            continue
        if current and _line_len(current, word) > max_chars:
            lines.append(current)
            current = []
        current.append(word)
    if current:
        lines.append(current)

    while len(lines) > max_lines:
        lines[-2] = lines[-2] + lines[-1]
        lines.pop()
    return lines


# --------------------------------------------------------------------------
# Build ASS puro (determinista)
# --------------------------------------------------------------------------

def _ass_time(seconds: float) -> str:
    """Segundos -> 'h:mm:ss.cc' (centisimas, formato de Dialogue)."""
    cs = max(1, int(round(float(seconds) * 100)))
    h, rem = divmod(cs, 360000)
    m, rem = divmod(rem, 6000)
    s, cs = divmod(rem, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _scaled(base: int, width: int, base_width: int = 1080) -> int:
    return max(1, int(round(float(base) * width / base_width)))


def _dialogue_text(
    words: list[dict],
    *,
    accent: int | None = None,
    max_chars: int,
    max_lines: int,
    highlight_color: str,
) -> str:
    """Texto de una Dialogue: palabras escapadas, opcional word resaltada.

    Base y lineas de accent comparten la MISMA particion en lineas
    (wrap_words sobre los words planos) para que libass posicione los glifos
    identicos; el accent solo recolorea su word (overrides ASS como datos).
    """
    plain = [w["word"] for w in words]
    lines = wrap_words(plain, max_chars=max_chars, max_lines=max_lines)
    acc = _bgr(highlight_color)
    rendered: list[str] = []
    pos = 0
    for line in lines:
        parts = []
        for _ in line:
            item = esc_ass(words[pos]["word"])
            if accent is not None and accent == pos:
                parts.append(f"{{\\c{acc}}}{item}{{\\c{_ASS_WHITE_BGR}}}")
            else:
                parts.append(item)
            pos += 1
        rendered.append(" ".join(parts))
    return "\\N".join(rendered)


def build_ass(
    captions: list[CaptionSegment],
    width: int,
    height: int,
    *,
    font: str,
    font_size: int = 64,
    margin_bottom: int = 120,
    outline_width: int = 3,
    highlight_color: str = "FF6600",
    highlight_enabled: bool = True,
    max_chars: int = 28,
    max_lines: int = 2,
) -> str:
    """Serializa los captions a un .ass COMPLETO y deterministico."""
    width = int(width)
    height = int(height)
    if width <= 0 or height <= 0:
        raise SubtitleDataError(f"dimensiones invalidas para ASS: {width}x{height}")

    fs = max(12, _scaled(int(font_size), width))
    outline = _scaled(int(outline_width), width)
    shadow = _scaled(2, width)
    margin = max(16, int(round(float(margin_bottom) * height / 1920)))

    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        "Collisions: Normal\n"
        f"PlayResX: {width}\n"
        f"PlayResY: {height}\n"
        f"ScaledBorderAndShadow: yes\n"
        "WrapStyle: 0\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
        "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Default,{font},{fs},{_ASS_WHITE_BGR},{_ASS_WHITE_BGR},"
        f"{_ASS_BLACK_BGR},{_ASS_DIM_BGR},1,0,0,0,100,100,0,0,1,{outline},"
        f"{shadow},2,10,10,{margin},1\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, "
        "Effect, Text\n"
    )

    anchor = f"{{\\an2\\pos({width // 2},{height - margin})}}"
    events: list[str] = []
    for cap in captions:
        words = [dict(w) for w in cap.words]
        if words:
            base_text = _dialogue_text(
                words,
                accent=None,
                max_chars=max_chars,
                max_lines=max_lines,
                highlight_color=highlight_color,
            )
        else:
            base_text = esc_ass(cap.text)
        events.append(
            f"Dialogue: 0,{_ass_time(cap.start)},{_ass_time(cap.end)},"
            f"Default,,0,0,0,,{anchor}{base_text}"
        )
        do_highlight = highlight_enabled and cap.highlight and bool(words)
        if do_highlight:
            for i, w in enumerate(words):
                wstart = max(w["start"], cap.start)
                wend = min(w["end"], cap.end)
                if wend - wstart < 0.01:
                    continue
                acc_text = _dialogue_text(
                    words,
                    accent=i,
                    max_chars=max_chars,
                    max_lines=max_lines,
                    highlight_color=highlight_color,
                )
                events.append(
                    f"Dialogue: 0,{_ass_time(wstart)},{_ass_time(wend)},"
                    f"Default,,0,0,0,,{anchor}{acc_text}"
                )
    return header + "\n".join(events) + "\n"


# --------------------------------------------------------------------------
# Render FFmpeg (offline, sin red)
# --------------------------------------------------------------------------

def _rel_to_project(path: Path) -> str:
    """Path absoluto -> relativo al proyecto (posix), para el filtro `ass`.

    Evita 'C:' y '\\' que el parser de filtergraph interpreta (verificado:
    'C:/...' rompe el parseo). Se asume CWD = PROJECT_ROOT en el subprocess.
    """
    return str(path.relative_to(config.PROJECT_ROOT)).replace("\\", "/")


def _validate_captioned(
    clip_path: Path, expected: tuple[int, int], source_duration: float
) -> float:
    """Valida el captioned REAL: existe, size>0, MISMA resolucion, audio y
    video, y duracion equivalente a la del vertical fuente."""
    if not clip_path.exists():
        raise ClipFilesystemError(f"clip captioned no existe: {clip_path}")
    try:
        size = clip_path.stat().st_size
    except OSError as exc:
        raise ClipFilesystemError(f"no se pudo leer el clip captioned: {exc}") from exc
    if size <= 0:
        raise ClipFilesystemError(f"clip captioned vacio (0 bytes): {clip_path}")

    width, height = probe_dimensions(clip_path)
    if (width, height) != expected:
        raise SubtitleRenderError(
            f"clip captioned {width}x{height} != esperado {expected[0]}x{expected[1]}: "
            f"{clip_path}"
        )
    if not _probe_has_video(clip_path):
        raise SubtitleRenderError(f"clip captioned sin stream de video: {clip_path}")
    if not _probe_has_audio(clip_path):
        raise SubtitleRenderError(f"clip captioned sin stream de audio: {clip_path}")

    duration = probe_duration(clip_path)
    if duration <= 0:
        raise SubtitleRenderError(f"clip captioned sin duracion valida (ffprobe): {clip_path}")
    if abs(duration - source_duration) > _DURATION_TOLERANCE_S:
        raise SubtitleRenderError(
            f"clip captioned dura {duration:.3f}s, vertical {source_duration:.3f}s "
            f"(tolerancia {_DURATION_TOLERANCE_S:.1f}s)"
        )
    return duration


def render_captioned(
    vertical: VerticalClipEntry,
    captions: list[CaptionSegment],
    job_id: str,
) -> SubtitleRenderResult:
    """Quema los captions en UN clip vertical -> clips_captioned/<clip_id>.mp4.

    Preserva resolucion (PlayRes = vertical), audio (-c:a copy) y duracion.
    El .ass vive en clips_captioned/subtitles/<clip_id>.ass (relativo al
    proyecto para el filtro `ass=filename=...`).
    """
    source_clip = config.output_dir() / vertical.path
    if not source_clip.exists():
        raise ClipFilesystemError(f"vertical no existe: {source_clip}")

    width, height = probe_dimensions(source_clip)
    if (width, height) == (0, 0):
        raise SubtitleRenderError(f"vertical sin video (imposible captioned): {source_clip}")

    font = resolve_font_family(str(config.get("subtitles.font", "")))
    ass = build_ass(
        captions,
        width,
        height,
        font=font,
        font_size=int(config.get("subtitles.font_size", 64)),
        margin_bottom=int(config.get("subtitles.margin_bottom", 120)),
        outline_width=int(config.get("subtitles.outline_width", 3)),
        highlight_color=str(config.get("subtitles.highlight_color", "FF6600")),
        highlight_enabled=bool(config.get("subtitles.highlight_enabled", True)),
        max_chars=int(config.get("subtitles.max_chars_per_line", 28)),
        max_lines=int(config.get("subtitles.max_lines", 2)),
    )

    dest_dir = config.output_dir() / job_id / "clips_captioned"
    ass_dir = dest_dir / "subtitles"
    ass_dir.mkdir(parents=True, exist_ok=True)
    dest_dir.mkdir(parents=True, exist_ok=True)
    ass_path = ass_dir / f"{vertical.clip_id}.ass"
    try:
        ass_path.write_text(ass, encoding="utf-8")
    except OSError as exc:
        raise ClipFilesystemError(f"no se pudo escribir el .ass: {exc}") from exc

    out_path = dest_dir / f"{vertical.clip_id}.{_VIDEO_EXT}"
    rel_ass = _rel_to_project(ass_path)

    ffmpeg = str(config.get("bins.ffmpeg", "ffmpeg"))
    preset = str(config.get("ffmpeg.preset", "medium"))
    crf = int(config.get("ffmpeg.crf", 23))
    cmd = [
        ffmpeg, "-y",
        "-i", str(source_clip),
        "-vf", f"ass=filename={rel_ass}",
        "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        "-c:a", "copy",
        "-movflags", "+faststart",
        str(out_path),
    ]
    try:
        _run(cmd, what=f"quemado de subtitulos de {vertical.clip_id}", tool="ffmpeg",
             cwd=str(config.PROJECT_ROOT))
    except ClipError as exc:
        raise SubtitleRenderError(exc.message) from exc

    source_duration = vertical.duration or probe_duration(source_clip)
    duration = _validate_captioned(out_path, (width, height), source_duration)

    result = SubtitleRenderResult(
        clip_id=vertical.clip_id,
        source_clip=str(source_clip.relative_to(config.output_dir())),
        output_clip=str(out_path.relative_to(config.output_dir())),
        subtitle_source="transcript",
        duration=round(duration, 3),
        caption_count=len(captions),
        style="dynamic",
        font=font,
        position="bottom_center",
        highlight_enabled=bool(config.get("subtitles.highlight_enabled", True)),
        qc_status="passed",
    )
    log.info(
        "clip_captioned_ready",
        job_id=job_id,
        clip=result.clip_id,
        path=str(out_path),
        captions=result.caption_count,
        font=font,
        resolution=f"{width}x{height}",
        duration_s=round(result.duration, 2),
    )
    return result


def render_captioned_clips(
    verticals: list[VerticalClipEntry],
    captions_by_clip: dict[str, list[CaptionSegment]],
    job_id: str,
) -> list[SubtitleRenderResult]:
    """Quema captions en todos los verticales + clips_captioned/manifest.json.

    Hereda source/source_type del manifest vertical (clips_vertical) para
    mantener la trazabilidad a la fuente original. Si no hay verticales, no
    genera nada. Los clips sin captions no se procesan.
    """
    if not verticals:
        log.info("captioned_skipped", job_id=job_id, reason="no_verticals")
        return []

    source_url = None
    source_type = None
    vert_manifest = config.output_dir() / job_id / "clips_vertical" / "manifest.json"
    if vert_manifest.exists():
        try:
            payload = json.loads(vert_manifest.read_text(encoding="utf-8"))
            source_url = payload.get("source")
            source_type = payload.get("source_type")
        except (OSError, ValueError) as exc:
            raise ClipFilesystemError(
                f"no se pudo leer el manifest vertical: {exc}"
            ) from exc

    results: list[SubtitleRenderResult] = []
    for vertical in verticals:
        caps = captions_by_clip.get(vertical.clip_id) or []
        if not caps:
            log.info(
                "captioned_skipped",
                job_id=job_id,
                clip=vertical.clip_id,
                reason="no_captions",
            )
            continue
        results.append(render_captioned(vertical, caps, job_id))

    write_captioned_manifest(
        job_id, results, source_url=source_url, source_type=source_type
    )
    return results


def write_captioned_manifest(
    job_id: str,
    results: list[SubtitleRenderResult],
    *,
    source_url: str | None = None,
    source_type: str | None = None,
) -> Path:
    """Persiste clips_captioned/manifest.json (envelope identico a las fases
    anteriores; cada entrada trae subtitle_enabled y los campos de Fase 5)."""
    dest_dir = config.output_dir() / job_id / "clips_captioned"
    try:
        dest_dir.mkdir(parents=True, exist_ok=True)
        path = dest_dir / "manifest.json"
        payload = {
            "job_id": job_id,
            "source": source_url,
            "source_type": source_type,
            "subtitles": [r.model_dump() for r in results],
        }
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError as exc:
        raise ClipFilesystemError(
            f"no se pudo escribir el manifest captioned: {exc}"
        ) from exc
    return path