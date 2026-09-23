"""Extracción de audio-only desde la fuente.

Dos ramas:
- "streaming": yt-dlp -f bestaudio -x (resuelve URLs firmadas, range requests).
- "direct_file": ffmpeg local contra el archivo crudo ya descargado en disco.

En AMBOS casos el audio sale a disco (streaming), nunca volcado entero en RAM.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import config
from core import ytdlp_opts
from core.errors import ExtractionError, YouTubeAcquisitionError

_AUDIO_EXT = "mp3"
_QUALITY = 5  # yt-dlp 0-9 / libmp3lame -q:a


def _run(cmd: list[str], *, what: str, ytdlp: bool = False) -> None:
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except FileNotFoundError as exc:
        raise ExtractionError(f"Binario no encontrado: {cmd[0]}") from exc
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
        raise ExtractionError(
            f"Fallo {what}\n  cmd: {' '.join(cmd)}\n  "
            + "\n  ".join(tail[-5:])
        )


def extract_stream_audio(url: str, dest_dir: Path) -> Path:
    """Descarga audio-only con yt-dlp (solo audio, nunca el video completo)."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    out_tpl = str(dest_dir / "audio.%(ext)s")

    # python -m yt_dlp garantiza encontrar el entrypoint en el venv actual,
    # sin depender del PATH del sistema.
    cmd = [
        sys.executable, "-m", "yt_dlp",
        *ytdlp_opts.common_ytdlp_args(),
        "-f", "bestaudio",
        "-x",
        "--audio-format", _AUDIO_EXT,
        "--audio-quality", str(_QUALITY),
        "-o", out_tpl,
        "--no-playlist",
        url,
    ]
    _run(cmd, what="extracción de audio (yt-dlp)", ytdlp=True)

    audio = dest_dir / f"audio.{_AUDIO_EXT}"
    if not audio.exists():
        # yt-dlp pudo dejar el audio con otro nombre si el formato difirió.
        found = list(dest_dir.glob("audio.*"))
        if not found:
            raise ExtractionError("yt-dlp no produjo ningún audio")
        audio = found[0]
    return audio


def extract_local_audio(video_path: Path, dest_dir: Path) -> Path:
    """Extrae la pista de audio de un archivo local ya descargado (ffmpeg)."""
    video_path = Path(video_path)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = dest_dir / f"audio.{_AUDIO_EXT}"

    ffmpeg = str(config.get("bins.ffmpeg", "ffmpeg"))
    cmd = [
        ffmpeg, "-i", str(video_path),
        "-vn",
        "-acodec", "libmp3lame",
        "-q:a", str(_QUALITY),
        "-y", str(out),
    ]
    _run(cmd, what="extracción de audio (ffmpeg local)")
    return out


def extract_audio(source: str, dest_dir: Path, *, source_type: str) -> Path:
    """Dispatch según la bifurcación de source_router."""
    if source_type == "streaming":
        return extract_stream_audio(source, dest_dir)
    return extract_local_audio(Path(source), dest_dir)