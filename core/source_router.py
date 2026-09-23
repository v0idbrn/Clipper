"""Bifurcación de fuente: streaming vs. direct_file.

PRIMER módulo que corre y decide todo lo demás. Si falla la detección acá,
el resto del pipeline no debería ni empezar.

Reglas (en orden de precedencia):
1. Path local existente                    -> "direct_file"
2. Host conocido de streaming              -> "streaming"
3. Host conocido de archivo crudo          -> "direct_file"
4. Extensión de archivo multimedia en URL  -> "direct_file"
5. Default                                 -> "streaming" (yt-dlp genérico)
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from core.errors import UnknownSourceError

SourceType = Literal["streaming", "direct_file"]

STREAMING_HOSTS = {
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "youtu.be",
    "twitch.tv",
    "www.twitch.tv",
    "vimeo.com",
    "player.vimeo.com",
    "tiktok.com",
    "www.tiktok.com",
    "facebook.com",
    "www.facebook.com",
    "instagram.com",
    "www.instagram.com",
    "soundcloud.com",
    "x.com",
    "twitter.com",
}

DIRECT_FILE_HOSTS = {
    "drive.google.com",
    "drive.usercontent.google.com",
    "docs.google.com",
    "we.tl",
    "wetransfer.com",
    "www.wetransfer.com",
    "dropbox.com",
    "www.dropbox.com",
    "dl.dropboxusercontent.com",
    "droplr.com",
}

# Extensiones de archivos multimedia crudos (siguiendo otras extensiones se
# cae al default "streaming" para no romper thumbnails/playlists/etc.)
MEDIA_EXTENSIONS = {".mp4", ".mkv", ".webm", ".mov", ".avi", ".flv",
                    ".m4a", ".m4v", ".mp3", ".wav", ".aac", ".ogg"}


def _host_of(url: str) -> str:
    try:
        return (urlparse(url).hostname or "").lower()
    except ValueError:
        return ""


def _extension_of(url: str) -> str:
    path = urlparse(url).path
    return Path(path).suffix.lower()


def detect_source_type(source: str) -> SourceType:
    """Clasifica la entrada como 'streaming' o 'direct_file'.

    source puede ser una URL o un path local.
    """
    if not source or not source.strip():
        raise UnknownSourceError("Fuente vacía")

    source = source.strip()

    # 1. Path local existente
    candidate = Path(source)
    if candidate.exists() and candidate.is_file():
        return "direct_file"

    # URLs
    if "://" not in source and not _host_of(source):
        raise UnknownSourceError(f"No se pudo clasificar la fuente: {source}")

    host = _host_of(source)
    ext = _extension_of(source)

    # 2. Streaming conocido
    if host in STREAMING_HOSTS or any(host.endswith(f".{h}") for h in STREAMING_HOSTS):
        return "streaming"
    # 3. Archivo crudo conocido
    if host in DIRECT_FILE_HOSTS or any(host.endswith(f".{h}") for h in DIRECT_FILE_HOSTS):
        return "direct_file"
    # 4. URL con extensión de archivo multimedia -> archivo directo
    if ext in MEDIA_EXTENSIONS:
        return "direct_file"

    # 5. Default optimista: la mayoría de share links son plataformas.
    return "streaming"