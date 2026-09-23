"""Descarga de archivos crudos (Drive/WeTransfer/Dropbox/links directos).

Regla de oro: el archivo completo va a DISCO, nunca a RAM. Se escribe en
streaming por bloques, con resume via HTTP Range si el servidor lo soporta.

Orden de operaciones (crítico):
1. HEAD para conocer Content-Length y si soporta Range (si es posible).
2. shutil.disk_usage() ANTES de abrir la primera conexión de descarga.
   Si no hay espacio para size + buffer, se aborta con DiskSpaceError
   sin haber escrito ni un byte.
3. Descarga en chunks de 1 MB al destino.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import requests

import config
from core.errors import DiskSpaceError, DownloadError

_DEFAULT_TIMEOUT = 60
_CHUNK_SIZE = 1024 * 1024  # 1 MB


def _required_free_bytes(file_size: int) -> int:
    buffer_mb = int(config.get("downloads.min_free_buffer_mb", 32))
    return file_size + buffer_mb * 1024 * 1024


def check_disk_space(dest_dir: Path, required_bytes: int) -> None:
    """Aborta con DiskSpaceError si no hay espacio libre suficiente.

    Corre ANTES de iniciar la descarga completa: es la validación previa
    que evita que la descarga falle a mitad sin explicación.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    try:
        free = shutil.disk_usage(dest_dir).free
    except OSError as exc:  # pragma: no cover - disco inaccesible
        raise DiskSpaceError(f"No se pudo consultar el espacio de {dest_dir}: {exc}")
    required = max(required_bytes, 1)
    if free < required:
        raise DiskSpaceError(
            f"Espacio insuficiente: se necesitan ~{required / 1024**3:.2f} GB "
            f"libres en {dest_dir} pero solo hay {free / 1024**3:.2f} GB. "
            "Abortando descarga antes de empezar."
        )


def _guess_filename(url: str, headers: dict[str, str] | None = None) -> str:
    headers = headers or {}
    disp = headers.get("Content-Disposition", "")
    if "filename=" in disp:
        name = disp.split("filename=", 1)[1].strip('";')
        if name:
            return name
    path = Path(requests.utils.urlparse(url).path)
    return path.name if path.name else "download.bin"


def download_full(url: str, dest_dir: Path, *, timeout: int = _DEFAULT_TIMEOUT) -> Path:
    """Descarga el archivo completo a disco y devuelve su Path.

    - Chequea espacio ANTES de descargar (si conocemos el tamaño).
    - Escribe en streaming (nunca buffer completo en RAM).
    - Resume desde un parcial existente si el servidor soporta Range.
    """
    try:
        head = requests.head(url, allow_redirects=True, timeout=timeout)
    except requests.RequestException as exc:
        raise DownloadError(f"No se pudo consultar el recurso {url}: {exc}")

    content_length = head.headers.get("Content-Length")
    file_size = int(content_length) if content_length and content_length.isdigit() else None
    supports_range = head.headers.get("Accept-Ranges") == "bytes"

    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / _guess_filename(url, head.headers)

    # Chequeo PREVIO de disco (la pieza que exige la spec).
    if file_size is not None:
        check_disk_space(dest_dir, _required_free_bytes(file_size))

    # Resume: si ya existe un parcial y el servidor soporta Range.
    headers: dict[str, str] = {}
    resume_offset = 0
    if dest.exists() and supports_range and file_size is not None:
        resume_offset = dest.stat().st_size
        if 0 < resume_offset < file_size:
            headers["Range"] = f"bytes={resume_offset}-"

    try:
        with requests.get(url, stream=True, headers=headers, allow_redirects=True, timeout=timeout) as resp:
            resp.raise_for_status()
            mode = "ab" if headers.get("Range") else "wb"
            with dest.open(mode) as f:
                for chunk in resp.iter_content(chunk_size=_CHUNK_SIZE):
                    if chunk:
                        f.write(chunk)
    except requests.RequestException as exc:
        raise DownloadError(f"Fallo descargando {url}: {exc}")

    # Verificación final: el archivo completo llegó (si conocíamos el tamaño).
    if file_size is not None and Path(dest).stat().st_size < file_size:
        # Descartamos el parcial corrupto para evitar entregar basura.
        Path(dest).unlink(missing_ok=True)
        raise DownloadError(
            f"Descarga incompleta: {Path(dest).stat().st_size if Path(dest).exists() else 0} "
            f"/ {file_size} bytes"
        )

    return dest