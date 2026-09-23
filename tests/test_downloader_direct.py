"""Tests de downloader_direct.

Verificación clave de la fase: check_disk_space corre ANTES de iniciar la
descarga — si no hay espacio, se aborta y la descarga (requests.get) jamás
se dispara.

Se mockea el "servidor" (requests.head/get) porque probar contra Drive real
de verdad es flaky en CI; la lógica de abort es exactamente la misma.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import core.downloader_direct as downloader_direct  # noqa: E402
from core.downloader_direct import download_full, check_disk_space  # noqa: E402
from core.errors import DiskSpaceError, DownloadError  # noqa: E402


class FakeResponse:
    def __init__(self, headers=None, status=200):
        self.headers = headers or {}
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeStream:
    """Simula requests.get(...).__enter__() -> resp con iter_content chunky."""

    def __init__(self, body: bytes, headers=None, status=200):
        self.body = body
        self.headers = headers or {}
        self.status_code = status

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def iter_content(self, chunk_size):
        for i in range(0, len(self.body), chunk_size):
            yield self.body[i:i + chunk_size]


def _free_space_bytes(tmp_path: Path) -> int:
    import shutil
    return shutil.disk_usage(tmp_path).free


class TestDiskSpaceCheck:
    def test_aborts_before_download_when_not_enough_space(self, tmp_path: Path, monkeypatch):
        """El caso que exige la spec: aborta ANTES, no como validación posterior."""
        called_get = {"n": 0}

        def fake_head(url, **kwargs):
            return FakeResponse(headers={"Content-Length": str(10**11)})  # 100 GB

        def fake_get(url, **kwargs):
            called_get["n"] += 1
            raise AssertionError("requests.get NO debe ejecutarse si falta espacio")

        monkeypatch.setattr(downloader_direct.requests, "head", fake_head)
        monkeypatch.setattr(downloader_direct.requests, "get", fake_get)

        with pytest.raises(DiskSpaceError):
            download_full("https://drive.google.com/huge.bin", tmp_path)

        assert called_get["n"] == 0, "La descarga se disparó pese al abort de disco"

    def test_succeeds_when_enough_space(self, tmp_path: Path, monkeypatch):
        size = 1024 * 1024  # 1 MB — cabe seguro en tmp_path
        body = b"x" * size

        def fake_head(url, **kwargs):
            return FakeResponse(headers={"Content-Length": str(size)})

        def fake_get(url, **kwargs):
            return FakeStream(body, headers={"Content-Length": str(size)})

        monkeypatch.setattr(downloader_direct.requests, "head", fake_head)
        monkeypatch.setattr(downloader_direct.requests, "get", fake_get)

        dest = download_full("https://dropbox.com/archivo.mp4", tmp_path)
        assert dest.exists()
        assert dest.stat().st_size == size

    def test_check_disk_space_direct(self, tmp_path: Path):
        free = _free_space_bytes(tmp_path)
        # Pedir 1 byte de más -> aborta
        with pytest.raises(DiskSpaceError):
            check_disk_space(tmp_path, free + 1)
        # Pedir muy poco -> pasa
        check_disk_space(tmp_path, 1024)


class TestDownloadBehavior:
    def test_streams_in_chunks_and_preserves_filename(self, tmp_path: Path, monkeypatch):
        body = b"abcdefgh" * 1000  # ~8 KB
        disp = 'attachment; filename="cliente_crudo.mp4"'

        def fake_head(url, **kwargs):
            return FakeResponse(headers={
                "Content-Length": str(len(body)),
                "Content-Disposition": disp,
            })

        def fake_get(url, **kwargs):
            return FakeStream(body)

        monkeypatch.setattr(downloader_direct.requests, "head", fake_head)
        monkeypatch.setattr(downloader_direct.requests, "get", fake_get)

        dest = download_full("https://drive.google.com/uc?id=X", tmp_path)
        assert dest.name == "cliente_crudo.mp4"
        assert dest.read_bytes() == body

    def test_resume_uses_range_header_when_partial_exists(self, tmp_path: Path, monkeypatch):
        body = b"0123456789" * 500
        partial = tmp_path / "raw.mp4"
        original = body[:2000]
        partial.write_bytes(original)

        captured_headers: dict = {}

        def fake_head(url, **kwargs):
            return FakeResponse(headers={
                "Content-Length": str(len(body)),
                "Accept-Ranges": "bytes",
            })

        def fake_get(url, **kwargs):
            captured_headers.update(kwargs.get("headers", {}))
            offset = 2000
            return FakeStream(body[offset:])

        monkeypatch.setattr(downloader_direct.requests, "head", fake_head)
        monkeypatch.setattr(downloader_direct.requests, "get", fake_get)

        dest = download_full("https://cdn.ejemplo.com/raw.mp4", tmp_path)
        assert captured_headers.get("Range") == "bytes=2000-"
        # El modo de escritura fue append: contenido real = original + resto.
        assert dest.read_bytes() == original + body[2000:]

    def test_incomplete_download_is_deleted(self, tmp_path: Path, monkeypatch):
        body = b"x" * 5000
        partial = tmp_path / "raw.mp4"
        partial.write_bytes(b"x" * 2000)

        def fake_head(url, **kwargs):
            # Sin Accept-Ranges: no resume. Contenido anunciado 5000.
            return FakeResponse(headers={"Content-Length": str(len(body))})

        def fake_get(url, **kwargs):
            # El servidor "corta" y manda menos de lo anunciado.
            return FakeStream(body[:3000])

        monkeypatch.setattr(downloader_direct.requests, "head", fake_head)
        monkeypatch.setattr(downloader_direct.requests, "get", fake_get)

        with pytest.raises(DownloadError):
            download_full("https://cdn.ejemplo.com/raw.mp4", tmp_path)
        assert not (tmp_path / "raw.mp4").exists()

    def test_head_failure_raises_download_error(self, tmp_path: Path, monkeypatch):
        def fake_head(url, **kwargs):
            from requests import RequestException
            raise RequestException("connection refused")
        monkeypatch.setattr(downloader_direct.requests, "head", fake_head)
        with pytest.raises(DownloadError):
            download_full("https://wetransfer.com/lo-que-sea", tmp_path)