"""Tests de source_router: streaming vs. direct_file.

La spec exige 4+ casos explícitos: YouTube, Twitch, Google Drive, WeTransfer.
Además: path local, URL con extensión .mp4, y fallo con fuente basura.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.source_router import detect_source_type  # noqa: E402
from core.errors import UnknownSourceError  # noqa: E402


class TestStreaming:
    def test_youtube_watch(self):
        assert detect_source_type("https://youtube.com/watch?v=dQw4w9WgXcQ") == "streaming"

    def test_youtube_youtu_be(self):
        assert detect_source_type("https://youtu.be/dQw4w9WgXcQ") == "streaming"

    def test_twitch_video(self):
        assert detect_source_type("https://www.twitch.tv/videos/1234567890") == "streaming"

    def test_vimeo(self):
        assert detect_source_type("https://vimeo.com/123456") == "streaming"


class TestDirectFile:
    def test_google_drive_view(self):
        assert detect_source_type("https://drive.google.com/file/d/ABC123/view?usp=sharing") == "direct_file"

    def test_wetransfer(self):
        assert detect_source_type("https://we.tl/t-abc123def456") == "direct_file"

    def test_wetransfer_domain(self):
        assert detect_source_type("https://wetransfer.com/downloads/xyz987") == "direct_file"

    def test_dropbox(self):
        assert detect_source_type("https://www.dropbox.com/s/ab12/raw.mp4?dl=0") == "direct_file"

    def test_direct_mp4_url(self):
        assert detect_source_type("https://cdn.ejemplo.com/videos/largo.mp4") == "direct_file"

    def test_local_file_path(self, tmp_path: Path):
        f = tmp_path / "cliente_crudo.mp4"
        f.write_bytes(b"\x00\x00\x00\x00")
        assert detect_source_type(str(f)) == "direct_file"

    def test_mkv_url(self):
        assert detect_source_type("http://ejemplo.com/stream.mkv") == "direct_file"


class TestFallbacks:
    def test_unknown_defaults_to_streaming(self):
        # Host no reconocido sin extensión -> fallback optimista a streaming.
        assert detect_source_type("https://midominio.com/watch?id=42") == "streaming"

    def test_empty_source_raises(self):
        with pytest.raises(UnknownSourceError):
            detect_source_type("")

    def test_garbage_raises(self):
        with pytest.raises(UnknownSourceError):
            detect_source_type("esto-no-es-una-fuente")