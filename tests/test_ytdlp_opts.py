"""Tests de core/ytdlp_opts.py — opciones comunes y clasificación de errores.

Cubren el gate de adquisición de YouTube:
- Defaults seguros (runtime node habilitado, cookies deshabilitadas).
- Cookies OPT-IN y configurables.
- Clasificación de errores de yt-dlp (429 / bot-check / runtime faltante).
- Ausencia de secretos: solo se reenvía el NOMBRE del navegador, nunca cookies.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from core import ytdlp_opts  # noqa: E402


@pytest.fixture(autouse=True)
def _restore_cfg():
    snapshot = dict(config._KEEP)
    yield
    config._KEEP.clear()
    config._KEEP.update(snapshot)


class TestDefaults:
    def test_js_runtime_default_is_node(self):
        # config.yaml define youtube.js_runtime: "node" (recomendado por yt-dlp).
        assert ytdlp_opts.js_runtime() == "node"
        assert ytdlp_opts.common_ytdlp_args() == ["--js-runtimes", "node"]

    def test_cookies_disabled_by_default(self):
        assert ytdlp_opts.cookies_from_browser() is None
        assert "--cookies-from-browser" not in ytdlp_opts.common_ytdlp_args()


class TestJsRuntime:
    def test_none_disables_runtime(self, monkeypatch):
        monkeypatch.setitem(config._KEEP, "youtube.js_runtime", "none")
        assert ytdlp_opts.js_runtime() is None
        assert ytdlp_opts.common_ytdlp_args() == []

    def test_empty_disables_runtime(self, monkeypatch):
        monkeypatch.setitem(config._KEEP, "youtube.js_runtime", "")
        assert ytdlp_opts.js_runtime() is None

    def test_explicit_path_is_passed_through(self, monkeypatch):
        monkeypatch.setitem(config._KEEP, "youtube.js_runtime", "node:C:/custom/node")
        assert ytdlp_opts.common_ytdlp_args() == ["--js-runtimes", "node:C:/custom/node"]

    def test_alternative_runtime_name(self, monkeypatch):
        monkeypatch.setitem(config._KEEP, "youtube.js_runtime", "deno")
        assert ytdlp_opts.common_ytdlp_args() == ["--js-runtimes", "deno"]


class TestCookies:
    def test_opt_in_browser(self, monkeypatch):
        monkeypatch.setitem(config._KEEP, "youtube.cookies_from_browser", "chrome")
        assert ytdlp_opts.cookies_from_browser() == "chrome"
        assert "--cookies-from-browser" in ytdlp_opts.common_ytdlp_args()
        assert "chrome" in ytdlp_opts.common_ytdlp_args()

    def test_only_browser_name_is_exposed(self, monkeypatch):
        """Seguridad: nunca se lee/persiste contenido de cookies; solo el nombre."""
        monkeypatch.setitem(config._KEEP, "youtube.cookies_from_browser", "firefox")
        args = ytdlp_opts.common_ytdlp_args()
        joined = " ".join(args)
        # No debe haber rutas a bases de cookies ni blobs.
        assert "cookies.sqlite" not in joined
        assert "Cookies" not in joined
        assert args[args.index("--cookies-from-browser") + 1] == "firefox"

    def test_full_args_order_with_both(self, monkeypatch):
        monkeypatch.setitem(config._KEEP, "youtube.js_runtime", "node")
        monkeypatch.setitem(config._KEEP, "youtube.cookies_from_browser", "edge")
        assert ytdlp_opts.common_ytdlp_args() == [
            "--js-runtimes", "node",
            "--cookies-from-browser", "edge",
        ]


class TestErrorClassification:
    def test_bot_check(self):
        hint = ytdlp_opts.diagnostic_hint(
            "ERROR: [youtube] x: Sign in to confirm you're not a bot. Use --cookies"
        )
        assert hint is not None
        assert "anti-bot" in hint
        assert "cookies_from_browser" in hint

    def test_429(self):
        hint = ytdlp_opts.diagnostic_hint("ERROR: HTTP Error 429: Too Many Requests")
        assert hint is not None
        assert "429" in hint or "rate limit" in hint.lower()

    def test_missing_js_runtime(self):
        hint = ytdlp_opts.diagnostic_hint(
            "WARNING: No supported JavaScript runtime could be found."
        )
        assert hint is not None
        assert "runtime" in hint.lower()

    def test_unknown_error_returns_none(self):
        assert ytdlp_opts.classify_ytdlp_error("ERROR: algo rarisimo pasó") is None
        assert ytdlp_opts.diagnostic_hint("ERROR: algo rarisimo pasó") is None

    def test_is_youtube_auth_error(self):
        assert ytdlp_opts.is_youtube_auth_error("Sign in to confirm you're not a bot")
        assert ytdlp_opts.is_youtube_auth_error("HTTP Error 429")
        assert not ytdlp_opts.is_youtube_auth_error("some other thing")
