"""Configuración global de AutoClipper.

Carga config.yaml + variables de entorno desde .env. Sin dependencias
fuera de la stdlib para mantener el core liviano.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
ENV_PATH = PROJECT_ROOT / ".env"

ENV_DEFAULTS = {
    "LLM_API_KEY": "",
    "LLM_BASE_URL": "",
}

_KEEP: dict[str, Any] = {}


def _load_dotenv() -> None:
    if not ENV_PATH.exists():
        return
    for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


def reload() -> None:
    """(Re)carga config.yaml y .env en el módulo."""
    _load_dotenv()
    for key, default in ENV_DEFAULTS.items():
        _KEEP[f"env.{key}"] = os.environ.get(key, default)

    import yaml

    try:
        data = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
    except FileNotFoundError:
        data = {}
    if not isinstance(data, dict):
        data = {}

    max_cost = data.get("max_api_cost_per_job", 1.50)
    _KEEP["max_api_cost_per_job"] = float(max_cost)

    clip = data.get("clip", {})
    _KEEP["clip.min_duration_seconds"] = float(clip.get("min_duration_seconds", 15))
    _KEEP["clip.max_duration_seconds"] = float(clip.get("max_duration_seconds", 90))
    _KEEP["clip.target_duration_seconds"] = float(
        clip.get("target_duration_seconds", 40)
    )
    _KEEP["clip.buffer_seconds"] = float(clip.get("buffer_seconds", 2))
    _KEEP["clip.output_resolution"] = str(clip.get("output_resolution", "1080x1920"))

    vertical = data.get("vertical", {})
    _KEEP["vertical.mode"] = str(vertical.get("mode", "center_crop"))
    _KEEP["vertical.max_upscale"] = float(vertical.get("max_upscale", 2.5))

    subtitles = data.get("subtitles", {})
    _KEEP["subtitles.enabled"] = bool(subtitles.get("enabled", True))
    _KEEP["subtitles.max_words_per_caption"] = int(
        subtitles.get("max_words_per_caption", 6)
    )
    _KEEP["subtitles.max_chars_per_line"] = int(
        subtitles.get("max_chars_per_line", 28)
    )
    _KEEP["subtitles.max_lines"] = int(subtitles.get("max_lines", 2))
    _KEEP["subtitles.font"] = str(subtitles.get("font", ""))
    _KEEP["subtitles.font_size"] = int(subtitles.get("font_size", 64))
    _KEEP["subtitles.margin_bottom"] = int(subtitles.get("margin_bottom", 120))
    _KEEP["subtitles.outline_width"] = int(subtitles.get("outline_width", 3))
    _KEEP["subtitles.highlight_enabled"] = bool(
        subtitles.get("highlight_enabled", True)
    )
    _KEEP["subtitles.highlight_color"] = str(
        subtitles.get("highlight_color", "FF6600")
    )

    models = data.get("models", {})
    _KEEP["models.whisper"] = str(models.get("whisper", "base"))
    _KEEP["models.llm"] = str(models.get("llm", "claude-haiku"))

    llm = data.get("llm", {})
    _KEEP["llm.mode"] = str(llm.get("mode", "mock"))
    _KEEP["llm.window_tokens"] = int(llm.get("window_tokens", 3500))
    _KEEP["llm.max_json_retries"] = int(llm.get("max_json_retries", 1))
    _KEEP["llm.timeout_seconds"] = int(llm.get("timeout_seconds", 60))
    _KEEP["llm.price_per_million_input_usd"] = float(
        llm.get("price_per_million_input_usd", 1.00)
    )
    _KEEP["llm.price_per_million_output_usd"] = float(
        llm.get("price_per_million_output_usd", 5.00)
    )
    _KEEP["llm.estimated_output_tokens"] = int(llm.get("estimated_output_tokens", 300))
    _KEEP["llm.candidates_per_window"] = int(llm.get("candidates_per_window", 12))

    whisper = data.get("whisper", {})
    _KEEP["whisper.cost_per_hour_usd"] = float(whisper.get("cost_per_hour_usd", 0.06))

    bins = data.get("bins", {})
    _KEEP["bins.ffmpeg"] = str(bins.get("ffmpeg", "ffmpeg"))
    _KEEP["bins.ffprobe"] = str(bins.get("ffprobe", "ffprobe"))

    downloads = data.get("downloads", {})
    _KEEP["downloads.min_free_buffer_mb"] = int(
        downloads.get("min_free_buffer_mb", 32)
    )

    youtube = data.get("youtube", {})
    _KEEP["youtube.js_runtime"] = str(youtube.get("js_runtime", "node"))
    _KEEP["youtube.cookies_from_browser"] = str(
        youtube.get("cookies_from_browser", "")
    )

    ffmpeg = data.get("ffmpeg", {})
    _KEEP["ffmpeg.preset"] = str(ffmpeg.get("preset", "medium"))
    _KEEP["ffmpeg.crf"] = int(ffmpeg.get("crf", 23))


def get(key: str, default: Any = None) -> Any:
    """Trae un valor de config; '.' separa niveles (ej. 'clip.buffer_seconds')."""
    value = _KEEP.get(key)
    return default if value is None else value


def output_dir() -> Path:
    root = PROJECT_ROOT / "outputs"
    root.mkdir(parents=True, exist_ok=True)
    return root


reload()