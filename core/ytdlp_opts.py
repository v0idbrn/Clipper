"""Opciones comunes de yt-dlp y diagnóstico de errores de adquisición (YouTube).

YouTube dejó de funcionar con una invocación "pelada" de yt-dlp: exige resolver
desafíos JavaScript con un runtime externo (wiki oficial: yt-dlp/yt-dlp/wiki/EJS)
y, en escenarios de rate-limit (429 / "Sign in to confirm you're not a bot"),
requiere cookies. Ambos son OPT-IN y configurables desde `config.yaml`:

    youtube:
      js_runtime: "node"            # node | node:/ruta | none
      cookies_from_browser: ""      # vacío = deshabilitado; ej "chrome"

Seguridad: acá nunca se lee, persiste ni imprime el CONTENIDO de las cookies.
Solo se reenvía a yt-dlp el nombre del navegador (opción oficial
`--cookies-from-browser`), que las lee localmente.
"""

from __future__ import annotations

import config

_DISABLED = {"", "none", "disabled", "off", "false", "0"}


def js_runtime() -> str | None:
    """Runtime JS configurado, o None si está deshabilitado."""
    value = str(config.get("youtube.js_runtime", "node") or "").strip()
    if value.lower() in _DISABLED:
        return None
    return value


def cookies_from_browser() -> str | None:
    """Navegador para `--cookies-from-browser`, o None si está deshabilitado."""
    value = str(config.get("youtube.cookies_from_browser", "") or "").strip()
    return value or None


def common_ytdlp_args() -> list[str]:
    """Flags de yt-dlp derivados de config (runtime JS + cookies opt-in).

    No incluye opciones por-llamada (formato, output, URL, etc.).
    """
    args: list[str] = []
    runtime = js_runtime()
    if runtime:
        args += ["--js-runtimes", runtime]
    browser = cookies_from_browser()
    if browser:
        args += ["--cookies-from-browser", browser]
    return args


# --------------------------------------------------------------------------
# Clasificación de errores de yt-dlp -> diagnóstico accionable
# --------------------------------------------------------------------------

# (substring en minúsculas, diagnóstico). El orden importa: gana el primero.
_SIGNATURES: tuple[tuple[str, str], ...] = (
    (
        "confirm you're not a bot",
        "YouTube pidió verificación anti-bot ('Sign in to confirm you're not a bot').",
    ),
    (
        "sign in",
        "YouTube pidió iniciar sesión.",
    ),
    (
        "http error 429",
        "YouTube respondió HTTP 429 (rate limit / demasiadas peticiones).",
    ),
    (
        "too many requests",
        "YouTube respondió 'Too Many Requests' (rate limit).",
    ),
    (
        "no supported javascript runtime",
        "No hay un runtime JavaScript disponible para yt-dlp.",
    ),
    (
        "javascript runtime",
        "yt-dlp no pudo usar un runtime JavaScript.",
    ),
    (
        "failed to decrypt with dpapi",
        "No se pudieron descifrar las cookies del navegador (DPAPI en Windows).",
    ),
    (
        "could not copy chrome cookie database",
        "No se pudo leer la base de cookies del navegador (¿está abierto?).",
    ),
    (
        "video unavailable",
        "El video no está disponible.",
    ),
    (
        "private video",
        "El video es privado.",
    ),
    (
        "members-only",
        "El video es exclusivo para miembros.",
    ),
)

_SUGGESTION = (
    "Sugerencia: verificá `youtube.js_runtime` (por defecto 'node') y, si YouTube "
    "exige autenticación, configurá `youtube.cookies_from_browser` (ej. \"chrome\"). "
    "Ver docs/pending_before_fase4.md."
)


def classify_ytdlp_error(stderr: str) -> str | None:
    """Devuelve un diagnóstico legible si el error de yt-dlp es reconocible."""
    low = (stderr or "").lower()
    for needle, diagnosis in _SIGNATURES:
        if needle in low:
            return diagnosis
    return None


def is_youtube_auth_error(stderr: str) -> bool:
    """True si el error se resuelve (o mitiga) con cookies/autenticación."""
    low = (stderr or "").lower()
    return any(
        n in low
        for n in ("confirm you're not a bot", "sign in", "429", "too many requests")
    )


def diagnostic_hint(stderr: str) -> str | None:
    """Diagnóstico + sugerencia de configuración, o None si no se reconoce."""
    diagnosis = classify_ytdlp_error(stderr)
    if diagnosis is None:
        return None
    return f"{diagnosis} {_SUGGESTION}"
