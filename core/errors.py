"""Excepciones del dominio con mapeo a exit codes del orquestador.

Cada error sabe a qué ExitCode corresponde, para que main.py traduzca
cualquier fallo a un código diferenciado sin duplicar lógica.
"""

from __future__ import annotations

from core.exit_codes import ExitCode


class AutoClipperError(Exception):
    exit_code: ExitCode = ExitCode.GENERIC

    def __init__(self, message: str = "") -> None:
        super().__init__(message)
        self.message = message


class UnknownSourceError(AutoClipperError):
    """No se pudo clasificar la entrada como streaming ni direct_file."""

    exit_code = ExitCode.UNKNOWN_SOURCE


class DownloadError(AutoClipperError):
    exit_code = ExitCode.DOWNLOAD_FAILED


class YouTubeAcquisitionError(DownloadError):
    """Falla de adquisición de YouTube vía yt-dlp: rate-limit/bot-check (429,
    "Sign in to confirm"), runtime JavaScript faltante o formato no disponible.

    Hereda el exit code de download failure (7) para NO inventar un código
    nuevo; se diferencia por tipo/mensaje para dar un diagnóstico accionable.
    """


class DiskSpaceError(AutoClipperError):
    exit_code = ExitCode.DISK_SPACE


class ExtractionError(AutoClipperError):
    """Falla extractando audio (yt-dlp o ffmpeg)."""

    exit_code = ExitCode.FFMPEG_FAILED


class ClipError(AutoClipperError):
    """Falla recortando un clip (ffmpeg local o sección de yt-dlp)."""

    exit_code = ExitCode.FFMPEG_FAILED


class HookValidationError(ClipError):
    """El hook es inválido (timestamps no finitos/negativos, fuera del medio,
    end <= start). Diferenciado por tipo; reusa el exit code de ClipError."""


class InsufficientDurationError(ClipError):
    """El medio completo dura menos que la ventana mínima exigida (<15s)."""


class WindowError(ClipError):
    """No existe una ventana válida para el hook (p.ej. el hook mismo supera
    la duración máxima y recortarlo perdería información semántica)."""


class FFmpegError(ClipError):
    """Falla de FFmpeg/yt-dlp al materializar el clip."""


class ClipFilesystemError(ClipError):
    """Falla de filesystem escribiendo/leyendo el clip o su manifest."""


class TranscriptionError(AutoClipperError):
    exit_code = ExitCode.TRANSCRIPTION_FAILED


class SubtitleError(ClipError):
    """Falla general de subtitulado dinamico (Fase 5): reusa el exit code de
    ClipError (5) para no inventar codigos nuevos."""


class SubtitleDataError(SubtitleError):
    """El transcript/captions son estructuralmente invalidos (timestamps no
    finitos, ruina temporal, texto vacio tras normalizar). Se diferencia por
    tipo/mensaje, no por exit code."""


class SubtitleFontError(SubtitleError):
    """No se pudo resolver una fuente segura (config invalida o ninguna fuente
    detectable). Nunca asumimos una fuente Windows sin verificarla."""


class SubtitleRenderError(SubtitleError):
    """Fallo de ffmpeg/libass al quemar los subtitulos en el clip."""


class LLMFailedError(AutoClipperError):
    """Falla del LLM: JSON mal formado de forma sostenida, error de red/API,
    o respuesta que no valida contra HookCandidate después de los reintentos."""

    exit_code = ExitCode.LLM_FAILED


class BudgetAbort(AutoClipperError):
    """Kill-switch: el job no puede afrontar el próximo paso pago."""

    exit_code = ExitCode.BUDGET_EXCEEDED