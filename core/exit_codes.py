"""Códigos de salida del CLI, diferenciados por tipo de error.

Estos códigos son el contrato con el orquestador comercial (Fase 4+): el
proceso externo los lee de exit code, no parsea texto libre.
"""

from enum import IntEnum


class ExitCode(IntEnum):
    OK = 0
    GENERIC = 1
    INVALID_URL = 2
    TRANSCRIPTION_FAILED = 3
    LLM_FAILED = 4
    FFMPEG_FAILED = 5
    BUDGET_EXCEEDED = 6
    DOWNLOAD_FAILED = 7
    DISK_SPACE = 8
    UNKNOWN_SOURCE = 9


# Aliases de uso frecuente en el código.
EXIT_OK = ExitCode.OK
EXIT_INVALID_URL = ExitCode.INVALID_URL
EXIT_BUDGET = ExitCode.BUDGET_EXCEEDED