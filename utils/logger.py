"""Logging estructurado en JSON lines a stdout.

Cada línea es un objeto JSON autocontenido para que un orquestador externo
pueda parsearlo sin tocar texto libre. Formato:
{"ts": "...", "level": "info", "event": "transcribing", "stage": "2", ...}

Uso:
    from utils.logger import get_logger
    log = get_logger()
    log.info("transcribing", progress=0.4, job_id="job_abc")
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from typing import Any


class JsonLogger:
    """Logger liviano que emite un JSON object por línea a stdout."""

    def __init__(self, stream: Any = None) -> None:
        self._stream = stream if stream is not None else sys.stdout

    def _emit(self, level: str, event: str, **fields: Any) -> None:
        record: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": level,
            "event": event,
        }
        record.update(fields)
        self._stream.write(json.dumps(record, default=str) + "\n")
        self._stream.flush()

    def debug(self, event: str, **fields: Any) -> None:
        self._emit("debug", event, **fields)

    def info(self, event: str, **fields: Any) -> None:
        self._emit("info", event, **fields)

    def warn(self, event: str, **fields: Any) -> None:
        self._emit("warn", event, **fields)

    def error(self, event: str, **fields: Any) -> None:
        self._emit("error", event, **fields)


_logger: JsonLogger | None = None


def get_logger() -> JsonLogger:
    """Devuelve el logger singleton; permite inyectar stream en tests."""
    global _logger
    if _logger is None:
        _logger = JsonLogger()
    return _logger


def reset_logger() -> None:
    """Resetea el singleton (para tests)."""
    global _logger
    _logger = None