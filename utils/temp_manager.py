"""Ciclo de vida de directorios temporales con limpieza garantizada.

Envuelve tempfile.TemporaryDirectory() en un context manager: aunque el
proceso crashee a mitad de pipeline, no queda basura ni "video fantasma"
ocupando disco.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any


class temp_dir:
    """Context manager sobre un tempdir con cleanup garantizado.

    Uso:
        with temp_dir() as tmp:
            path = tmp.path / "audio.mp3"
            ...
        # tmp.path ya no existe al salir.
    """

    def __init__(self, *, prefix: str = "autoclipper_") -> None:
        self._prefix = prefix
        self._tf: tempfile.TemporaryDirectory[Any] | None = None
        self.path: Path | None = None

    def __enter__(self) -> "temp_dir":
        self._tf = tempfile.TemporaryDirectory(prefix=self._prefix)
        self.path = Path(self._tf.name)
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._tf is not None:
            self._tf.cleanup()
        self.path = None
        self._tf = None