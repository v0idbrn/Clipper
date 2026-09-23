"""Schemas centrales de AutoClipper (Pydantic).

Validan la salida de módulos externos (Whisper, LLM) ANTES de que un error
mal formado se propague al resto del pipeline.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class TranscriptSegment(BaseModel):
    """Un segmento de transcripción con timestamps a nivel de palabra."""

    start: float = Field(ge=0, description="Inicio del segmento en segundos.")
    end: float = Field(ge=0, description="Fin del segmento en segundos.")
    text: str = Field(min_length=1, description="Texto transcrito.")
    words: list[dict] = Field(
        default_factory=list,
        description="Lista de {start, end, word} a nivel de palabra.",
    )

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


class HookCandidate(BaseModel):
    """Fragmento del video candidato a clip, rankeado por score."""

    start: float = Field(ge=0, description="Inicio del hook en segundos (origen).")
    end: float = Field(
        ge=0, description="Fin del hook en segundos (origen)."
    )
    score: float = Field(
        ge=0.0, le=1.0, description="Score de retención/viralidad 0-1."
    )
    title: str = Field(
        min_length=1, description="Título/gancho comercial del clip."
    )
    reason: str = Field(
        min_length=1, description="Por qué el LLM eligió este fragmento."
    )
    content: Literal["EDITORIAL", "PROMOTIONAL", "UNCERTAIN"] = Field(
        default="EDITORIAL",
        description="Clasificación de propósito: editorial, promocional o duda.",
    )
    ad_reason: str = Field(
        default="",
        description="Evidencia de publicidad si content != EDITORIAL.",
    )

    @field_validator("content", mode="before")
    @classmethod
    def _normalize_content(cls, v):
        if v is None or v == "":
            return "EDITORIAL"
        s = str(v).strip().upper()
        if s in ("EDITORIAL", "PROMOTIONAL", "UNCERTAIN"):
            return s
        return "UNCERTAIN"

    @model_validator(mode="after")
    def _end_after_start(self):
        if self.end < self.start:
            raise ValueError("end debe ser >= start (duración negativa)")
        return self

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


class ClipWindow(BaseModel):
    """Ventana narrativa final (15-90 s) anclada a un hook semántico.

    La construye `core.clip_window.build_clip_window` de forma determinista y
    SIN LLM. El hook queda completamente contenido en [start, end]. El LLM
    identifica el hook; el sistema decide la duración final del clip.
    """

    start: float = Field(ge=0, description="Inicio de la ventana en el medio (s).")
    end: float = Field(ge=0, description="Fin de la ventana en el medio (s).")
    hook_start: float = Field(ge=0, description="Inicio del hook semántico (s).")
    hook_end: float = Field(ge=0, description="Fin del hook semántico (s).")

    @model_validator(mode="after")
    def _validate(self):
        for name in ("start", "end", "hook_start", "hook_end"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} no es finito (NaN/Infinity)")
        if self.end <= self.start:
            raise ValueError("end debe ser > start (ventana degenerada)")
        if self.hook_start < self.start:
            raise ValueError("hook_start debe ser >= start")
        if self.hook_end > self.end:
            raise ValueError("hook_end debe ser <= end")
        if self.hook_end < self.hook_start:
            raise ValueError("hook_end debe ser >= hook_start")

        # Rango 15-90 s configurable (clip.min/max_duration_seconds).
        import config  # import local: mantiene el schema liviano al importar

        min_d = float(config.get("clip.min_duration_seconds", 15))
        max_d = float(config.get("clip.max_duration_seconds", 90))
        if not (min_d <= self.duration <= max_d):
            raise ValueError(
                f"duración {self.duration:.3f}s fuera de [{min_d}, {max_d}]"
            )
        return self

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    @property
    def hook_offset(self) -> float:
        """Segundos de contexto previo desde el inicio de la ventana al hook."""
        return max(0.0, self.hook_start - self.start)


class ClipManifestEntry(BaseModel):
    """Metadata de un clip final ya renderizado en disco."""

    clip_id: str = Field(description="ID único del clip.")
    path: str = Field(description="Path relativo al .mp4 final.")
    hook: HookCandidate
    duration: float = Field(gt=0, description="Duración real del clip en segundos.")
    source_start: float = Field(ge=0, description="Timestamp de inicio en el origen.")
    source_end: float = Field(ge=0, description="Timestamp de fin en el origen.")
    resolution: str = Field(default="1080x1920")
    format: str = Field(default="mp4")
    watermark: bool = Field(default=False, description="True si es variante bloqueada.")
    qc_status: str = Field(
        default="pending", description="pending|passed|failed|blocked_preview."
    )


# Tolerancia de aspect ratio (enteros y rounding a pares hacen imposible el
# 9:16 exacto a cualquier resolución; valores pequeños amplifican el error).
_RATIO_TOLERANCE = 0.02  # 2% respecto al 9:16 objetivo.


class CropGeometry(BaseModel):
    """Geometría determinista del crop central horizontal -> vertical.

    La calcula `core.vertical.calculate_vertical_crop` (función pura, sin
    FFmpeg). El crop recorta de la fuente una región central de aspecto 9:16;
    `target_width/height` es la resolución final del clip vertical (siempre
    par, para yuv420p, y nunca mayor que el target salvo que se evite el
    upscaling absurdo).
    """

    source_width: int = Field(gt=0)
    source_height: int = Field(gt=0)
    crop_width: int = Field(gt=0)
    crop_height: int = Field(gt=0)
    crop_x: int = Field(ge=0)
    crop_y: int = Field(ge=0)
    target_width: int = Field(gt=0)
    target_height: int = Field(gt=0)

    @property
    def source_ratio(self) -> float:
        return self.source_width / self.source_height

    @property
    def crop_ratio(self) -> float:
        return self.crop_width / self.crop_height

    @property
    def target_ratio(self) -> float:
        return self.target_width / self.target_height

    @model_validator(mode="after")
    def _validate(self):
        for name in (
            "source_width", "source_height",
            "crop_width", "crop_height", "crop_x", "crop_y",
            "target_width", "target_height",
        ):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} no es finito (NaN/Infinity)")
        if self.crop_x + self.crop_width > self.source_width:
            raise ValueError("crop se sale del frame por la derecha")
        if self.crop_y + self.crop_height > self.source_height:
            raise ValueError("crop se sale del frame por abajo")
        if self.target_width % 2 != 0 or self.target_height % 2 != 0:
            raise ValueError(f"dimensiones objetivo impares: "
                             f"{self.target_width}x{self.target_height}")
        for ratio in (self.crop_ratio, self.target_ratio):
            if abs(ratio - 9 / 16) / (9 / 16) > _RATIO_TOLERANCE:
                raise ValueError(
                    f"aspect ratio {ratio:.4f} fuera de 9:16 (tol 2%)"
                )
        return self

    @property
    def ffmpeg_crop(self) -> str:
        """'crop=w:h:x:y' listo para -vf de FFmpeg."""
        return f"crop={self.crop_width}:{self.crop_height}:{self.crop_x}:{self.crop_y}"


class VerticalClipEntry(BaseModel):
    """Metadata de un clip vertical 9:16 ya renderizado en disco.

    Cada entrada es trazable hasta su source clip horizontal (mismo clip_id),
    las dimensiones origen, la geometría de crop, la resolución de salida,
    modo de composición, duración real, path de salida y status QC.
    """

    clip_id: str = Field(description="ID del clip (mismo índice que el horizontal).")
    source_clip: str = Field(description="Path relativo del clip horizontal fuente.")
    source_width: int = Field(gt=0)
    source_height: int = Field(gt=0)
    output_width: int = Field(gt=0)
    output_height: int = Field(gt=0)
    aspect_ratio: str = Field(default="9:16")
    composition_mode: str = Field(default="center_crop")
    crop: CropGeometry
    duration: float = Field(gt=0, description="Duración real del vertical (s).")
    path: str = Field(description="Path relativo del .mp4 vertical.")
    qc_status: str = Field(
        default="pending", description="pending|passed|failed."
    )

    @model_validator(mode="after")
    def _validate(self):
        if self.output_width <= 0 or self.output_height <= 0:
            raise ValueError("dimensiones de salida deben ser > 0")
        ratio = self.output_width / self.output_height
        if abs(ratio - 9 / 16) / (9 / 16) > _RATIO_TOLERANCE:
            raise ValueError(f"output {self.output_width}x{self.output_height} "
                             f"no es 9:16 (tol 2%)")
        if self.aspect_ratio != "9:16":
            raise ValueError(f"aspect_ratio debe ser '9:16', no {self.aspect_ratio!r}")
        return self


# ---------------------------------------------------------------------------
# Subtitulos dinamicos (Fase 5)
# ---------------------------------------------------------------------------

# Tolerancia de solapamiento entre word y caption (rounding float).
_WORD_MARGIN_S = 1e-4


class CaptionSegment(BaseModel):
    """Un caption (2-6 palabras) con timing relativo al CLIP (0..duracion).

    El texto se normaliza ANTES; el renderizador nunca vuelve a interpretar
    el texto como comandos (ASS: texto = datos en una linea de Dialogue).
    `words` repite el desglose a nivel de palabra (start/end dentro del
    caption) para permitir word highlight determinista.
    """

    text: str = Field(min_length=1, description="Texto normalizado del caption.")
    start: float = Field(ge=0, description="Inicio en segundos (clip-relativo).")
    end: float = Field(ge=0, description="Fin en segundos (clip-relativo).")
    words: list[dict] = Field(
        default_factory=list,
        description="Lista de {start, end, word} a nivel de palabra.",
    )
    style: str = Field(default="dynamic")
    highlight: bool = Field(default=True, description="Word highlight habilitado.")

    @model_validator(mode="after")
    def _validate(self):
        for name in ("start", "end"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} no es finito (NaN/Infinity)")
        if self.end <= self.start:
            raise ValueError("end debe ser > start (caption degenerado)")
        for w in self.words:
            if not isinstance(w, dict) or "word" not in w or "start" not in w or "end" not in w:
                raise ValueError(f"word invalida (necesita word/start/end): {w!r}")
            ws = w["start"]
            we = w["end"]
            for name, v in (("start", ws), ("end", we)):
                try:
                    vf = float(v)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"word {name} no numerico: {v!r}") from exc
                if not math.isfinite(vf):
                    raise ValueError(f"word {name} no finito: {v!r}")
            if not isinstance(w["word"], str) or not w["word"].strip():
                raise ValueError(f"word sin texto: {w['word']!r}")
            if ws < self.start - _WORD_MARGIN_S or we > self.end + _WORD_MARGIN_S:
                raise ValueError(
                    f"word {w['word']!r} ({ws:.3f}-{we:.3f}) fuera del caption "
                    f"({self.start:.3f}-{self.end:.3f})"
                )
            if we < ws:
                raise ValueError(f"word end < word start: {w!r}")
        return self

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


class SubtitleRenderResult(BaseModel):
    """Metadata de un clip vertical 9:16 ya captioned en disco (Fase 5).

    Trazabilidad hasta su source vertical (mismo clip_id), la fuente de los
    subtitulos (transcript), la cantidad de captions renderizadas, el estilo
    aplicado, la fuente, la posicion y el status QC. Cada entrada es trazable
    sin abrir el clip.
    """

    clip_id: str = Field(description="ID del clip (mismo indice que el vertical).")
    source_clip: str = Field(description="Path relativo del clip vertical fuente.")
    output_clip: str = Field(description="Path relativo del clip captioned.")
    subtitle_source: str = Field(
        default="transcript", description="transcript|none."
    )
    duration: float = Field(gt=0, description="Duracion real del captioned (s).")
    caption_count: int = Field(ge=0, description="Numero de captions quemadas.")
    style: str = Field(default="dynamic", description="Estilo de caption aplicado.")
    font: str = Field(default="", description="Familia de fuente resuelta.")
    position: str = Field(default="bottom_center", description="Posicion en pantalla.")
    highlight_enabled: bool = Field(
        default=True, description="Word highlight activo en este render."
    )
    qc_status: str = Field(
        default="pending", description="pending|passed|failed."
    )

    @model_validator(mode="after")
    def _validate(self):
        if self.subtitle_source not in ("transcript", "none"):
            raise ValueError(f"subtitle_source invalido: {self.subtitle_source!r}")
        if self.position not in ("bottom_center",):
            raise ValueError(f"position invalido: {self.position!r}")
        return self


class JobResult(BaseModel):
    """Resultado global de una corrida del pipeline."""

    job_id: str = Field(description="ID del job.")
    source_url: str = Field(description="URL/archivo de origen.")
    source_type: str = Field(
        default="streaming", description="streaming|direct_file."
    )
    n_clips_requested: int = Field(ge=1)
    clips: list[ClipManifestEntry] = Field(default_factory=list)
    total_cost_usd: float = Field(default=0.0, ge=0)
    status: str = Field(default="completed")
    started_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    finished_at: datetime | None = None