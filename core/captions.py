"""Subtitulos dinamicos (Fase 5) - logica pura de captions (sin I/O, sin FFmpeg).

Contrato:

    transcript (list[TranscriptSegment], tiempo absoluto del medio)
        ->
    captions_for_window(segments, window_start, window_end)
        ->
    list[CaptionSegment]   (timestamps RELATIVOS al clip: 0..duracion)

Reglas (deterministas):
- Normalizacion de texto ANTES de renderizar: el texto es dato, nunca comandos.
- Word timing: usa TranscriptSegment.words SI existen y son estructuralmente
  validos; agrupa 2-6 palabras por caption dentro de cada segmento.
- Sin word timing (no word_timestamps=True): caption = segmento completo
  (segment-level timing), documentado como limitacion.
- Filtrado al window: se descartan captions sin solape; los solapados se
  clampan y se desplazan a tiempo de clip (0..duracion). Errores de timing
  no finitos -> SubtitleDataError (la data del transcript podrida no debe
  sobrevivir al pipeline).
"""

from __future__ import annotations

import math

from core.errors import SubtitleDataError
from models.schemas import CaptionSegment, TranscriptSegment

# Minimo tolerable de duracion de un caption tras clamp (rounding/dura en ms).
_MIN_CAPTION_S = 0.05


def normalize_text(raw: str) -> str:
    """Normaliza una cadena: colapsa whitespace/newlines a un espacio unico.

    Reemplaza caracteres de control (NUL, etc.) con espacio para evitar
    problemas en el render ASS.
    """
    if raw is None:
        return ""
    cleaned = "".join(ch if (ord(ch) >= 32 or ch in "\t\n\r") else " " for ch in raw)
    return " ".join(cleaned.split())


def _finite(name: str, value: float, what: str) -> None:
    try:
        v = float(value)
    except (TypeError, ValueError) as exc:
        raise SubtitleDataError(
            f"{what}: {name} no numerico: {value!r}"
        ) from exc
    if not math.isfinite(v):
        raise SubtitleDataError(f"{what}: {name} no finito (NaN/Infinity)")
    return None


def words_from_segment(seg: TranscriptSegment):
    """Devuelve las words del segmento SI son validas; None si no hay.

    Levanta SubtitleDataError solo si hay words pero estan podridas
    (no finitas / rangos rotos). Un segmento sin words NO es error: es el
    caso segment-level timing.
    """
    if not seg.words:
        return None
    out = []
    for w in seg.words:
        if not isinstance(w, dict) or "word" not in w:
            raise SubtitleDataError("word invalida (necesita word/start/end)")
        text = normalize_text(str(w.get("word", "")))
        if not text:
            continue
        _finite("start", w.get("start"), "word")
        _finite("end", w.get("end"), "word")
        ws = float(w["start"])
        we = float(w["end"])
        if we < ws:
            raise SubtitleDataError(f"word end < start: {w!r}")
        out.append({"start": ws, "end": we, "word": text})
    return out if out else None


def _chunk_words(
    words: list[dict],
    *,
    max_words: int,
    min_words: int = 2,
) -> list[list[dict]]:
    """Agrupa palabras en captions de <= max_words palabras.

    El ultimo grupo de 1 sola palabra se fusiona con el anterior si no
    excede max_words; si queda aislada, se mantiene (min_words es blando).
    """
    if max_words < 1:
        raise SubtitleDataError(f"max_words_per_caption invalido: {max_words!r}")
    chunks: list[list[dict]] = []
    current: list[dict] = []
    for w in words:
        if len(current) >= max_words:
            chunks.append(current)
            current = []
        current.append(w)
    if current:
        chunks.append(current)

    if len(chunks) >= 2 and len(chunks[-1]) < min_words:
        merged = chunks[-2] + chunks[-1]
        if len(merged) <= max_words:
            chunks[-2] = merged
            chunks.pop()
    return chunks


def captions_from_segment(
    seg: TranscriptSegment,
    *,
    max_words: int,
) -> list[CaptionSegment]:
    """Construye los captions de UN segmento (texto normalizado).

    Con word timing: agrupa palabras (2-6) en captions y arma el texto desde
    los words (no desde seg.text, para que caption.text == concatenacion real).
    Sin word timing: caption unico con el texto completo del segmento.
    """
    words = words_from_segment(seg)
    captions: list[CaptionSegment] = []

    if words is None:
        text = normalize_text(seg.text)
        if not text:
            return captions
        _finite("start", seg.start, "segment")
        _finite("end", seg.end, "segment")
        start = float(seg.start)
        end = float(seg.end)
        if end <= start:
            raise SubtitleDataError(
                f"segmento degenerado (end<=start): {seg.start!r}-{seg.end!r}"
            )
        captions.append(
            CaptionSegment(text=text, start=start, end=end, highlight=True)
        )
        return captions

    for chunk in _chunk_words(words, max_words=max_words):
        start = chunk[0]["start"]
        end = chunk[-1]["end"]
        if end <= start:
            raise SubtitleDataError(f"caption degenerado: {start!r}-{end!r}")
        text = " ".join(w["word"] for w in chunk)
        captions.append(
            CaptionSegment(
                text=text,
                start=start,
                end=end,
                words=chunk,
                highlight=True,
            )
        )
    return captions


def _clamp_and_shift(
    cap: CaptionSegment,
    window_start: float,
    window_end: float,
    *,
    max_words: int,
) -> CaptionSegment | None:
    """Recorta un caption al window y lo desplaza a tiempo de CLIP.

    Descarta captions sin solape o que quedan degenerados tras el clamp.
    Las words tambien se recortan/desplazan (y se re-filtran por el caption
    resultante) para que el highlight nunca dibuje fuera de su caption.
    """
    start = max(cap.start, window_start)
    end = min(cap.end, window_end)
    if end - start < _MIN_CAPTION_S:
        return None

    words = [dict(w) for w in cap.words]
    kept = []
    for w in words:
        ws = max(w["start"], start)
        we = min(w["end"], end)
        if we - ws < 0:
            continue
        kept.append({"start": ws, "end": we, "word": w["word"]})
    if not kept:
        # Sin word timing (segment-level): el caption entero queda.
        kept = words

    # Re-agrupar si el clamp dejo un solo word aislado (evita ruido).
    if len(kept) >= 2:
        kept = _chunk_words(kept, max_words=max_words)[-1]

    text = " ".join(w["word"] for w in kept) if kept else cap.text
    try:
        return CaptionSegment(
            text=text,
            start=round(start - window_start, 3),
            end=round(end - window_start, 3),
            words=[
                {
                    "start": round(w["start"] - window_start, 3),
                    "end": round(w["end"] - window_start, 3),
                    "word": w["word"],
                }
                for w in kept
            ],
            highlight=cap.highlight,
        )
    except Exception as exc:  # noqa: BLE001 - pydantic re-valida la estructura
        raise SubtitleDataError(f"caption invalido tras clamp: {exc}") from exc


def captions_for_window(
    segments: list[TranscriptSegment],
    window_start: float,
    window_end: float,
    *,
    max_words: int | None = None,
) -> list[CaptionSegment]:
    """Construye los captions de UN clip (window en tiempo absoluto del medio).

    Timepo de salida: relativo al clip (0..window_end-window_start).
    Determinista y sin I/O. Devuelve lista ordenada por start.
    """
    if max_words is None:
        import config  # import local: funcion pura hasta el knob

        max_words = int(config.get("subtitles.max_words_per_caption", 6))

    _finite("window_start", window_start, "window")
    _finite("window_end", window_end, "window")
    if window_end <= window_start:
        raise SubtitleDataError(
            f"window degenerada: {window_start!r}-{window_end!r}"
        )

    captions: list[CaptionSegment] = []
    for seg in segments:
        _finite("start", seg.start, "segment")
        _finite("end", seg.end, "segment")
        seg_end = float(seg.end)
        seg_start = float(seg.start)
        if seg_end <= seg_start:
            continue  # segmentos rotos del transcript se ignoran aqui
        if seg_start >= window_end or seg_end <= window_start:
            continue  # sin solape con el clip
        for cap in captions_from_segment(seg, max_words=max_words):
            shifted = _clamp_and_shift(
                cap, window_start, window_end, max_words=max_words
            )
            if shifted is not None:
                captions.append(shifted)

    captions.sort(key=lambda c: (c.start, c.end))
    return captions