"""Selección determinista de la ventana narrativa de un clip (Fase 3).

Separa explícitamente dos responsabilidades:

    SEMANTIC DETECTION   (LLM, Fase 2)  -> HookCandidate (punto/intervalo)
            |
    WINDOW SELECTION     (este módulo)  -> ClipWindow de 15-90 s (SIN LLM)
            |
    MEDIA CUT            (core.clipper) -> clip_000.mp4 (FFmpeg / yt-dlp)

El LLM identifica el hook; el sistema determina la duración final del clip.
Heurística simple, explicable y estable (nada de ML ni de otro LLM):

1. incluir el hook completo (prioridad máxima);
2. conservar contexto previo al hook (pre-roll ~35% de la ventana);
3. conservar desarrollo posterior;
4. apuntar a clip.target_duration_seconds (~30-45 s) cuando el medio lo permite;
5. respetar siempre los límites [0, media_duration];
6. nunca salir del rango [min_duration, max_duration] (15-90 s por defecto);
7. si el transcript trae word timestamps, alinear start/end a límites reales
   de palabra (P1: eliminar cortes a mitad de palabra/frase).
"""

from __future__ import annotations

import math

import config
from core.errors import (
    HookValidationError,
    InsufficientDurationError,
    WindowError,
)
from models.schemas import ClipWindow, HookCandidate
from utils.logger import get_logger

log = get_logger()

# Porcentaje de la ventana destinado a contexto PREVIO al hook. El resto va
# después del hook (desarrollo posterior). Ajustable en el futuro si hiciera
# falta, pero hoy es una constante explicable.
PRE_ROLL_RATIO = 0.35

# Dos ventanas se consideran la misma si arrancan y terminan casi igual.
DEDUP_EPS_SECONDS = 1.0

# Solapamiento máximo tolerado (fracción de la ventana más corta). Por encima
# de esto, se descarta la ventana de menor prioridad para no generar clips
# con contenido repetido. 0.15 = máximo 15% de solapamiento (~6s en clips
# de 40s), suficiente para transiciones naturales pero sin repetición
# sustancial.
MAX_OVERLAP_RATIO = 0.15


def _bounds() -> tuple[float, float, float]:
    min_d = float(config.get("clip.min_duration_seconds", 15))
    max_d = float(config.get("clip.max_duration_seconds", 90))
    target = float(config.get("clip.target_duration_seconds", 40))
    return min_d, max_d, target


def _word_boundaries(words: list, media_duration: float) -> list[tuple[float, float]]:
    """Valida word timestamps y devuelve boundaries (start, end) ordenados.

    Reglas:
    - usa únicamente word["start"] y word["end"];
    - rechaza NaN/Infinity, timestamps negativos y end < start (WindowError);
    - excluye words completamente fuera de [0, media_duration];
    - clamp parcial al medio cuando la word lo solapa;
    - si no queda ningún boundary válido -> WindowError.
    """
    if not words:
        raise WindowError("transcript sin word timestamps")

    valid: list[tuple[float, float]] = []
    for w in words:
        if not isinstance(w, dict) or "start" not in w or "end" not in w:
            raise WindowError(f"word timestamp malformado (falta start/end): {w!r}")
        try:
            s = float(w["start"])
            e = float(w["end"])
        except (TypeError, ValueError) as exc:
            raise WindowError(f"word timestamp no numérico: {w!r}") from exc
        if not math.isfinite(s) or not math.isfinite(e):
            raise WindowError(f"word timestamp no finito (NaN/Infinity): {w!r}")
        if s < 0 or e < 0:
            raise WindowError(f"word timestamp negativo: start={s} end={e}")
        if e < s:
            raise WindowError(f"word degenerado (end < start): start={s} end={e}")
        # Completamente fuera del medio -> se excluye (no fabricar timestamps).
        if e <= 0.0 or s >= media_duration:
            continue
        cs = max(0.0, s)
        ce = min(media_duration, e)
        if ce > cs:
            valid.append((cs, ce))

    if not valid:
        raise WindowError("transcript sin word timestamps válidos")
    return sorted(valid)


def snap_to_word_boundaries(
    start: float,
    end: float,
    words: list,
    *,
    hook_start: float,
    hook_end: float,
    min_duration: float,
    max_duration: float,
    media_duration: float,
) -> tuple[float, float]:
    """Alinea la propuesta [start, end] a límites reales de word timestamps.

    Contrato determinista:
    - candidatos SOLO word.start / word.end (nunca se inventan timestamps);
    - start <= hook_start (más cercano a la propuesta; empate -> el mayor);
    - end >= hook_end (más cercano a la propuesta; empate -> el menor);
    - 15 <= duration <= 90 (o el rango [min_duration, max_duration] recibido);
    - si la primera combinación no cumple duración, BFS por desviación
      creciente total = i + j sobre las listas ya ordenadas por distancia;
    - si ninguna combinación válida existe -> WindowError.

    Devuelve (start, end) alineados.
    """
    bounds = _word_boundaries(words, media_duration)
    points = sorted({p for se in bounds for p in se})

    start_cands = [p for p in points if p <= hook_start]
    if not start_cands:
        raise WindowError(
            f"sin word boundary <= hook_start ({hook_start:.3f}) para el snap"
        )
    start_cands.sort(key=lambda p: (abs(p - start), -p))

    end_cands = [p for p in points if p >= hook_end]
    if not end_cands:
        raise WindowError(
            f"sin word boundary >= hook_end ({hook_end:.3f}) para el snap"
        )
    end_cands.sort(key=lambda p: (abs(p - end), p))

    max_total = len(start_cands) + len(end_cands) - 1
    for total in range(max_total + 1):
        for i in range(min(total, len(start_cands) - 1) + 1):
            j = total - i
            if j >= len(end_cands):
                continue
            s = start_cands[i]
            e = end_cands[j]
            if s < 0.0 or e > media_duration or e <= s:
                continue
            if min_duration <= (e - s) <= max_duration:
                return s, e

    raise WindowError(
        f"sin combinación de word boundaries con duración en "
        f"[{min_duration}, {max_duration}]"
    )


def build_clip_window(
    hook_start: float,
    hook_end: float,
    media_duration: float | None,
    *,
    min_duration: float | None = None,
    max_duration: float | None = None,
    target_duration: float | None = None,
    words: list | None = None,
) -> ClipWindow:
    """Construye la ventana narrativa [start, end] para un hook.

    Determinista (mismas entradas -> misma salida), sin LLM ni red.

    `words`:
    - None (default): comportamiento legacy SIN snapping.
    - [] o lista con word timestamps: aplica snapping a límites reales de
      palabra tras los clamps finales; [] -> WindowError.

    Levanta:
    - HookValidationError: timestamps NaN/infinitos/negativos, hook fuera del
      medio, o end <= start.
    - InsufficientDurationError: duración del medio desconocida/inválida o
      menor que la ventana mínima.
    - WindowError: el hook mismo supera la duración máxima, o no existe
      combinación de word boundaries válida (no se recorta silenciosamente
      información semántica ni se fabrican timestamps).
    """
    min_d, max_d, target = _bounds()
    if min_duration is not None:
        min_d = float(min_duration)
    if max_duration is not None:
        max_d = float(max_duration)
    if target_duration is not None:
        target = float(target_duration)

    # 1) Timestamps finitos y no negativos.
    for name, value in (("hook_start", hook_start), ("hook_end", hook_end)):
        if not math.isfinite(value):
            raise HookValidationError(f"{name} no es finito (NaN/Infinity): {value!r}")
    if hook_start < 0 or hook_end < 0:
        raise HookValidationError(
            f"timestamps negativos no válidos: start={hook_start}, end={hook_end}"
        )
    if hook_end < hook_start:
        raise HookValidationError("hook degenerado: end <= start")

    # 2) Configuración y duración del medio.
    if not math.isfinite(min_d) or not math.isfinite(max_d) or min_d <= 0 or max_d < min_d:
        raise WindowError(f"config de duración inválida: min={min_d} max={max_d}")
    if media_duration is None or not math.isfinite(media_duration) or media_duration <= 0:
        raise InsufficientDurationError("duración del medio desconocida o inválida")
    if media_duration < min_d:
        raise InsufficientDurationError(
            f"el medio dura {media_duration:.3f}s < mínimo {min_d:.3f}s"
        )
    if hook_start >= media_duration:
        raise HookValidationError(
            f"hook fuera del medio: start={hook_start:.3f} >= {media_duration:.3f}"
        )

    # El hook no puede terminar más allá del medio.
    hook_end = min(hook_end, media_duration)
    hook_span = hook_end - hook_start
    if hook_span > max_d:
        raise WindowError(
            f"el hook dura {hook_span:.3f}s > máximo {max_d:.3f}s; "
            "recortarlo perdería información semántica"
        )

    target = min(max_d, max(min_d, target))

    # 3) Duración deseada: target, pero nunca menos que el propio hook.
    desired = max(hook_span, target)
    desired = min(desired, max_d, media_duration)

    # Anclar el hook con pre-roll, intentando conservar contexto previo.
    start = hook_start - PRE_ROLL_RATIO * desired
    if start + desired < hook_end:  # el hook debe entrar completo
        start = hook_end - desired
    if start < 0:
        start = 0.0
    if start + desired > media_duration:
        start = media_duration - desired
    if start < 0:
        start = 0.0
    if start > hook_start:  # nunca arrancar después del hook
        start = hook_start

    end = start + desired
    if end > media_duration:
        end = media_duration
        start = max(0.0, end - desired)
    if end < hook_end:  # garantizar el hook completo
        end = hook_end
        start = max(0.0, end - desired)
        if start > hook_start:
            start = hook_start

    # 4) Defensa final: rango [min_d, max_d] y límites del medio.
    if end - start < min_d:
        end = min(media_duration, start + min_d)
        if end - start < min_d:
            start = max(0.0, end - min_d)
    if end - start > max_d:
        start = max(start, end - max_d)
        if start > hook_start:
            start = hook_start

    start = max(0.0, min(start, media_duration))
    end = min(media_duration, max(end, start))
    if end <= start:
        raise WindowError("ventana degenerada tras el ajuste")

    # Word-boundary snapping (P1): alinear a timestamps reales del transcript
    # DESPUÉS de los clamps finales y ANTES de construir ClipWindow.
    if words is not None:
        start, end = snap_to_word_boundaries(
            start,
            end,
            words,
            hook_start=hook_start,
            hook_end=hook_end,
            min_duration=min_d,
            max_duration=max_d,
            media_duration=media_duration,
        )

    return ClipWindow(
        start=start, end=end, hook_start=hook_start, hook_end=hook_end
    )


def window_for_hook(
    hook: HookCandidate,
    media_duration: float | None,
    *,
    min_duration: float | None = None,
    max_duration: float | None = None,
    target_duration: float | None = None,
    words: list | None = None,
) -> ClipWindow:
    """Conveniencia: construye la ventana de un HookCandidate."""
    return build_clip_window(
        hook.start,
        hook.end,
        media_duration,
        min_duration=min_duration,
        max_duration=max_duration,
        target_duration=target_duration,
        words=words,
    )


def _overlap_seconds(a: ClipWindow, b: ClipWindow) -> float:
    return max(0.0, min(a.end, b.end) - max(a.start, b.start))


def _is_redundant(window: ClipWindow, kept: list[ClipWindow]) -> bool:
    """True si `window` duplica o solapa en exceso una ventana ya conservada."""
    for other in kept:
        if (
            abs(window.start - other.start) <= DEDUP_EPS_SECONDS
            and abs(window.end - other.end) <= DEDUP_EPS_SECONDS
        ):
            return True
        shorter = min(window.duration, other.duration)
        if shorter > 0 and _overlap_seconds(window, other) / shorter >= MAX_OVERLAP_RATIO:
            return True
    return False


def build_clip_windows(
    hooks: list[HookCandidate],
    media_duration: float | None,
    *,
    min_duration: float | None = None,
    max_duration: float | None = None,
    target_duration: float | None = None,
    words: list | None = None,
) -> list[tuple[HookCandidate, ClipWindow]]:
    """Convierte N hooks en ventanas no redundantes, ordenadas de forma estable.

    - Un hook inválido/imposible se descarta con log (no tumba el job entero).
    - Un medio globalmente demasiado corto (<min_duration) es error explícito.
    - Si se solicitó snapping (words is not None) y no hay words pero sí hooks,
      falla explícito con WindowError (nunca 0 clips silenciosos con exit 0).
    - Deduplica ventanas casi idénticas y descarta solapamientos excesivos,
      priorizando el hook de mayor score.
    - Orden final determinista: por (inicio de ventana, fin, inicio del hook,
      -score), para que clip_000..clip_NNN sean estables.
    """
    if not hooks:
        return []

    if words is not None and not words:
        raise WindowError(
            "transcript sin word timestamps: se solicitó snapping (words=[]) "
            "con hooks presentes"
        )

    built: list[tuple[HookCandidate, ClipWindow]] = []
    for hook in hooks:
        try:
            window = window_for_hook(
                hook,
                media_duration,
                min_duration=min_duration,
                max_duration=max_duration,
                target_duration=target_duration,
                words=words,
            )
        except InsufficientDurationError:
            raise  # condición global del medio, no del hook
        except (HookValidationError, WindowError) as exc:
            log.warn(
                "clip_window_skipped",
                hook_start=round(hook.start, 3),
                hook_end=round(hook.end, 3),
                error=str(exc),
            )
            continue
        built.append((hook, window))

    # Prioridad para elegir entre ventanas redundantes: score alto primero;
    # desempate determinista por posición temporal.
    ordered = sorted(
        built, key=lambda hw: (-hw[0].score, hw[1].start, hw[1].end)
    )
    kept: list[tuple[HookCandidate, ClipWindow]] = []
    for hook, window in ordered:
        if _is_redundant(window, [w for _, w in kept]):
            continue
        kept.append((hook, window))

    # Orden estable de salida para nombrar clip_000, clip_001, ...
    kept.sort(key=lambda hw: (hw[1].start, hw[1].end, hw[0].start, -hw[0].score))
    return kept
