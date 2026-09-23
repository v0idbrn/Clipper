"""Tests del Window Builder determinista (Fase 3) — core/clip_window.py.

Cubren la transformación HookCandidate -> ClipWindow SIN LLM:
- validación de timestamps (negativos, NaN, Infinity, end<=start, fuera del medio);
- ventanas (inicio/medio/final, medios de 15/30/90/>90 s, hooks cortos/largos,
  ajuste contra límites, rango 15-90 s);
- múltiples hooks (orden estable, deduplicación, solapamiento excesivo,
  hooks inválidos que no tumban el job).

Todos deterministas: nada de red ni de FFmpeg.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from core import clip_window as cw  # noqa: E402
from core.errors import (  # noqa: E402
    HookValidationError,
    InsufficientDurationError,
    WindowError,
)
from models.schemas import HookCandidate  # noqa: E402


def _hook(start: float, end: float, score: float = 0.8) -> HookCandidate:
    return HookCandidate(start=start, end=end, score=score, title="hook", reason="test")


@pytest.fixture(autouse=True)
def _restore_cfg():
    snapshot = dict(config._KEEP)
    yield
    config._KEEP.clear()
    config._KEEP.update(snapshot)


class TestValidation:
    def test_valid_hook_builds_window(self):
        window = cw.build_clip_window(50.0, 55.0, 120.0)
        assert window.start <= 50.0 <= window.hook_start
        assert window.hook_end <= window.end
        assert 15.0 <= window.duration <= 90.0

    def test_negative_hook_start_raises(self):
        with pytest.raises(HookValidationError):
            cw.build_clip_window(-1.0, 5.0, 120.0)

    def test_negative_hook_end_raises(self):
        with pytest.raises(HookValidationError):
            cw.build_clip_window(0.0, -5.0, 120.0)

    def test_hook_outside_video_raises(self):
        with pytest.raises(HookValidationError):
            cw.build_clip_window(130.0, 135.0, 120.0)

    def test_nan_raises(self):
        with pytest.raises(HookValidationError):
            cw.build_clip_window(float("nan"), 5.0, 120.0)

    def test_infinity_raises(self):
        with pytest.raises(HookValidationError):
            cw.build_clip_window(5.0, float("inf"), 120.0)

    def test_end_before_start_raises(self):
        with pytest.raises(HookValidationError):
            cw.build_clip_window(40.0, 30.0, 120.0)

    def test_media_shorter_than_min_raises(self):
        with pytest.raises(InsufficientDurationError):
            cw.build_clip_window(5.0, 8.0, 12.0)

    def test_unknown_media_duration_raises(self):
        with pytest.raises(InsufficientDurationError):
            cw.build_clip_window(1.0, 2.0, None)

    def test_hook_span_over_max_raises(self):
        # El hook mismo dura 120s > 90: rechazo explícito, no recorte silencioso.
        with pytest.raises(WindowError):
            cw.build_clip_window(0.0, 120.0, 200.0)

    def test_hook_end_beyond_media_is_clamped(self):
        # Hook que termina pasado el medio: se clampa el hook, no se rechaza.
        window = cw.build_clip_window(115.0, 130.0, 120.0)
        assert window.hook_end == pytest.approx(120.0)
        assert window.end <= 120.0


class TestWindows:
    def test_hook_at_beginning_starts_at_zero(self):
        window = cw.build_clip_window(0.0, 3.0, 120.0)
        assert window.start == 0.0
        assert 15.0 <= window.duration <= 90.0

    def test_hook_in_middle_contains_hook(self):
        window = cw.build_clip_window(60.0, 66.0, 200.0)
        assert window.start <= 60.0
        assert window.end >= 66.0
        assert 15.0 <= window.duration <= 90.0

    def test_hook_near_end_shifts_window_backwards(self):
        window = cw.build_clip_window(115.0, 118.0, 120.0)
        assert window.end == pytest.approx(120.0)
        assert window.start >= 0.0
        assert window.duration >= 15.0

    def test_media_exactly_15_seconds(self):
        window = cw.build_clip_window(5.0, 8.0, 15.0)
        assert window.start == pytest.approx(0.0)
        assert window.end == pytest.approx(15.0)
        assert window.duration == pytest.approx(15.0)

    def test_media_30_seconds(self):
        window = cw.build_clip_window(10.0, 14.0, 30.0)
        assert window.duration == pytest.approx(30.0)
        assert window.end <= 30.0

    def test_media_90_seconds(self):
        window = cw.build_clip_window(40.0, 50.0, 90.0)
        assert 15.0 <= window.duration <= 90.0

    def test_media_larger_than_90_targets_30_45(self):
        window = cw.build_clip_window(100.0, 105.0, 600.0)
        assert 30.0 <= window.duration <= 45.0

    def test_short_hook_is_expanded(self):
        window = cw.build_clip_window(50.0, 53.0, 300.0)
        assert window.duration >= 15.0
        assert window.start <= 50.0 and window.end >= 53.0

    def test_long_hook_window_at_least_hook(self):
        window = cw.build_clip_window(20.0, 70.0, 300.0)  # span 50s
        assert window.duration >= 50.0
        assert window.start <= 20.0 and window.end >= 70.0

    def test_respects_media_bounds(self):
        window = cw.build_clip_window(1.0, 4.0, 20.0)
        assert 0.0 <= window.start
        assert window.end <= 20.0

    def test_result_always_in_range_when_media_allows(self):
        for media in (15.0, 20.0, 45.0, 90.0, 120.0, 600.0):
            window = cw.build_clip_window(media * 0.5, min(media * 0.5 + 3, media), media)
            assert 15.0 <= window.duration <= 90.0

    def test_deterministic(self):
        assert cw.build_clip_window(33.3, 39.9, 200.0) == cw.build_clip_window(
            33.3, 39.9, 200.0
        )

    def test_hook_offset_is_previous_context(self):
        window = cw.build_clip_window(50.0, 55.0, 200.0)
        assert window.hook_offset == pytest.approx(window.hook_start - window.start)
        assert window.hook_offset >= 0.0


class TestMultipleHooks:
    def test_orders_by_window_start(self):
        hooks = [_hook(100.0, 105.0, 0.9), _hook(10.0, 15.0, 0.5),
                 _hook(200.0, 205.0, 0.7)]
        selected = cw.build_clip_windows(hooks, 400.0)
        starts = [w.start for _, w in selected]
        assert starts == sorted(starts)

    def test_dedups_identical_windows(self):
        # Medio de 20s: ambas ventanas colapsan al medio completo -> 1 clip.
        hooks = [_hook(2.0, 5.0, 0.9), _hook(10.0, 14.0, 0.5)]
        selected = cw.build_clip_windows(hooks, 20.0)
        assert len(selected) == 1

    def test_dedup_keeps_higher_score(self):
        hooks = [_hook(2.0, 5.0, 0.9), _hook(2.0, 5.0, 0.5)]
        selected = cw.build_clip_windows(hooks, 300.0)
        assert len(selected) == 1
        assert selected[0][0].score == 0.9

    def test_distinct_windows_kept(self):
        hooks = [_hook(10.0, 16.0, 0.8), _hook(120.0, 126.0, 0.7)]
        selected = cw.build_clip_windows(hooks, 300.0)
        assert len(selected) == 2

    def test_excessive_overlap_dropped(self):
        hooks = [_hook(100.0, 110.0, 0.9), _hook(105.0, 115.0, 0.8)]
        selected = cw.build_clip_windows(hooks, 400.0)
        assert len(selected) == 1

    def test_invalid_hook_skipped_not_fatal(self):
        hooks = [_hook(100.0, 105.0, 0.9), _hook(500.0, 505.0, 0.5)]
        selected = cw.build_clip_windows(hooks, 300.0)
        assert len(selected) == 1

    def test_empty_hooks_returns_empty(self):
        assert cw.build_clip_windows([], 300.0) == []

    def test_naming_order_is_stable(self):
        hooks = [_hook(120.0, 126.0, 0.7), _hook(10.0, 16.0, 0.8)]
        selected = cw.build_clip_windows(hooks, 300.0)
        # clip_000 arranca antes que clip_001.
        assert selected[0][1].start < selected[1][1].start


def _snap(
    start: float,
    end: float,
    words: list,
    *,
    hook_start: float,
    hook_end: float,
    min_d: float = 15.0,
    max_d: float = 90.0,
    media: float = 100.0,
) -> tuple[float, float]:
    return cw.snap_to_word_boundaries(
        start,
        end,
        words,
        hook_start=hook_start,
        hook_end=hook_end,
        min_duration=min_d,
        max_duration=max_d,
        media_duration=media,
    )


def _points(words: list) -> set[float]:
    pts: set[float] = set()
    for w in words:
        pts.add(float(w["start"]))
        pts.add(float(w["end"]))
    return pts


def _dense_words(t0: float, t1: float, step: float = 0.5) -> list[dict]:
    words: list[dict] = []
    t = t0
    while t < t1 - 1e-9:
        words.append(
            {
                "start": round(t, 3),
                "end": round(min(t + step, t1), 3),
                "word": f"w{len(words)}",
            }
        )
        t = round(t + step, 3)
    return words


class TestWordBoundarySnap:
    """P1: alineación de ventanas a word timestamps reales del transcript."""

    # A. Propuesta 10.25 -> 20.75 cae en boundaries reales del transcript.
    def test_a_proposed_snaps_to_real_boundaries(self):
        words = _dense_words(0.0, 60.0, 0.5)
        s, e = _snap(10.25, 20.75, words, hook_start=11.0, hook_end=20.0)
        pts = _points(words)
        assert s in pts
        assert e in pts
        assert s <= 11.0
        assert e >= 20.0
        assert 15.0 <= e - s <= 90.0

    # B. Start DENTRO de una palabra -> snap a un límite válido anterior
    # (nunca queda estrictamente dentro del intervalo de la palabra).
    def test_b_start_inside_word_snaps_to_boundary(self):
        words = (
            _dense_words(0.0, 10.0, 0.5)
            + [{"start": 10.0, "end": 14.0, "word": "long"}]
            + _dense_words(14.0, 60.0, 0.5)
        )
        # propuesta 11.5 cae dentro de "long" [10.0, 14.0]
        s, e = _snap(11.5, 40.0, words, hook_start=13.0, hook_end=39.0)
        pts = _points(words)
        assert s in pts
        assert not (10.0 < s < 14.0)
        assert s <= 13.0
        assert 15.0 <= e - s <= 90.0

    # C. End DENTRO de una palabra -> snap a un límite válido posterior.
    def test_c_end_inside_word_snaps_to_boundary(self):
        words = (
            _dense_words(0.0, 30.0, 0.5)
            + [{"start": 30.0, "end": 34.0, "word": "long"}]
            + _dense_words(34.0, 60.0, 0.5)
        )
        # propuesta 32.0 cae dentro de "long" [30.0, 34.0]
        s, e = _snap(10.0, 32.0, words, hook_start=11.0, hook_end=33.0)
        pts = _points(words)
        assert e in pts
        assert not (30.0 < e < 34.0)
        assert e >= 33.0
        assert 15.0 <= e - s <= 90.0

    # D. El hook queda siempre contenido: start <= hook_start, end >= hook_end.
    def test_d_hook_fully_contained(self):
        words = _dense_words(0.0, 60.0, 0.5)
        hook_start, hook_end = 17.3, 25.8
        s, e = _snap(12.0, 30.0, words, hook_start=hook_start, hook_end=hook_end)
        assert s <= hook_start
        assert e >= hook_end
        assert s in _points(words)
        assert e in _points(words)

    # E. Duración final siempre 15-90s vía build_clip_window(words=...).
    def test_e_duration_always_in_range(self):
        words = _dense_words(0.0, 300.0, 0.5)
        pts = _points(words)
        for hs, he in ((10.0, 15.0), (50.0, 55.0), (100.0, 140.0), (250.0, 255.0)):
            window = cw.build_clip_window(hs, he, 300.0, words=words)
            assert 15.0 <= window.duration <= 90.0
            assert window.start in pts
            assert window.end in pts
            assert window.start <= hs
            assert window.end >= he

    # F. Near-start: propuesta 0.0 -> primer word.start alcanzable (0.1).
    def test_f_near_start(self):
        words = (
            [{"start": 0.1, "end": 0.5, "word": "first"}]
            + _dense_words(0.5, 60.0, 0.5)
        )
        s, e = _snap(0.0, 30.0, words, hook_start=2.0, hook_end=25.0)
        pts = _points(words)
        assert s in pts
        assert s == 0.1
        assert s <= 2.0
        assert e >= 25.0
        assert 15.0 <= e - s <= 90.0

    # G. Near-end: propuesta end=media_duration -> último boundary real.
    def test_g_near_end(self):
        media = 100.0
        words = _dense_words(0.0, 99.5, 0.5)
        pts = _points(words)
        s, e = _snap(60.0, 100.0, words, hook_start=61.0, hook_end=70.0, media=media)
        assert e in pts
        assert e == 99.5
        assert e >= 70.0
        assert s <= 61.0
        assert 15.0 <= e - s <= 90.0

    # H. words=[] (snapping solicitado sin timestamps) -> WindowError.
    def test_h_empty_words_raises(self):
        with pytest.raises(WindowError):
            cw.build_clip_window(10.0, 20.0, 120.0, words=[])
        with pytest.raises(WindowError):
            _snap(5.0, 40.0, [], hook_start=10.0, hook_end=20.0, media=120.0)

    # I. NaN / Infinity / end<start / todos fuera del medio -> WindowError.
    def test_i_invalid_words_raise(self):
        kw = dict(hook_start=10.0, hook_end=20.0, media=120.0)
        with pytest.raises(WindowError):
            _snap(
                5.0,
                40.0,
                [{"start": float("nan"), "end": 1.0, "word": "x"}],
                **kw,
            )
        with pytest.raises(WindowError):
            _snap(
                5.0,
                40.0,
                [{"start": 0.0, "end": float("inf"), "word": "x"}],
                **kw,
            )
        with pytest.raises(WindowError):
            _snap(
                5.0,
                40.0,
                [{"start": 5.0, "end": 4.0, "word": "x"}],
                **kw,
            )
        with pytest.raises(WindowError):
            # todos los timestamps caen completamente fuera del medio
            _snap(
                5.0,
                40.0,
                [{"start": 200.0, "end": 210.0, "word": "x"}],
                **kw,
            )
        with pytest.raises(WindowError):
            _snap(
                5.0,
                40.0,
                [{"start": -1.0, "end": 2.0, "word": "x"}],
                **kw,
            )

    # J. Mismo input -> mismo output (determinismo puro).
    def test_j_deterministic(self):
        words = _dense_words(0.0, 60.0, 0.5)
        a = _snap(10.25, 20.75, words, hook_start=11.0, hook_end=20.0)
        b = _snap(10.25, 20.75, words, hook_start=11.0, hook_end=20.0)
        assert a == b

    # Words parcialmente fuera del medio: se excluyen/clampean, no fallan.
    def test_words_partially_outside_media_excluded(self):
        media = 50.0
        words = (
            _dense_words(0.0, 48.0, 1.0)
            + [{"start": 49.0, "end": 55.0, "word": "straddle"}]
            + [{"start": 60.0, "end": 61.0, "word": "outside"}]
        )
        # "outside" se excluye; "straddle" se clampa a 50.0 (fin del medio).
        s, e = _snap(10.0, 45.0, words, hook_start=12.0, hook_end=40.0, media=media)
        pts = _points(words)
        assert s in pts
        assert e in pts
        assert e <= media
        assert s <= 12.0
        assert e >= 40.0
        assert 15.0 <= e - s <= 90.0

    # build_clip_windows: words=[] con hooks presentes -> falla explícita.
    def test_build_clip_windows_empty_words_raises(self):
        with pytest.raises(WindowError):
            cw.build_clip_windows([_hook(10.0, 20.0)], 120.0, words=[])

    # words=None conserva el legacy SIN snapping (no cambia la semántica).
    def test_words_none_keeps_legacy_behavior(self):
        legacy = cw.build_clip_window(50.0, 55.0, 200.0)
        explicit_none = cw.build_clip_window(50.0, 55.0, 200.0, words=None)
        assert legacy == explicit_none


class TestOverlapLimit:
    """Regresión: MAX_OVERLAP_RATIO=0.15 impide clips con >15% de solapamiento."""

    def test_hooks_30s_apart_overlap_dropped(self):
        """Hooks a 30s de distancia: ventanas de 40s se solapan ~10s (25%).
        Con MAX_OVERLAP_RATIO=0.15, el de menor score se descarta."""
        hooks = [_hook(100.0, 106.0, 0.9), _hook(130.0, 136.0, 0.8)]
        selected = cw.build_clip_windows(hooks, 400.0)
        # Solo queda el de mayor score (0.9)
        assert len(selected) == 1
        assert selected[0][0].score == 0.9

    def test_hooks_50s_apart_both_kept(self):
        """Hooks a 50s de distancia: ventanas de 40s NO se solapan."""
        hooks = [_hook(100.0, 106.0, 0.9), _hook(150.0, 156.0, 0.8)]
        selected = cw.build_clip_windows(hooks, 400.0)
        assert len(selected) == 2

    def test_overlap_exact_15pct_boundary(self):
        """Solapamiento exacto en la frontera: 6s overlap / 40s = 15%.
        Con >= en el filtro, 15% exacto se descarta (conservador)."""
        hooks = [_hook(100.0, 106.0, 0.9), _hook(134.0, 140.0, 0.8)]
        selected = cw.build_clip_windows(hooks, 400.0)
        # 15% exacto: se descarta (>= umbral)
        assert len(selected) == 1
        assert selected[0][0].score == 0.9

    def test_overlap_above_15pct_dropped(self):
        """7s overlap / 40s = 17.5% > 15%: el de menor score se descarta."""
        hooks = [_hook(100.0, 106.0, 0.9), _hook(133.0, 139.0, 0.8)]
        selected = cw.build_clip_windows(hooks, 400.0)
        assert len(selected) == 1
        assert selected[0][0].score == 0.9
