"""Tests de llm_analyzer.py (Fase 2).

Cubren:
- Troceo en ventanas de ~window_tokens con timestamps embebidos.
- Estimación de costo ANTES de cada llamada al LLM + kill-switch (BudgetAbort
  sin tocar la red) — mismo patrón que transcriber.py.
- Validación estricta contra HookCandidate y reintento ante JSON mal formado.
- El caso explícito que exige la spec: JSON roto dos veces seguidas
  => LLMFailedError con exit code 4, sin crasheo silencioso.
- Filtro de duración 15-90s (configurable por config.yaml).

Toda la red se mockea: jamás se llama a un LLM real en el suite.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from core import cost_guard, llm_analyzer  # noqa: E402
from core.errors import BudgetAbort, LLMFailedError  # noqa: E402
from core.exit_codes import ExitCode  # noqa: E402
from models.schemas import TranscriptSegment  # noqa: E402


def _seg(start: float, end: float, text: str) -> TranscriptSegment:
    return TranscriptSegment(start=start, end=end, text=text, words=[])


@pytest.fixture(autouse=True)
def _clean():
    cost_guard._TRACKERS.clear()
    yield
    cost_guard._TRACKERS.clear()


def _sample_segments(n: int = 5, gap: float = 5.0, dur: float = 5.0) -> list[TranscriptSegment]:
    return [_seg(i * gap, i * gap + dur, f"segmento numero {i} para detectar el hook") for i in range(n)]


# --------------------------------------------------------------------------
# Troceo en ventanas
# --------------------------------------------------------------------------

class TestChunking:
    def test_single_window_when_small(self):
        windows = llm_analyzer.chunk_transcript(_sample_segments(3), window_tokens=5000)
        assert len(windows) == 1

    def test_splits_by_token_budget(self):
        # Ventanas chicas fuerzan varias particiones.
        segs = _sample_segments(10)
        windows = llm_analyzer.chunk_transcript(segs, window_tokens=60)
        assert len(windows) > 1
        # Cada ventana preserva el orden original.
        all_segs = [s for w in windows for s in w]
        assert [s.start for s in all_segs] == [s.start for s in segs]

    def test_window_prompt_embeds_timestamps(self):
        segs = _sample_segments(2)
        prompt = llm_analyzer._window_prompt(segs, n_candidates=5)
        assert "[0.0 - 5.0]" in prompt
        assert "segmento numero 0" in prompt
        assert "[5.0 - 10.0]" in prompt


# --------------------------------------------------------------------------
# Estimación de costo antes de cada llamada (kill-switch)
# --------------------------------------------------------------------------

class TestBudgetBeforeCall:
    def test_aborts_before_llm_when_budget_tiny(self, monkeypatch):
        """Spec Fase 2: estimar costo ANTES de cada llamada. Si el presupuesto
        no da, BudgetAbort y el LLM jamás se invoca."""
        monkeypatch.setitem(config._KEEP, "max_api_cost_per_job", 0.0000001)
        calls = {"n": 0}

        def fake_generate(window, n_candidates, **kw):
            calls["n"] += 1
            raise AssertionError("El LLM no debe llamarse si falla el budget")

        monkeypatch.setattr(llm_analyzer, "generate_window", fake_generate)
        cost_guard.reset("job_llm_tiny")

        with pytest.raises(BudgetAbort):
            llm_analyzer.find_hooks(_sample_segments(3), "job_llm_tiny")
        assert calls["n"] == 0, "Se llamó al LLM pese al budget abort"

    def test_aborts_across_windows_when_spent_accumulates(self, monkeypatch):
        """El presupuesto se consulta por CADA ventana; un gasto previo grande
        frena la llamada siguiente antes de que llegue a la red."""
        limit = float(config.get("max_api_cost_per_job", 1.50))
        cost_guard.reset("job_llm_spent")
        cost_guard.spend(limit - 0.00001, job_id="job_llm_spent", model="llm")

        calls = {"n": 0}

        def fake_generate(window, n_candidates, **kw):
            calls["n"] += 1
            raise AssertionError("El LLM no debe llamarse")

        monkeypatch.setattr(llm_analyzer, "generate_window", fake_generate)

        with pytest.raises(BudgetAbort):
            llm_analyzer.find_hooks(_sample_segments(3), "job_llm_spent")
        assert calls["n"] == 0

    def test_estimates_cost_and_registers_spend(self, monkeypatch):
        """Con presupuesto normal, el gasto estimado se registra por llamada."""
        cost_guard.reset("job_llm_ok")

        def fake_generate(window, n_candidates, **kw):
            return llm_analyzer._mock_hook_response(window, n_candidates)

        monkeypatch.setattr(llm_analyzer, "generate_window", fake_generate)
        hooks = llm_analyzer.find_hooks(_sample_segments(4), "job_llm_ok")
        assert cost_guard.spent_total("job_llm_ok") > 0.0
        assert hooks  # el mock produce candidatos (si duran 15-90s)


# --------------------------------------------------------------------------
# Validación estricta + reintento + fallo sostenido (exit 4)
# --------------------------------------------------------------------------

class TestStrictValidationAndRetry:
    def test_valid_json_parsed_into_hook_candidates(self):
        raw = (
            '[{"start": 10.0, "end": 40.0, "score": 0.9, '
            '"title": "El secreto", "reason": "gancho"}]'
        )
        hooks = llm_analyzer.parse_hook_response(raw)
        assert len(hooks) == 1
        assert hooks[0].score == 0.9

    def test_malformed_json_raises_llm_failed(self):
        with pytest.raises(LLMFailedError):
            llm_analyzer.parse_hook_response("esto no es json")

    def test_not_a_list_raises_llm_failed(self):
        with pytest.raises(LLMFailedError):
            llm_analyzer.parse_hook_response('{"start": 0}')

    def test_schema_violation_raises_llm_failed(self):
        # start > end o campos faltantes => no válida contra HookCandidate.
        with pytest.raises(LLMFailedError):
            llm_analyzer.parse_hook_response(
                '[{"start": 50, "end": 10, "score": 0.9, "title": "x", "reason": "y"}]'
            )
        with pytest.raises(LLMFailedError):
            llm_analyzer.parse_hook_response('[{"start": 0, "end": 10}]')

    def test_markdown_fenced_json_accepted(self):
        raw = "```json\n[{\"start\": 0, \"end\": 30, \"score\": 0.5, \"title\": \"t\", \"reason\": \"r\"}]\n```"
        hooks = llm_analyzer.parse_hook_response(raw)
        assert len(hooks) == 1

    def test_retries_once_then_succeeds(self, monkeypatch):
        """Primera respuesta rota -> reintento; la segunda es válida. El job
        NO falla: se recupera con el retry."""
        cost_guard.reset("job_retry")
        segs = _sample_segments(2, gap=20.0, dur=20.0)  # 0-20 y 20-40 (duran 15-90s)
        ok = llm_analyzer._mock_hook_response(segs, n_candidates=3)
        responses = iter(["basura no json", ok])
        calls = {"n": 0}

        def fake_generate(window, n_candidates, **kw):
            calls["n"] += 1
            return next(responses)

        monkeypatch.setattr(llm_analyzer, "generate_window", fake_generate)
        hooks = llm_analyzer.find_hooks(segs, "job_retry")
        assert calls["n"] == 2
        assert hooks

    def test_broken_json_twice_raises_llm_failed_exit_4(self, monkeypatch):
        """El caso explícito de la spec: el LLM devuelve JSON roto DOS veces
        seguidas. El job debe abortar con LLMFailedError -> exit code 4,
        no crashear sin explicación."""
        cost_guard.reset("job_broken")

        def fake_generate(window, n_candidates, **kw):
            return "no te doy json valido"

        monkeypatch.setattr(llm_analyzer, "generate_window", fake_generate)

        with pytest.raises(LLMFailedError) as excinfo:
            llm_analyzer.find_hooks(_sample_segments(2), "job_broken")
        assert int(excinfo.value.exit_code) == int(ExitCode.LLM_FAILED)
        assert "JSON" in excinfo.value.message or "no valida" in excinfo.value.message

    def test_api_misconfigured_raises_llm_failed(self, monkeypatch):
        """mode='api' sin LLM_BASE_URL/LLM_API_KEY => error claro, no KeyError."""
        monkeypatch.setitem(config._KEEP, "llm.mode", "api")
        monkeypatch.setitem(config._KEEP, "env.LLM_BASE_URL", "")
        monkeypatch.setitem(config._KEEP, "env.LLM_API_KEY", "")
        cost_guard.reset("job_noconfig")
        with pytest.raises(LLMFailedError):
            llm_analyzer.find_hooks(_sample_segments(2), "job_noconfig")


# --------------------------------------------------------------------------
# Filtro de duración 15-90s configurable
# --------------------------------------------------------------------------

class TestDurationFilter:
    def test_filters_out_of_range_candidates(self, monkeypatch):
        """Solo hooks >max_duration (120s) se descartan. Hooks cortos (10s)
        son VÁLIDOS porque clip_window.py los expande a 15-90s con contexto."""
        raw = (
            '[{"start": 0, "end": 10, "score": 0.9, "title": "corto", "reason": "a"},'
            '{"start": 20, "end": 60, "score": 0.7, "title": "ok", "reason": "b"},'
            '{"start": 100, "end": 220, "score": 0.8, "title": "largo", "reason": "c"}]'
        )

        def fake_generate(window, n_candidates, **kw):
            return raw

        monkeypatch.setattr(llm_analyzer, "generate_window", fake_generate)
        cost_guard.reset("job_dur")
        hooks = llm_analyzer.find_hooks(_sample_segments(3, gap=10.0, dur=10.0), "job_dur")
        # "corto" (10s) pasa porque clip_window.py lo expande; "largo" (120s) se descarta
        assert [h.title for h in hooks] == ["corto", "ok"]

    def test_duration_bounds_follow_config(self, monkeypatch):
        """Cambiar clip.max_duration_seconds cambia el filtro. Hooks cortos
        siempre pasan (clip_window.py los expande); solo >max se descarta."""
        monkeypatch.setitem(config._KEEP, "clip.max_duration_seconds", 30)
        raw = (
            '[{"start": 0, "end": 20, "score": 0.9, "title": "corto", "reason": "a"},'
            '{"start": 0, "end": 40, "score": 0.6, "title": "largo", "reason": "b"}]'
        )

        def fake_generate(window, n_candidates, **kw):
            return raw

        monkeypatch.setattr(llm_analyzer, "generate_window", fake_generate)
        cost_guard.reset("job_dur2")
        hooks = llm_analyzer.find_hooks(_sample_segments(3), "job_dur2")
        # "corto" (20s) pasa; "largo" (40s > max 30s) se descarta
        assert [h.title for h in hooks] == ["corto"]

    def test_ranking_by_score(self, monkeypatch):
        raw = (
            '[{"start": 0, "end": 40, "score": 0.5, "title": "medio", "reason": "a"},'
            '{"start": 50, "end": 90, "score": 0.95, "title": "alto", "reason": "b"},'
            '{"start": 100, "end": 140, "score": 0.2, "title": "bajo", "reason": "c"}]'
        )

        def fake_generate(window, n_candidates, **kw):
            return raw

        monkeypatch.setattr(llm_analyzer, "generate_window", fake_generate)
        cost_guard.reset("job_rank")
        hooks = llm_analyzer.find_hooks(_sample_segments(5), "job_rank")
        assert [h.title for h in hooks] == ["alto", "medio", "bajo"]

    def test_n_candidates_topk(self, monkeypatch):
        raw = (
            '[{"start": 0, "end": 40, "score": 0.5, "title": "a", "reason": "x"},'
            '{"start": 50, "end": 90, "score": 0.9, "title": "b", "reason": "x"},'
            '{"start": 100, "end": 140, "score": 0.7, "title": "c", "reason": "x"}]'
        )

        def fake_generate(window, n_candidates, **kw):
            return raw

        monkeypatch.setattr(llm_analyzer, "generate_window", fake_generate)
        cost_guard.reset("job_topk")
        hooks = llm_analyzer.find_hooks(_sample_segments(5), "job_topk", n_candidates=2)
        assert [h.title for h in hooks] == ["b", "c"]

    def test_empty_transcript_returns_empty(self, monkeypatch):
        responses = {"n": 0}

        def fake_generate(window, n_candidates, **kw):
            responses["n"] += 1
            raise AssertionError("No debe llamarse el LLM con transcripción vacía")

        monkeypatch.setattr(llm_analyzer, "generate_window", fake_generate)
        cost_guard.reset("job_empty")
        assert llm_analyzer.find_hooks([], "job_empty") == []
        assert responses["n"] == 0


# --------------------------------------------------------------------------
# Estimador de tokens y costo
# --------------------------------------------------------------------------

class TestTokenCostEstimates:
    def test_tokens_scale_with_text_length(self):
        short_t = llm_analyzer.estimate_prompt_tokens("corto")
        long_t = llm_analyzer.estimate_prompt_tokens("curso " * 200)
        assert 0 < short_t < long_t

    def test_cost_scales_with_tokens(self):
        c1 = llm_analyzer.estimate_llm_cost(1000, 300)
        c2 = llm_analyzer.estimate_llm_cost(100_000, 300)
        assert c2 > c1
        # Nunca negativo, incluso con input 0.
        assert llm_analyzer.estimate_llm_cost(0, 300) >= 0.0

    def test_cost_uses_config_rates(self, monkeypatch):
        monkeypatch.setitem(config._KEEP, "llm.price_per_million_input_usd", 3.0)
        monkeypatch.setitem(config._KEEP, "llm.price_per_million_output_usd", 15.0)
        cost = llm_analyzer.estimate_llm_cost(1_000_000, 0)
        assert cost == pytest.approx(3.0)


# --------------------------------------------------------------------------
# P1: contrato de title + validación local + fallback
# --------------------------------------------------------------------------

class TestTitleContract:
    """Tests A-J del P1 truncated hook titles."""

    # A. Título completo -> válido.
    def test_a_complete_title_valid(self):
        title = "The United States' Role in World Affairs"
        hook = (
            "This is a fight between a slave world and a free world, just as "
            "the United States in 1862 could not remain half slave and half free."
        )
        assert llm_analyzer.is_valid_title(title, hook) is True
        assert llm_analyzer.title_issues(title, hook) == []

    # B. Título truncado en preposición -> inválido.
    def test_b_truncated_title_invalid(self):
        title = "Just as the United States in"
        hook = (
            "This is a fight between a slave world and a free world, just as "
            "the United States in 1862 could not remain half slave and half free."
        )
        issues = llm_analyzer.title_issues(title, hook)
        assert "dangling_end" in issues
        assert llm_analyzer.is_valid_title(title, hook) is False

    # C. Ellipsis usado para esconder truncamiento -> inválido.
    def test_c_ellipsis_truncation_invalid(self):
        title = "How the Government..."
        assert "ellipsis_truncation" in llm_analyzer.title_issues(title, "")
        assert llm_analyzer.is_valid_title(title, "") is False

    # D. Título vacío -> inválido.
    def test_d_empty_title_invalid(self):
        assert llm_analyzer.title_issues("") == ["empty"]
        assert llm_analyzer.title_issues("   ") == ["empty"]
        assert llm_analyzer.is_valid_title("") is False

    # E. Título demasiado largo -> inválido.
    def test_e_too_long_title_invalid(self):
        title = "palabra " * 30  # > 120 chars
        assert len(title) > llm_analyzer._TITLE_MAX_LEN
        assert "too_long" in llm_analyzer.title_issues(title, "")

    # F. Caracteres de control -> sanitizados (no quedan en el output).
    def test_f_control_chars_sanitized(self):
        raw = "Title\x00\x07 with\x1f controls"
        cleaned = llm_analyzer.sanitize_title(raw)
        assert cleaned == "Title with controls"
        assert "control_chars" in llm_analyzer.title_issues(raw, "")
        # Tras sanitizar y revalidar sin otros issues -> usable.
        assert llm_analyzer.is_valid_title(cleaned, "") is True

    # G. Título válido que termina en palabra corta de contenido -> NO rechazar.
    def test_g_short_content_word_ending_ok(self):
        title = "A fight for a free world"
        hook = "This is a fight between a slave world and a free world."
        assert llm_analyzer.is_valid_title(title, hook) is True
        # También una sola content-word corta al final.
        assert llm_analyzer.is_valid_title("Slave world and free", hook) is True
        assert llm_analyzer.is_valid_title("Never give up", "") is True

    # H. Título no respaldado por el hook (varias content-words, solape 0).
    def test_h_unsupported_title_invalid(self):
        title = "Quantum blockchain cooking secrets revealed"
        hook = "We have a unique kind of effluent from the Hanford Reactors."
        issues = llm_analyzer.title_issues(title, hook)
        assert "unsupported_by_hook" in issues

    # I. Fallback funciona: inválido -> representación real del hook.
    def test_i_fallback_works(self):
        from models.schemas import HookCandidate

        hook_text = (
            "These men live for 34 days on the open sea in a rubber life raft "
            "eight feet by four feet with no food but that which the sea provides."
        )
        window = [_seg(100.0, 120.0, hook_text)]
        bad = HookCandidate(
            start=102.0, end=118.0, score=0.9,
            title="These men live for 34 days on the open sea in",
            reason="r",
        )
        out = llm_analyzer.enforce_title_contract([bad], window)
        assert out[0].title != "These men live for 34 days on the open sea in"
        assert llm_analyzer.is_valid_title(out[0].title, hook_text) is True
        # Fiel al hook: comparte content-words reales.
        assert "34" in out[0].title or "men" in out[0].title.lower() or "sea" in out[0].title.lower()

    # J. Input idéntico -> resultado determinista (mock LLM determinista).
    def test_j_deterministic_titles(self, monkeypatch):
        raw = (
            '[{"start": 0, "end": 40, "score": 0.9, '
            '"title": "Just as the United States in", "reason": "r"}]'
        )

        def fake_generate(window, n_candidates, **kw):
            return raw

        monkeypatch.setattr(llm_analyzer, "generate_window", fake_generate)
        cost_guard.reset("job_det_title")
        h1 = llm_analyzer.find_hooks(_sample_segments(3), "job_det_title")
        cost_guard.reset("job_det_title2")
        h2 = llm_analyzer.find_hooks(_sample_segments(3), "job_det_title2")
        assert h1 and h2
        assert h1[0].title == h2[0].title
        assert llm_analyzer.is_valid_title(h1[0].title) is True

    def test_prompt_includes_title_contract(self):
        prompt = llm_analyzer._window_prompt(_sample_segments(2), n_candidates=3)
        assert "standalone" in prompt
        assert "..." in prompt or "ellipsis" in prompt.lower() or "puntos" in prompt
        assert "title" in prompt

    def test_mock_titles_not_midword_truncated(self):
        segs = [
            _seg(0.0, 20.0, "Hola mundo esto es una demostracion del mock analyzer offline."),
        ]
        raw = llm_analyzer._mock_hook_response(segs, n_candidates=1)
        import json
        items = json.loads(raw)
        t = items[0]["title"]
        assert t
        assert not t.endswith("...")
        # No cortado a mitad de palabra por el viejo [:60].
        assert llm_analyzer.is_valid_title(t, segs[0].text) is True


# --------------------------------------------------------------------------
# P1: advertisement selection (bloqueo de anuncios, sin falsos positivos)
# --------------------------------------------------------------------------

_LONGINES_HOOK = "The only watch in history to win 10 world's fair grand prizes."
_LONGINES_CTX = (
    "No other name on a watch means so much as Longine. "
    "The world's most honored watch. The only watch in history to win 10 "
    "world's fair grand prizes. 28 gold medals. And yet you may buy and own, "
    "or buy and proudly give the Longine watch for as little as 71.50. "
    "And other beautiful Longine watches sold only by authorized jeweler agencies."
)


class TestAdvertisementContract:
    """Tests A-J del P1 advertisement selection."""

    def test_a_obvious_advertisement_marked(self):
        content, reason = llm_analyzer.classify_advertisement(
            "Buy now and save on your next order.",
            "Special offer for as little as 19.99. Order now.",
        )
        assert content == "PROMOTIONAL"
        assert reason

    def test_b_brand_mention_stays_editorial(self):
        content, _ = llm_analyzer.classify_advertisement(
            "Apple reported record revenue this quarter in the interview.",
            "The CEO discussed supply chain challenges without pitching a product.",
        )
        assert content == "EDITORIAL"

    def test_c_interview_product_not_auto_discarded(self):
        content, _ = llm_analyzer.classify_advertisement(
            "She explained how the new battery chemistry works in the lab.",
            "No call to action, no price, just technical explanation from the founder.",
        )
        assert content == "EDITORIAL"

    def test_d_sponsor_marked(self):
        content, reason = llm_analyzer.classify_advertisement(
            "This program is sponsored by the Longine company.",
            "Brought to you by authorized dealers nationwide.",
        )
        assert content == "PROMOTIONAL"
        assert "sponsor" in reason or "cta" in reason or "retail" in reason

    def test_e_offer_marked(self):
        content, _ = llm_analyzer.classify_advertisement(
            "Available now for only $49 — order today.",
            "Buy now and get free shipping.",
        )
        assert content == "PROMOTIONAL"

    def test_f_historical_product_stays_editorial(self):
        content, _ = llm_analyzer.classify_advertisement(
            "In 1957 the Longine Chronoscope was introduced to collectors.",
            "Historians describe the launch as a milestone in mid-century design.",
        )
        assert content == "EDITORIAL"

    def test_g_insufficient_evidence_uncertain_not_reject(self):
        content, reason = llm_analyzer.classify_advertisement(
            "The only candidate left in the race so far is unclear.",
            "",
        )
        # Sin señal fuerte de promo: no hard-reject.
        assert content in ("EDITORIAL", "UNCERTAIN")
        assert content != "PROMOTIONAL"

        # Claim de producto solo, sin CTA/precio/sponsor -> UNCERTAIN (no bloqueo).
        content2, _ = llm_analyzer.classify_advertisement(
            "The world's most honored watch brand statement.",
            "No price, no call to action in the surrounding transcript.",
        )
        assert content2 == "UNCERTAIN"

    def test_h_deterministic_same_input_same_output(self):
        a1 = llm_analyzer.classify_advertisement(_LONGINES_HOOK, _LONGINES_CTX)
        a2 = llm_analyzer.classify_advertisement(_LONGINES_HOOK, _LONGINES_CTX)
        assert a1 == a2
        assert a1[0] == "PROMOTIONAL"

    def test_i_longines_regression_not_selected(self, monkeypatch):
        segs = [
            _seg(741.0, 760.0, "Exposite Longine watches in matching styles. "
                 "Each bride's watch, a diminutive replica of the groom's watch."),
            _seg(760.0, 766.0, "No other name on a watch means so much as Longine. "
                 "The world's most honored watch."),
            _seg(766.0, 771.0, _LONGINES_HOOK),
            _seg(771.0, 790.0, "28 gold medals. And yet you may buy and own, "
                 "or buy and proudly give the Longine watch for as little as 71.50."),
            _seg(790.0, 800.0, "And other beautiful Longine watches sold only "
                 "by authorized jeweler agencies."),
        ]
        ad_json = (
            '[{"start": 766.0, "end": 771.0, "score": 0.95, '
            '"title": "The only watch in history to win 10 world\'s fair grand prizes.", '
            '"reason": "Bold claim", "content": "EDITORIAL"}, '
            '{"start": 759.0, "end": 762.8, "score": 0.90, '
            '"title": "No other name on a watch means so much as Longine.", '
            '"reason": "Brand statement", "content": "EDITORIAL"}]'
        )

        def fake_generate(window, n_candidates, **kw):
            return ad_json

        monkeypatch.setattr(llm_analyzer, "generate_window", fake_generate)
        cost_guard.reset("job_longines")
        hooks = llm_analyzer.find_hooks(segs, "job_longines", n_candidates=5)
        # Ninguno de los candidatos publicitarios debe quedar seleccionado.
        for h in hooks:
            assert "Longine" not in h.title or "sold only" not in h.title.lower()
            assert "grand prizes" not in h.title.lower()
            assert "No other name on a watch" not in h.title
        # Los dos ads del bloque Longines fueron bloqueados (no en la lista).
        assert all("grand prizes" not in h.title.lower() for h in hooks)
        assert hooks == [] or all(
            h.content in ("EDITORIAL", "UNCERTAIN") for h in hooks
        )

    def test_j_strong_editorial_hook_still_selectable(self, monkeypatch):
        segs = [
            _seg(0.0, 10.0, "Senator, you claim the bill protects workers."),
            _seg(10.0, 20.0, "But the data shows the opposite happened last year."),
            _seg(20.0, 30.0, "Opponents call it the largest policy failure in decades."),
        ]
        editorial_json = (
            '[{"start": 0.0, "end": 25.0, "score": 0.92, '
            '"title": "Senator, you claim the bill protects workers.", '
            '"reason": "Direct challenge", "content": "EDITORIAL"}]'
        )

        def fake_generate(window, n_candidates, **kw):
            return editorial_json

        monkeypatch.setattr(llm_analyzer, "generate_window", fake_generate)
        cost_guard.reset("job_editorial")
        hooks = llm_analyzer.find_hooks(segs, "job_editorial", n_candidates=3)
        assert len(hooks) == 1
        assert hooks[0].content == "EDITORIAL"
        assert "Senator" in hooks[0].title

    def test_ranking_ad_not_in_final_candidates(self, monkeypatch):
        segs = [
            _seg(0.0, 10.0, "Editorial opening about elections."),
            _seg(10.0, 20.0, "More analysis from the newsroom desk."),
            _seg(766.0, 771.0, _LONGINES_HOOK),
            _seg(771.0, 795.0, "And yet you may buy and own for as little as 71.50. "
                 "Sold only by authorized jeweler agencies."),
        ]
        mixed = (
            '[{"start": 766.0, "end": 771.0, "score": 0.99, '
            '"title": "The only watch in history to win 10 world\'s fair grand prizes.", '
            '"reason": "Bold claim", "content": "EDITORIAL"}, '
            '{"start": 0.0, "end": 15.0, "score": 0.70, '
            '"title": "Editorial opening about elections.", '
            '"reason": "News hook", "content": "EDITORIAL"}]'
        )

        def fake_generate(window, n_candidates, **kw):
            return mixed

        monkeypatch.setattr(llm_analyzer, "generate_window", fake_generate)
        cost_guard.reset("job_rank_ad")
        hooks = llm_analyzer.find_hooks(segs, "job_rank_ad", n_candidates=5)
        assert hooks
        assert all("grand prizes" not in h.title.lower() for h in hooks)
        assert any("elections" in h.title.lower() for h in hooks)

    def test_prompt_includes_advertisement_contract(self):
        prompt = llm_analyzer._window_prompt(_sample_segments(2), n_candidates=3)
        assert "content" in prompt
        assert "PROMOTIONAL" in prompt
        assert "EDITORIAL" in prompt
        assert "publicit" in prompt.lower() or "promocion" in prompt.lower()

    def test_enforce_filter_separates_blocked(self):
        from models.schemas import HookCandidate

        segs = [
            _seg(766.0, 771.0, _LONGINES_HOOK),
            _seg(771.0, 795.0, "You may buy and own for as little as 71.50. "
                 "Sold only by authorized jeweler agencies."),
            _seg(0.0, 10.0, "A calm historical narration about river ecology."),
        ]
        ads = HookCandidate(
            start=766.0, end=771.0, score=0.95,
            title="The only watch in history to win 10 world's fair grand prizes.",
            reason="claim",
        )
        ed = HookCandidate(
            start=0.0, end=10.0, score=0.5,
            title="A calm historical narration about river ecology.",
            reason="calm",
        )
        selected, blocked = llm_analyzer.enforce_advertisement_filter([ads, ed], segs)
        assert [h.title for h in blocked] == [ads.title]
        assert ed in selected
        assert ads not in selected
        assert ads.content == "PROMOTIONAL"
        assert ed.content == "EDITORIAL"