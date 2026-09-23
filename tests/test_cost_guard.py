"""Tests unitarios Fase 0/1.

- cost_guard.check_budget() aborta cuando el costo estimado supera el límite.
- El registro de gasto es inmutable (patrón CostTracker de cost-aware-llm-pipeline).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import cost_guard  # noqa: E402
import config  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_trackers():
    yield
    cost_guard._TRACKERS.clear()


# ---------- check_budget() ----------

def test_check_budget_returns_true_when_under_limit():
    cost_guard.reset("job_under")
    assert cost_guard.check_budget(estimated_cost=0.10, job_id="job_under") is True


def test_check_budget_aborts_when_estimated_exceeds_limit():
    limit = float(config.get("max_api_cost_per_job", 1.50))
    cost_guard.reset("job_over")
    assert cost_guard.check_budget(estimated_cost=limit + 1.0, job_id="job_over") is False


def test_check_budget_aborts_when_accumulated_push_over_limit():
    cost_guard.reset("job_accum")
    cost_guard.spend(1.49, job_id="job_accum")
    assert cost_guard.check_budget(estimated_cost=0.10, job_id="job_accum") is False


def test_check_budget_accepts_when_accumulated_fits():
    cost_guard.reset("job_fits")
    cost_guard.spend(1.00, job_id="job_fits")
    limit = float(config.get("max_api_cost_per_job", 1.50))
    assert cost_guard.check_budget(estimated_cost=limit - 1.00, job_id="job_fits") is True


def test_negative_estimate_never_passes():
    cost_guard.reset("job_neg")
    assert cost_guard.check_budget(estimated_cost=-0.01, job_id="job_neg") is False


# ---------- inmutabilidad (CostTracker/CostRecord) ----------

def test_spend_rejects_negative():
    cost_guard.reset("job_neg")
    with pytest.raises(ValueError):
        cost_guard.spend(-0.01, job_id="job_neg")


def test_tracker_add_returns_new_tracker_no_mutation():
    t = cost_guard.CostTracker(budget_limit=1.50)
    record = cost_guard.CostRecord.for_flat_amount(0.50, model="whisper")
    t2 = t.add(record)

    # t queda intacto, t2 tiene el registro
    assert t.total_cost == 0.0
    assert t2.total_cost == 0.50
    assert t.records == ()
    assert len(t2.records) == 1


def test_tracker_is_frozen():
    t = cost_guard.CostTracker(budget_limit=1.50)
    with pytest.raises(Exception):
        t.total_cost = 999.0  # frozen dataclass -> FrozentDataclassError
    # y no se pierde el budget_limit original
    assert t.budget_limit == 1.50


def test_spend_registers_immutable_records_across_job():
    cost_guard.reset("job_rec")
    r1 = cost_guard.spend(0.25, job_id="job_rec", model="whisper")
    r2 = cost_guard.spend(0.15, job_id="job_rec", model="llm")
    assert (r1, r2) == (
        cost_guard.CostRecord(model="whisper", input_tokens=0, output_tokens=0, cost_usd=0.25),
        cost_guard.CostRecord(model="llm", input_tokens=0, output_tokens=0, cost_usd=0.15),
    )
    assert cost_guard.spent_total("job_rec") == 0.40


def test_over_budget_property():
    t = cost_guard.CostTracker(budget_limit=1.00).add(
        cost_guard.CostRecord.for_flat_amount(1.50)
    )
    assert t.over_budget is True
    assert t.estimated_over_budget(0.01) is True


def test_reset_clears_tracker_spend():
    cost_guard.reset("job_reset")
    cost_guard.spend(0.50, job_id="job_reset")
    cost_guard.reset("job_reset")
    assert cost_guard.spent_total("job_reset") == 0.0