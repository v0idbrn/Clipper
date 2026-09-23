"""Kill-switch de presupuesto por job.

Politica: "no gastar un centavo si el próximo paso revienta el límite".

check_budget() se invoca DESDE ADENTRO de transcriber.py / llm_analyzer.py
antes de cada llamada paga — no es un chequeo de una sola vez al final.

Patrón base: CostTracker/CostRecord de `cost-aware-llm-pipeline` (ECC).
Frozen dataclasses = inmutabilidad del registro de gasto: una vez que se
anota un CostRecord no se puede pisar por accidente en medio de un job
concurrente. (ver docs/reused_patterns.md, sección 1).
"""

from __future__ import annotations

from dataclasses import dataclass

import config


@dataclass(frozen=True, slots=True)
class CostRecord:
    """Un gasto puntual inmutable ya incurrido."""

    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float

    @classmethod
    def for_flat_amount(cls, cost_usd: float, *, model: str = "unknown") -> "CostRecord":
        """Registro simple sin desglose de tokens (estimadores baratos)."""
        if cost_usd < 0:
            raise ValueError("cost_usd must be >= 0")
        return cls(model=model, input_tokens=0, output_tokens=0, cost_usd=cost_usd)


@dataclass(frozen=True, slots=True)
class CostTracker:
    """Acumulado inmutable: cada add() devuelve un tracker NUEVO."""

    budget_limit: float = 1.00
    records: tuple[CostRecord, ...] = ()

    def add(self, record: CostRecord) -> "CostTracker":
        """Devuelve nuevo tracker con el record agregado (nunca muta self)."""
        return CostTracker(
            budget_limit=self.budget_limit,
            records=(*self.records, record),
        )

    @property
    def total_cost(self) -> float:
        return sum(r.cost_usd for r in self.records)

    @property
    def over_budget(self) -> bool:
        return self.total_cost > self.budget_limit

    def estimated_over_budget(self, estimated_cost: float) -> bool:
        """True si total + estimado violentaría el límite."""
        if estimated_cost < 0:
            return True
        return self.total_cost + estimated_cost > self.budget_limit


class BudgetExceededError(Exception):
    """Se lanza cuando un job excede (o violentaría) el presupuesto máximo."""

    def __init__(self, spent: float, estimated: float, limit: float) -> None:
        super().__init__(
            f"Budget exceeded: spent=${spent:.4f} + estimated=${estimated:.4f} "
            f"> limit=${limit:.4f}"
        )
        self.spent = spent
        self.estimated = estimated
        self.limit = limit


def _limit() -> float:
    return float(config.get("max_api_cost_per_job", 1.50))


# El contenedor es mutable (dict), los trackers adentro son inmutables.
# Reasignamos la referencia job_id -> new_tracker en cada add(); el tracker
# viejo sigue siendo válido para auditorías pasadas.
_TRACKERS: dict[str, CostTracker] = {}


def get_tracker(job_id: str) -> CostTracker:
    """Devuelve (o crea) el tracker inmutable del job."""
    if job_id not in _TRACKERS:
        _TRACKERS[job_id] = CostTracker(budget_limit=_limit())
    return _TRACKERS[job_id]


def reset(job_id: str) -> None:
    """Borra el tracker de un job (al terminar el job)."""
    _TRACKERS.pop(job_id, None)


def check_budget(estimated_cost: float, job_id: str) -> bool:
    """True si el job puede afrontar estimated_cost sin exceder el límite.

    Devuelve False (abortar) si total + estimated > limit. Jamás lanza
    excepción: la degradación controlada del caller decide cómo abortar.
    """
    tracker = get_tracker(job_id)
    return not tracker.estimated_over_budget(estimated_cost)


def spend(amount: float, job_id: str, *, model: str = "unknown") -> CostRecord:
    """Registra un gasto real como CostRecord inmutable. Devuelve el record."""
    record = CostRecord.for_flat_amount(amount, model=model)
    _TRACKERS[job_id] = get_tracker(job_id).add(record)
    return record


def spent_total(job_id: str) -> float:
    """Gasto total acumulado del job (para auditoría/reporte)."""
    return get_tracker(job_id).total_cost