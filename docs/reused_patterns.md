# Reused Patterns — AutoClipper

Notas de evaluación de skills en `F:/Gigs/Bussiness/ECC/skills/` y decisión de
reutilización. **REGLA: estos patrones solo se aplican en la fase indicada.**
No se importan antes. El orden fase-por-fase es un contrato.

---

## 1. `CostTracker` / `CostRecord` — de `cost-aware-llm-pipeline`

**Origen:** `ECC/skills/cost-aware-llm-pipeline/SKILL.md`

**Qué es:** patrón de tracking de costo de LLM con frozen dataclasses. Una vez
que se registra un gasto no se puede pisar por accidente (inmutabilidad), lo
que lo hace seguro ante jobs concurrentes.

```python
@dataclass(frozen=True, slots=True)
class CostRecord:
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float

@dataclass(frozen=True, slots=True)
class CostTracker:
    budget_limit: float = 1.00
    records: tuple[CostRecord, ...] = ()

    def add(self, record: CostRecord) -> "CostTracker":
        return CostTracker(budget_limit=self.budget_limit, records=(*self.records, record))

    @property
    def total_cost(self) -> float: ...
    @property
    def over_budget(self) -> bool: ...
```

**Cuándo usar:** **Fase 0** — como base REAL de `core/cost_guard.py`, no solo
inspiración conceptual. `check_budget(estimated_cost, job_id)` sigue siendo la
API pública que exige la spec, pero respaldada internamente por un
`CostTracker` por job. Confirmado por el usuario: es más robusto que el esqueleto
original de "función simple".

**Qué fue descartado del skill (no aplica todavía / no aplica):**
- **Model routing por complejidad** (select_model): Fase 11, selección adaptativa
  de modelo Whisper. No ahora.
- **Narrow retry (3 intentos, backoff exp)**: aplica desde Fase 2 (llm_analyzer).
- **Prompt caching**: solo aplica cuando hay system prompt largo, Fase 2+.

---

## 2. Crop vertical 9:16 y detección de silencios — de `video-editing`

**Origen:** `ECC/skills/video-editing/SKILL.md`

### 2a. Crop estático 16:9 → 9:16

```bash
ffmpeg -i input.mp4 -vf "crop=ih*9/16:ih,scale=1080:1920" vertical.mp4
```

**Cuándo usar:** **Fase 4** — crop vertical estático del MVP end-to-end.
NO usar en Fase 0 ni Fase 1.

**Contexto:** el crop estático (centrado con blur de fondo) es lo mínimo que hace
que el output sea entregable; el face-tracking dinámico recién llega en Fase 5.

### 2b. Detección de silencios / muletillas

```bash
ffmpeg -i input.mp4 -af silencedetect=noise=-30dB:d=2 -f null - 2>&1 | grep silence
```

**Cuándo usar:** **Fase 7** — jump cuts automáticos de "eh", "este", pausas
largas antes del crop final. NO usar en Fase 0 ni Fase 1.

**Contexto:** es lo primero que hace un editor humano y lo que más tiempo ahorra;
sin esto el clip "se siente" amateur. Hoy no corresponde: Fase 1 solo llega hasta
transcripción.

---

## 3. `remotion-video-creation` — DESCARTADO (no aplica)

**Origen:** `ECC/skills/remotion-video-creation/`

**Por qué no:** composición de video programática en React (Remotion). Stack
incompatible con AutoClipper, que quema subtítulos con ffmpeg/.ass y procesa
frames con OpenCV/MediaPipe. El concepto de "captions con word highlight" es
conceptualmente afín a la Fase 6 (subtítulos estilo Hormozi), pero la
implementación React no se traslada. No se reutiliza nada.