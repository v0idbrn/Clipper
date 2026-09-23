"""Detección de hooks (Fase 2) — windowed LLM analysis.

Trocea la transcripción en ventanas de ~window_tokens con timestamps
embebidos, pide al LLM candidatos HookCandidate en JSON estricto, valida
contra el schema Pydantic, filtra por duración 15-90s y rankea por score.

Contrato económico (mismo patrón que transcriber.py):
- ANTES de cada llamada al LLM se estima el costo por tokens del prompt y se
  consulta cost_guard.check_budget(). Si el estimado revienta el presupuesto,
  BudgetAbort SIN tocar la red.
- El gasto real estimado se registra con cost_guard.spend() por ventana.

Modos (config llm.mode):
- "api":  llama al LLM real vía LLM_BASE_URL/LLM_API_KEY (OpenAI-compatible).
- "mock": analizador sintético determinista SIN red. Para desarrollo y para el
  suite de tests offline. NUNCA reemplaza al LLM real en producción.
"""

from __future__ import annotations

import json
import re

import requests

import config
from core import cost_guard
from core.errors import BudgetAbort, LLMFailedError
from models.schemas import HookCandidate, TranscriptSegment
from utils.logger import get_logger

log = get_logger()

_CHARS_PER_TOKEN = 4.0
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)

# Contrato de title (P1): standalone, no truncado, fiel al hook.
_TITLE_MAX_LEN = 120
# Final en function-word/coma => casi siempre truncado sintáctico. No incluye
# content-words cortas (world, war, free...) para no rechazar títulos válidos.
_DANGLING_END_RE = re.compile(
    r"(?:[,;:-]\s*|\b(?:a|an|the|of|in|to|and|or|but|because|when|that|which|"
    r"who|whom|whose|with|for|on|at|from|by|as|is|are|was|were|be|been|being|"
    r"have|has|had|do|does|did|will|would|can|could|should|may|might|must|"
    r"if|then|than|so|about|into|via|per)\s*)+$",
    re.IGNORECASE,
)
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# Contrato de advertisement (P1): señales FUERTES de propósito promocional.
# Una mención de marca/producto NO basta; se exige evidencia de venta/promo.
_AD_SPONSOR_RE = re.compile(
    r"\b(?:sponsored by|brought to you by|broadcast on behalf of|"
    r"in association with|presented by)\b",
    re.IGNORECASE,
)
_AD_CTA_RE = re.compile(
    r"\b(?:buy now|order now|order today|call now|click here|"
    r"you may buy|buy and (?:own|proudly|today)|order yours|"
    r"shop now|purchase (?:now|today)|subscribe (?:now|today)|"
    r"for as little as|available (?:now|exclusively)|"
    r"special (?:offer|price|deal)|limited[- ]time (?:offer|deal|sale)|"
    r"promo(?:code|tion code)|only \$\s*\d|for only \$\s*\d)\b",
    re.IGNORECASE,
)
_AD_RETAIL_RE = re.compile(
    r"\b(?:sold only|authorized (?:dealer|jeweler|dealers|jewelers|agent|agents)|"
    r"authorized service center|while supplies last|order line)\b",
    re.IGNORECASE,
)
_AD_PRICE_RE = re.compile(
    r"(?:\$|usd\s*)?\b\d{1,3}[.,]\d{2}\b|\bonly\s+\$?\d+(?:\.\d+)?\b",
    re.IGNORECASE,
)
_AD_PROMO_CLAIM_RE = re.compile(
    r"\b(?:the only .{0,40} in history|world'?s most (?:honored|expensive|popular)|"
    r"no other (?:name|watch|brand) .{0,30} like)\b",
    re.IGNORECASE,
)
_AD_CONTEXT_SPAN_PRE_S = 20.0
_AD_CONTEXT_SPAN_POST_S = 45.0
_CONTENT_RANK = {"EDITORIAL": 2, "UNCERTAIN": 1}
# Palabras vacías EN/ES para solape title↔hook (validación de respaldo).
_TITLE_STOPWORDS = frozenset(
    """a an the of in to and or but because when that which who whom whose with
    for on at from by as is are was were be been being have has had do does did
    will would can could should may might must if then than so about into over
    under up down out off via per this these those such very just also
    el la los las un una unos unas y o pero porque cuando que cual quien con
    para en del al como es son fue ser estar hay no si mas más bien ya solo sólo
    """.split()
)


# --------------------------------------------------------------------------
# Estimación de costo (kill-switch, ANTES de cada llamada)
# --------------------------------------------------------------------------

def estimate_prompt_tokens(text: str) -> int:
    """Aproximación de tokens de un texto (~4 chars/token). Suficiente para
    el kill-switch; el costo real se audita por separado."""
    return max(1, round(len(text) / _CHARS_PER_TOKEN))


def estimate_llm_cost(input_tokens: int, output_estimate: int) -> float:
    """Costo estimado (USD) de una llamada: input + salida esperada."""
    p_in = float(config.get("llm.price_per_million_input_usd", 1.00))
    p_out = float(config.get("llm.price_per_million_output_usd", 5.00))
    return (input_tokens / 1e6) * p_in + (output_estimate / 1e6) * p_out


# --------------------------------------------------------------------------
# Troceo en ventanas con timestamps embebidos
# --------------------------------------------------------------------------

def _segment_line(seg: TranscriptSegment) -> str:
    return f"[{seg.start:.1f} - {seg.end:.1f}] {seg.text}"


def _segment_tokens(seg: TranscriptSegment) -> int:
    return estimate_prompt_tokens(_segment_line(seg))


def chunk_transcript(
    segments: list[TranscriptSegment],
    window_tokens: int | None = None,
) -> list[list[TranscriptSegment]]:
    """Divide la transcripción en ventanas de ~window_tokens (default 3500)."""
    limit = window_tokens or int(config.get("llm.window_tokens", 3500))
    windows: list[list[TranscriptSegment]] = []
    current: list[TranscriptSegment] = []
    current_tokens = 0
    for seg in segments:
        seg_tokens = _segment_tokens(seg)
        if current and current_tokens + seg_tokens > limit:
            windows.append(current)
            current = []
            current_tokens = 0
        current.append(seg)
        current_tokens += seg_tokens
    if current:
        windows.append(current)
    return windows


def _window_prompt(window: list[TranscriptSegment], n_candidates: int) -> str:
    """Prompt de usuario con timestamps embebidos y la consigna de formato."""
    lines = "\n".join(_segment_line(seg) for seg in window)
    return (
        "A continuacion, una transcripcion de video con timestamps "
        "[inicio - fin] en segundos. Detecta los fragmentos que funcionarian "
        "como HOOK de un clip vertical (algo que engancha al espectador en los "
        f"primeros 3 segundos). Devuelve EXCLUSIVAMENTE un JSON array de maximo "
        f"{n_candidates} objetos, sin texto adicional, con el formato exacto: "
        '[{"start": <sec>, "end": <sec>, "score": <0.0-1.0>, '
        '"title": "<titulo corto>", "reason": "<por que>", '
        '"content": "EDITORIAL|PROMOTIONAL|UNCERTAIN"}]. '
        "El campo score indica retencion/impacto, mayor es mejor.\n\n"
        "REGLA CRITICA para el campo 'title' (titulo standalone, max "
        f"{_TITLE_MAX_LEN} caracteres):\n"
        "1. Describe el significado COMPLETO del segmento entre start y end; "
        "es lo que el espectador vera en pantalla y debe coincidir con el audio "
        "del clip.\n"
        "2. Debe ser una frase o etiqueta standalone y gramaticalmente cerrada; "
        "NO es una copia del primer o ultimo fragmento del transcript.\n"
        "3. NO termines abruptamente en preposiciones/conectores (in, to, of, "
        "and, because, when, that, the, a, ...) ni en coma; si el pensamiento "
        "no cierra, elige otro angulo del MISMO segmento que si cierre.\n"
        "4. NO uses puntos suspensivos (...) para ocultar truncamiento.\n"
        "5. NO inventes hechos fuera del segmento ni cambies el sentido del "
        "hook; NO uses palabras de segmentos adyacentes o posteriores.\n"
        "6. Conciso (short-form), idioma predominante del contenido, no generico "
        "ni aplicable a cualquier clip.\n"
        "7. Si el segmento no permite un titulo fiable, devuelve string vacio "
        '("") en title en lugar de inventar.\n'
        "Si la frase mas impactante esta fuera del segmento, reporta ESE "
        "segmento como hook, no el anterior.\n\n"
        "REGLA CRITICA para el campo 'content' (clasificacion del proposito):\n"
        '- "EDITORIAL": noticia, entrevista, documental, historia, tutorial, '
        "opinion o contenido informativo/entretenido. Una mencion de marca, "
        "empresa o producto NO es publicidad por si sola.\n"
        '- "PROMOTIONAL": el proposito PRINCIPAL del fragmento es vender, '
        "promocionar o cerrar una oferta (compra, precio, sponsored, CTA "
        "comercial, claim promocional de producto).\n"
        '- "UNCERTAIN": evidencia insuficiente o ambigua; NO uses '
        "PROMOTIONAL solo por mencionar una marca.\n"
        "NO selecciones como hook un bloque PROMOTIONAL; si hay duda, usa "
        'UNCERTAIN (score bajo) en vez de PROMOTIONAL.\n\n'
        f"TRANSCRIPCION:\n{lines}"
    )


# --------------------------------------------------------------------------
# Validacion local determinista de title + fallback (P1)
# --------------------------------------------------------------------------

def sanitize_title(raw: str) -> str:
    """Limpia caracteres de control y colapsa espacios. No inventa contenido."""
    if not isinstance(raw, str):
        return ""
    cleaned = _CONTROL_CHARS_RE.sub("", raw)
    cleaned = cleaned.replace(" ", " ")
    return " ".join(cleaned.split())


def title_issues(title: str, hook_text: str = "") -> list[str]:
    """Lista conservadora de problemas del title (vacía = válido).

    No construye parser gramatical; solo señales deterministas de truncamiento
    o mala fe. No rechaza títulos cortos que terminan en content-word.
    """
    issues: list[str] = []
    t = sanitize_title(title)
    if not t:
        issues.append("empty")
        return issues
    if len(t) > _TITLE_MAX_LEN:
        issues.append("too_long")
    if "..." in t or t.rstrip().endswith("…"):
        issues.append("ellipsis_truncation")
    if _CONTROL_CHARS_RE.search(title or ""):
        issues.append("control_chars")
    # Termina en conector/preposición o puntuación de continuación.
    if _DANGLING_END_RE.search(t):
        issues.append("dangling_end")
    # Título idéntico a un fragmento obviamente incompleto del hook:
    # hook_text termina en dangling y title == tail exacto del hook.
    ht = sanitize_title(hook_text)
    if ht and t and _DANGLING_END_RE.search(ht) and t.lower() == ht.lower():
        issues.append("copies_incomplete_hook")
    elif ht and t.lower() in (ht.lower(), ht.lower().rstrip(".")):
        # Copia literal completa del hook solo es problema si el hook mismo
        # está incompleto (cubierto arriba); si el hook cierra, title==hook OK.
        pass
    # Respaldo por el hook: si el title tiene varias content-words y NINGUNA
    # aparece en el hook, no está respaldado. Conservador: no exige solape
    # para títulos de 1-2 palabras (paráfrasis cortas válidas).
    if hook_text:
        title_keys = {
            w for w in re.findall(r"[A-Za-zÀ-ÿ0-9']+", t.lower())
            if w not in _TITLE_STOPWORDS and len(w) > 2
        }
        hook_keys = set(re.findall(r"[A-Za-zÀ-ÿ0-9']+", ht.lower()))
        if len(title_keys) >= 3 and hook_keys and not (title_keys & hook_keys):
            issues.append("unsupported_by_hook")
    return issues


def is_valid_title(title: str, hook_text: str = "") -> bool:
    return not title_issues(title, hook_text)


def _first_complete_clause(text: str, max_len: int = _TITLE_MAX_LEN) -> str:
    """Primera oración/clausula cerrada de texto real del hook, sin inventar."""
    t = sanitize_title(text)
    if not t:
        return ""
    # Preferir fin de oración dentro del límite.
    m = re.match(r"^[^.!?]*[.!?]", t)
    if m:
        sent = sanitize_title(m.group(0))
        if 8 <= len(sent) <= max_len and not _DANGLING_END_RE.search(sent):
            return sent
    # Si el texto cabe completo y no dangling -> usarlo tal cual.
    if len(t) <= max_len and not _DANGLING_END_RE.search(t):
        return t
    # Cortar en word-boundary antes de max_len; luego solo recortar el
    # TRASERO dangling (no palabras del medio: evita salad de auxiliares).
    prefix = t[:max_len]
    cut = max(prefix.rfind(", "), prefix.rfind(" "))
    if cut < 20:
        cut = prefix.rfind(" ")
    if cut > 0:
        words = t[:cut].split()
    else:
        words = prefix.split()
    while words and _DANGLING_END_RE.search(" ".join(words)):
        words.pop()
    out = sanitize_title(" ".join(words)).rstrip(",;:- ")
    if len(out) >= 8 and not _DANGLING_END_RE.search(out):
        return out
    # Fallback final: primeras palabras del hook sin dangling final.
    words = t.split()
    acc: list[str] = []
    for w in words:
        trial = " ".join(acc + [w])
        if len(trial) > max_len - 1:
            break
        acc.append(w)
    while acc and _DANGLING_END_RE.search(" ".join(acc)):
        acc.pop()
    out = sanitize_title(" ".join(acc)).rstrip(",;:- ")
    return out if len(out) >= 8 else ""


def fallback_title(hook_text: str) -> str:
    """Fallback seguro: representación textual real del hook, sin inventar."""
    return _first_complete_clause(hook_text)


def enforce_title_contract(
    hooks: list[HookCandidate],
    window: list[TranscriptSegment],
) -> list[HookCandidate]:
    """Aplica validación local + fallback in-place en los titles.

    Flujo: LLM title -> sanitize -> válido? aceptar : fallback_title(hook_text).
    Si el fallback también falla, title queda como issue-string de revisión
    humana determinista (no se inventa información externa).
    """
    for h in hooks:
        hook_text = _hook_text(window, h.start, h.end)
        raw = h.title
        cleaned = sanitize_title(raw)
        if is_valid_title(cleaned, hook_text):
            h.title = _cosmetic_title(cleaned)
            continue
        fb = fallback_title(hook_text)
        if fb and is_valid_title(fb, hook_text):
            log.warn(
                "title_fallback",
                llm_title=raw[:80],
                fallback=fb[:80],
                issues=title_issues(cleaned, hook_text),
            )
            h.title = _cosmetic_title(fb)
        else:
            # Señal explícita de baja confianza; no inventar.
            marker = cleaned or fb or "[title_invalid]"
            h.title = marker[:_TITLE_MAX_LEN]
            log.warn(
                "title_invalid_kept",
                title=h.title[:80],
                issues=title_issues(cleaned, hook_text) or ["fallback_failed"],
            )
    return hooks


def _cosmetic_title(title: str) -> str:
    """Ajuste cosmético seguro: punto final en títulos largos sin puntuación.

    No altera palabras ni sentido; solo cierra la etiqueta para que no parezca
    truncada al publicar.
    """
    t = sanitize_title(title)
    if len(t) >= 40 and t[-1] not in ".!?…\"'":
        if not _DANGLING_END_RE.search(t):
            t = t + "."
    return t[:_TITLE_MAX_LEN] if len(t) > _TITLE_MAX_LEN else t


def _hook_text(window: list[TranscriptSegment], start: float, end: float) -> str:
    """Texto real de los segmentos que solapan el hook [start, end]."""
    parts: list[str] = []
    for seg in window:
        if seg.end < start or seg.start > end:
            continue
        parts.append(seg.text)
    return " ".join(parts).strip()


# --------------------------------------------------------------------------
# Advertisement signal + filtro determinista de seguridad (P1)
# --------------------------------------------------------------------------

def ad_context_text(
    window: list[TranscriptSegment],
    start: float,
    end: float,
) -> str:
    """Texto del hook + contexto temporal inmediato (pre/post roll corto)."""
    lo = max(0.0, start - _AD_CONTEXT_SPAN_PRE_S)
    hi = end + _AD_CONTEXT_SPAN_POST_S
    return _hook_text(window, lo, hi)


def classify_advertisement(
    hook_text: str,
    context_text: str = "",
    llm_content: str = "EDITORIAL",
) -> tuple[str, str]:
    """Clasifica propósito del candidato de forma conservadora.

    Devuelve (content, ad_reason). NO descarta por marca/producto/dinero solos;
    exige evidencia fuerte de promoción (sponsor, CTA de compra, retail,
    precio combinado con claim/CTA, o claim promocional + señal comercial).
    """
    llm = (llm_content or "EDITORIAL").strip().upper()
    if llm not in ("EDITORIAL", "PROMOTIONAL", "UNCERTAIN"):
        llm = "UNCERTAIN"

    hook = (hook_text or "").strip()
    ctx = (context_text or "").strip()
    scoped = f"{hook}\n{ctx}"

    sponsor = bool(_AD_SPONSOR_RE.search(scoped))
    cta = bool(_AD_CTA_RE.search(scoped))
    retail = bool(_AD_RETAIL_RE.search(scoped))
    price = bool(_AD_PRICE_RE.search(scoped))
    hook_claim = bool(_AD_PROMO_CLAIM_RE.search(hook))
    ctx_claim = bool(_AD_PROMO_CLAIM_RE.search(ctx))
    hook_commercial = bool(
        _AD_CTA_RE.search(hook) or _AD_SPONSOR_RE.search(hook) or _AD_RETAIL_RE.search(hook)
    )

    # Evidencia dura en cualquier parte del scope: sponsor/CTA/retail son
    # inequívocos de promoción (aunque estén solo en el contexto inmediato).
    hard_signal = sponsor or cta or retail
    # Claim promocional del hook necesita respaldo comercial en contexto
    # (precio o claim CTA/retail/sponsor) para no matar un superlativo editorial.
    claim_plus_backing = hook_claim and (price or cta or retail or sponsor or ctx_claim)

    if hard_signal or claim_plus_backing:
        tags = [
            name
            for name, on in (
                ("sponsor", sponsor),
                ("cta", cta),
                ("retail", retail),
                ("price", price),
                ("product_claim", hook_claim),
            )
            if on
        ]
        content, reason = "PROMOTIONAL", "strong_promo_evidence:" + ",".join(tags)
    elif hook_claim or ctx_claim or price:
        # Señal débil sola (claim de producto o precio aislado) -> no bloquear.
        content, reason = "UNCERTAIN", "weak_promo_signal"
    else:
        content, reason = "EDITORIAL", ""

    # LLM como señal adicional, nunca como único motivo de bloqueo duro
    # sin al menos una señal local (evita falsos positivos por confusión LLM).
    if llm == "PROMOTIONAL":
        if content == "EDITORIAL" and (price or hook_claim or ctx_claim or hard_signal):
            content, reason = "PROMOTIONAL", f"llm+local:{reason or 'llm_promotional'}"
        elif content == "EDITORIAL":
            content, reason = "UNCERTAIN", "llm_promotional_without_local_signal"
    elif llm == "UNCERTAIN" and content == "EDITORIAL":
        content, reason = "UNCERTAIN", "llm_uncertain"

    return content, reason


def enforce_advertisement_filter(
    hooks: list[HookCandidate],
    window: list[TranscriptSegment],
) -> tuple[list[HookCandidate], list[HookCandidate]]:
    """Etiqueta content/ad_reason in-place y separa (selected, blocked).

    PROMOTIONAL fuerte -> bloqueado (no entra en ranking final; se conserva
    en la lista blocked para trazabilidad/audit). UNCERTAIN permanece con
    prioridad menor en rank_hooks.
    """
    selected: list[HookCandidate] = []
    blocked: list[HookCandidate] = []
    for h in hooks:
        hook_text = _hook_text(window, h.start, h.end)
        ctx = ad_context_text(window, h.start, h.end)
        prior = h.content
        content, reason = classify_advertisement(hook_text, ctx, prior)
        h.content = content
        h.ad_reason = reason
        if content == "PROMOTIONAL":
            blocked.append(h)
            log.warn(
                "ad_blocked",
                start=h.start,
                end=h.end,
                score=h.score,
                title=h.title[:80],
                ad_reason=reason,
                llm_content=prior,
            )
        else:
            selected.append(h)
    return selected, blocked


def rank_hooks(hooks: list[HookCandidate]) -> list[HookCandidate]:
    """Ordena por content (EDITORIAL > UNCERTAIN) y luego por score desc."""
    return sorted(
        hooks,
        key=lambda h: (_CONTENT_RANK.get(h.content, 1), h.score),
        reverse=True,
    )


# --------------------------------------------------------------------------
# Parsing estricto contra el schema
# --------------------------------------------------------------------------

def _extract_json(raw: str) -> str:
    """Saca el JSON de una respuesta que puede venir con markdown fences."""
    if not raw or not raw.strip():
        raise LLMFailedError("El LLM devolvio una respuesta vacia")
    match = _JSON_FENCE_RE.search(raw)
    if match:
        return match.group(1).strip()
    return raw.strip()


def parse_hook_response(raw: str, *, window_start: float = 0.0) -> list[HookCandidate]:
    """Valida la respuesta cruda contra HookCandidate. Lanza LLMFailedError."""
    try:
        payload = json.loads(_extract_json(raw))
    except json.JSONDecodeError as exc:
        raise LLMFailedError(f"JSON mal formado del LLM: {exc}") from exc

    if not isinstance(payload, list):
        raise LLMFailedError("El LLM devolvio un JSON que no es una lista")

    try:
        return [HookCandidate.model_validate(item) for item in payload]
    except Exception as exc:  # noqa: BLE001 - pydantic validation errors
        raise LLMFailedError(f"Respuesta del LLM no valida contra HookCandidate: {exc}")


# --------------------------------------------------------------------------
# Caminos de generacion: API real / mock determinista
# --------------------------------------------------------------------------

def _mock_hook_response(window: list[TranscriptSegment], n_candidates: int) -> str:
    """Mock determinista SIN red: agrupa segmentos consecutivos en ventanas de
    duracion ~30-45s y les asigna un score heuristico estable por texto."""
    candidates: list[dict] = []
    i = 0
    while i < len(window) and len(candidates) < n_candidates:
        start = window[i].start
        end = window[i].end
        text_bits = [window[i].text]
        j = i + 1
        while j < len(window) and (window[j].end - start) < 30:
            end = window[j].end
            text_bits.append(window[j].text)
            j += 1
        text = " ".join(text_bits)
        upper = text.upper()
        score = 0.5
        if "?" in text or "!" in text:
            score += 0.25
        for kw in ("HOW", "WHY", "WHAT", "SECRET", "NADA", "TODOS", "USTED",
                   "NUNCA", "MEJOR", "PEOR", "DINERO"):
            if kw in upper:
                score += 0.15
        score = round(min(1.0, score), 2)
        candidates.append({
            "start": round(start, 2),
            "end": round(end, 2),
            "score": score,
            # Título standalone real del grupo (sin [:60] a mitad de palabra).
            "title": fallback_title(text) or "Hook",
            "reason": "Mock: grupo de segmentos de alto contenido (offline).",
            "content": "EDITORIAL",
        })
        i = j
    return json.dumps(candidates, ensure_ascii=False)


def _rate_limit_wait(resp: requests.Response) -> float:
    header = resp.headers.get("x-ratelimit-reset-tokens")
    if not header:
        header = resp.headers.get("Retry-After", "")
    raw = str(header).strip().lower().rstrip("s").strip()
    try:
        return min(180.0, max(2.0, float(raw) + 1.0))
    except ValueError:
        return 10.0


def _api_call(messages: list[dict]) -> str:
    """Llamada real al LLM (endpoint OpenAI-compatible). Levanta LLMFailedError."""
    import time as _time

    base_url = str(config.get("env.LLM_BASE_URL", "")).rstrip("/")
    api_key = str(config.get("env.LLM_API_KEY", ""))
    model = str(config.get("models.llm", "claude-haiku"))
    timeout = int(config.get("llm.timeout_seconds", 60))
    max_rate_retries = 4

    if not base_url or not api_key:
        raise LLMFailedError(
            "llm.mode='api' requiere LLM_BASE_URL y LLM_API_KEY en .env"
        )

    url = f"{base_url}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {"model": model, "messages": messages}
    for attempt in range(max_rate_retries + 1):
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
            if resp.status_code == 429:
                if attempt >= max_rate_retries:
                    raise LLMFailedError(
                        f"Rate limit agotado llamando al LLM ({url}): {resp.status_code}"
                    )
                wait = _rate_limit_wait(resp)
                log.warn("llm_rate_limited", attempt=attempt, wait_s=round(wait, 1))
                _time.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except LLMFailedError:
            raise
        except requests.RequestException as exc:
            raise LLMFailedError(f"Fallo llamando al LLM ({url}): {exc}") from exc
        except (KeyError, ValueError, IndexError) as exc:
            raise LLMFailedError(f"Respuesta inesperada del LLM: {exc}") from exc
    raise LLMFailedError(f"Rate limit agotado llamando al LLM ({url})")


def generate_window(
    window: list[TranscriptSegment], n_candidates: int, *, mode: str | None = None
) -> str:
    """Devuelve el verbo crudo del analizador (api real o mock)."""
    if mode is None:
        mode = str(config.get("llm.mode", "mock"))
    if mode == "api":
        system = (
            "Eres un editor de video especializado en encontrar hooks. "
            "Siempre respondes con JSON valido y nada mas."
        )
        return _api_call([
            {"role": "system", "content": system},
            {"role": "user", "content": _window_prompt(window, n_candidates)},
        ])
    if mode == "mock":
        return _mock_hook_response(window, n_candidates)
    raise LLMFailedError(f"llm.mode desconocido: {mode!r}")


# --------------------------------------------------------------------------
# Orquestador principal
# --------------------------------------------------------------------------

def _duration_bounds() -> tuple[float, float]:
    min_d = float(config.get("clip.min_duration_seconds", 15))
    max_d = float(config.get("clip.max_duration_seconds", 90))
    return min_d, max_d


def _in_duration_range(candidate: HookCandidate) -> bool:
    """Solo descarta hooks CORTOS (0s o degenerados) y LARGOS (>max_duration).

    Hooks cortos son VÁLIDOS: clip_window.py los expande a 15-90s con contexto.
    Solo rechazamos hooks que excedan la duración máxima del clip.
    """
    _, max_d = _duration_bounds()
    return candidate.duration > 0 and candidate.duration <= max_d


def find_hooks(
    segments: list[TranscriptSegment],
    job_id: str,
    *,
    n_candidates: int | None = None,
) -> list[HookCandidate]:
    """Analiza la transcripción en ventanas y devuelve hooks rankeados.

    Kill-switch: se estima el costo por tokens del prompt ANTES de cada llamada
    al LLM. Si el estimado excede lo que queda del presupuesto, BudgetAbort.
    Si el LLM devuelve JSON mal formado, se reintenta hasta max_json_retries;
    agotados los reintentos -> LLMFailedError (exit 4).
    """
    if not segments:
        log.info("hooks_skipped", job_id=job_id, reason="empty_transcript")
        return []

    per_window = int(config.get("llm.candidates_per_window", 12))
    max_retries = int(config.get("llm.max_json_retries", 1))
    model = str(config.get("models.llm", "claude-haiku"))
    output_est = int(config.get("llm.estimated_output_tokens", 300))

    windows = chunk_transcript(segments)
    hooks: list[HookCandidate] = []
    blocked_hooks: list[HookCandidate] = []

    for idx, window in enumerate(windows):
        prompt = _window_prompt(window, per_window)
        input_tokens = estimate_prompt_tokens(prompt)
        estimated = estimate_llm_cost(input_tokens, output_est)

        for attempt in range(max_retries + 1):
            # Kill-switch ANTES de cada llamada (mismo patron que transcriber).
            if not cost_guard.check_budget(estimated, job_id):
                spent = cost_guard.spent_total(job_id)
                limit = float(config.get("max_api_cost_per_job", 1.50))
                log.error(
                    "budget_abort",
                    job_id=job_id,
                    stage="llm_analyze",
                    window=idx,
                    estimated_usd=round(estimated, 6),
                    spent_usd=round(spent, 6),
                    limit_usd=limit,
                )
                raise BudgetAbort(
                    f"LLM estimado ${estimated:.6f} excede el presupuesto restante "
                    f"de ${max(0.0, limit - spent):.6f} del job {job_id} (ventana {idx})"
                )

            raw = generate_window(window, per_window)
            # El gasto se registra por cada llamada paga realizada, validada o no.
            cost_guard.spend(estimated, job_id, model=model)

            try:
                parsed = parse_hook_response(raw)
            except LLMFailedError as exc:
                if attempt < max_retries:
                    log.warn(
                        "llm_retry",
                        job_id=job_id,
                        window=idx,
                        attempt=attempt + 1,
                        error=exc.message,
                    )
                    continue
                log.error(
                    "llm_failed",
                    job_id=job_id,
                    window=idx,
                    attempts=attempt + 1,
                    error=exc.message,
                )
                raise exc

            # P1 titles: validación local determinista + fallback (misma llamada).
            enforce_title_contract(parsed, window)
            # P1 ads: señal LLM + filtro determinista de seguridad por ventana.
            selected, blocked = enforce_advertisement_filter(parsed, window)
            hooks.extend(selected)
            blocked_hooks.extend(blocked)
            log.info(
                "hooks_window_ok",
                job_id=job_id,
                window=idx,
                candidates=len(selected),
                blocked_ads=len(blocked),
                attempts=attempt + 1,
            )
            break

    # Filtro de duracion configurable + advertisement filter + ranking.
    hooks = [h for h in hooks if _in_duration_range(h)]
    hooks = rank_hooks(hooks)

    limit = n_candidates or int(config.get("llm.candidates_per_window", 12))
    selected = hooks[:limit]
    # Trazabilidad: anuncios bloqueados no entran en la selección final.
    if blocked_hooks:
        log.info(
            "ads_blocked_audit",
            job_id=job_id,
            blocked=len(blocked_hooks),
            titles=[h.title[:60] for h in blocked_hooks],
        )
    return selected