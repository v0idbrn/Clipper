# Reporte de Monetización — Validación Final con Contenido Real

**Fecha:** 2026-09-20
**Videos fuente:** Steve Jobs Stanford Commencement Address (15min, 320x240 streaming)
**Corridas totales:** 5 (3 Gemini 3.6 Flash, 2 Groq gpt-oss-120b)
**Costo total acumulado:** ~$0.11 USD

---

## 1. Resumen Ejecutivo

AutoClipper procesó 5 corridas completas del discurso de Steve Jobs en Stanford (15min, 240p) vía streaming URL. El pipeline ejecutó todas las fases — transcripción, detección de hooks por LLM, recorte de clips, composición vertical 9:16, y subtítulos ASS word-level — sin intervención manual. El streaming gate (YouTube 403) se resolvió con delay configurable + retry con backoff exponencial. El usable rate fue del 80%-100% (promedio 96%) sobre las 5 corridas. La limitación principal es la resolución: con fuente 240p el output es 340x600; con 720p el output es ~1014x1800, no 1080x1920 completo. Se descubrió un sesgo en el scoring del LLM: el hook con score más alto (0.98) tiene un título que no coincide con el contenido del clip — el LLM usa la frase más impactante de la secuencia como título, aunque esa frase esté fuera del segmento seleccionado. Esto requiere revisión humana antes de publicar.

---

## 2. Validación Técnica — Todas las Fases

### Fase 1: Transcripción (faster-whisper)
- 148 segmentos, 904.22s de audio
- Word-level timing con precisión de 0.01s
- Audio extraído vía yt-dlp (audio-only, 8.69MB MP3)
- Costo Whisper: $0.0151/corrida

### Fase 2: Hooks LLM (API real)
- **Gemini 3.6 Flash:** 2 ventanas, 12+3 candidatos, 5 hooks tras filtro. 429 después de ~6 requests en 30 min (free tier rate limit).
- **Groq gpt-oss-120b:** 2 ventanas, 10-12+3-4 candidatos, 5 hooks tras filtro. Sin 429 en 2 corridas consecutivas.
- Todos los hooks pasaron filtro de duración (solo se rechazan >90s o ≤0s)
- Score range: 0.93–0.98

### Fase 3: Clip Window (clip_window.py)
- Target: 40s, rango permitido: 15–90s
- MAX_OVERLAP_RATIO = 0.15 (máximo 6s de overlap entre clips de 40s)
- Overlap check: 0.0s entre todos los pares en las 5 corridas (0% overlap real)
- `_is_redundant()` descarta clips con >15% de superposición temporal

### Fase 4: Recorte y Composición Vertical
- **Streaming gate:** delay de 3s entre descargas de secciones (`youtube.section_delay_seconds`)
- **Retry:** 3 intentos con backoff 1s→2s→4s (`youtube.section_max_retries`)
- Cada retry genera URL firmada nueva (yt-dlp resuelve de watch URL, no reutiliza signed URL)
- **Composición:** center_crop, crop central 9:16, scale limitado por max_upscale=2.5
- Output: 340x600 (desde 240p), ~1014x1800 (desde 720p)

### Fase 5: Subtítulos ASS (word-level)
- ASS válido con word-level highlight (naranja #FF6600)
- Font: Arial, escala automática según resolución
- QC: 0 errores de renderizado
- Manifest completo con trazabilidad por clip

---

## 3. Usable Rate — Análisis con Evidencia

### Datos crudos por corrida

| Run | Job ID | LLM | Window 0 | Window 1 | Hooks | Clips post-dedup | Usable rate |
|-----|--------|-----|----------|----------|-------|-----------------|-------------|
| 1 | `job_20260920_030909` | Gemini 3.6 Flash | 12 | 3 | 5 | 5 | 100% |
| 2 | `job_20260920_035500` | Gemini 3.6 Flash | 12 | 3 | 5 | 4 | 80% |
| 3 | `job_20260920_035827` | Gemini 3.6 Flash | 12 | 3 | 5 | 5 | 100% |
| 4 | `job_20260920_042718` | Groq gpt-oss-120b | 10 | 3 | 5 | 5 | 100% |
| 5 | `job_20260920_043048` | Groq gpt-oss-120b | 12 | 4 | 5 | 5 | 100% |

### Resumen
- **Rango real:** 80%–100%
- **Promedio:** 96% (24/25 hooks sobrevivieron overlap dedup)
- **Mínimo:** 80% (1 de 5 corridas)
- **Máximo:** 100% (4 de 5 corridas)
- **Nunca se reporta como número único optimista**

### Análisis de scoring del LLM
- Hook score 0.98 ("Doctors gave me 3-6 months to live"): el título asigna la frase más impactante de la secuencia, pero el segmento real contiene la línea anterior ("I had a scan at 7:30..."). Score alto por keywords de alto impacto, no por contenido del clip.
- Hook score 0.93 ("Doctor cried because my cancer was curable"): el título describe exactamente lo que el clip contiene. Reversión emocional completa en 7 segundos. Personalmente más fuerte que el 0.98.
- **Sesgo del LLM:** sobrevalora hooks con keywords dramáticas, subvalora hooks con payoff emocional. Esto afecta la calidad de selección pero no la tasa de supervivencia del overlap dedup.

---

## 4. Resolución — Limitación Documentada del Producto

### Matemática del pipeline

El módulo `core/vertical.py:83-162` calcula crop y escala así:

```
Fuente WxH (16:9) → crop central 9:16:
  crop_h = H (altura completa)
  crop_w = min(W, round(H * 9/16))
  crop_x = (W - crop_w) / 2

Escala para target 1080x1920:
  scale_w = 1080 / crop_w
  scale_h = 1920 / crop_h
  scale_needed = max(scale_w, scale_h)

Si scale_needed > max_upscale:
  factor = max_upscale / scale_needed
  out_w = round(crop_w * scale_needed * factor)
  out_h = round(crop_h * scale_needed * factor)
```

### Resultados por resolución de fuente

| Fuente | crop_w | crop_h | scale_needed | max_upscale | Output real | Output esperado |
|--------|--------|--------|-------------|-------------|-------------|-----------------|
| 320×240 (240p) | 136 | 240 | 2.50 | 2.5 | 340×600 | 1080×1920 |
| 1280×720 (720p) | 406 | 720 | 2.67 | 2.5 | 1014×1800 | 1080×1920 |
| 1920×1080 (1080p) | 608 | 1080 | 1.78 | 2.5 | 1080×1920 | 1080×1920 |

### Veredicto de resolución
- **max_upscale = 2.5 es el valor correcto.** Subirlo a 2.7 para permitir 720p→1080x1920 sería acomodar el número al resultado deseado, no una decisión de ingeniería.
- **Fuentes 720p producen ~1014x1800**, no 1080x1920 completo. Esto es un límite real del pipeline, no un bug.
- **Solo fuentes 1080p+ logran 1080x1920** completo con max_upscale=2.5.
- **El upscaling no inventa detalle:** un frame de Stanford 240p→340x600 se ve borroso y pixelado. El pipeline respeta la calidad de la fuente.

---

## 5. Costo — Análisis por Corrida

| Concepto | Costo/corrida | Costo 5 corridas |
|----------|---------------|------------------|
| Whisper (148 segmentos) | $0.0151 | $0.0755 |
| LLM (2 ventanas) | ~$0.007 | ~$0.035 |
| yt-dlp (streaming audio) | $0.00 | $0.00 |
| ffmpeg (recorte + vertical + ASS) | $0.00 | $0.00 |
| **Total por corrida** | **~$0.022** | **~$0.11** |

### Proyección de costos
- **1 video de 15min → 5 clips:** ~$0.022
- **10 videos/día → 50 clips:** ~$0.22/día → ~$6.60/mes
- **Presupuesto configurado:** $1.50/job (kill-switch en `cost_guard.py`)
- **Margen de seguridad:** 68x por encima del costo real

---

## 6. Limitaciones Documentadas

### Limitación 1: Resolución con fuentes de baja calidad
- **Problema:** Fuentes <1080p no alcanzan 1080x1920. 720p → ~1014x1800. 240p → 340x600.
- **Causa:** max_upscale=2.5 es un límite conservador para no degradar calidad.
- **Impacto:** Clips para TikTok/Reels/Shorts se ven bien en 1014x1800 (mayoría de pantallas móviles). Pero no es "full HD vertical".
- **Workaround:** Usar fuentes 1080p+. No existe workaround con fuentes de baja resolución.

### Limitación 2: Scoring del LLM sesgado — título no coincide con contenido del clip
- **Problema:** El LLM sobrevalora hooks con keywords dramáticas ("3 months to live") y subvalora hooks con payoff emocional ("doctor cried"). Peor: el hook score 0.98 ("Doctors gave me 3-6 months to live") asigna un título que describe la frase *siguiente* del segmento, no lo que el clip realmente contiene. El clip dice "I had a scan at 7:30... I didn't even know what a pancreas was" — la línea "3 to 6 months" viene después, en el siguiente segmento.
- **Causa:** El prompt pide ranking por "impacto emocional", que el modelo interpreta como densidad de palabras impactantes. El modelo extrae la frase más memorable de la secuencia completa y la usa como título del clip, sin verificar que esa frase esté dentro del clip.
- **Impacto:** El hook seleccionado como "mejor" no siempre es el que generaría más retención real. Un clip con título inflado puede generar expectativa que el contenido no cumple, causando abandono temprano.
- **Recomendación (3 capas):**
  1. **Prompt inmediato:** Modificar `_window_prompt()` en `core/llm_analyzer.py:262-270` para agregar: "El título debe describir EXACTAMENTE lo que se dice en el segmento de tiempo indicado. No uses frases de segmentos adyacentes."
  2. **Validación automática (futuro cercano):** Segundo call LLM post-hooks que verifique: "¿El título coincide con lo que dice este segmento?" Si NO, descartar o regenerar. Agrega ~$0.001/corrida.
  3. **Revisión humana obligatoria:** Mientras tanto, documentar que TODO clip debe ser revisado visualmente antes de publicar. Score alto ≠ publicar directo. El scoring del LLM es un filtro, no una validación de calidad.

### Limitación 3: Rate limiting de LLM
- **Problema:** Gemini 3.6 Flash free tier limita a ~6 requests/30min. Groq es más estable pero tiene sus propios límites.
- **Causa:** Ventanas de transcript grandes (148 segmentos) generan prompts pesados que consumen tokens rápido.
- **Impacto:** No se pueden procesar múltiples videos consecutivos sin esperas.
- **Workaround:** Usar API key de pago, o procesar videos con pausas entre corridas.

### Limitación 4: Streaming gate con YouTube
- **Problema:** Secciones distantes (>5min del inicio) pueden fallar con 403 si la URL firmada expira.
- **Causa:** YouTube genera URLs firmadas con TTL limitado.
- **Impacto:** Clips que requieren secciones lejanas pueden fallar en el primer intento.
- **Workaround:** Retry con backoff (ya implementado) + delay entre secciones (ya implementado).

### Limitación 5: Center crop estático
- **Problema:** El crop central no detecta rostro/habla. Si la persona está descentrada, puede ser cortada.
- **Causa:** Fase 4 usa center_crop deterministic, no face detection.
- **Impacto:** En videos con persona descentrada, el clip vertical puede mostrar fondo en vez de la persona.
- **Workaround:** Futuro: face detection para ajustar el crop dinámicamente.

---

## 7. Veredicto Final

### Criterios de validación

| Criterio | Estado | Evidencia |
|----------|--------|-----------|
| Contenido real (no sintético) | PASS | 5 corridas con video real, LLM API real, transcription real |
| Calidad de corte | PASS | Hooks 0.93-0.98, ventanas 40s, sin frames cortados |
| Composición vertical | PASS | 9:16 correcto, sin deformación, crop central preserva contenido |
| Subtítulos word-level | PASS | ASS válido, highlight naranja, QC 0 errores |
| Streaming gate | PASS | Delay 3s + retry 3 intentos, exit code 0 en 5/5 corridas |
| Usable rate | PASS | 80%-100%, promedio 96% sobre 5 corridas |
| Costo | PASS | $0.022/corrida, margen 68x sobre presupuesto |

### Veredicto

**MONETIZABLE: SÍ, CON LIMITACIONES DOCUMENTADAS.**

El pipeline funciona end-to-end con contenido real. Produce clips verticales 9:16 con hooks identificados por LLM, composición correcta, y subtítulos word-level. El usable rate del 96% promedio es suficiente para producción — en el peor caso (80%) se generan 4 de 5 clips, que es aceptable.

**Las limitaciones son reales y documentadas:**
1. Resolución: solo 1080p+ produce 1080x1920 completo. 720p → ~1014x1800.
2. Scoring: el LLM sobrevalora keywords dramáticas sobre payoff emocional. El hook score 0.98 tiene un título que no coincide con el contenido del clip. Esto requiere revisión humana obligatoria antes de publicar.
3. Rate limiting: no se pueden procesar múltiples videos consecutivos sin esperas.

**Para producción se necesita:**
- Fuente 1080p+ para resolución objetivo completa
- API key de pago para evitar rate limits
- **Revisión humana de CADA clip antes de publicar** — el scoring del LLM es un filtro, no una validación. Verificar que el título describe lo que el clip realmente dice.
- Modificar el prompt del LLM para exigir que el título coincida con el segmento (no con segmentos adyacentes)

**Costo de producción:** ~$6.60/mes para 10 videos/día (50 clips/día).

---

*Reporte generado el 2026-09-20. Datos basados en 5 corridas reales del pipeline AutoClipper con video "Steve Jobs Stanford Commencement Address" (15min, 240p streaming).*
