# Reporte de Monetización — Validación con Contenido Real

**Fecha:** 2026-09-20
**Video fuente:** "Me at the zoo" (Jawed Karim, 19s, 320x240)
**Job ID:** job_20260920_014702_01fb5d47
**Costo total:** $0.002 USD

---

## Datos de la Ejecución Real

### 1. Transcripción (faster-whisper, word-level)
- **4 segmentos**, 19.01s de audio
- Words con timing preciso (ej: `" All" @ 0.9-1.36s`, `" elephants." @ 2.72-3.22s`)
- Costo whisper: $0.0003

### 2. Hooks LLM (Gemini 3.6 Flash, API real)
| # | Score | Hook | Texto | Duración |
|---|-------|------|-------|----------|
| 1 | 0.85 | "Lo genial de estos animales" | 3.9-12.5s | 8.6s |
| 2 | 0.75 | "Frente a los elefantes" | 0.9-3.2s | 2.3s |

- **2 candidatos** devueltos por el LLM
- **2 hooks** pasaron el filtro de duración
- Costo LLM: ~$0.0017

### 3. Clip Seleccionado
- **Hook:** "Lo genial de estos animales" (score 0.85)
- **Ventana:** 0.0-19.01s (video completo, 19s)
- **Razón:** El video es demasiado corto (19s < target 40s), así que se usa la ventana completa

### 4. Composición Vertical (center_crop)
- **Fuente:** 320x240 (4:3)
- **Crop central:** 136x240 (centro del frame, incluye a la persona)
- **Salida:** 340x600 (9:16, escalado limitado por max_upscale=2.5)
- **Audio:** copiado sin re-compresión

### 5. Subtítulos ASS (word-level highlight)
- **8 captions** con timing de palabra
- **Font:** Arial, tamaño 20 (escala automática para 340x600)
- **Highlight:** palabra actual en naranja (`#FF6600`)
- **Posición:** centro-inferior (`\an2\pos(170,562)`)
- **QC:** 0 errores (formato ASS válido, sin texto vacío)

---

## CRITERIO DE MONETIZACIÓN — Evaluación

### ✅ Criterio 1: Contenido Real (NO sintético)
- **Video:** "Me at the zoo" — primer video de YouTube, persona real hablando
- **Transcripción:** faster-whisper con word-level timing real
- **LLM:** Gemini 3.6 Flash con API key real, sin mocks
- **Resultado:** PASS

### ✅ Criterio 2: Calidad de Corte
- **Hook identificado:** "The cool thing about these guys..." (score 0.85)
- **Ventana narrativa:** clip_window.py expandió el hook de 8.6s a 19s (video completo)
- **Corte limpio:** sin frames cortados, audio sincronizado
- **Resultado:** PASS

### ✅ Criterio 3: Composición Vertical
- **Aspecto 9:16:** 340x600 (correcto)
- **Persona visible:** crop central incluye a Jawed Karim en el centro del frame
- **Sin deformación:** escala uniforme, sin stretching
- **Límite de upscale:** respeta max_upscale=2.5 (no inventa detalle)
- **Resultado:** PASS

### ✅ Criterio 4: Subtítulos
- **ASS válido:** 57 líneas, formato correcto
- **Word-level highlight:** cada palabra tiene timing individual
- **Legible:** tamaño adecuado para 340x600, contraste alto (contorno negro)
- **QC:** 0 errores de renderizado
- **Resultado:** PASS

### ✅ Criterio 5: Integridad del Pipeline
- **Flujo completo:** probe → transcribe → hooks → clip → vertical → subtitles
- **Manifest:** clips_raw, clips_vertical, clips_captioned con trazabilidad
- **Costo:** $0.002 USD (dentro del presupuesto de $1.50)
- **Resultado:** PASS

### ✅ Criterio 6: Reproducibilidad
- **Tests:** 284/284 passing
- **Determinismo:** mismas entradas → misma salida (mock mode para tests)
- **Modo API:** funciona con Gemini sin cambios de código
- **Resultado:** PASS

---

## Limitaciones Detectadas

1. **Video corto (19s):** No permite generar múltiples clips (min_duration=15s). Con videos de 60s+ se generarían 2-3 clips.

2. **Resolución baja (320x240):** El upscale está limitado por max_upscale=2.5, produciendo 340x600 en vez de 1080x1920. Con fuentes 1080p+, se lograría la resolución objetivo.

3. **Center crop estático:** No detecta rostro/habla. Para videos con persona descentrada, el crop podría cortar. En este video, la persona está centrada y se preserva.

---

## Veredicto

**MONETIZABLE: SÍ** — El pipeline produce clips verticales con:
- Contenido real (no sintético)
- Hooks identificados por LLM real
- Composición vertical correcta (9:16, sin deformación)
- Subtítulos word-level con highlight
- QC pass, manifest completo
- Costo mínimo ($0.002/video)

**Para producción se necesita:**
- Videos de 60s+ para generar múltiples clips
- Fuentes 1080p+ para resolución objetivo 1080x1920
- Evaluación de crop con persona descentrada (futuro: face detection)
