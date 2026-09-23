# Fase 3 — Hook → Clip Window → Clip real

> El LLM identifica el hook (punto/intervalo semántico). **El sistema determina
> la duración final del clip.** La ventana narrativa la construye código
> determinista, sin LLM, sin ML y sin red.

---

## Pipeline

```
LLM (Fase 2, sin cambios)
        ↓
HookCandidate          start/end/score/title/reason  (models/schemas.py)
        ↓  core/clip_window.build_clip_window  (DETERMINISTA, sin LLM)
ClipWindow             15-90 s, hook contenido        (models/schemas.py)
        ↓  core/clipper (FFmpeg / yt-dlp)
clip_000.mp4           horizontal, video+audio intactos
```

Separación de responsabilidades:

| Etapa | Módulo | Decide |
|---|---|---|
| SEMANTIC DETECTION | `core/llm_analyzer.py` (Fase 2) | dónde está el hook |
| WINDOW SELECTION | `core/clip_window.py` (Fase 3) | cuánto dura el clip |
| MEDIA CUT | `core/clipper.py` (Fase 3) | materializa la ventana |

---

## `ClipWindow`

Modelo mínimo y tipado en `models/schemas.py`:

```
start, end, hook_start, hook_end   (+ duration, hook_offset como propiedades)
```

Validaciones (Pydantic): `start >= 0`, `end > start`, hook contenido
(`hook_start >= start`, `hook_end <= end`), timestamps finitos (rechaza
NaN/Infinity) y duración dentro de `[clip.min_duration_seconds,
clip.max_duration_seconds]` (15-90 s por defecto, configurable).

## Window Builder (`core/clip_window.py`)

`build_clip_window(hook_start, hook_end, media_duration, *, min_duration,
max_duration, target_duration) -> ClipWindow`

Heurística simple, explicable y estable:

1. incluir el hook completo (prioridad máxima);
2. conservar contexto previo: pre-roll = 35% de la ventana (`PRE_ROLL_RATIO`);
3. conservar desarrollo posterior;
4. apuntar a `clip.target_duration_seconds` (default 40 s; ideal 30-45);
5. respetar `[0, media_duration]`;
6. nunca salir de `[min_duration, max_duration]`.

Casos especiales:

| Caso | Comportamiento |
|---|---|
| hook cerca del comienzo | `start = 0`; usa el contexto posterior |
| hook cerca del final | desplaza la ventana hacia atrás (`end = media_duration`) |
| medio < `min_duration` | `InsufficientDurationError` (rechazo explícito) |
| hook mismo > `max_duration` | `WindowError` (no recorta semántica en silencio) |
| hook termina pasado el medio | se clampa `hook_end` al medio |
| hook fuera del medio | `HookValidationError` |

## Múltiples hooks (`build_clip_windows`)

- ventana independiente por hook; un hook inválido/imposible se descarta con
  log (no tumba el job);
- **orden determinista** final por `(start, end, hook_start, -score)`;
- **deduplicación**: ventanas que arrancan/terminan ≤1 s de diferencia, o que
  solapan ≥70% de la más corta, se colapsan conservando el hook de mayor score;
- nombres estables por índice: `clip_000.mp4`, `clip_001.mp4`, ...

## FFmpeg / streaming / direct_file

`core/clipper.py` materializa la `ClipWindow` con las dos rutas existentes:

- **streaming**: `yt-dlp --download-sections "*{start}-{end}"`
  `--force-keyframes-at-cuts` (descarga SOLO la sección; nunca el video completo);
- **direct_file**: `ffmpeg -ss {start} -i raw -t {duration}` con re-encode
  (preset/crf de config).

No se cambia resolución/aspect ratio, no se agregan filtros ni subtítulos, no se
carga el video completo en RAM.

**Validación del output** (`_validate_clip`): existe, tamaño > 0, duración real
vía ffprobe dentro de 15-90 s (tolerancia 1 s por keyframes), y se registra
presencia de video/audio. No alcanza con el exit code 0 de FFmpeg.

## Manifest

`clips_raw/manifest.json` (extendido mínimamente) relaciona por job:
`source`, `source_type` y, por clip: `hook`, `source_start`, `source_end`,
`duration` real, `path`, `clip_id`, `qc_status`. Permite reproducir qué decisión
produjo cada clip.

## Errores (exit codes existentes, sin inventar)

| Tipo | Exit code |
|---|---|
| `HookValidationError` (hook inválido) | 5 |
| `InsufficientDurationError` (medio < 15 s) | 5 |
| `WindowError` (ventana imposible / hook > 90 s) | 5 |
| `FFmpegError` (falla FFmpeg/yt-dlp) | 5 |
| `ClipFilesystemError` (filesystem) | 5 |
| `YouTubeAcquisitionError` (adquisición YouTube) | 7 |

Sin retries a nivel aplicación.

---

## Tests

- `tests/test_clip_window.py` (nuevo): validación (negativos, NaN, Infinity,
  end≤start, fuera del medio, medio <15 s, hook >90 s), ventanas (inicio/medio/
  final, medios 15/30/90/>90, hooks cortos/largos, límites, rango 15-90),
  múltiples hooks (orden, dedup, solapamiento, hooks inválidos).
- `tests/test_clipper.py` (actualizado): comando de sección con la ventana,
  errores diferenciados, validación de output, E2E real offline (single y
  múltiple ventana).
- `tests/test_e2e_flow.py` (actualizado): E2E por CLI con dedup y manifest.
- Suite completa: **139 passed, 0 failed** (antes: 105).

## Prueba real

### A) Offline real (direct_file, FFmpeg real) — **PASS**

Video sintético de 120 s + dos `HookCandidate` controlados:

```
clip_000  window 0.0-40.0    hook 12.0-18.0   probe duration=40.0  video=True audio=True
clip_001  window 61.0-101.0  hook 75.0-90.0   probe duration=40.0  video=True audio=True
```

Ambas ventanas de 40 s (= `clip.target_duration_seconds`), dentro de 15-90 s,
con video y audio presentes. Sin mocks en la ruta de media.

### B) Streaming real (YouTube público corto, yt-dlp) — **NOT AVAILABLE**

El video del gate (`jNQXAC9IVRw`) volvió a responder bot-check / challenge:

- sin cookies: `Sign in to confirm you're not a bot`;
- `cookies_from_browser=chrome`: `Failed to decrypt with DPAPI`;
- `cookies_from_browser=firefox`: `n challenge solving failed` / `The page
  needs to be reloaded`.

Es una limitación **externa** de YouTube (el gate de adquisición está cerrado y
no se re-investiga). La materialización vía streaming queda cubierta por tests
con `_run` mockeado, pero **no** se declara una prueba streaming real que no se
ejecutó.

---

## Limitaciones conocidas

- `find_hooks` (Fase 2) conserva su filtro de duración 15-90 s sobre el span del
  LLM. El Window Builder **acepta cualquier hook** (incluido ~3 s), pero un hook
  corto solo llega al builder si el filtro de Fase 2 se relaja en una
  integración posterior (ver `docs/pending_before_fase3_gate.md`). No se tocó
  Fase 2.
- Los límites de ventana usan timestamps del hook, no fronteras de segmentos de
  transcripción (se decidió no acoplar el builder a la transcripción; snapping a
  segmentos queda para más adelante).
- `PRE_ROLL_RATIO` es constante (0.35); solo `target_duration_seconds` es
  configurable.
- Streaming real no verificable por bot-check externo (ver arriba).

## Estado

```
Phase 3 status: CLOSED
```
