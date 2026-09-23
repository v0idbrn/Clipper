# Pendientes operativos — antes de cerrar Fase 4

> Gate obligatorio: estos ítems NO son temas de negocio (no se resuelven con
> features nuevas); son fallas de la **ruta principal de ingesta** que se
> confirmaron en la verificación real de Fase 1 y deben estar resueltos antes
> de que la Fase 4 deje el pipeline en estado productivo end-to-end.

---

## P1. yt-dlp: YouTube responde 429 / "Sign in to confirm you're not a bot"

**Estado: RESUELTO (gate de adquisición cerrado).** Ver detalle abajo.

### Causa encontrada

El pipeline invocaba `yt-dlp` "pelado", sin runtime JavaScript ni cookies. Con
yt-dlp 2026.x, YouTube dejó de funcionar sin un runtime JS externo (wiki
oficial: <https://github.com/yt-dlp/yt-dlp/wiki/EJS>). La advertencia era
reproducible el 2026-09-16:

```
WARNING: [youtube] No supported JavaScript runtime could be found. Only deno is
enabled by default; to use another runtime add --js-runtimes RUNTIME[:PATH].
YouTube extraction without a JS runtime has been deprecated, and some formats
may be missing.
```

El 429 / bot-check ("Sign in to confirm you're not a bot") era el bloqueo
histórico (rate-limiting tras varias corridas). **No se pudo reproducir el
429 en esta verificación** (la red sirvió normalmente), pero sí se reprodujo la
advertencia de runtime JS y la ausencia de una ruta de cookies.

Causas: **B (runtime JS requerido)** confirmada + **C (ausencia de cookies)**
para escenarios de rate-limit/edad/región + **D (configuración incompleta)**,
porque el proyecto no exponía ninguna de las dos opciones.

### Solución elegida (mínima y oficial de yt-dlp)

1. **Runtime JS configurable** (`--js-runtimes`): por defecto `node` (Node.js
   ≥22 en PATH), que es el runtime recomendado por yt-dlp para esta plataforma.
   Se puede fijar una ruta (`node:C:/ruta/a/node`), cambiar de runtime o
   deshabilitar (`none`).
2. **Cookies OPT-IN desde navegador** (`--cookies-from-browser`): deshabilitado
   por defecto. Solo se reenvía el NOMBRE del navegador; AutoClipper nunca lee,
   persiste ni imprime el contenido de las cookies.
3. **Errores de YouTube diferenciados**: un fallo de adquisición (429,
   bot-check, runtime faltante, video no disponible) se convierte en
   `YouTubeAcquisitionError` (exit **7** — download failure), reusando los exit
   codes existentes, con diagnóstico y sugerencia de configuración.
4. **Sin reintentos a nivel aplicación**: hard stop tras UNA invocación de
   yt-dlp (yt-dlp maneja sus reintentos internos acotados). No hay loops ni
   spam de requests.

Archivos: `core/ytdlp_opts.py` (nuevo), `core/extractor_audio.py`,
`core/clipper.py`, `core/errors.py`, `config.py`, `config.yaml`.

### Configuración

```yaml
youtube:
  js_runtime: "node"            # "node" | "node:/ruta/a/node" | "none"
  cookies_from_browser: ""      # "" = deshabilitado; ej "chrome", "firefox", "edge"
```

### Cómo usarla

- Por defecto ya funciona si hay **Node.js ≥22** en el `PATH`.
- Si YouTube pide autenticación (429 / bot-check), habilitar cookies:
  `youtube.cookies_from_browser: "chrome"` (el navegador debe existir
  localmente y, en Windows, puede requerir que esté cerrado).
- Si no hay Node, `js_runtime: "none"` evita la advertencia (YouTube puede
  perder formatos).

### Evidencia del test real

Prueba real controlada (video público corto `jNQXAC9IVRw`, 19 s), por la ruta
real `core.extractor_audio.extract_stream_audio` con las opciones nuevas:

```
JS_RUNTIME_CONFIG = node
COMMON_ARGS = ['--js-runtimes', 'node']
REAL_ACQUISITION = OK
AUDIO_FILE = audio.mp3
SIZE_BYTES = 120621
DURATION_S = 19.005542
CLEANUP = OK
```

También se validó la ruta de recorte de video (`bestvideo*+bestaudio/best`
`--download-sections`, usada por Fase 3): produjo `clip_000.mp4` (124 KB) sin
la advertencia de runtime. El 429 **no** se pudo reproducir en esta ventana de
tiempo.

### Tests ejecutados

- `tests/test_ytdlp_opts.py` (nuevo): defaults seguros, cookies opt-in,
  clasificación de errores, sin secretos.
- `tests/test_extractor_youtube.py` (nuevo): mapeo 429/bot-check/runtime →
  exit 7; genérico → exit 5; sin retry; flags en el comando.
- `tests/test_clipper.py`: + casos de YouTube en la ruta de clips.
- Suite completa: **105 passed, 0 failed** (antes: 78).

### Qué NO se resolvió / limitaciones

- El **429 real no se reprodujo**, así que la mitigación por cookies es
  opt-in y no fue exercitada contra un 429 en vivo. Si vuelve a aparecer, el
  fix es configurar `youtube.cookies_from_browser`.
- No se implementó `--cookies <file>` (solo `--cookies-from-browser`), ni
  proxies, ni impersonación: fuera de alcance y no necesarios hoy.
- No se agregó reintento con backoff: por diseño hay hard stop (sin loops ni
  spam de requests).
- La descarga sigue siendo audio-only para transcripción y por secciones para
  clips (nunca el video completo en RAM), sin cambios en la arquitectura.

### `YouTube acquisition gate: CLOSED`

> Nota: la Fase 1 (transcripción) jamás dependió de YouTube para sus tests
> (ver `tests/fixtures/` y `docs/testing_offline.md`).
