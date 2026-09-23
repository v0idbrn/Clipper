# AutoClipper

[English](README.md) | [Español](README.es.md)

Convierte video largo en clips verticales cortos con hooks seleccionados por IA, subtítulos quemados y un filtro comercial de anuncios.

## Qué hace

AutoClipper ejecuta un pipeline de extremo a extremo:

1. **Ingesta** — URL de YouTube/streaming o archivo local (`direct_file`).
2. **Transcripción** — Whisper con timestamps a nivel de palabra.
3. **Detección de hooks** — análisis LLM por ventanas + contrato local de títulos + filtro de publicidad (`EDITORIAL` / `PROMOTIONAL` / `UNCERTAIN`).
4. **Recorte** — ventanas narrativas deterministas de 15–90s alineadas a límites de palabra.
5. **Vertical** — composición 9:16 con center-crop.
6. **Subtítulos** — ASS dinámico con highlight opcional de palabra, quemado en el vertical.

La salida queda en `outputs/<job_id>/`:

- `transcript.json` / `hooks.json`
- `clips_raw/` — cortes horizontales
- `clips_vertical/` — 9:16
- `clips_captioned/` — vertical + subtítulos quemados + archivos `.ass`

## Requisitos

- Python 3.10+
- [FFmpeg](https://ffmpeg.org/) y `ffprobe` en el `PATH`
- Opcional: Node.js ≥ 22 (desafíos JS de yt-dlp en YouTube)
- Un endpoint LLM compatible con OpenAI (API key en `.env`)

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Crear `.env` (nunca commitear este archivo):

```bash
LLM_API_KEY=tu_key
LLM_BASE_URL=https://tu-endpoint-llm/v1
```

## Configuración

Los límites operativos viven en `config.yaml`:

| Clave | Significado |
| --- | --- |
| `max_api_cost_per_job` | Kill-switch de presupuesto USD por job |
| `clip.min/max/target_duration_seconds` | Límites de duración de la ventana |
| `llm.mode` | `api` (real) o `mock` (tests offline) |
| `models.whisper` | Tamaño de faster-whisper |
| `subtitles.enabled` | Quemar subtítulos ASS |

## Uso

```bash
python main.py run --url "https://www.youtube.com/watch?v=..." --n-clips 5
python main.py run --url "C:/ruta/archivo local.mp4" --n-clips 5
```

Los códigos de salida están en `core/exit_codes.py` (URL inválida, aborto de presupuesto, fallo del LLM, etc.).

## Tests

La suite es offline (no requiere red):

```bash
pytest tests/ -q
```

Los fixtures están en `tests/fixtures/` (tono sintético, habla corta, video sintético). Ver `docs/testing_offline.md`.

## Estructura del proyecto

```
main.py           Entrada CLI
config.yaml       Límites operativos
config.py         Cargador de configuración
core/             Etapas del pipeline (ingesta, transcripción, hooks, clip, vertical, captions)
models/           Schemas Pydantic
utils/            Logging, directorios temporales
tests/            Suite Pytest + fixtures
docs/             Notas de diseño e informes
```

## Seguridad y notas comerciales

- **Cost guard** estima el gasto del LLM *antes* de cada llamada y aborta si se pasa del presupuesto.
- **Contrato de título** rechaza títulos de hook truncados / no standalone con fallback local.
- **Filtro de publicidad** bloquea segmentos promocionales fuertes (sponsor, CTA, retail, precio+claim) para que no terminen como hooks finales; menciones editoriales de marcas siguen siendo seleccionables.
- El snapping a límites de palabra evita cortes a mitad de palabra.

## Licencia

MIT — ver [LICENSE](LICENSE).
