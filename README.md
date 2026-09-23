# AutoClipper

[English](README.md) | [Español](README.es.md)

Turn long-form video into short vertical clips with AI-selected hooks, burned-in captions, and a commercial safety filter for ads.

## What it does

AutoClipper runs an end-to-end pipeline:

1. **Ingest** — YouTube/streaming URL or local file (`direct_file`).
2. **Transcribe** — Whisper with word-level timestamps.
3. **Find hooks** — LLM window analysis + local title contract + advertisement filter (`EDITORIAL` / `PROMOTIONAL` / `UNCERTAIN`).
4. **Clip** — deterministic 15–90s narrative windows snapped to word boundaries.
5. **Vertical** — 9:16 center-crop composition.
6. **Captions** — dynamic ASS subtitles with optional word highlight, burned into the vertical.

Output lands in `outputs/<job_id>/`:

- `transcript.json` / `hooks.json`
- `clips_raw/` — horizontal cuts
- `clips_vertical/` — 9:16
- `clips_captioned/` — vertical + burned subtitles + `.ass` files

## Requirements

- Python 3.10+
- [FFmpeg](https://ffmpeg.org/) and `ffprobe` on `PATH`
- Optional: Node.js ≥ 22 (yt-dlp YouTube JS challenges)
- An OpenAI-compatible LLM endpoint (API key in `.env`)

## Install

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Create `.env` (never commit this file):

```bash
LLM_API_KEY=your_key
LLM_BASE_URL=https://your-llm-endpoint/v1
```

## Configuration

Business and runtime limits live in `config.yaml`:

| Key | Meaning |
| --- | --- |
| `max_api_cost_per_job` | USD budget kill-switch per job |
| `clip.min/max/target_duration_seconds` | Window length bounds |
| `llm.mode` | `api` (real) or `mock` (offline tests) |
| `models.whisper` | faster-whisper size |
| `subtitles.enabled` | Burn ASS captions |

## Usage

```bash
python main.py run --url "https://www.youtube.com/watch?v=..." --n-clips 5
python main.py run --url "C:/path/to/local.mp4" --n-clips 5
```

Exit codes are defined in `core/exit_codes.py` (invalid URL, budget abort, LLM failure, etc.).

## Tests

The suite is offline (no network required):

```bash
pytest tests/ -q
```

Fixtures live in `tests/fixtures/` (synthetic tone, short speech, synthetic video). See `docs/testing_offline.md`.

## Project layout

```
main.py           CLI entry
config.yaml       Operational limits
config.py         Config loader
core/             Pipeline stages (ingest, transcribe, hooks, clip, vertical, captions)
models/           Pydantic schemas
utils/            Logging, temp dirs
tests/            Pytest suite + fixtures
docs/             Design notes and reports
```

## Safety & commercial notes

- **Cost guard** estimates LLM spend *before* each call and aborts over budget.
- **Title contract** rejects truncated / non-standalone hook titles with a local fallback.
- **Advertisement filter** blocks strong promotional segments (sponsor, CTA, retail, price+claim) from becoming final hooks; editorial brand mentions stay selectable.
- Word-boundary snapping keeps cuts off mid-word.

## License

MIT — see [LICENSE](LICENSE).
