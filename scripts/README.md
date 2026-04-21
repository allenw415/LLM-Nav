# Scripts

This directory contains repository-facing CLI helpers around the core modules.
They are grouped by purpose instead of by implementation history.

## Layout

```text
scripts/
  data/
  demo/
```

## `data/`

Data and artifact preparation scripts. These are not runtime navigation entrypoints.

- `build_british_museum_artifacts.py`: build normalized room graph, pano graph, and grounding template artifacts
- `batch_floor_room_grounding.py`: ground arbitrary floor panos in fixed-size batches
- `batch_room_grounding.py`: batch-generate pano-to-room grounding candidates and review files
- `pano2room_grounding.py`: run one-off pano-to-room grounding while operating on grounding artifacts
- `summarize_room_grounding.py`: summarize grounding records for one room

### Commands

`build_british_museum_artifacts.py`

```bash
python3 scripts/data/build_british_museum_artifacts.py
```

Args:
- `--explicit-map-path`
- `--pano-graph-path`
- `--output-dir`

`pano2room_grounding.py`

```bash
python3 scripts/data/pano2room_grounding.py --pano-id "7grGsbOXqpEMDLgTG6VfmQ" --full-output
```

Args:
- `--artifacts-dir`
- `--manifest-path`
- `--pano-id`
- `--room-id` (repeatable)
- `--limit`
- `--profile`
- `--model-provider`
- `--model-name`
- `--api-key`
- `--api-base`
- `--api-kind`
- `--gemini-api-key`
- `--gemini-model`
- `--vlm-timeout`
- `--render-api-key`
- `--render-output-dir`
- `--heading-mode` (`museum | cardinal | grounding | graph`)
- `--max-captures`
- `--pitch`
- `--fov`
- `--width`
- `--height`
- `--candidate-scope` (`same-floor | all`)
- `--debug-trace`
- `--full-output`

`batch_floor_room_grounding.py`

```bash
python3 scripts/data/batch_floor_room_grounding.py --floor 0 --offset 0 --limit 100 --heading-mode museum --max-captures 8 --fov 45
```

Default outputs go under `dataset/sites/british_museum/normalized/room_grounding_batches/`:
- `floor0_batch_0000_100.json`
- `floor0_batch_0000_100.review.json`
- `floor0_batch_0000_100.manual.json`

Args:
- `--artifacts-dir`
- `--floor`
- `--offset`
- `--limit`
- `--output-path`
- `--review-output-path`
- `--manual-output-path`
- `--profile`
- `--model-provider`
- `--model-name`
- `--api-key`
- `--api-base`
- `--api-kind`
- `--gemini-api-key`
- `--gemini-model`
- `--vlm-timeout`
- `--render-api-key`
- `--render-output-dir`
- `--heading-mode`
- `--max-captures`
- `--pitch`
- `--fov`
- `--width`
- `--height`
- `--candidate-scope`
- `--min-confidence`
- `--debug-trace`

`batch_room_grounding.py`

```bash
python3 scripts/data/batch_room_grounding.py --room-id "Room 8" --room-id "Room 23"
```

Args:
- `--artifacts-dir`
- `--room-id` (repeatable)
- `--expansion-strategy` (`confidence-region-growing | fixed-hops`)
- `--max-hops`
- `--floor`
- `--limit`
- `--output-path`
- `--review-output-path`
- `--manual-output-path`
- `--compact-output-path`
- `--profile`
- `--model-provider`
- `--model-name`
- `--api-key`
- `--api-base`
- `--api-kind`
- `--min-confidence`
- `--expansion-confidence`
- `--gemini-api-key`
- `--gemini-model`
- `--vlm-timeout`
- `--render-api-key`
- `--render-output-dir`
- `--heading-mode`
- `--max-captures`
- `--pitch`
- `--fov`
- `--width`
- `--height`
- `--candidate-scope`
- `--dry-run`
- `--debug-trace`

`summarize_room_grounding.py`

```bash
python3 scripts/data/summarize_room_grounding.py --room-id "Room 8"
```

Args:
- `--room-id`
- `--gemini-path`
- `--manual-path`

## `demo/`

Manual smoke tests and module-level demos.
These are useful when debugging one stage or a short chain of stages without treating them as formal benchmarks.

LLM/VLM demo scripts can now target either:
- OpenAI `Responses API`
- an OpenAI-compatible self-hosted server exposing `/v1/chat/completions`

Use `--*-api-kind` to switch transport (`responses` or `chat_completions`) and `--*-api-base` to point at your server.

For `.env`-only switching, prefer an active-profile layout instead of repeating the same keys twice:

```env
ST_NAV_ACTIVE_PROFILE=ollama

ST_NAV_PROFILE_OLLAMA_MODEL_PROVIDER=ollama
ST_NAV_PROFILE_OLLAMA_MODEL_NAME=gemma4:26b
ST_NAV_PROFILE_OLLAMA_API_BASE=http://127.0.0.1:11434/v1
ST_NAV_PROFILE_OLLAMA_API_KEY=ollama
ST_NAV_PROFILE_OLLAMA_API_KIND=chat_completions
ST_NAV_PROFILE_OLLAMA_REQUEST_TIMEOUT=180
ST_NAV_PROFILE_OLLAMA_NUM_CTX=4096
ST_NAV_PROFILE_OLLAMA_TEMPERATURE=0

ST_NAV_PROFILE_OPENAI_MODEL_PROVIDER=openai
ST_NAV_PROFILE_OPENAI_MODEL_NAME=gpt-5-mini
ST_NAV_PROFILE_OPENAI_API_BASE=https://api.openai.com/v1
ST_NAV_PROFILE_OPENAI_API_KEY=YOUR_OPENAI_KEY
ST_NAV_PROFILE_OPENAI_API_KIND=responses
ST_NAV_PROFILE_OPENAI_REQUEST_TIMEOUT=30

ST_NAV_PROFILE_GEMINI_MODEL_PROVIDER=gemini
ST_NAV_PROFILE_GEMINI_MODEL_NAME=gemma-4-26b-a4b-it
ST_NAV_PROFILE_GEMINI_API_KEY=YOUR_GEMINI_API_KEY
ST_NAV_PROFILE_GEMINI_API_KIND=responses
ST_NAV_PROFILE_GEMINI_REQUEST_TIMEOUT=60
```

Then switch providers by editing only:

```env
ST_NAV_ACTIVE_PROFILE=ollama
```

or

```env
ST_NAV_ACTIVE_PROFILE=openai
```

or

```env
ST_NAV_ACTIVE_PROFILE=gemini
```

- `parse_instruction.py`: inspect the instruction parser output
- `resolve_source_pano.py`: inspect source-room to source-pano resolution
- `plan_room_route.py`: inspect shortest-room-route planning from explicit room ids
- `run_pano_perception.py`: run perception directly on one pano id
- `run_localization.py`: run localization on synthetic, manifest-based, or cached perception inputs

### Commands

`parse_instruction.py`

```bash
python3 scripts/demo/parse_instruction.py --instruction "Find the way from Room 4 to Room 23."
```

Args:
- `--artifacts-dir`
- `--instruction`
- `--llm-api-key`
- `--llm-model`
- `--llm-api-kind` (`responses | chat_completions`)
- `--llm-api-base`

`resolve_source_pano.py`

```bash
python3 scripts/demo/resolve_source_pano.py --source-room-id "Room 8"
```

Args:
- `--artifacts-dir`
- `--source-room-id`
- `--debug`

`plan_room_route.py`

```bash
python3 scripts/demo/plan_room_route.py --source-room-id "Room 8" --target-room-id "Room 23"
```

Args:
- `--artifacts-dir`
- `--source-room-id`
- `--target-room-id`
- `--waypoint-room-id` (repeatable)

`run_pano_perception.py`

```bash
python3 scripts/demo/run_pano_perception.py --pano-id "7grGsbOXqpEMDLgTG6VfmQ"
```

Args:
- `--artifacts-dir`
- `--pano-id`
- `--llm-api-key`
- `--detector-model`
- `--detector-api-kind` (`responses | chat_completions`)
- `--detector-api-base`
- `--vlm-timeout`
- `--render-api-key`
- `--render-output-dir`
- `--heading-mode` (`museum | cardinal | graph`)
- `--pitch`
- `--fov`
- `--width`
- `--height`
- `--current-heading`
- `--demo-trace`
- `--output-path`

`run_localization.py`

```bash
python3 scripts/demo/run_localization.py --mode perception-json --perception-json-path outputs/step1_perception.json
```

Args:
- `--mode` (`synthetic | manifest | perception-json`)
- `--artifacts-dir`
- `--manifest-path`
- `--perception-json-path`
- `--prior-localization-json`
- `--start-pano-id`
- `--start-room-id`
- `--current-heading`
- `--localizer` (`heuristic | llm`)
- `--llm-model`
- `--llm-api-key`
- `--llm-api-kind` (`responses | chat_completions`)
- `--llm-api-base`
- `--llm-timeout`
- `--prior-room` (repeatable, e.g. `Room 10=0.7`)
- `--top-k`
- `--json`
- `--full-json`
- `--output-path`

## Keep vs Remove

The current set is intentionally kept because each script covers a distinct use case:

- keep `data/` scripts because they produce, inspect, or summarize offline artifacts
- keep `demo/` scripts because they isolate one module or a short workflow for debugging

If a future script does not clearly fit one of these two roles, it should probably not live here.
