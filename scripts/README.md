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
- `diagnose_pano_seed_coverage.py`: inspect nearest existing pano nodes around a candidate room seed coordinate
- `export_pano_visualization.py`: export pano graph viewer data, GeoJSON, Gephi/GraphML, Graphviz DOT, and publication SVGs
- `merge_pano_seed_crawl.py`: merge supplemental room-seed Street View crawls into an existing raw floor crawl
- `rebuild_pano_room_grounding.py`: rebuild compact pano-to-room mapping from reviewed batch files
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

`diagnose_pano_seed_coverage.py`

```bash
python3 scripts/data/diagnose_pano_seed_coverage.py --lat 51.5189169422846 --lng -0.12809724778801693 --floor 0
```

Use this before a supplemental crawl to distinguish missing Street View coverage
from missing room grounding. It prints the nearest current pano nodes, distance,
room label, and grounding source.

Args:
- `--lat`
- `--lng`
- `--floor`
- `--limit`
- `--pano-graph-path`
- `--grounding-path`

`merge_pano_seed_crawl.py`

```bash
python3 scripts/data/merge_pano_seed_crawl.py \
  --incoming-raw-path artifacts/pano_seed_crawls/streetview_panos_0_room18_seed.json
```

The default base is `dataset/sites/british_museum/pano_graph/raw/streetview_panos_0_1.json`
and the default preview output is `artifacts/pano_seed_crawls/streetview_panos_0_merged.preview.json`.
Inspect the summary first; if the added pano count looks correct, re-run with
`--overwrite-base`, then rebuild grouped, processed, normalized, grounding, and
visualization artifacts.

Args:
- `--base-raw-path`
- `--incoming-raw-path` (repeatable)
- `--output-path`
- `--overwrite-base`

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
- `--fov` (default `90`)
- `--width`
- `--height`
- `--candidate-scope` (`same-floor | all`)
- `--debug-trace`
- `--full-output`

`export_pano_visualization.py`

```bash
python3 scripts/data/export_pano_visualization.py
python3 -m http.server 8000 --directory artifacts/pano_visualization/british_museum
```

Open `http://127.0.0.1:8000/` after export. The viewer reads `viewer_data.json`
and can optionally load Google Street View when `.env.js` defines
`window.GMAPS_API_KEY = "..."`.

Default outputs go under `artifacts/pano_visualization/british_museum/`:
- `viewer_data.json`
- `pano_nodes.geojson`
- `pano_edges.geojson`
- `pano_graph.gexf`
- `pano_graph.graphml`
- `pano_graph_floor0.dot`
- `publication/floor_*_overview.svg`

Args:
- `--artifacts-dir`
- `--pano-graph-path`
- `--room-graph-path`
- `--grounding-path`
- `--output-dir`
- `--dot-floor`
- `--dot-room-id` (repeatable)
- `--route-source-pano-id`
- `--route-target-pano-id`
- `--copy-viewer` / `--no-copy-viewer`

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

`rebuild_pano_room_grounding.py`

```bash
python3 scripts/data/rebuild_pano_room_grounding.py
```

Use this after editing `room_grounding_batches/*.manual.json`. It rebuilds
`dataset/sites/british_museum/normalized/pano_room_grounding.json` without
rerunning model grounding.

Args:
- `--artifacts-dir`
- `--batch-dir`
- `--output-path`

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
- `eval_pano_perception_grounding.py`: sample grounded panos per room, run perception, and compute visual-localization accuracy
- `run_localization.py`: run localization on synthetic, manifest-based, or cached perception inputs
- `run_navigation.py`: run the end-to-end navigation loop with subgoals, candidate paths, and reasoning traces

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

The perception step now uses one multi-view VLM call for entity recognition,
inside/outside entity classification, and observation-only visual localization
when room graph context is available. The sibling `*_detections.json` cache
stores `cache_version`, `candidate_room_ids`, all entities with
`location_scope`, and `visual_localization`; older entity-only caches still load
with entities treated as `inside`.

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

`eval_pano_perception_grounding.py`

```bash
python3 scripts/demo/eval_pano_perception_grounding.py --samples-per-room 5 --seed 0
```

This samples up to five grounded panoramas per room from
`pano_room_grounding.json`, runs `run_pano_perception.py` for each sample, and
scores `visual_localization.predicted_room_id` against the grounding label.

Args:
- `--artifacts-dir`
- `--grounding-path`
- `--samples-per-room` (default `5`)
- `--seed`
- `--room-id` (repeatable)
- `--max-total`
- `--output-dir`
- `--summary-output-path`
- `--reuse-existing-output`
- `--force`
- `--render-output-dir`
- `--render-api-key`
- `--llm-api-key`
- `--detector-model`
- `--detector-api-kind` (`responses | chat_completions`)
- `--detector-api-base`
- `--vlm-timeout`
- `--heading-mode` (`museum | cardinal | graph`)
- `--fov` (default `90`)
- `--print-failures`

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
- `--localizer` (`bayesian-filter | heuristic | llm | visual-vlm | spatial-alignment-a | spatial-alignment-b`)
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

`run_navigation.py`

```bash
python3 scripts/demo/run_navigation.py --instruction "Find the way from Room 8 to Room 23."
```

Args:
- `--instruction`
- `--artifacts-dir`
- `--localizer` (`heuristic | llm | visual-vlm | spatial-alignment-a | spatial-alignment-b`; default `visual-vlm`)
- `--manifest-map-json`
- `--step-budget`
- `--start-heading`
- `--llm-model`
- `--llm-api-key`
- `--llm-api-kind` (`responses | chat_completions`)
- `--llm-api-base`
- `--llm-timeout`
- `--render-api-key`
- `--render-output-dir`
- `--render-heading-mode` (`museum | cardinal | graph`)
- `--render-pitch`
- `--render-fov` (default `90`)
- `--render-width`
- `--render-height`
- `--output-path`

## Keep vs Remove

The current set is intentionally kept because each script covers a distinct use case:

- keep `data/` scripts because they produce, inspect, or summarize offline artifacts
- keep `demo/` scripts because they isolate one module or a short workflow for debugging

If a future script does not clearly fit one of these two roles, it should probably not live here.
