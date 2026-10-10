# SA-Nav

Navigation using Spatial Alignment over Street View panorama graphs and
site-specific map assets.

## Current Structure

The repository is split into runtime code, review-facing examples, non-runtime
tools, and persistent dataset assets:

- `sa_nav/`: runtime navigation modules
- `examples/`: focused core demos and complete navigation-flow demos
- `tools/`: evaluation, visualization, and data-preparation tooling
- `sa_nav_data/`: offline data preparation and grounding helpers
- `dataset/`: site assets and acquisition provenance

## Naming

The project name is **SA-Nav**. Python packages use `sa_nav` and
`sa_nav_data`, and project-specific environment variables use the `SA_NAV_*`
prefix. Existing experiment artifacts may still contain legacy `st_nav.*` or
`llm_nav_*` identifiers; those recorded files are retained as immutable
historical data, while newly generated artifacts use the SA-Nav naming.

## Repository Layout

```text
SA-Nav/
  sa_nav/
    common/
    decision/
    spatial/
    perception/
    execution/
    pipeline/
  examples/
    core/
    flows/
  tools/
    data/
    evaluation/
    visualization/
  sa_nav_data/
    normalize.py
    room_grounder.py
  dataset/
    README.md
    sites/
      british_museum/
        README.md
        explicit_map/
        pano_graph/
        normalized/
    pipelines/
      google_streetview/
        control.py
        run_control.py
        scripts/
        web/
```

## Key Directories

- `sa_nav/`: runtime code for parsing, spatial reasoning, perception, and execution
- `sa_nav/execution/`: episode step loop after a task and source pano are known
- `sa_nav/pipeline/`: end-to-end workflow orchestration from instruction to final trace; use
  `build_navigation_pipeline(...)` to assemble the standard runtime components
- `sa_nav/decision/`: instruction parsing and action policy modules
- `sa_nav/spatial/`: grounding, localization, routing, and state update logic
- `sa_nav/perception/`: rendering, detection, and perception providers
- `examples/core/`: demos for individual navigation capabilities
- `examples/flows/`: demos for complete navigation flows
- `tools/evaluation/`: benchmark and evaluation entrypoints
- `tools/pano_viewer/`: panorama graph viewer source and exporter
- `tools/data/`: dataset preparation, grounding, and normalization tools
- `sa_nav_data/`: offline data-centric code such as graph normalization and pano-to-room grounding
- `dataset/sites/`: site-specific assets consumed by the navigation system
- `dataset/sites/british_museum/explicit_map/`: manually curated semantic map
- `dataset/sites/british_museum/pano_graph/`: processed Street View pano graph
- `dataset/sites/british_museum/normalized/`: normalized artifacts consumed by the runtime modules
- `dataset/pipelines/google_streetview/scripts/`: data preparation scripts for
  pano graph generation
- `dataset/pipelines/google_streetview/web/`: browser-based Street View viewer
- `dataset/pipelines/google_streetview/control.py`: Playwright controller for
  the viewer
- `dataset/pipelines/google_streetview/run_control.py`: small CLI entrypoint for
  interactive checks and screenshots

## Single-call Spatial Alignment Experiment

Run the standalone eight-image alignment experiment without changing the
navigation pipeline:

```bash
python3 -m sa_nav.cli.run_spatial_alignment \
  --pano-id JK44-iXUvic1mv3GUGMaUQ \
  --render-output-dir renders/room_grounding_fov90 \
  --candidate-top-k 5
```

The command reads the panorama manifest and its sibling
`*_manifest_detections.json` cache, selects candidates from the cached raw
room scores, and makes one VLM request containing all eight images and their
geocentric headings. It does not run perception when the score cache is
missing. The default diagnostic output is:

```text
outputs/spatial_alignment/<pano-id>_single_call.json
```

## Map-grounded Spatial Alignment Experiment

Run the standalone map-grounded experiment with one north-up semantic map and
eight clockwise panorama views. It reuses cached Top-5 room candidates but does
not send their scores or ranks to the alignment VLM:

```bash
python3 -m sa_nav.cli.run_map_grounded_spatial_alignment \
  --pano-id JK44-iXUvic1mv3GUGMaUQ \
  --render-output-dir renders/room_grounding_fov90 \
  --candidate-top-k 5 \
  --repeat 3
```

The command is independent from navigation and never reruns perception. Its
default output contains the generated candidate-highlighted map, three run
records, and an agreement summary under:

```text
outputs/spatial_alignment_map_grounded/<pano-id>/
```

Compare the current single-full prompt against the map-grounded prompt on the
same cached panoramas and candidates with:

```bash
python3 tools/evaluation/localization/eval_map_grounded_spatial_alignment.py \
  --samples-per-room 5 \
  --repeat 3 \
  --modes single-full map-grounded
```

The evaluator only scores cases whose render manifest and detection cache
already exist; missing inputs are recorded as skipped and are not regenerated.

## Map-only Passage Heading Experiment

Run one passage-heading case using only a north-up map plus the current and
target room IDs:

```bash
python3 -m sa_nav.cli.run_map_grounded_heading \
  --current-room-id "Room 6" \
  --target-room-id "Room 7" \
  --repeat 3
```

Run the full Level 0 catalog of 76 directed adjacent-room edges, repeated three
times each (228 VLM calls), with the active `gemini` profile:

```bash
venv/bin/python tools/evaluation/navigation/eval_map_grounded_heading.py \
  --repeat 3 \
  --output-dir outputs/map_grounded_heading/all_level0_directed
```

To test the same 76 cases with the original unmodified map (no blue/green room
overlays, C/T markers, or compass sidebar), use a separate output directory:

```bash
venv/bin/python tools/evaluation/navigation/eval_map_grounded_heading.py \
  --map-mode plain \
  --repeat 3 \
  --output-dir outputs/map_grounded_heading/plain_map_all_level0_directed
```

In plain mode all cases reuse the same original map image; only the current and
target room IDs change between requests.

The batch evaluator writes `runs.jsonl`, `case_summaries.jsonl`,
`review_cases.jsonl`, `summary.json`, and `meeting_summary.md`. Highlighted
mode also writes 76 generated semantic maps; plain mode creates no per-case map
copies. Room-graph headings are used only after each response for
evaluation and are never included in the VLM request. This experiment measures
room-level passage-side inference; it does not provide an agent position or
claim to solve position-dependent first-action headings.

## Local Data Retention

The repository keeps runtime code, dataset sources, experiment tooling, pano
viewer tooling, and the local environment needed to reproduce existing runs.
Large generated data remain under `outputs/`, `renders/`, and
`artifacts/pano_viewer/` and are intentionally excluded from Git.

The canonical 715-case navigation dataset and ablations are indexed in
`outputs/README.md`. Their paths, row counts, byte sizes, and SHA-256 digests
are recorded in `outputs/experiment_manifest_715.json`. These files cover:

- ground-truth offset with and without trajectory memory;
- estimated offset with and without trajectory memory;
- four-view versus eight-view spatial alignment;
- Visual, Bayesian, and Spatial localization-stage ablations.

Retry snapshots may be removed only when every JSON record is identical to the
retained canonical record for the same condition and test ID. Results with any
content difference are retained as experimental history.

## Notes

- Runtime package exports live in `sa_nav/__init__.py`.
- Generated artifacts under `renders/`, `outputs/`, `artifacts/pano_viewer/`,
  legacy `artifacts/pano_visualization/`, and room grounding batch outputs are ignored by git.
- Dataset layout details are documented in
  `dataset/README.md`.
- Site-specific notes for the British Museum assets are documented in
  `dataset/sites/british_museum/README.md`.
- Command examples are documented in `examples/`; non-runtime tools are
  documented in `tools/`.
