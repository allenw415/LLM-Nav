# ST-Nav

Navigation using a spatial transforming method over Street View pano graphs and
site-specific map assets.

## Current Structure

The repository is split into runtime code, review-facing examples, non-runtime
tools, and persistent dataset assets:

- `st_nav/`: runtime navigation modules
- `examples/`: focused core demos and complete navigation-flow demos
- `tools/`: evaluation, visualization, and data-preparation tooling
- `st_nav_data/`: offline data preparation and grounding helpers
- `dataset/`: site assets and acquisition provenance

## Repository Layout

```text
ST-Nav/
  st_nav/
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
  st_nav_data/
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

- `st_nav/`: runtime code for parsing, spatial reasoning, perception, and execution
- `st_nav/execution/`: episode step loop after a task and source pano are known
- `st_nav/pipeline/`: end-to-end workflow orchestration from instruction to final trace; use
  `build_navigation_pipeline(...)` to assemble the standard runtime components
- `st_nav/decision/`: instruction parsing and action policy modules
- `st_nav/spatial/`: grounding, localization, routing, and state update logic
- `st_nav/perception/`: rendering, detection, and perception providers
- `examples/core/`: demos for individual navigation capabilities
- `examples/flows/`: demos for complete navigation flows
- `tools/evaluation/`: benchmark and evaluation entrypoints
- `tools/pano_viewer/`: panorama graph viewer source and exporter
- `tools/data/`: dataset preparation, grounding, and normalization tools
- `st_nav_data/`: offline data-centric code such as graph normalization and pano-to-room grounding
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

## Notes

- Runtime package exports live in `st_nav/__init__.py`.
- Generated artifacts under `renders/`, `outputs/`, `artifacts/pano_viewer/`,
  legacy `artifacts/pano_visualization/`, and room grounding batch outputs are ignored by git.
- Dataset layout details are documented in
  `dataset/README.md`.
- Site-specific notes for the British Museum assets are documented in
  `dataset/sites/british_museum/README.md`.
- Command examples are documented in `examples/`; non-runtime tools are
  documented in `tools/`.
