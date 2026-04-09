# ST-Nav

Navigation using a spatial transforming method over Street View pano graphs and
site-specific map assets.

## Current Structure

The repository is now split into three main code/data areas:

- `st_nav/`: runtime navigation modules
- `st_nav_data/`: offline data preparation and grounding helpers
- `scripts/`: repository-facing CLIs for data workflows and manual demos

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
  scripts/
    data/
    demo/
```

## Key Directories

- `st_nav/`: runtime code for parsing, spatial reasoning, perception, and execution
- `st_nav/decision/`: instruction parsing and action policy modules
- `st_nav/spatial/`: grounding, localization, routing, and state update logic
- `st_nav/perception/`: rendering, detection, and perception providers
- `st_nav/pipeline/`: thin workflow glue such as source resolution and end-to-end navigation flow
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
- `scripts/`: repository-facing helper CLIs grouped into `data/` and `demo/`

## Notes

- Runtime package exports live in `st_nav/__init__.py`.
- Generated artifacts under `renders/` and `outputs/` are ignored by git.
- Dataset layout details are documented in
  `dataset/README.md`.
- Site-specific notes for the British Museum assets are documented in
  `dataset/sites/british_museum/README.md`.
- Script usage notes are documented in
  `scripts/README.md`.
