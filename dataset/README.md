# Dataset Layout

The dataset is organized with two top-level concerns:

- `sites/`: site-specific data assets that the system can load later
- `pipelines/`: tooling and scripts used to generate or inspect those assets

In the current repo layout, the dataset is consumed by `st_nav/`, while
normalization and grounding-related preprocessing code lives in `st_nav_data/`
and executable tooling belongs in `tools/`.

Example:

```text
dataset/
  sites/
    british_museum/
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

## `sites/`

Site directories contain persistent data assets.

- `explicit_map/`: manually curated room graph source
- `pano_graph/`: raw and processed Street View pano graph assets
- `normalized/`: runtime-ready artifacts such as `room_graph.json`,
  `pano_graph.json`, and `pano_room_grounding.json`

## `pipelines/`

Pipeline directories contain acquisition and inspection tooling rather than
runtime navigation code. New executable tooling should generally live under
`tools/`, while `dataset/pipelines/` should preserve data-source provenance.
