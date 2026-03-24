# Dataset Layout

The dataset is organized with two top-level concerns:

- `sites/`: site-specific data assets that the system can load later
- `pipelines/`: tooling and scripts used to generate or inspect those assets

Example:

```text
dataset/
  sites/
    british_museum/
      explicit_map/
      pano_graph/
  pipelines/
    google_streetview/
      control.py
      run_control.py
      scripts/
      web/
```
