# ST-Nav

Navigation using a spatial transforming method over Street View pano graphs and
site-specific map assets.

## Repository Layout

```text
ST-Nav/
  dataset/
    README.md
    sites/
      british_museum/
        README.md
        explicit_map/
        pano_graph/
    pipelines/
      google_streetview/
        control.py
        run_control.py
        scripts/
        web/
```

## Key Directories

- `dataset/sites/`: site-specific assets consumed by the navigation system
- `dataset/sites/british_museum/explicit_map/`: manually curated semantic map
- `dataset/sites/british_museum/pano_graph/`: processed Street View pano graph
- `dataset/pipelines/google_streetview/scripts/`: data preparation scripts for
  pano graph generation
- `dataset/pipelines/google_streetview/web/`: browser-based Street View viewer
- `dataset/pipelines/google_streetview/control.py`: Playwright controller for
  the viewer
- `dataset/pipelines/google_streetview/run_control.py`: small CLI entrypoint for
  interactive checks and screenshots

## Notes

- Dataset layout details are documented in
  `dataset/README.md`.
- Site-specific notes for the British Museum assets are documented in
  `dataset/sites/british_museum/README.md`.
