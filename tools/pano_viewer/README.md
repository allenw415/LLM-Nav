# Pano Viewer

Panorama graph viewer and export targets for normalized panorama graphs. The
viewer can also open a navigation episode or Pilot evaluation JSON, play its
panorama trajectory, and keep Google Street View synchronized to the current
pano ID and action heading.

Generate artifacts:

```bash
python3 tools/pano_viewer/export.py
```

The exporter copies viewer assets from `tools/pano_viewer/web`. It also reads
`GMAPS_API_KEY` from the project `.env` and writes a generated `.env.js` next to
the exported viewer files so the Street View panel can load.

Serve the copied viewer:

```bash
python3 tools/pano_viewer/server.py
```

Then open <http://127.0.0.1:8000>. Choose or drop a trajectory JSON in the
Trajectory panel. Playback updates the selected graph node and, after you click
`Load` once, the Google Street View pano and heading.

The bundled server also exposes local captures under `renders/` and
`outputs/` to the viewer, so the current rendered panorama frame can be
previewed and exported as a WebM video. A plain `python3 -m http.server` still
supports graph and Street View playback, but not local panorama frames.

The export writes:

- `viewer_data.json` for the browser viewer
- `pano_nodes.geojson` and `pano_edges.geojson` for map tooling
- `pano_graph.gexf` and `pano_graph.graphml` for Gephi/Sigma-style workflows
- `pano_graph_floor0.dot` for Graphviz
- `publication/floor_*_overview.svg` for report figures
- `trajectory.js` for trajectory parsing and manual Street View synchronization
- `.env.js` for the local Street View API key when `GMAPS_API_KEY` is set

Street View uses the generated `.env.js` in the exported viewer folder:

```js
window.GMAPS_API_KEY = "YOUR_KEY";
```

Regenerate it after changing `.env`:

```bash
python3 tools/pano_viewer/export.py
```

Keep `.env.js` out of git.
