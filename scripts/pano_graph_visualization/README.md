# Panorama Graph Visualization

Static research viewer and export targets for normalized panorama graphs.

Generate artifacts:

```bash
python3 scripts/data/export_pano_visualization.py
```

Serve the copied viewer:

```bash
python3 -m http.server 8000 --directory artifacts/pano_visualization/british_museum
```

The export writes:

- `viewer_data.json` for the browser viewer
- `pano_nodes.geojson` and `pano_edges.geojson` for map tooling
- `pano_graph.gexf` and `pano_graph.graphml` for Gephi/Sigma-style workflows
- `pano_graph_floor0.dot` for Graphviz
- `publication/floor_*_overview.svg` for report figures

To enable Street View in the right panel, place `.env.js` next to the exported
viewer files:

```js
window.GMAPS_API_KEY = "YOUR_KEY";
```
