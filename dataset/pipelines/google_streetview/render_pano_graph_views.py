from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from st_nav.perception import PanoramaRenderer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render four pano views through the perception-layer renderer."
    )
    parser.add_argument(
        "--graph-path",
        default="dataset/sites/british_museum/pano_graph/processed/panos.json",
        help="Path to processed pano graph JSON.",
    )
    parser.add_argument(
        "--pano-id",
        required=True,
        help="Current panoID to render.",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("GMAPS_API_KEY"),
        help="Google Street View Static API key. Defaults to GMAPS_API_KEY.",
    )
    parser.add_argument(
        "--output-dir",
        default="renders/pano_graph",
        help="Output directory relative to repo root.",
    )
    parser.add_argument(
        "--heading-mode",
        choices=["museum", "cardinal", "graph"],
        default="cardinal",
        help="Use true cardinal headings 0/90/180/270, museum-map headings 330/60/150/240, or graph-aligned headings.",
    )
    parser.add_argument("--pitch", type=float, default=0.0)
    parser.add_argument("--fov", type=int, default=90)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=640)
    return parser


def load_graph(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Graph must be a dict: {path}")
    return payload


def main() -> int:
    args = build_parser().parse_args()
    if not args.api_key:
        raise RuntimeError("Missing API key. Pass --api-key or set GMAPS_API_KEY.")

    graph_path = (PROJECT_ROOT / args.graph_path).resolve()
    output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    renderer = PanoramaRenderer(load_graph(graph_path))
    manifest = renderer.render(
        pano_id=args.pano_id,
        api_key=args.api_key,
        output_dir=output_dir,
        heading_mode=args.heading_mode,
        pitch=args.pitch,
        fov=args.fov,
        width=args.width,
        height=args.height,
        graph_path=graph_path,
    )

    print(f"[saved] manifest -> {manifest['manifest_path']}")
    for capture in manifest["captures"]:
        print(f"[saved] {capture['label']} -> {capture['path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
