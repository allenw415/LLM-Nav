from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sa_nav.evaluation import sync_floor0_assets


DEFAULT_DATASET_SOURCE = (
    "/home/allenw4/memory-based-sa-nav/outputs/navigation_paths/"
    "floor0_semantic_targets_topology_constrained"
)
DEFAULT_DATASET_DESTINATION = (
    "outputs/navigation_paths/floor0_semantic_targets_topology_constrained"
)
DEFAULT_RENDER_SOURCE = (
    "/home/allenw4/memory-based-sa-nav/renders/room_grounding_fov90"
)
DEFAULT_RENDER_DESTINATION = "renders/room_grounding_fov90"
DEFAULT_GRAPH = "dataset/sites/british_museum/normalized/pano_graph.json"


def resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate.resolve() if candidate.is_absolute() else (PROJECT_ROOT / candidate).resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Copy the benchmark dataset and source-authoritative floor render cache."
    )
    parser.add_argument("--dataset-source", default=DEFAULT_DATASET_SOURCE)
    parser.add_argument("--dataset-destination", default=DEFAULT_DATASET_DESTINATION)
    parser.add_argument("--render-source", default=DEFAULT_RENDER_SOURCE)
    parser.add_argument("--render-destination", default=DEFAULT_RENDER_DESTINATION)
    parser.add_argument("--pano-graph", default=DEFAULT_GRAPH)
    parser.add_argument("--floor", default="0")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Perform the copy. Without this flag the command is a read-only dry run.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    graph_path = resolve(args.pano_graph)
    pano_graph = json.loads(graph_path.read_text(encoding="utf-8"))
    report = sync_floor0_assets(
        dataset_source=resolve(args.dataset_source),
        dataset_destination=resolve(args.dataset_destination),
        render_source=resolve(args.render_source),
        render_destination=resolve(args.render_destination),
        pano_graph=pano_graph,
        graph_path=graph_path,
        floor=args.floor,
        apply=args.apply,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
