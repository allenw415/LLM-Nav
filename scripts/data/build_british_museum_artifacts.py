from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from st_nav import build_grounding_template
from st_nav_data.normalize import (
    BRITISH_MUSEUM_DIRECTION_OVERRIDES,
    BRITISH_MUSEUM_EXCLUDED_EDGES,
    BRITISH_MUSEUM_EXPERIMENT_ROOM_IDS,
    BRITISH_MUSEUM_ROOM_CANONICAL_IDS,
    BRITISH_MUSEUM_TRANSITION_OVERRIDES,
    load_json,
    normalize_pano_graph,
    normalize_room_graph,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build normalized ST-Nav artifacts under dataset/."
    )
    parser.add_argument(
        "--explicit-map-path",
        default="dataset/sites/british_museum/explicit_map/explicit_map.json",
    )
    parser.add_argument(
        "--pano-graph-path",
        default="dataset/sites/british_museum/pano_graph/processed/panos.json",
    )
    parser.add_argument(
        "--output-dir",
        default="dataset/sites/british_museum/normalized",
        help="Directory where normalized artifacts are written.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    explicit_map = load_json(PROJECT_ROOT / args.explicit_map_path)
    pano_graph = load_json(PROJECT_ROOT / args.pano_graph_path)

    normalized_room_graph = normalize_room_graph(
        explicit_map,
        allowed_room_ids=BRITISH_MUSEUM_EXPERIMENT_ROOM_IDS,
        canonical_room_ids=BRITISH_MUSEUM_ROOM_CANONICAL_IDS,
        ensure_bidirectional=True,
        direction_overrides=BRITISH_MUSEUM_DIRECTION_OVERRIDES,
        transition_overrides=BRITISH_MUSEUM_TRANSITION_OVERRIDES,
        excluded_edges=BRITISH_MUSEUM_EXCLUDED_EDGES,
    )
    normalized_pano_graph = normalize_pano_graph(pano_graph)
    room_grounding = build_grounding_template(normalized_room_graph)

    output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs = {
        "room_graph.json": normalized_room_graph,
        "pano_graph.json": normalized_pano_graph,
        "room_grounding.template.json": room_grounding,
    }
    for filename, payload in outputs.items():
        path = output_dir / filename
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[saved] {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
