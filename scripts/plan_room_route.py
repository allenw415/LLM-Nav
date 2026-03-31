from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from st_nav import GroundingIndex, SpatialEngine
from st_nav.env import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute the shortest room route from explicit room inputs.")
    parser.add_argument("--artifacts-dir", default="dataset/sites/british_museum/normalized")
    parser.add_argument("--source-room-id", required=True)
    parser.add_argument("--target-room-id", required=True)
    parser.add_argument("--waypoint-room-id", action="append", default=[])
    return parser


def load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict JSON: {path}")
    return payload


def main() -> int:
    args = build_parser().parse_args()
    artifacts_dir = (PROJECT_ROOT / args.artifacts_dir).resolve()
    room_graph = load_json(artifacts_dir / "room_graph.json")
    pano_graph = load_json(artifacts_dir / "pano_graph.json")
    grounding = load_json(artifacts_dir / "room_grounding.template.json")

    spatial = SpatialEngine(
        room_graph=room_graph,
        pano_graph=pano_graph,
        grounding_index=GroundingIndex(grounding),
    )
    shortest_path = spatial.shortest_room_route(
        args.source_room_id,
        args.target_room_id,
        args.waypoint_room_id,
    )

    print(
        json.dumps(
            {
                "source_room_id": args.source_room_id,
                "target_room_id": args.target_room_id,
                "waypoint_room_ids": args.waypoint_room_id,
                "shortest_path": shortest_path,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
