from __future__ import annotations

import argparse
from _common import PROJECT_ROOT, ensure_project_root_on_path, load_normalized_artifacts, render_json

ensure_project_root_on_path()

from st_nav import GroundingIndex, SpatialEngine, load_dotenv

load_dotenv(PROJECT_ROOT / ".env")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute the shortest room route from explicit room inputs.")
    parser.add_argument("--artifacts-dir", default="dataset/sites/british_museum/normalized")
    parser.add_argument("--source-room-id", required=True)
    parser.add_argument("--target-room-id", required=True)
    parser.add_argument("--waypoint-room-id", action="append", default=[])
    return parser


def main() -> int:
    args = build_parser().parse_args()
    artifacts = load_normalized_artifacts(
        args.artifacts_dir,
        room_graph=True,
        pano_graph=True,
        grounding=True,
    )

    spatial = SpatialEngine(
        room_graph=artifacts.room_graph or {},
        pano_graph=artifacts.pano_graph or {},
        grounding_index=GroundingIndex(artifacts.grounding or {}),
    )
    shortest_path = spatial.shortest_room_route(
        args.source_room_id,
        args.target_room_id,
        args.waypoint_room_id,
    )

    print(
        render_json(
            {
                "source_room_id": args.source_room_id,
                "target_room_id": args.target_room_id,
                "waypoint_room_ids": args.waypoint_room_id,
                "shortest_path": shortest_path,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
