from __future__ import annotations

import argparse
from _common import PROJECT_ROOT, ensure_project_root_on_path, load_normalized_artifacts, render_json

ensure_project_root_on_path()

from st_nav import GroundingIndex, SourcePanoResolver, load_dotenv

load_dotenv(PROJECT_ROOT / ".env")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Resolve a source room into its representative pano.")
    parser.add_argument("--artifacts-dir", default="dataset/sites/british_museum/normalized")
    parser.add_argument("--source-room-id", required=True)
    parser.add_argument("--debug", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    artifacts = load_normalized_artifacts(args.artifacts_dir, grounding=True)

    resolver = SourcePanoResolver(GroundingIndex(artifacts.grounding or {}))
    resolution = resolver.resolve(args.source_room_id)

    payload = {
        "source_room_id": resolution.source_room_id,
        "source_pano_id": resolution.pano_id,
    }
    if args.debug:
        payload["candidate_pano_ids"] = resolution.candidate_pano_ids
        payload["resolution_method"] = resolution.resolution_method

    print(render_json(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
