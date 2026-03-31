from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from st_nav import GroundingIndex, SourcePanoResolver
from st_nav.env import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Resolve a source room into its representative pano.")
    parser.add_argument("--artifacts-dir", default="dataset/sites/british_museum/normalized")
    parser.add_argument("--source-room-id", required=True)
    parser.add_argument("--debug", action="store_true")
    return parser


def load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict JSON: {path}")
    return payload


def main() -> int:
    args = build_parser().parse_args()
    artifacts_dir = (PROJECT_ROOT / args.artifacts_dir).resolve()
    grounding = load_json(artifacts_dir / "room_grounding.template.json")

    resolver = SourcePanoResolver(GroundingIndex(grounding))
    resolution = resolver.resolve(args.source_room_id)

    payload = {
        "source_room_id": resolution.source_room_id,
        "source_pano_id": resolution.pano_id,
    }
    if args.debug:
        payload["candidate_pano_ids"] = resolution.candidate_pano_ids
        payload["resolution_method"] = resolution.resolution_method

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
