from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from st_nav_data.pano_room_grounding import rebuild_pano_room_grounding_from_batches, write_pano_room_grounding

DEFAULT_ARTIFACTS_DIR = PROJECT_ROOT / "dataset/sites/british_museum/normalized"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild pano_room_grounding.json from room grounding batch files.")
    parser.add_argument("--artifacts-dir", default=str(DEFAULT_ARTIFACTS_DIR))
    parser.add_argument("--batch-dir")
    parser.add_argument(
        "--manual-path",
        action="append",
        help="Extra manual annotation file. Defaults to room_grounding.manual.json when it exists.",
    )
    parser.add_argument("--output-path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    artifacts_dir = Path(args.artifacts_dir)
    batch_dir = Path(args.batch_dir) if args.batch_dir else artifacts_dir / "room_grounding_batches"
    default_manual_path = artifacts_dir / "room_grounding.manual.json"
    manual_paths = list(args.manual_path or [])
    if not args.manual_path and default_manual_path.exists():
        manual_paths.append(str(default_manual_path))
    output_path = Path(args.output_path) if args.output_path else artifacts_dir / "pano_room_grounding.json"

    payload = rebuild_pano_room_grounding_from_batches(batch_dir, manual_paths=manual_paths)
    write_pano_room_grounding(output_path, payload)
    print(json.dumps({"output_path": str(output_path), "summary": payload["summary"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
