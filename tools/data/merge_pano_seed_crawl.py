from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from st_nav_data.pano_graph_merge import merge_raw_crawl_payloads


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge one or more supplemental Street View seed crawls into a base raw floor crawl."
    )
    parser.add_argument(
        "--base-raw-path",
        default="dataset/sites/british_museum/pano_graph/raw/streetview_panos_0_1.json",
        help="Existing raw floor crawl JSON to merge into.",
    )
    parser.add_argument(
        "--incoming-raw-path",
        action="append",
        required=True,
        help="Supplemental raw crawl JSON. Repeat this argument for multiple room seed crawls.",
    )
    parser.add_argument(
        "--output-path",
        default="artifacts/pano_seed_crawls/streetview_panos_0_merged.preview.json",
        help="Merged raw crawl JSON output. Keep this separate until you have inspected the summary.",
    )
    parser.add_argument(
        "--overwrite-base",
        action="store_true",
        help="Write the merged payload back to --base-raw-path instead of --output-path.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    base_path = (PROJECT_ROOT / args.base_raw_path).resolve()
    incoming_paths = [(PROJECT_ROOT / path).resolve() for path in args.incoming_raw_path]
    output_path = base_path if args.overwrite_base else (PROJECT_ROOT / args.output_path).resolve()

    base_payload = load_json(base_path)
    incoming_payloads = [load_json(path) for path in incoming_paths]
    merged, summary = merge_raw_crawl_payloads(base_payload, incoming_payloads)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[saved] {output_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not args.overwrite_base:
        print("Inspect this output first. Re-run with --overwrite-base when the summary looks correct.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
