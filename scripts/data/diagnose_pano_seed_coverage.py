from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from st_nav_data.pano_visualization import extract_grounding_mapping


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def floor_matches(value: Any, floor: str | None) -> bool:
    if floor is None:
        return True
    try:
        return float(value) == float(floor)
    except (TypeError, ValueError):
        return str(value) == floor


def distance_m(lat_a: float, lng_a: float, lat_b: float, lng_b: float) -> float:
    dlat = (lat_a - lat_b) * 111_320
    dlng = (lng_a - lng_b) * 111_320 * math.cos(math.radians(lat_b))
    return math.hypot(dlat, dlng)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Find current nearest pano graph nodes and room labels around a candidate room seed."
    )
    parser.add_argument("--lat", type=float, required=True)
    parser.add_argument("--lng", type=float, required=True)
    parser.add_argument("--floor", default="0")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument(
        "--pano-graph-path",
        default="dataset/sites/british_museum/normalized/pano_graph.json",
    )
    parser.add_argument(
        "--grounding-path",
        default="dataset/sites/british_museum/normalized/pano_room_grounding.json",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    pano_graph = load_json(PROJECT_ROOT / args.pano_graph_path)
    grounding_payload = load_json(PROJECT_ROOT / args.grounding_path)
    grounding, sources = extract_grounding_mapping(grounding_payload)

    rows = []
    for pano_id, node in pano_graph.items():
        lat = node.get("lat")
        lng = node.get("lng")
        if lat is None or lng is None or not floor_matches(node.get("floor"), args.floor):
            continue
        rows.append(
            {
                "distance_m": distance_m(float(lat), float(lng), args.lat, args.lng),
                "pano_id": pano_id,
                "lat": lat,
                "lng": lng,
                "floor": node.get("floor"),
                "room_id": grounding.get(pano_id),
                "source": sources.get(pano_id, "unknown"),
            }
        )

    rows.sort(key=lambda row: row["distance_m"])
    nearest = rows[: max(args.limit, 0)]
    print(json.dumps({"seed": {"lat": args.lat, "lng": args.lng, "floor": args.floor}, "nearest": nearest}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
