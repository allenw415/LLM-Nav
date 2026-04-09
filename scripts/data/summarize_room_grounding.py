from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize grounding results and manual annotations for one room.")
    parser.add_argument("--room-id", required=True)
    parser.add_argument("--gemini-path", default="dataset/sites/british_museum/normalized/room_grounding.gemini.json")
    parser.add_argument("--manual-path", default="dataset/sites/british_museum/normalized/room_grounding.manual.json")
    return parser


def load_results(path: Path) -> list[dict]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    results = payload.get("results", [])
    if not isinstance(results, list):
        return []
    return [record for record in results if isinstance(record, dict)]


def main() -> int:
    args = build_parser().parse_args()
    room_id = args.room_id

    gemini_records = load_results((PROJECT_ROOT / args.gemini_path).resolve())
    manual_records = load_results((PROJECT_ROOT / args.manual_path).resolve())
    manual_by_pano = {
        record["pano_id"]: record for record in manual_records if isinstance(record.get("pano_id"), str) and record["pano_id"]
    }

    relevant_records = []
    for record in gemini_records:
        pano_id = record.get("pano_id")
        if not isinstance(pano_id, str) or not pano_id:
            continue

        gemini_match = record.get("predicted_room_id") == room_id
        frontier_match = room_id in record.get("frontier_room_ids", [])
        expansion_match = room_id in record.get("expansion_room_ids", [])
        manual_record = manual_by_pano.get(pano_id, {})
        manual_match = manual_record.get("manual_room_id") == room_id

        if not any([gemini_match, frontier_match, expansion_match, manual_match]):
            continue

        relevant_records.append(
            {
                "pano_id": pano_id,
                "region_depth": record.get("region_depth"),
                "predicted_room_id": record.get("predicted_room_id"),
                "confidence": record.get("confidence"),
                "alternative_room_ids": record.get("alternative_room_ids", []),
                "summary": record.get("summary"),
                "manifest_path": record.get("manifest_path"),
                "manual_status": manual_record.get("manual_status", "pending"),
                "manual_room_id": manual_record.get("manual_room_id"),
                "notes": manual_record.get("notes", ""),
            }
        )

    grouped = {
        "accepted": [],
        "boundary": [],
        "ambiguous": [],
        "rejected": [],
        "pending": [],
        "other": [],
    }
    for record in relevant_records:
        status = record.get("manual_status")
        if status not in grouped:
            status = "other"
        grouped[status].append(record)

    for records in grouped.values():
        records.sort(key=lambda item: (item.get("region_depth") or 0, str(item["pano_id"])))

    summary = {
        "room_id": room_id,
        "total": len(relevant_records),
        "accepted": len(grouped["accepted"]),
        "boundary": len(grouped["boundary"]),
        "ambiguous": len(grouped["ambiguous"]),
        "rejected": len(grouped["rejected"]),
        "pending": len(grouped["pending"]),
        "other": len(grouped["other"]),
    }
    print(json.dumps({"summary": summary, "groups": grouped}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
