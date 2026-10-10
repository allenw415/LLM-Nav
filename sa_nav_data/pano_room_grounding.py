from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .room_grounder import build_compact_pano_room_mapping, merge_records_by_pano_id

JsonDict = dict[str, Any]


def _visual_profile_anchor_entities(node: JsonDict, *, max_items: int = 6) -> list[str]:
    profile = node.get("visual_profile")
    if not isinstance(profile, dict):
        return []
    anchors: list[str] = []
    short_description = profile.get("short_description")
    if isinstance(short_description, str) and short_description:
        anchors.append(short_description)
    for key in ("visual_cues", "possible_text_labels"):
        for value in profile.get(key, []):
            if isinstance(value, str) and value and value not in anchors:
                anchors.append(value)
            if len(anchors) >= max_items:
                return anchors
    return anchors[:max_items]


def load_results_file(path: str | Path) -> list[JsonDict]:
    resolved = Path(path)
    if not resolved.exists():
        return []
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict JSON: {resolved}")
    results = payload.get("results", [])
    if not isinstance(results, list):
        return []
    return [record for record in results if isinstance(record, dict)]


def find_batch_result_paths(batch_dir: str | Path) -> tuple[list[Path], list[Path]]:
    resolved = Path(batch_dir)
    raw_paths = sorted(
        path
        for path in resolved.glob("floor*_batch_*.json")
        if not path.name.endswith(".manual.json") and not path.name.endswith(".review.json")
    )
    manual_paths = sorted(resolved.glob("floor*_batch_*.manual.json"))
    return raw_paths, manual_paths


def build_room_grounding_from_pano_room_mapping(room_graph: JsonDict, pano_room_grounding: JsonDict | None) -> JsonDict:
    entries: JsonDict = {}
    for room_id, node in room_graph.items():
        if not isinstance(room_id, str) or not room_id:
            continue
        node = node if isinstance(node, dict) else {}
        aliases = [value for value in list(node.get("aliases") or []) if isinstance(value, str) and value]
        anchor_entities = []
        for key in ("title", "category"):
            value = node.get(key)
            if isinstance(value, str) and value:
                anchor_entities.append(value)
        anchor_entities.extend(value for value in _visual_profile_anchor_entities(node) if value not in anchor_entities)
        entries[room_id] = {
            "room_id": room_id,
            "floor": str(node.get("floor", "unknown")),
            "pano_ids": [],
            "aliases": aliases,
            "anchor_entities": anchor_entities,
            "notes": "Derived from pano_room_grounding.json.",
        }

    mappings = (pano_room_grounding or {}).get("mappings", pano_room_grounding or {})
    if isinstance(mappings, dict):
        for pano_id, room_id in mappings.items():
            if not isinstance(pano_id, str) or not pano_id:
                continue
            if not isinstance(room_id, str) or not room_id or room_id == "null":
                continue
            entry = entries.setdefault(
                room_id,
                {
                    "room_id": room_id,
                    "floor": "unknown",
                    "pano_ids": [],
                    "aliases": [],
                    "anchor_entities": [],
                    "notes": "Derived from pano_room_grounding.json.",
                },
            )
            pano_ids = entry.setdefault("pano_ids", [])
            if isinstance(pano_ids, list) and pano_id not in pano_ids:
                pano_ids.append(pano_id)

    return {room_id: entries[room_id] for room_id in sorted(entries.keys())}


def rebuild_pano_room_grounding_from_batches(
    batch_dir: str | Path,
    *,
    manual_paths: list[str | Path] | None = None,
    source_name: str = "merged_from_room_grounding_batches",
) -> JsonDict:
    raw_paths, batch_manual_paths = find_batch_result_paths(batch_dir)
    raw_records: list[JsonDict] = []
    manual_records: list[JsonDict] = []
    extra_manual_records: list[JsonDict] = []
    extra_manual_paths = [Path(path) for path in manual_paths or []]

    for path in raw_paths:
        raw_records.extend(load_results_file(path))
    for path in batch_manual_paths:
        manual_records.extend(load_results_file(path))
    for path in extra_manual_paths:
        records = load_results_file(path)
        extra_manual_records.extend(records)
        manual_records.extend(records)

    merged_raw = merge_records_by_pano_id([], raw_records)
    raw_pano_ids = {
        record.get("pano_id")
        for record in merged_raw
        if isinstance(record, dict) and isinstance(record.get("pano_id"), str) and record.get("pano_id")
    }
    extra_manual_only_records = [
        {"pano_id": record["pano_id"]}
        for record in extra_manual_records
        if isinstance(record.get("pano_id"), str)
        and record.get("pano_id")
        and record.get("pano_id") not in raw_pano_ids
    ]
    if extra_manual_only_records:
        merged_raw = merge_records_by_pano_id(merged_raw, extra_manual_only_records)

    compact_mapping = build_compact_pano_room_mapping(
        merged_raw,
        manual_records=manual_records,
    )

    return {
        "summary": {
            "source": source_name,
            "batch_file_count": len(raw_paths),
            "manual_batch_file_count": len(batch_manual_paths),
            "extra_manual_file_count": len([path for path in extra_manual_paths if path.exists()]),
            "grounding_result_count": len(merged_raw),
            "extra_manual_only_record_count": len(extra_manual_only_records),
            "manual_record_count": len(manual_records),
            "mapping_count": len(compact_mapping["mappings"]),
        },
        **compact_mapping,
    }


def write_pano_room_grounding(path: str | Path, payload: JsonDict) -> None:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
