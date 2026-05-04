from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .room_grounder import build_compact_pano_room_mapping, merge_records_by_pano_id

JsonDict = dict[str, Any]


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
