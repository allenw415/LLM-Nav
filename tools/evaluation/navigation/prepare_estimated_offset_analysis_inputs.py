from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError(f"Expected JSON object at {path}:{line_number}")
        rows.append(payload)
    return rows


def index_by_test_id(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        test_id = row.get("test_id")
        if not isinstance(test_id, str) or not test_id:
            raise ValueError(f"{label} contains a row without test_id")
        if test_id in indexed:
            raise ValueError(f"{label} contains duplicate test_id {test_id}")
        indexed[test_id] = row
    return indexed


def is_completed(row: dict[str, Any]) -> bool:
    return row.get("error") is None and row.get("stop_reason") != "error"


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_grounding(path: Path) -> dict[str, str | None]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = (
        payload.get("records")
        or payload.get("grounding")
        or payload.get("panoramas")
        or payload.get("mappings")
    )
    null_sentinels = {"", "null", "none", "unknown", "unlabeled"}

    def normalize(value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return None if text.lower() in null_sentinels else text

    if isinstance(records, dict):
        output: dict[str, str | None] = {}
        for pano_id, value in records.items():
            if isinstance(value, dict):
                output[str(pano_id)] = normalize(value.get("room_id"))
            else:
                output[str(pano_id)] = normalize(value)
        return output
    if isinstance(records, list):
        return {
            str(record["pano_id"]): normalize(record.get("room_id"))
            for record in records
            if isinstance(record, dict) and record.get("pano_id")
        }
    raise ValueError(f"Unsupported grounding schema: {path}")


def build_derived_area_catalog(
    source_path: Path,
    output_path: Path,
    grounding_path: Path,
    result_rows: list[dict[str, Any]],
) -> list[str]:
    catalog = json.loads(source_path.read_text(encoding="utf-8"))
    rooms = catalog.get("rooms")
    if not isinstance(rooms, list):
        raise ValueError("Area catalog rooms must be a list")
    area_room_ids = {
        str(room["room_id"])
        for room in rooms
        if isinstance(room, dict) and room.get("room_id")
    }
    exclusions = {
        str(room_id) for room_id in catalog.get("excluded_route_room_ids", [])
    }
    grounding = load_grounding(grounding_path)
    observed_labels: set[str] = set()
    for row in result_rows:
        pano_path = row.get("pano_path")
        if not isinstance(pano_path, list):
            continue
        for pano_id in pano_path:
            room_id = grounding.get(str(pano_id))
            if isinstance(room_id, str) and room_id:
                observed_labels.add(room_id)
    unknown = sorted(observed_labels - area_room_ids - exclusions)
    missing_gallery_areas = [room_id for room_id in unknown if room_id.startswith("Room ")]
    if missing_gallery_areas:
        raise ValueError(
            "Area-bearing gallery IDs are missing from the catalog: "
            f"{missing_gallery_areas}"
        )
    added_transit_exclusions = [room_id for room_id in unknown if not room_id.startswith("Room ")]
    catalog["excluded_route_room_ids"] = sorted(exclusions | set(added_transit_exclusions))
    catalog["derived_for_analysis"] = {
        "reason": "Exclude non-gallery transit nodes observed in estimated-offset trajectories.",
        "added_transit_exclusions": added_transit_exclusions,
        "source_catalog": str(source_path),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(catalog, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return added_transit_exclusions


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--without-results", type=Path, required=True)
    parser.add_argument("--with-results", type=Path, required=True)
    parser.add_argument("--area-catalog", type=Path, required=True)
    parser.add_argument("--grounding", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    dataset_rows = load_jsonl(args.dataset)
    without_rows = load_jsonl(args.without_results)
    with_rows = load_jsonl(args.with_results)
    dataset = index_by_test_id(dataset_rows, "dataset")
    without = index_by_test_id(without_rows, "without results")
    with_trajectory = index_by_test_id(with_rows, "with results")
    if set(dataset) != set(without) or set(dataset) != set(with_trajectory):
        raise ValueError("Dataset and result test IDs must match before filtering")

    without_completed_ids = {test_id for test_id, row in without.items() if is_completed(row)}
    with_completed_ids = {
        test_id for test_id, row in with_trajectory.items() if is_completed(row)
    }
    common_completed_ids = without_completed_ids & with_completed_ids
    ordered_ids = [str(row["test_id"]) for row in dataset_rows]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "dataset_common": args.output_dir / "dataset_common_completed.jsonl",
        "without_common": args.output_dir / "without_common_completed.jsonl",
        "with_common": args.output_dir / "with_common_completed.jsonl",
        "without_completed": args.output_dir / "without_completed.jsonl",
        "with_completed": args.output_dir / "with_completed.jsonl",
        "area_catalog": args.output_dir / "gallery_areas_with_transit_exclusions.json",
    }
    write_jsonl(
        output_paths["dataset_common"],
        [dataset[test_id] for test_id in ordered_ids if test_id in common_completed_ids],
    )
    write_jsonl(
        output_paths["without_common"],
        [without[test_id] for test_id in ordered_ids if test_id in common_completed_ids],
    )
    write_jsonl(
        output_paths["with_common"],
        [with_trajectory[test_id] for test_id in ordered_ids if test_id in common_completed_ids],
    )
    write_jsonl(
        output_paths["without_completed"],
        [without[test_id] for test_id in ordered_ids if test_id in without_completed_ids],
    )
    write_jsonl(
        output_paths["with_completed"],
        [with_trajectory[test_id] for test_id in ordered_ids if test_id in with_completed_ids],
    )
    added_transit_exclusions = build_derived_area_catalog(
        args.area_catalog,
        output_paths["area_catalog"],
        args.grounding,
        [
            without[test_id]
            for test_id in ordered_ids
            if test_id in without_completed_ids
        ]
        + [
            with_trajectory[test_id]
            for test_id in ordered_ids
            if test_id in with_completed_ids
        ],
    )

    audit = {
        "schema": "llm_nav_estimated_offset_analysis_inputs",
        "dataset_case_count": len(dataset),
        "without_completed_count": len(without_completed_ids),
        "with_completed_count": len(with_completed_ids),
        "common_completed_count": len(common_completed_ids),
        "without_error_test_ids": sorted(set(dataset) - without_completed_ids),
        "with_error_test_ids": sorted(set(dataset) - with_completed_ids),
        "common_excluded_test_ids": sorted(set(dataset) - common_completed_ids),
        "added_transit_area_exclusions": added_transit_exclusions,
        "sources": {
            "dataset": str(args.dataset),
            "without_results": str(args.without_results),
            "with_results": str(args.with_results),
            "area_catalog": str(args.area_catalog),
            "grounding": str(args.grounding),
        },
        "outputs": {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in output_paths.items()
        },
    }
    audit_path = args.output_dir / "input_audit.json"
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
