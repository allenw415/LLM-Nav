from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import tempfile
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    "outputs/navigation_paths/floor0_semantic_targets_topology_constrained/"
    "navigation_paths.jsonl"
)
DEFAULT_TEST_SET = (
    "outputs/navigation_paths/floor0_semantic_targets_topology_constrained/"
    "room_count_2to6_ratio_1p0_2p0_per30_sparse715/test_set.jsonl"
)
DEFAULT_OUTPUT_DIR = (
    "outputs/navigation_paths/floor0_semantic_targets_topology_constrained/"
    "thesis_24909"
)

CANDIDATE_FILENAME = "candidate_tasks_24909.psv"
TEST_TASK_IDS_FILENAME = "test_task_ids_715.psv"
CROSSWALK_FILENAME = "task_id_crosswalk.psv"
SUMMARY_FILENAME = "summary.json"
BUNDLE_FILENAME = "thesis_24909_bundle.zip"

CANDIDATE_FIELDS = (
    "task_id",
    "panorama_id_start",
    "panorama_id_end",
    "room_start",
    "room_end",
    "visited_room_count",
    "detour_ratio",
    "straight_line_distance_m",
    "shortest_path_distance_m",
    "shortest_path_room_sequence",
    "shortest_path_panorama_id_sequence",
)
TEST_TASK_ID_FIELDS = ("test_id", "task_id")
CROSSWALK_FIELDS = ("task_id", "legacy_path_id")
TOPOLOGY_CONSTRAINT_MODE = "room_topology_bfs_constrained_dijkstra"


@dataclass(frozen=True)
class ExportExpectations:
    source_task_count: int
    candidate_task_count: int
    eligible_start_panorama_count: int
    semantic_target_group_count: int
    targets_per_start: int
    test_task_count: int
    excluded_start_panoramas_by_room: tuple[tuple[str, int], ...]


THESIS_EXPECTATIONS = ExportExpectations(
    source_task_count=25_245,
    candidate_task_count=24_909,
    eligible_start_panorama_count=1_083,
    semantic_target_group_count=24,
    targets_per_start=23,
    test_task_count=715,
    excluded_start_panoramas_by_room=(("Room 18a", 6), ("Room 18b", 8)),
)


@dataclass(frozen=True)
class ThesisExport:
    candidate_rows: list[dict]
    test_task_id_rows: list[dict]
    crosswalk_rows: list[dict]
    diagnostics: dict


def resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate.resolve() if candidate.is_absolute() else (PROJECT_ROOT / candidate).resolve()


def load_jsonl(path: str | Path) -> list[dict]:
    records: list[dict] = []
    source = Path(path)
    for line_number, raw_line in enumerate(
        source.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw_line.strip():
            continue
        payload = json.loads(raw_line)
        if not isinstance(payload, dict):
            raise ValueError(f"Expected an object at {source}:{line_number}.")
        records.append(payload)
    return records


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identifier_sort_key(identifier: object) -> tuple[str, int, str]:
    value = str(identifier)
    match = re.fullmatch(r"(.*?)(\d+)", value)
    if match is None:
        return value, -1, value
    return match.group(1), int(match.group(2)), value


def _require_nonempty_string(record: dict, field: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{record.get('path_id')} has invalid {field}: {value!r}")
    if "|" in value or "\n" in value or "\r" in value:
        raise ValueError(f"{record.get('path_id')} {field} is not PSV-safe: {value!r}")
    return value


def _require_string_sequence(record: dict, field: str) -> list[str]:
    value = record.get(field)
    if not isinstance(value, list) or not value:
        raise ValueError(f"{record.get('path_id')} has invalid {field}: {value!r}")
    for item in value:
        if not isinstance(item, str) or not item:
            raise ValueError(f"{record.get('path_id')} has invalid {field}: {value!r}")
        if "|" in item or "\n" in item or "\r" in item:
            raise ValueError(
                f"{record.get('path_id')} {field} is not PSV-safe: {item!r}"
            )
    return value


def _validate_source_record(record: dict) -> None:
    path_id = _require_nonempty_string(record, "path_id")
    start_pano_id = _require_nonempty_string(record, "start_pano_id")
    end_pano_id = _require_nonempty_string(record, "end_pano_id")
    start_room_id = _require_nonempty_string(record, "start_room_id")
    end_room_id = _require_nonempty_string(record, "end_room_id")
    _require_nonempty_string(record, "target_group_id")
    acceptable_rooms = _require_string_sequence(record, "acceptable_target_room_ids")
    shortest_rooms = _require_string_sequence(record, "shortest_path_rooms")
    shortest_panos = _require_string_sequence(record, "shortest_path_panos")

    if record.get("path_constraint_mode") != TOPOLOGY_CONSTRAINT_MODE:
        raise ValueError(f"{path_id} is not topology-constrained.")
    if record.get("topology_validation_status") != "valid":
        raise ValueError(f"{path_id} is not topology-valid.")
    if start_room_id in acceptable_rooms:
        raise ValueError(f"{path_id} targets its own semantic target group.")
    if shortest_panos[0] != start_pano_id or shortest_panos[-1] != end_pano_id:
        raise ValueError(f"{path_id} panorama sequence endpoints do not match.")
    if shortest_rooms[0] != start_room_id or shortest_rooms[-1] != end_room_id:
        raise ValueError(f"{path_id} room sequence endpoints do not match.")

    visited_room_count = record.get("visited_room_count")
    if not isinstance(visited_room_count, int) or visited_room_count != len(shortest_rooms):
        raise ValueError(f"{path_id} visited_room_count does not match its room sequence.")

    try:
        detour_ratio = float(record["detour_ratio"])
        path_distance_m = float(record["path_distance_m"])
        straight_line_distance_m = float(record["straight_line_distance_m"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{path_id} has invalid distance fields.") from error
    if not all(
        math.isfinite(value)
        for value in (detour_ratio, path_distance_m, straight_line_distance_m)
    ):
        raise ValueError(f"{path_id} has non-finite distance fields.")
    if path_distance_m < 0.0 or straight_line_distance_m <= 0.0:
        raise ValueError(f"{path_id} has non-positive distance fields.")
    if not math.isclose(
        detour_ratio,
        path_distance_m / straight_line_distance_m,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError(f"{path_id} has an inconsistent detour_ratio.")


def _target_groups(records: Iterable[dict]) -> dict[str, tuple[str, ...]]:
    groups: dict[str, tuple[str, ...]] = {}
    for record in records:
        group_id = _require_nonempty_string(record, "target_group_id")
        acceptable_rooms = tuple(
            sorted(set(_require_string_sequence(record, "acceptable_target_room_ids")))
        )
        previous = groups.setdefault(group_id, acceptable_rooms)
        if previous != acceptable_rooms:
            raise ValueError(
                f"Target group {group_id!r} has inconsistent acceptable rooms: "
                f"{previous!r} versus {acceptable_rooms!r}."
            )
    if not groups:
        raise ValueError("No semantic target groups were found.")
    return groups


def _room_to_target_group(
    target_groups: dict[str, tuple[str, ...]],
) -> dict[str, str]:
    room_to_group: dict[str, str] = {}
    for group_id, room_ids in target_groups.items():
        for room_id in room_ids:
            previous = room_to_group.setdefault(room_id, group_id)
            if previous != group_id:
                raise ValueError(
                    f"Room {room_id!r} belongs to both {previous!r} and {group_id!r}."
                )
    return room_to_group


def _candidate_row(record: dict, *, task_id: str) -> dict:
    return {
        "task_id": task_id,
        "panorama_id_start": record["start_pano_id"],
        "panorama_id_end": record["end_pano_id"],
        "room_start": record["start_room_id"],
        "room_end": record["end_room_id"],
        "visited_room_count": record["visited_room_count"],
        "detour_ratio": record["detour_ratio"],
        "straight_line_distance_m": record["straight_line_distance_m"],
        "shortest_path_distance_m": record["path_distance_m"],
        "shortest_path_room_sequence": " -> ".join(record["shortest_path_rooms"]),
        "shortest_path_panorama_id_sequence": " -> ".join(
            record["shortest_path_panos"]
        ),
    }


def build_thesis_export(
    source_records: Sequence[dict],
    test_records: Sequence[dict],
    *,
    task_id_prefix: str = "TASK",
) -> ThesisExport:
    if not task_id_prefix or "|" in task_id_prefix:
        raise ValueError("task_id_prefix must be a non-empty PSV-safe string.")
    if not source_records:
        raise ValueError("The source route corpus is empty.")

    source_path_ids: set[str] = set()
    source_pairs: set[tuple[str, str]] = set()
    start_room_by_pano: dict[str, str] = {}
    for record in source_records:
        _validate_source_record(record)
        path_id = str(record["path_id"])
        if path_id in source_path_ids:
            raise ValueError(f"Duplicate source path_id: {path_id}")
        source_path_ids.add(path_id)
        pair = (str(record["start_pano_id"]), str(record["target_group_id"]))
        if pair in source_pairs:
            raise ValueError(f"Duplicate start/target pair: {pair!r}")
        source_pairs.add(pair)
        start_pano_id = str(record["start_pano_id"])
        start_room_id = str(record["start_room_id"])
        previous_room = start_room_by_pano.setdefault(start_pano_id, start_room_id)
        if previous_room != start_room_id:
            raise ValueError(
                f"Start panorama {start_pano_id!r} has inconsistent rooms."
            )

    target_groups = _target_groups(source_records)
    room_to_group = _room_to_target_group(target_groups)
    eligible_start_rooms = set(room_to_group)
    filtered_records = sorted(
        (
            record
            for record in source_records
            if str(record["start_room_id"]) in eligible_start_rooms
        ),
        key=lambda record: _identifier_sort_key(record["path_id"]),
    )
    removed_records = [
        record
        for record in source_records
        if str(record["start_room_id"]) not in eligible_start_rooms
    ]

    target_groups_by_start: dict[str, set[str]] = defaultdict(set)
    for record in filtered_records:
        target_groups_by_start[str(record["start_pano_id"])].add(
            str(record["target_group_id"])
        )
    all_target_group_ids = set(target_groups)
    for start_pano_id, actual_group_ids in target_groups_by_start.items():
        start_room_id = start_room_by_pano[start_pano_id]
        own_group_id = room_to_group[start_room_id]
        expected_group_ids = all_target_group_ids - {own_group_id}
        if actual_group_ids != expected_group_ids:
            missing = sorted(expected_group_ids - actual_group_ids)
            extra = sorted(actual_group_ids - expected_group_ids)
            raise ValueError(
                f"Start panorama {start_pano_id!r} has the wrong target groups; "
                f"missing={missing}, extra={extra}."
            )

    task_id_width = max(5, len(str(len(filtered_records))))
    candidate_rows: list[dict] = []
    crosswalk_rows: list[dict] = []
    task_id_by_path_id: dict[str, str] = {}
    for index, record in enumerate(filtered_records, start=1):
        task_id = f"{task_id_prefix}{index:0{task_id_width}d}"
        path_id = str(record["path_id"])
        candidate_rows.append(_candidate_row(record, task_id=task_id))
        crosswalk_rows.append({"task_id": task_id, "legacy_path_id": path_id})
        task_id_by_path_id[path_id] = task_id

    test_ids: set[str] = set()
    test_path_ids: set[str] = set()
    test_task_id_rows: list[dict] = []
    for record in sorted(
        test_records, key=lambda item: _identifier_sort_key(item.get("test_id"))
    ):
        test_id = _require_nonempty_string(record, "test_id")
        path_id = _require_nonempty_string(record, "path_id")
        if test_id in test_ids:
            raise ValueError(f"Duplicate test_id: {test_id}")
        if path_id in test_path_ids:
            raise ValueError(f"Duplicate test path_id: {path_id}")
        test_ids.add(test_id)
        test_path_ids.add(path_id)
        task_id = task_id_by_path_id.get(path_id)
        if task_id is None:
            raise ValueError(
                f"Test case {test_id} / {path_id} is absent from the thesis corpus."
            )
        test_task_id_rows.append({"test_id": test_id, "task_id": task_id})

    excluded_start_panos_by_room: dict[str, set[str]] = defaultdict(set)
    removed_tasks_by_room: Counter[str] = Counter()
    for record in removed_records:
        room_id = str(record["start_room_id"])
        excluded_start_panos_by_room[room_id].add(str(record["start_pano_id"]))
        removed_tasks_by_room[room_id] += 1

    route_counts = Counter(
        len(group_ids) for group_ids in target_groups_by_start.values()
    )
    diagnostics = {
        "source_task_count": len(source_records),
        "candidate_task_count": len(candidate_rows),
        "semantic_target_group_count": len(target_groups),
        "semantic_target_groups": [
            {
                "target_group_id": group_id,
                "acceptable_target_room_ids": list(target_groups[group_id]),
            }
            for group_id in sorted(target_groups)
        ],
        "eligible_start_room_ids": sorted(eligible_start_rooms),
        "eligible_start_panorama_count": len(target_groups_by_start),
        "tasks_per_start_distribution": {
            str(count): frequency for count, frequency in sorted(route_counts.items())
        },
        "excluded_start_panorama_count": sum(
            len(panos) for panos in excluded_start_panos_by_room.values()
        ),
        "excluded_start_panoramas_by_room": {
            room_id: len(excluded_start_panos_by_room[room_id])
            for room_id in sorted(excluded_start_panos_by_room)
        },
        "removed_task_count": len(removed_records),
        "removed_tasks_by_start_room": {
            room_id: removed_tasks_by_room[room_id]
            for room_id in sorted(removed_tasks_by_room)
        },
        "test_task_count": len(test_task_id_rows),
        "task_id_prefix": task_id_prefix,
        "task_id_width": task_id_width,
        "task_id_first": candidate_rows[0]["task_id"] if candidate_rows else None,
        "task_id_last": candidate_rows[-1]["task_id"] if candidate_rows else None,
    }
    return ThesisExport(
        candidate_rows=candidate_rows,
        test_task_id_rows=test_task_id_rows,
        crosswalk_rows=crosswalk_rows,
        diagnostics=diagnostics,
    )


def validate_expectations(
    export: ThesisExport,
    expectations: ExportExpectations = THESIS_EXPECTATIONS,
) -> None:
    diagnostics = export.diagnostics
    expected_scalars = {
        "source_task_count": expectations.source_task_count,
        "candidate_task_count": expectations.candidate_task_count,
        "eligible_start_panorama_count": expectations.eligible_start_panorama_count,
        "semantic_target_group_count": expectations.semantic_target_group_count,
        "test_task_count": expectations.test_task_count,
    }
    for field, expected in expected_scalars.items():
        actual = diagnostics.get(field)
        if actual != expected:
            raise ValueError(f"Expected {field}={expected}, found {actual}.")

    expected_distribution = {
        str(expectations.targets_per_start): expectations.eligible_start_panorama_count
    }
    if diagnostics["tasks_per_start_distribution"] != expected_distribution:
        raise ValueError(
            "Unexpected tasks-per-start distribution: "
            f"{diagnostics['tasks_per_start_distribution']!r}."
        )

    expected_exclusions = dict(expectations.excluded_start_panoramas_by_room)
    if diagnostics["excluded_start_panoramas_by_room"] != expected_exclusions:
        raise ValueError(
            "Unexpected excluded start panoramas: "
            f"{diagnostics['excluded_start_panoramas_by_room']!r}."
        )
    expected_removed = sum(
        count * expectations.semantic_target_group_count
        for count in expected_exclusions.values()
    )
    if diagnostics["removed_task_count"] != expected_removed:
        raise ValueError(
            f"Expected removed_task_count={expected_removed}, "
            f"found {diagnostics['removed_task_count']}."
        )

    task_ids = [str(row["task_id"]) for row in export.candidate_rows]
    width = int(diagnostics["task_id_width"])
    expected_task_ids = [
        f"{diagnostics['task_id_prefix']}{index:0{width}d}"
        for index in range(1, expectations.candidate_task_count + 1)
    ]
    if task_ids != expected_task_ids:
        raise ValueError("task_id values are not unique and contiguous.")
    if len(export.crosswalk_rows) != expectations.candidate_task_count:
        raise ValueError("The task-ID crosswalk has the wrong row count.")


def _atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            temporary_path = Path(stream.name)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _atomic_write_psv(
    path: Path,
    rows: Iterable[dict],
    *,
    fieldnames: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=fieldnames,
                delimiter="|",
                lineterminator="\n",
                extrasaction="raise",
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
            temporary_path = Path(stream.name)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _atomic_write_bundle(output_dir: Path, member_names: Sequence[str]) -> Path:
    bundle_path = output_dir / BUNDLE_FILENAME
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            dir=output_dir,
            prefix=f".{bundle_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
        with zipfile.ZipFile(
            temporary_path, mode="w", compression=zipfile.ZIP_DEFLATED
        ) as archive:
            for member_name in member_names:
                archive.write(output_dir / member_name, arcname=member_name)
        os.replace(temporary_path, bundle_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return bundle_path


def _read_psv(path: Path, expected_fields: Sequence[str]) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="|")
        if reader.fieldnames != list(expected_fields):
            raise ValueError(
                f"{path} has header {reader.fieldnames!r}, expected {list(expected_fields)!r}."
            )
        rows = list(reader)
    if any(None in row for row in rows):
        raise ValueError(f"{path} contains a row with too many columns.")
    return rows


def validate_export_directory(
    output_dir: str | Path,
    expectations: ExportExpectations = THESIS_EXPECTATIONS,
) -> dict:
    directory = Path(output_dir)
    summary_path = directory / SUMMARY_FILENAME
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(summary, dict):
        raise ValueError(f"Expected a JSON object in {summary_path}.")

    candidate_path = directory / CANDIDATE_FILENAME
    test_path = directory / TEST_TASK_IDS_FILENAME
    crosswalk_path = directory / CROSSWALK_FILENAME
    candidate_rows = _read_psv(candidate_path, CANDIDATE_FIELDS)
    test_rows = _read_psv(test_path, TEST_TASK_ID_FIELDS)
    crosswalk_rows = _read_psv(crosswalk_path, CROSSWALK_FIELDS)

    if len(candidate_rows) != expectations.candidate_task_count:
        raise ValueError(f"{candidate_path} has {len(candidate_rows)} data rows.")
    if len(test_rows) != expectations.test_task_count:
        raise ValueError(f"{test_path} has {len(test_rows)} data rows.")
    if len(crosswalk_rows) != expectations.candidate_task_count:
        raise ValueError(f"{crosswalk_path} has {len(crosswalk_rows)} data rows.")

    width = max(5, len(str(expectations.candidate_task_count)))
    expected_task_ids = [
        f"TASK{index:0{width}d}"
        for index in range(1, expectations.candidate_task_count + 1)
    ]
    candidate_task_ids = [row["task_id"] for row in candidate_rows]
    if candidate_task_ids != expected_task_ids:
        raise ValueError("Candidate task IDs are not TASK00001...TASK24909.")
    if [row["task_id"] for row in crosswalk_rows] != expected_task_ids:
        raise ValueError("Crosswalk task IDs do not match the candidate rows.")

    candidate_task_id_set = set(candidate_task_ids)
    test_ids = [row["test_id"] for row in test_rows]
    test_task_ids = [row["task_id"] for row in test_rows]
    if len(test_ids) != len(set(test_ids)):
        raise ValueError("Test IDs are not unique.")
    if len(test_task_ids) != len(set(test_task_ids)):
        raise ValueError("Test task IDs are not unique.")
    if not set(test_task_ids).issubset(candidate_task_id_set):
        raise ValueError("At least one test task ID is absent from the candidate corpus.")

    start_counts: Counter[str] = Counter()
    for row in candidate_rows:
        room_sequence = row["shortest_path_room_sequence"].split(" -> ")
        pano_sequence = row["shortest_path_panorama_id_sequence"].split(" -> ")
        if room_sequence[0] != row["room_start"] or room_sequence[-1] != row["room_end"]:
            raise ValueError(f"{row['task_id']} has inconsistent room endpoints.")
        if (
            pano_sequence[0] != row["panorama_id_start"]
            or pano_sequence[-1] != row["panorama_id_end"]
        ):
            raise ValueError(f"{row['task_id']} has inconsistent panorama endpoints.")
        if int(row["visited_room_count"]) != len(room_sequence):
            raise ValueError(f"{row['task_id']} has an inconsistent visited-room count.")
        ratio = float(row["detour_ratio"])
        straight = float(row["straight_line_distance_m"])
        shortest = float(row["shortest_path_distance_m"])
        if not math.isclose(
            ratio, shortest / straight, rel_tol=1e-12, abs_tol=1e-12
        ):
            raise ValueError(f"{row['task_id']} has an inconsistent detour ratio.")
        start_counts[row["panorama_id_start"]] += 1
    if len(start_counts) != expectations.eligible_start_panorama_count:
        raise ValueError("The emitted candidate corpus has the wrong number of starts.")
    if set(start_counts.values()) != {expectations.targets_per_start}:
        raise ValueError("Not every emitted start panorama has the expected task count.")

    expected_hashes = summary.get("output_files")
    if not isinstance(expected_hashes, dict):
        raise ValueError("summary.json has no output_files object.")
    for filename in (CANDIDATE_FILENAME, TEST_TASK_IDS_FILENAME, CROSSWALK_FILENAME):
        metadata = expected_hashes.get(filename)
        if not isinstance(metadata, dict) or metadata.get("sha256") != sha256_file(
            directory / filename
        ):
            raise ValueError(f"Hash mismatch for {filename}.")

    bundle_path = directory / BUNDLE_FILENAME
    expected_members = {
        CANDIDATE_FILENAME,
        TEST_TASK_IDS_FILENAME,
        CROSSWALK_FILENAME,
        SUMMARY_FILENAME,
    }
    with zipfile.ZipFile(bundle_path, mode="r") as archive:
        if archive.testzip() is not None:
            raise ValueError(f"{bundle_path} contains a corrupt member.")
        if set(archive.namelist()) != expected_members:
            raise ValueError(f"{bundle_path} has unexpected members.")
        for filename in expected_members:
            if archive.read(filename) != (directory / filename).read_bytes():
                raise ValueError(f"ZIP member {filename} does not match its source file.")

    return {
        "status": "valid",
        "candidate_task_count": len(candidate_rows),
        "eligible_start_panorama_count": len(start_counts),
        "targets_per_start": expectations.targets_per_start,
        "test_task_count": len(test_rows),
        "task_id_first": candidate_task_ids[0],
        "task_id_last": candidate_task_ids[-1],
        "bundle": str(bundle_path),
    }


def export_thesis_candidate_tasks(
    *,
    input_path: Path,
    test_set_path: Path,
    output_dir: Path,
    expectations: ExportExpectations = THESIS_EXPECTATIONS,
) -> dict:
    source_records = load_jsonl(input_path)
    test_records = load_jsonl(test_set_path)
    export = build_thesis_export(source_records, test_records)
    validate_expectations(export, expectations)

    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = output_dir / CANDIDATE_FILENAME
    test_task_ids_path = output_dir / TEST_TASK_IDS_FILENAME
    crosswalk_path = output_dir / CROSSWALK_FILENAME
    _atomic_write_psv(
        candidate_path, export.candidate_rows, fieldnames=CANDIDATE_FIELDS
    )
    _atomic_write_psv(
        test_task_ids_path,
        export.test_task_id_rows,
        fieldnames=TEST_TASK_ID_FIELDS,
    )
    _atomic_write_psv(
        crosswalk_path, export.crosswalk_rows, fieldnames=CROSSWALK_FIELDS
    )

    diagnostics = export.diagnostics
    summary = {
        "schema": "llm_nav_thesis_candidate_tasks_summary",
        "schema_version": 1,
        "generator": {
            "name": "export_thesis_candidate_tasks",
            "selection_rule": (
                "Keep starts whose room belongs to one of the 24 semantic target "
                "groups, then retain the other 23 groups for each start."
            ),
            "path_recomputation": False,
            "task_id_policy": "contiguous_TASK00001_to_TASK24909",
        },
        "source": {
            "navigation_paths": str(input_path),
            "navigation_paths_sha256": sha256_file(input_path),
            "navigation_path_count": len(source_records),
            "test_set": str(test_set_path),
            "test_set_sha256": sha256_file(test_set_path),
            "test_set_count": len(test_records),
        },
        "task_definition": {
            "formula": "1083 start panoramas x 23 remaining semantic targets = 24909 tasks",
            "semantic_target_interpretation": (
                "There are 24 semantic target groups in total. The start panorama's "
                "own group is excluded, leaving 23 targets per start."
            ),
            "distance_unit": "meters",
            "visited_room_count_definition": (
                "Room nodes on the reference shortest path, including start, end, "
                "and required circulation nodes."
            ),
            "detour_ratio_definition": (
                "shortest_path_distance_m / straight_line_distance_m"
            ),
            "room_end_definition": (
                "The actual reference endpoint room selected for the semantic target group."
            ),
            "sequence_separator": " -> ",
            "column_separator": "|",
        },
        **diagnostics,
        "output_files": {
            filename: {
                "path": str(output_dir / filename),
                "sha256": sha256_file(output_dir / filename),
            }
            for filename in (
                CANDIDATE_FILENAME,
                TEST_TASK_IDS_FILENAME,
                CROSSWALK_FILENAME,
            )
        },
        "zip_bundle": {
            "filename": BUNDLE_FILENAME,
            "members": [
                CANDIDATE_FILENAME,
                TEST_TASK_IDS_FILENAME,
                CROSSWALK_FILENAME,
                SUMMARY_FILENAME,
            ],
        },
    }
    _atomic_write_json(output_dir / SUMMARY_FILENAME, summary)
    _atomic_write_bundle(
        output_dir,
        (
            CANDIDATE_FILENAME,
            TEST_TASK_IDS_FILENAME,
            CROSSWALK_FILENAME,
            SUMMARY_FILENAME,
        ),
    )
    return validate_export_directory(output_dir, expectations)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export the thesis-defined 24,909 British Museum Level-0 candidate "
            "navigation tasks and map the 715 evaluation cases to contiguous task IDs."
        )
    )
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--test-set", default=DEFAULT_TEST_SET)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Validate an existing output directory without rewriting it.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_dir = resolve_project_path(args.output_dir)
    if args.verify_only:
        result = validate_export_directory(output_dir)
    else:
        result = export_thesis_candidate_tasks(
            input_path=resolve_project_path(args.input),
            test_set_path=resolve_project_path(args.test_set),
            output_dir=output_dir,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
