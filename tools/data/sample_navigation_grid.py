from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import random
import re
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    "outputs/navigation_paths/floor0_semantic_targets_topology_constrained/"
    "navigation_paths.jsonl"
)
DEFAULT_PRESERVE_TEST_SET = (
    "outputs/navigation_paths/floor0_semantic_targets_topology_constrained/"
    "room_count_2to6_ratio_1p0_2p0_per10_sparse249/test_set.jsonl"
)
DEFAULT_QUERIES_BY_TARGET = (
    "outputs/navigation_paths/floor0_semantic_targets_topology_constrained/"
    "room_count_2to6_ratio_1p0_2p0_per10_sparse249/queries_by_target.json"
)
DEFAULT_FAIL_EDGES = (
    "dataset/sites/british_museum/room_graph/manual_annotations/"
    "passage_selection_fail_edges_floor0.json"
)
DEFAULT_OUTPUT_DIR = (
    "outputs/navigation_paths/floor0_semantic_targets_topology_constrained/"
    "room_count_2to6_ratio_1p0_2p0_per20_sparse485"
)
DEFAULT_VISITED_ROOM_COUNTS = (2, 3, 4, 5, 6)
DEFAULT_EXCLUDED_ROOM_IDS = ("Room 18a", "Room 18b")
DEFAULT_PER_GRID_CELL = 20
GENERATOR_SCHEMA_VERSION = 1
GENERATOR_PROVENANCE = {
    "name": "llm_nav_room_count_ratio_grid_sampler",
    "ported_from": (
        "memory-based-sa-nav/tools/data/sample_navigation_test_set.py:"
        "room-count-ratio-grid"
    ),
    "selection_algorithm": "partial_preserve_seeded_greedy_v1",
}
TEST_SET_CSV_FIELDS = (
    "test_id",
    "path_id",
    "query",
    "target_group_id",
    "acceptable_target_room_ids",
    "reference_end_room_id",
    "start_pano_id",
    "start_room_id",
    "end_pano_id",
    "visited_room_count",
    "ratio_stratum",
    "detour_ratio",
    "path_distance_m",
    "straight_line_distance_m",
    "pano_step_count",
    "passage_profile",
    "known_failed_passage_edges_on_path",
    "shortest_path_rooms",
)


def resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate.resolve() if candidate.is_absolute() else (PROJECT_ROOT / candidate).resolve()


def load_jsonl(path: str | Path) -> list[dict]:
    records: list[dict] = []
    path = Path(path)
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw_line.strip():
            continue
        payload = json.loads(raw_line)
        if not isinstance(payload, dict):
            raise ValueError(f"Expected an object at {path}:{line_number}.")
        records.append(payload)
    return records


def load_json_object(path: str | Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        stream.write(text)
        temporary_path = Path(stream.name)
    os.replace(temporary_path, path)


def write_json(path: Path, payload: object) -> None:
    _atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    text = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        for record in records
    )
    _atomic_write_text(path, text)


def write_csv(path: Path, records: Iterable[dict]) -> None:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=TEST_SET_CSV_FIELDS)
    writer.writeheader()
    for record in records:
        row = {field: record.get(field) for field in TEST_SET_CSV_FIELDS}
        row["acceptable_target_room_ids"] = " | ".join(
            record.get("acceptable_target_room_ids", [])
        )
        row["shortest_path_rooms"] = " -> ".join(record.get("shortest_path_rooms", []))
        row["known_failed_passage_edges_on_path"] = " | ".join(
            f"{edge['from_room_id']} -> {edge['to_room_id']}"
            for edge in record.get("known_failed_passage_edges_on_path", [])
        )
        writer.writerow(row)
    _atomic_write_text(path, stream.getvalue())


def normalize_passage_room_id(room_id: object) -> str:
    normalized = str(room_id).strip()
    if normalized.casefold() == "north stairs":
        return "North stairs"
    return normalized


def load_directed_fail_edges(path: Path) -> tuple[tuple[str, str], ...]:
    payload = load_json_object(path)
    values = payload.get("directed_fail_edges")
    if not isinstance(values, list):
        raise ValueError(f"{path} does not contain directed_fail_edges.")
    edges: list[tuple[str, str]] = []
    for index, item in enumerate(values):
        if not isinstance(item, dict):
            raise ValueError(f"Invalid fail edge at index {index}: {item!r}")
        source = normalize_passage_room_id(item.get("from_room_id", ""))
        target = normalize_passage_room_id(item.get("to_room_id", ""))
        if not source or not target:
            raise ValueError(f"Fail edge {index} has an empty endpoint.")
        edges.append((source, target))
    if len(edges) != len(set(edges)):
        raise ValueError("directed_fail_edges contains duplicates.")
    return tuple(edges)


def _format_ratio_boundary(value: float) -> str:
    formatted = f"{float(value):.10f}".rstrip("0").rstrip(".")
    return formatted if "." in formatted else f"{formatted}.0"


def build_ratio_grid_strata(
    *,
    lower_bound: float,
    upper_bound: float,
    bin_width: float,
) -> tuple[dict, ...]:
    if not all(math.isfinite(value) for value in (lower_bound, upper_bound, bin_width)):
        raise ValueError("Ratio grid bounds and width must be finite.")
    if bin_width <= 0.0 or upper_bound <= lower_bound:
        raise ValueError("Ratio grid requires positive width and upper > lower.")
    raw_bin_count = (upper_bound - lower_bound) / bin_width
    bin_count = int(round(raw_bin_count))
    if bin_count <= 0 or not math.isclose(raw_bin_count, bin_count, abs_tol=1e-9):
        raise ValueError("Ratio range must be an integer multiple of bin width.")
    return tuple(
        {
            "index": index,
            "label": (
                f"{_format_ratio_boundary(round(lower_bound + index * bin_width, 10))}-"
                f"{_format_ratio_boundary(round(lower_bound + (index + 1) * bin_width, 10))}"
            ),
            "lower_bound": float(round(lower_bound + index * bin_width, 10)),
            "upper_bound": float(round(lower_bound + (index + 1) * bin_width, 10)),
            "lower_inclusive": True,
            "upper_inclusive": False,
        }
        for index in range(bin_count)
    )


def ratio_grid_stratum_for_value(
    value: float,
    *,
    lower_bound: float = 1.0,
    upper_bound: float = 2.0,
    bin_width: float = 0.2,
) -> dict | None:
    ratio = float(value)
    strata = build_ratio_grid_strata(
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        bin_width=bin_width,
    )
    if ratio < lower_bound or ratio >= upper_bound:
        return None
    index = int(math.floor((ratio - lower_bound) / bin_width + 1e-12))
    if index < 0 or index >= len(strata):
        return None
    return dict(strata[index])


def _normalize_room_ids(room_ids: Iterable[str]) -> tuple[str, ...]:
    values: list[str] = []
    seen: set[str] = set()
    for room_id in room_ids:
        value = normalize_passage_room_id(room_id)
        if value and value not in seen:
            values.append(value)
            seen.add(value)
    return tuple(values)


def excluded_room_hits(record: dict, excluded_room_ids: Iterable[str]) -> tuple[str, ...]:
    excluded = set(_normalize_room_ids(excluded_room_ids))
    if not excluded:
        return ()
    referenced: list[str] = []
    for field in (
        "start_room_id",
        "reference_end_room_id",
        "target_room_id",
        "end_room_id",
    ):
        value = record.get(field)
        if value is not None:
            referenced.append(normalize_passage_room_id(value))
    for field in ("acceptable_target_room_ids", "shortest_path_rooms"):
        referenced.extend(
            normalize_passage_room_id(value)
            for value in record.get(field, [])
        )
    return tuple(sorted(excluded.intersection(referenced)))


def _known_failed_passage_edges_on_path(
    record: dict,
    fail_edge_set: set[tuple[str, str]],
) -> list[dict[str, str]]:
    rooms = [normalize_passage_room_id(room_id) for room_id in record["shortest_path_rooms"]]
    hits: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for edge in zip(rooms, rooms[1:]):
        if edge in fail_edge_set and edge not in seen:
            hits.append(edge)
            seen.add(edge)
    return [
        {"from_room_id": source, "to_room_id": target}
        for source, target in hits
    ]


def _start_target_pair_key(record: dict) -> tuple[str, str]:
    return str(record["start_room_id"]), str(record["target_group_id"])


def _ordered_room_sequence_key(record: dict) -> tuple[str, ...]:
    return tuple(str(room_id) for room_id in record["shortest_path_rooms"])


def _grid_cell_key(
    record: dict,
    *,
    lower_bound: float,
    upper_bound: float,
    bin_width: float,
) -> tuple[int, int]:
    stratum = ratio_grid_stratum_for_value(
        float(record["detour_ratio"]),
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        bin_width=bin_width,
    )
    if stratum is None:
        raise ValueError(f"Record {record.get('path_id')} falls outside the requested ratio grid.")
    return int(record["visited_room_count"]), int(stratum["index"])


def _prepare_candidates(
    records: Iterable[dict],
    fail_edges: Iterable[tuple[str, str]],
    *,
    room_counts: Iterable[int],
    lower_bound: float,
    upper_bound: float,
    bin_width: float,
    excluded_room_ids: Iterable[str],
) -> tuple[tuple[int, ...], tuple[dict, ...], dict[tuple[int, int], list[dict]], dict]:
    requested_room_counts = tuple(sorted({int(value) for value in room_counts}))
    if not requested_room_counts or any(value <= 0 for value in requested_room_counts):
        raise ValueError("room_counts must contain positive integers.")
    requested_set = set(requested_room_counts)
    strata = build_ratio_grid_strata(
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        bin_width=bin_width,
    )
    fail_edge_set = {
        (normalize_passage_room_id(source), normalize_passage_room_id(target))
        for source, target in fail_edges
    }
    excluded_ids = _normalize_room_ids(excluded_room_ids)
    cells: dict[tuple[int, int], list[dict]] = defaultdict(list)
    excluded_room_count = 0
    excluded_below = 0
    excluded_above = 0
    excluded_referenced_count = 0
    excluded_reference_counts: Counter[str] = Counter()
    for source in records:
        room_count = int(source["visited_room_count"])
        if room_count not in requested_set:
            excluded_room_count += 1
            continue
        ratio = float(source["detour_ratio"])
        stratum = ratio_grid_stratum_for_value(
            ratio,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            bin_width=bin_width,
        )
        if stratum is None:
            if ratio < lower_bound:
                excluded_below += 1
            else:
                excluded_above += 1
            continue
        room_hits = excluded_room_hits(source, excluded_ids)
        if room_hits:
            excluded_referenced_count += 1
            excluded_reference_counts.update(room_hits)
            continue
        hits = _known_failed_passage_edges_on_path(source, fail_edge_set)
        record = dict(source)
        record.update(
            {
                "ratio_stratum": stratum["label"],
                "ratio_stratum_lower_bound": stratum["lower_bound"],
                "ratio_stratum_upper_bound": stratum["upper_bound"],
                "ratio_stratum_lower_inclusive": True,
                "ratio_stratum_upper_inclusive": False,
                "ratio_stratum_type": "fixed_width_detour_ratio",
                "passage_profile": "risk" if hits else "reliable",
                "known_failed_passage_edges_on_path": hits,
                "sampling_cell_id": f"rooms_{room_count}/{stratum['label']}",
            }
        )
        cells[(room_count, int(stratum["index"]))].append(record)
    diagnostics = {
        "excluded_outside_requested_room_counts_count": excluded_room_count,
        "excluded_below_ratio_range_count": excluded_below,
        "excluded_at_or_above_ratio_range_count": excluded_above,
        "excluded_room_ids": list(excluded_ids),
        "excluded_referenced_room_candidate_count": excluded_referenced_count,
        "excluded_room_reference_counts": {
            room_id: excluded_reference_counts[room_id]
            for room_id in excluded_ids
        },
    }
    return requested_room_counts, strata, cells, diagnostics


def _validate_preserved_record(previous: dict, candidate: dict) -> None:
    generated_fields = {
        "query",
        "query_id",
        "test_id",
        "ratio_stratum",
        "ratio_stratum_lower_bound",
        "ratio_stratum_upper_bound",
        "ratio_stratum_lower_inclusive",
        "ratio_stratum_upper_inclusive",
        "ratio_stratum_type",
        "passage_profile",
        "known_failed_passage_edges_on_path",
        "sampling_cell_id",
        "sampling_cell_candidate_count",
        "sampling_cell_selected_count",
        "sampling_cell_sparse",
    }
    for key, value in candidate.items():
        if key in generated_fields:
            continue
        if previous.get(key) != value:
            raise ValueError(
                f"Preserved case {previous.get('test_id')} changed field {key}: "
                f"{previous.get(key)!r} != {value!r}."
            )


def _next_test_id_state(records: Iterable[dict], prefix: str) -> tuple[int, int]:
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)$")
    numbers: list[int] = []
    widths: list[int] = []
    for record in records:
        test_id = str(record.get("test_id", ""))
        match = pattern.fullmatch(test_id)
        if match is None:
            raise ValueError(f"Preserved test ID does not match {prefix}<digits>: {test_id}")
        numbers.append(int(match.group(1)))
        widths.append(len(match.group(1)))
    return max(numbers, default=0) + 1, max([3, *widths])


def _validate_query_mapping(record: dict, query: dict) -> None:
    group_id = str(record["target_group_id"])
    if query.get("target_group_id") not in (None, group_id):
        raise ValueError(f"Query mapping target mismatch for {group_id}.")
    mapped_rooms = query.get("acceptable_target_room_ids")
    if isinstance(mapped_rooms, list) and mapped_rooms != record.get("acceptable_target_room_ids"):
        raise ValueError(f"Query mapping acceptable rooms mismatch for {group_id}.")
    if not isinstance(query.get("query_id"), str) or not isinstance(query.get("query"), str):
        raise ValueError(f"Query mapping is incomplete for {group_id}.")


def build_partial_grid(
    records: Iterable[dict],
    previous_test_records: Iterable[dict],
    fail_edges: Iterable[tuple[str, str]],
    queries_by_target: dict[str, dict],
    *,
    seed: int = 42,
    room_counts: Iterable[int] = DEFAULT_VISITED_ROOM_COUNTS,
    lower_bound: float = 1.0,
    upper_bound: float = 2.0,
    bin_width: float = 0.2,
    per_cell: int = DEFAULT_PER_GRID_CELL,
    allow_sparse_cells: bool = False,
    excluded_room_ids: Iterable[str] = DEFAULT_EXCLUDED_ROOM_IDS,
    test_id_prefix: str = "TEST",
) -> tuple[list[dict], list[dict], dict, dict, dict]:
    if per_cell <= 0:
        raise ValueError("per_cell must be positive.")
    source_records = list(records)
    previous_records = [dict(record) for record in previous_test_records]
    requested_room_counts, strata, cells, filter_diagnostics = _prepare_candidates(
        source_records,
        fail_edges,
        room_counts=room_counts,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        bin_width=bin_width,
        excluded_room_ids=excluded_room_ids,
    )
    all_cell_keys = [
        (room_count, int(stratum["index"]))
        for room_count in requested_room_counts
        for stratum in strata
    ]
    incomplete_cells = [
        {
            "visited_room_count": room_count,
            "ratio_stratum": strata[index]["label"],
            "candidate_count": len(cells.get((room_count, index), [])),
            "shortfall": per_cell - len(cells.get((room_count, index), [])),
        }
        for room_count, index in all_cell_keys
        if len(cells.get((room_count, index), [])) < per_cell
    ]
    if incomplete_cells and not allow_sparse_cells:
        details = ", ".join(
            f"rooms_{item['visited_room_count']}/{item['ratio_stratum']}="
            f"{item['candidate_count']}"
            for item in incomplete_cells
        )
        raise ValueError(
            f"Full grid requires at least {per_cell} candidates per cell; "
            f"insufficient cells: {details}."
        )

    eligible_by_path_id = {
        str(record["path_id"]): record
        for candidates in cells.values()
        for record in candidates
    }
    source_by_path_id = {str(record["path_id"]): record for record in source_records}
    if len(eligible_by_path_id) != sum(len(candidates) for candidates in cells.values()):
        raise ValueError("Eligible candidate pool contains duplicate path IDs.")
    if len(source_by_path_id) != len(source_records):
        raise ValueError("Source candidate corpus contains duplicate path IDs.")

    previous_test_ids = [str(record.get("test_id")) for record in previous_records]
    previous_path_ids = [str(record.get("path_id")) for record in previous_records]
    if len(previous_test_ids) != len(set(previous_test_ids)):
        raise ValueError("Preserved test IDs are not unique.")
    if len(previous_path_ids) != len(set(previous_path_ids)):
        raise ValueError("Preserved path IDs are not unique.")

    preserved_cell_counts: Counter[tuple[int, int]] = Counter()
    for previous in previous_records:
        path_id = str(previous["path_id"])
        candidate = eligible_by_path_id.get(path_id)
        if candidate is None:
            raise ValueError(
                f"Preserved case {previous.get('test_id')} / {path_id} is no longer eligible."
            )
        _validate_preserved_record(previous, source_by_path_id[path_id])
        if excluded_room_hits(previous, excluded_room_ids):
            raise ValueError(f"Preserved case {previous.get('test_id')} references an excluded room.")
        cell = _grid_cell_key(
            previous,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            bin_width=bin_width,
        )
        expected_stratum = strata[cell[1]]["label"]
        if previous.get("ratio_stratum") != expected_stratum:
            raise ValueError(f"Preserved case {previous.get('test_id')} has the wrong ratio stratum.")
        preserved_cell_counts[cell] += 1

    target_count_by_cell = {
        cell: min(per_cell, len(cells.get(cell, [])))
        for cell in all_cell_keys
    }
    for cell, count in preserved_cell_counts.items():
        if cell not in target_count_by_cell or count > target_count_by_cell[cell]:
            raise ValueError(f"Preserved grid overfills cell {cell}: {count}.")

    rng = random.Random(seed)
    priority: dict[str, int] = {}
    for candidates in cells.values():
        candidates.sort(key=lambda record: str(record["path_id"]))
        rng.shuffle(candidates)
        for record in candidates:
            priority[str(record["path_id"])] = len(priority)

    target_counts = Counter(str(record["target_group_id"]) for record in previous_records)
    start_room_counts = Counter(str(record["start_room_id"]) for record in previous_records)
    start_pano_counts = Counter(str(record["start_pano_id"]) for record in previous_records)
    pair_counts = Counter(_start_target_pair_key(record) for record in previous_records)
    sequence_counts = Counter(_ordered_room_sequence_key(record) for record in previous_records)
    used_path_ids = set(previous_path_ids)
    selected: list[dict] = []
    cells_to_fill = [
        cell
        for cell in all_cell_keys
        if preserved_cell_counts[cell] < target_count_by_cell[cell]
    ]
    cell_order = sorted(
        cells_to_fill,
        key=lambda cell: (len(cells.get(cell, [])), cell[0], cell[1]),
    )
    for cell in cell_order:
        candidates = [
            record
            for record in cells.get(cell, [])
            if str(record["path_id"]) not in used_path_ids
        ]
        required = target_count_by_cell[cell] - preserved_cell_counts[cell]
        if len(candidates) < required:
            raise ValueError(f"Grid cell {cell} has {len(candidates)} candidates for {required} slots.")
        for _ in range(required):
            chosen = min(
                candidates,
                key=lambda record: (
                    start_pano_counts[str(record["start_pano_id"])],
                    pair_counts[_start_target_pair_key(record)],
                    sequence_counts[_ordered_room_sequence_key(record)],
                    target_counts[str(record["target_group_id"])],
                    start_room_counts[str(record["start_room_id"])],
                    priority[str(record["path_id"])],
                ),
            )
            candidates.remove(chosen)
            record = dict(chosen)
            record["sampling_cell_candidate_count"] = len(cells[cell])
            record["sampling_cell_selected_count"] = target_count_by_cell[cell]
            record["sampling_cell_sparse"] = len(cells[cell]) < per_cell
            selected.append(record)
            used_path_ids.add(str(record["path_id"]))
            target_counts[str(record["target_group_id"])] += 1
            start_room_counts[str(record["start_room_id"])] += 1
            start_pano_counts[str(record["start_pano_id"])] += 1
            pair_counts[_start_target_pair_key(record)] += 1
            sequence_counts[_ordered_room_sequence_key(record)] += 1

    selected.sort(
        key=lambda record: (
            int(record["visited_room_count"]),
            float(record["ratio_stratum_lower_bound"]),
            str(record["path_id"]),
        )
    )
    next_test_number, test_id_width = _next_test_id_state(previous_records, test_id_prefix)
    extension_records: list[dict] = []
    for offset, source in enumerate(selected):
        record = dict(source)
        group_id = str(record["target_group_id"])
        query = queries_by_target.get(group_id)
        if not isinstance(query, dict):
            raise ValueError(f"Missing query mapping for target group {group_id}.")
        _validate_query_mapping(record, query)
        record["test_id"] = f"{test_id_prefix}{next_test_number + offset:0{test_id_width}d}"
        record["query_id"] = query["query_id"]
        record["query"] = query["query"]
        extension_records.append(record)

    combined_records = previous_records + extension_records
    expected_actual_count = sum(target_count_by_cell.values())
    if len(combined_records) != expected_actual_count:
        raise ValueError(f"Expected {expected_actual_count} combined records, got {len(combined_records)}.")
    if len({str(record["test_id"]) for record in combined_records}) != len(combined_records):
        raise ValueError("Combined grid contains duplicate test IDs.")
    if len({str(record["path_id"]) for record in combined_records}) != len(combined_records):
        raise ValueError("Combined grid contains duplicate path IDs.")

    for record in combined_records:
        cell = _grid_cell_key(
            record,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            bin_width=bin_width,
        )
        candidate_count = len(cells.get(cell, []))
        record["sampling_cell_candidate_count"] = candidate_count
        record["sampling_cell_selected_count"] = target_count_by_cell[cell]
        record["sampling_cell_sparse"] = candidate_count < per_cell

    selected_cell_counts = Counter(
        _grid_cell_key(
            record,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            bin_width=bin_width,
        )
        for record in combined_records
    )
    candidate_profiles = Counter(
        (cell[0], cell[1], str(record["passage_profile"]))
        for cell, candidates in cells.items()
        for record in candidates
    )
    selected_profiles = Counter(
        (
            int(record["visited_room_count"]),
            _grid_cell_key(
                record,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                bin_width=bin_width,
            )[1],
            str(record["passage_profile"]),
        )
        for record in combined_records
    )
    grid_cells: list[dict] = []
    for room_count, index in all_cell_keys:
        candidate_count = len(cells.get((room_count, index), []))
        selected_count = selected_cell_counts[(room_count, index)]
        grid_cells.append(
            {
                "visited_room_count": room_count,
                "ratio_stratum": strata[index]["label"],
                "ratio_stratum_lower_bound": strata[index]["lower_bound"],
                "ratio_stratum_upper_bound": strata[index]["upper_bound"],
                "candidate_count": candidate_count,
                "selected_count": selected_count,
                "preserved_count": preserved_cell_counts[(room_count, index)],
                "added_count": selected_count - preserved_cell_counts[(room_count, index)],
                "shortfall": per_cell - selected_count,
                "sparse": 0 < candidate_count < per_cell,
                "empty": candidate_count == 0,
                "candidate_reliable_count": candidate_profiles[(room_count, index, "reliable")],
                "candidate_risk_count": candidate_profiles[(room_count, index, "risk")],
                "selected_reliable_count": selected_profiles[(room_count, index, "reliable")],
                "selected_risk_count": selected_profiles[(room_count, index, "risk")],
            }
        )

    diagnostics = {
        "eligible_candidate_count": sum(len(candidates) for candidates in cells.values()),
        **filter_diagnostics,
        "room_count_values": list(requested_room_counts),
        "requested_visited_room_counts": list(requested_room_counts),
        "ratio_strata": [dict(stratum) for stratum in strata],
        "grid_cell_count": len(grid_cells),
        "nonempty_grid_cell_count": sum(not cell["empty"] for cell in grid_cells),
        "empty_grid_cell_count": sum(cell["empty"] for cell in grid_cells),
        "full_grid_cell_count": sum(cell["candidate_count"] >= per_cell for cell in grid_cells),
        "sparse_grid_cell_count": sum(cell["sparse"] for cell in grid_cells),
        "requested_sample_count": len(grid_cells) * per_cell,
        "actual_sample_count": len(combined_records),
        "sample_shortfall_count": len(grid_cells) * per_cell - len(combined_records),
        "preserved_count": len(previous_records),
        "extension_count": len(extension_records),
        "allow_sparse_cells": bool(allow_sparse_cells),
        "incomplete_cells": incomplete_cells,
        "grid_cells": grid_cells,
        "selection_strategy": (
            "Preserve prior cases; fill rare cells first; within each cell greedily "
            "minimize reuse of start panorama, start-room/target pair, ordered room "
            "sequence, target group, and start room; seeded path order breaks ties."
        ),
    }
    manifest = {
        "schema_version": 1,
        "preserved_count": len(previous_records),
        "added_count": len(extension_records),
        "combined_count": len(combined_records),
        "preserved_test_ids": [str(record["test_id"]) for record in previous_records],
        "added_test_ids": [str(record["test_id"]) for record in extension_records],
        "added_records": [
            {
                "test_id": record["test_id"],
                "path_id": record["path_id"],
                "sampling_cell_id": record["sampling_cell_id"],
                "visited_room_count": record["visited_room_count"],
                "ratio_stratum": record["ratio_stratum"],
                "detour_ratio": record["detour_ratio"],
            }
            for record in extension_records
        ],
    }
    rerun = {
        "schema_version": 1,
        "count": len(extension_records),
        "test_ids": [str(record["test_id"]) for record in extension_records],
    }
    return combined_records, extension_records, diagnostics, manifest, rerun


def sampling_summary_notes(diagnostics: dict, *, per_cell: int) -> tuple[str, str]:
    undersupplied = [
        cell
        for cell in diagnostics["grid_cells"]
        if int(cell["selected_count"]) < per_cell
    ]
    selection_bias_note = (
        "This is a cell-balanced diagnostic set with a per-cell cap, not a random "
        "sample of the candidate-path distribution."
    )
    if not undersupplied:
        return (
            selection_bias_note + " Every requested cell reaches the cap.",
            f"Report each cell denominator. Every requested cell contains {per_cell} cases.",
        )

    cell_details = ", ".join(
        f"rooms_{cell['visited_room_count']}/{cell['ratio_stratum']}="
        f"{cell['selected_count']}/{per_cell}"
        for cell in undersupplied
    )
    full_cell_count = int(diagnostics["grid_cell_count"]) - len(undersupplied)
    return (
        selection_bias_note
        + f" {len(undersupplied)} naturally undersupplied cells contribute all eligible "
        "unique paths.",
        f"Report each cell denominator. Undersupplied cells: {cell_details}. "
        f"The other {full_cell_count} requested cells contain {per_cell} cases each.",
    )


def build_summary(
    records: list[dict],
    extension_records: list[dict],
    diagnostics: dict,
    *,
    seed: int,
    lower_bound: float,
    upper_bound: float,
    bin_width: float,
    per_cell: int,
    source_path: Path,
    preserve_path: Path | None,
    queries_path: Path,
    fail_edges_path: Path,
    fail_edge_count: int,
) -> dict:
    room_counts = Counter(int(record["visited_room_count"]) for record in records)
    ratio_counts = Counter(str(record["ratio_stratum"]) for record in records)
    target_counts = Counter(str(record["target_group_id"]) for record in records)
    start_room_counts = Counter(str(record["start_room_id"]) for record in records)
    profile_counts = Counter(str(record["passage_profile"]) for record in records)
    start_pano_counts = Counter(str(record["start_pano_id"]) for record in records)
    pair_counts = Counter(_start_target_pair_key(record) for record in records)
    sequence_counts = Counter(_ordered_room_sequence_key(record) for record in records)
    fail_edge_counts: Counter[str] = Counter()
    for record in records:
        for edge in record.get("known_failed_passage_edges_on_path", []):
            fail_edge_counts[f"{edge['from_room_id']} -> {edge['to_room_id']}"] += 1
    selection_bias_note, reporting_scope = sampling_summary_notes(
        diagnostics,
        per_cell=per_cell,
    )
    return {
        "schema": "llm_nav_navigation_grid_sample_summary",
        "schema_version": GENERATOR_SCHEMA_VERSION,
        "generator": dict(GENERATOR_PROVENANCE),
        "sampling_design": "partial_preserved_visited_room_count_x_fixed_width_detour_ratio_grid",
        "selection_bias_note": selection_bias_note,
        "reporting_scope": reporting_scope,
        "seed": seed,
        "source_navigation_paths": str(source_path),
        "source_navigation_paths_sha256": sha256_file(source_path),
        "source_navigation_path_count": len(load_jsonl(source_path)),
        "preserved_test_set": str(preserve_path) if preserve_path else None,
        "preserved_test_set_sha256": sha256_file(preserve_path) if preserve_path else None,
        "queries_by_target": str(queries_path),
        "queries_by_target_sha256": sha256_file(queries_path),
        "known_fail_edges_annotation_path": str(fail_edges_path),
        "known_fail_edges_sha256": sha256_file(fail_edges_path),
        "known_directed_fail_edge_count": fail_edge_count,
        "sample_count": len(records),
        "extension_sample_count": len(extension_records),
        "ratio_range": {
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "lower_inclusive": True,
            "upper_inclusive": False,
        },
        "ratio_bin_width": bin_width,
        "per_grid_cell_cap": per_cell,
        **diagnostics,
        "visited_room_count_sample_counts": {
            str(room_count): room_counts[room_count]
            for room_count in diagnostics["room_count_values"]
        },
        "ratio_stratum_sample_counts": {
            stratum["label"]: ratio_counts[stratum["label"]]
            for stratum in diagnostics["ratio_strata"]
        },
        "passage_profile_counts": {
            "reliable": profile_counts["reliable"],
            "risk": profile_counts["risk"],
        },
        "selected_fail_edge_exposure_counts": dict(sorted(fail_edge_counts.items())),
        "uniqueness": {
            "test_id_count": len({str(record["test_id"]) for record in records}),
            "path_id_count": len({str(record["path_id"]) for record in records}),
            "start_panorama_count": len(start_pano_counts),
            "start_panorama_max_reuse": max(start_pano_counts.values(), default=0),
            "start_room_target_group_pair_count": len(pair_counts),
            "start_room_target_group_pair_max_reuse": max(pair_counts.values(), default=0),
            "ordered_room_sequence_count": len(sequence_counts),
            "ordered_room_sequence_max_reuse": max(sequence_counts.values(), default=0),
        },
        "target_group_count": len(target_counts),
        "target_group_sample_counts": dict(sorted(target_counts.items())),
        "start_room_count": len(start_room_counts),
        "start_room_sample_counts": dict(sorted(start_room_counts.items())),
        "selected_path_ids": [str(record["path_id"]) for record in records],
    }


def write_sampling_grid_csv(path: Path, grid_cells: Iterable[dict]) -> None:
    fields = (
        "visited_room_count",
        "ratio_stratum",
        "ratio_stratum_lower_bound",
        "ratio_stratum_upper_bound",
        "candidate_count",
        "selected_count",
        "preserved_count",
        "added_count",
        "shortfall",
        "sparse",
        "empty",
        "candidate_reliable_count",
        "candidate_risk_count",
        "selected_reliable_count",
        "selected_risk_count",
    )
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for cell in grid_cells:
        writer.writerow({field: cell.get(field) for field in fields})
    _atomic_write_text(path, stream.getvalue())


def validate_grid_records(
    records: Iterable[dict],
    *,
    excluded_room_ids: Iterable[str],
    lower_bound: float,
    upper_bound: float,
    bin_width: float,
) -> None:
    for record in records:
        if record.get("path_constraint_mode") != "room_topology_bfs_constrained_dijkstra":
            raise ValueError(f"{record.get('test_id')} is not topology-constrained.")
        if record.get("topology_validation_status") != "valid":
            raise ValueError(f"{record.get('test_id')} is not topology-valid.")
        if excluded_room_hits(record, excluded_room_ids):
            raise ValueError(f"{record.get('test_id')} references an excluded room.")
        stratum = ratio_grid_stratum_for_value(
            float(record["detour_ratio"]),
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            bin_width=bin_width,
        )
        if stratum is None or record.get("ratio_stratum") != stratum["label"]:
            raise ValueError(f"{record.get('test_id')} has an invalid ratio stratum.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build or extend a deterministic visited-room-count by fixed-width "
            "detour-ratio navigation test grid."
        )
    )
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--preserve-test-set", default=DEFAULT_PRESERVE_TEST_SET)
    parser.add_argument("--queries-by-target", default=DEFAULT_QUERIES_BY_TARGET)
    parser.add_argument("--fail-edges", default=DEFAULT_FAIL_EDGES)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--visited-room-counts", type=int, nargs="+", default=list(DEFAULT_VISITED_ROOM_COUNTS))
    parser.add_argument("--ratio-lower", type=float, default=1.0)
    parser.add_argument("--ratio-upper", type=float, default=2.0)
    parser.add_argument("--ratio-bin-width", type=float, default=0.2)
    parser.add_argument("--per-grid-cell", type=int, default=DEFAULT_PER_GRID_CELL)
    parser.add_argument("--exclude-room-ids", nargs="*", default=list(DEFAULT_EXCLUDED_ROOM_IDS))
    parser.add_argument("--test-id-prefix", default="TEST")
    parser.add_argument(
        "--allow-sparse-cells",
        action="store_true",
        help="Allow a cell to contribute every eligible path when fewer than the cap exist.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    input_path = resolve_project_path(args.input)
    preserve_path = resolve_project_path(args.preserve_test_set) if args.preserve_test_set else None
    queries_path = resolve_project_path(args.queries_by_target)
    fail_edges_path = resolve_project_path(args.fail_edges)
    output_dir = resolve_project_path(args.output_dir)
    source_records = load_jsonl(input_path)
    previous_records = load_jsonl(preserve_path) if preserve_path else []
    queries_by_target = load_json_object(queries_path)
    fail_edges = load_directed_fail_edges(fail_edges_path)
    combined, extension, diagnostics, manifest, rerun = build_partial_grid(
        source_records,
        previous_records,
        fail_edges,
        queries_by_target,
        seed=args.seed,
        room_counts=args.visited_room_counts,
        lower_bound=args.ratio_lower,
        upper_bound=args.ratio_upper,
        bin_width=args.ratio_bin_width,
        per_cell=args.per_grid_cell,
        allow_sparse_cells=args.allow_sparse_cells,
        excluded_room_ids=args.exclude_room_ids,
        test_id_prefix=args.test_id_prefix,
    )
    validate_grid_records(
        combined,
        excluded_room_ids=args.exclude_room_ids,
        lower_bound=args.ratio_lower,
        upper_bound=args.ratio_upper,
        bin_width=args.ratio_bin_width,
    )
    summary = build_summary(
        combined,
        extension,
        diagnostics,
        seed=args.seed,
        lower_bound=args.ratio_lower,
        upper_bound=args.ratio_upper,
        bin_width=args.ratio_bin_width,
        per_cell=args.per_grid_cell,
        source_path=input_path,
        preserve_path=preserve_path,
        queries_path=queries_path,
        fail_edges_path=fail_edges_path,
        fail_edge_count=len(fail_edges),
    )
    manifest.update(
        {
            "generator": dict(GENERATOR_PROVENANCE),
            "source_navigation_paths_sha256": summary["source_navigation_paths_sha256"],
            "preserved_test_set_sha256": summary["preserved_test_set_sha256"],
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "test_set.jsonl", combined)
    write_csv(output_dir / "test_set.csv", combined)
    write_jsonl(output_dir / "extension_test_set.jsonl", extension)
    write_csv(output_dir / "extension_test_set.csv", extension)
    write_json(output_dir / "queries_by_target.json", queries_by_target)
    write_json(output_dir / "sample_summary.json", summary)
    write_sampling_grid_csv(output_dir / "sampling_grid.csv", diagnostics["grid_cells"])
    write_json(output_dir / "extension_manifest.json", manifest)
    write_json(output_dir / "rerun_test_ids.json", rerun)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "sample_count": len(combined),
                "preserved_count": len(previous_records),
                "extension_count": len(extension),
                "grid_cell_count": diagnostics["grid_cell_count"],
                "sparse_grid_cell_count": diagnostics["sparse_grid_cell_count"],
                "sample_shortfall_count": diagnostics["sample_shortfall_count"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
