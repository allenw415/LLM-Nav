from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from st_nav.spatial.routing import RoutePlanner  # noqa: E402
from tools.data.plot_thesis_candidate_heatmaps import (  # noqa: E402
    CandidateMetric,
    _atomic_write_json,
    _atomic_write_matrix_csv,
    _write_heatmap,
    build_heatmap_data,
    sha256_file,
)


DEFAULT_ARTIFACTS_DIR = "dataset/sites/british_museum/normalized"
DEFAULT_START_SOURCE = (
    "outputs/navigation_paths/floor0_semantic_targets_topology_constrained/"
    "thesis_24909/candidate_tasks_24909.psv"
)
DEFAULT_OUTPUT_DIR = (
    "outputs/navigation_paths/floor0_to_room33_cross_floor/"
    "detour_visited_room_heatmaps"
)
DEFAULT_EXPECTED_START_COUNT = 1_083
DEFAULT_START_FLOOR = "0"
DEFAULT_TARGET_FLOOR = "1"
DEFAULT_TARGET_ROOM_ID = "Room 33"
DEFAULT_BIN_WIDTH = 0.2

TASK_FIELDS = (
    "task_id",
    "panorama_id_start",
    "panorama_id_end",
    "room_start",
    "room_end",
    "start_floor",
    "target_floor",
    "visited_room_count",
    "topology_room_node_count",
    "circulation_node_count",
    "detour_ratio",
    "straight_line_distance_m",
    "shortest_path_distance_m",
    "pano_step_count",
    "cross_floor_transition_count",
    "shortest_path_room_sequence",
    "shortest_path_panorama_id_sequence",
)


def resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate.resolve() if candidate.is_absolute() else (PROJECT_ROOT / candidate).resolve()


def load_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def normalized_room_id(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    if not cleaned or cleaned.lower() == "null":
        return None
    return cleaned


def load_pano_room_mappings(path: Path) -> dict[str, str]:
    payload = load_json(path)
    if not isinstance(payload, dict) or not isinstance(payload.get("mappings"), dict):
        raise ValueError(f"Expected a mappings object in {path}.")
    return {
        str(pano_id): room_id
        for pano_id, value in payload["mappings"].items()
        if (room_id := normalized_room_id(value)) is not None
    }


def load_canonical_starts(path: Path) -> dict[str, str]:
    delimiter = "|" if path.suffix.lower() == ".psv" else ","
    starts: dict[str, str] = {}
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter=delimiter)
        required = {"panorama_id_start", "room_start"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} is missing fields: {sorted(missing)!r}.")
        for line_number, row in enumerate(reader, start=2):
            pano_id = str(row.get("panorama_id_start") or "").strip()
            room_id = str(row.get("room_start") or "").strip()
            if not pano_id or not room_id:
                raise ValueError(f"{path}:{line_number} has an empty start panorama or room.")
            previous = starts.setdefault(pano_id, room_id)
            if previous != room_id:
                raise ValueError(
                    f"{path}:{line_number} maps start panorama {pano_id!r} "
                    f"to both {previous!r} and {room_id!r}."
                )
    if not starts:
        raise ValueError(f"{path} contains no candidate starts.")
    return dict(sorted(starts.items()))


def select_floor_graph(
    pano_graph: dict[str, dict],
    *,
    floors: set[str],
) -> dict[str, dict]:
    selected_ids = {
        pano_id
        for pano_id, record in pano_graph.items()
        if isinstance(record, dict) and str(record.get("floor")) in floors
    }
    selected: dict[str, dict] = {}
    for pano_id in sorted(selected_ids):
        record = dict(pano_graph[pano_id])
        record["neighbors"] = [
            dict(edge)
            for edge in record.get("neighbors", [])
            if isinstance(edge, dict) and edge.get("target_pano_id") in selected_ids
        ]
        selected[pano_id] = record
    return selected


def lat_lng(record: dict | None) -> tuple[float, float] | None:
    if not isinstance(record, dict):
        return None
    lat = record.get("lat")
    lng = record.get("lng")
    if not isinstance(lat, (int, float)) or not isinstance(lng, (int, float)):
        return None
    return float(lat), float(lng)


def haversine_m(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    earth_radius_m = 6_371_000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lng2 - lng1)
    a = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    )
    return 2.0 * earth_radius_m * math.asin(math.sqrt(a))


def distance_between_panos_m(
    pano_graph: dict[str, dict],
    source_pano_id: str,
    target_pano_id: str,
) -> float | None:
    source = lat_lng(pano_graph.get(source_pano_id))
    target = lat_lng(pano_graph.get(target_pano_id))
    if source is None or target is None:
        return None
    return haversine_m(source[0], source[1], target[0], target[1])


def pano_edge_distances(
    pano_graph: dict[str, dict],
) -> dict[tuple[str, str], float]:
    distances: dict[tuple[str, str], float] = {}
    for source_pano_id, node in pano_graph.items():
        for edge in node.get("neighbors", []):
            target_pano_id = edge.get("target_pano_id")
            if not isinstance(target_pano_id, str) or target_pano_id not in pano_graph:
                continue
            distance = distance_between_panos_m(
                pano_graph,
                source_pano_id,
                target_pano_id,
            )
            if distance is not None:
                distances[(source_pano_id, target_pano_id)] = distance
    return distances


ConstrainedState = tuple[str, int]


def constrained_shortest_pano_path(
    pano_graph: dict[str, dict],
    pano_room_mappings: dict[str, str],
    *,
    start_pano_id: str,
    target_pano_ids: Iterable[str],
    topology_room_route: Sequence[str],
    edge_distances: dict[tuple[str, str], float],
) -> tuple[float, str, list[str]] | None:
    if not topology_room_route or start_pano_id not in pano_graph:
        return None
    if pano_room_mappings.get(start_pano_id) != topology_room_route[0]:
        return None
    target_ids = {pano_id for pano_id in target_pano_ids if pano_id in pano_graph}
    if not target_ids:
        return None

    start_state: ConstrainedState = (start_pano_id, 0)
    distances: dict[ConstrainedState, float] = {start_state: 0.0}
    parents: dict[ConstrainedState, ConstrainedState] = {}
    queue: list[tuple[float, str, int]] = [(0.0, start_pano_id, 0)]
    goal_state: ConstrainedState | None = None

    while queue:
        current_distance, pano_id, route_index = heapq.heappop(queue)
        state = (pano_id, route_index)
        if current_distance != distances.get(state):
            continue
        if route_index == len(topology_room_route) - 1 and pano_id in target_ids:
            goal_state = state
            break
        for edge in pano_graph[pano_id].get("neighbors", []):
            target_pano_id = edge.get("target_pano_id")
            if not isinstance(target_pano_id, str) or target_pano_id not in pano_graph:
                continue
            target_room_id = pano_room_mappings.get(target_pano_id)
            next_route_index = route_index
            if target_room_id is not None:
                if target_room_id == topology_room_route[route_index]:
                    pass
                elif (
                    route_index + 1 < len(topology_room_route)
                    and target_room_id == topology_room_route[route_index + 1]
                ):
                    next_route_index += 1
                else:
                    continue
            edge_distance = edge_distances.get((pano_id, target_pano_id))
            if edge_distance is None:
                continue
            next_state = (target_pano_id, next_route_index)
            candidate_distance = current_distance + edge_distance
            if candidate_distance < distances.get(next_state, math.inf):
                distances[next_state] = candidate_distance
                parents[next_state] = state
                heapq.heappush(
                    queue,
                    (candidate_distance, target_pano_id, next_route_index),
                )

    if goal_state is None:
        return None
    path = [goal_state[0]]
    cursor = goal_state
    while cursor != start_state:
        cursor = parents[cursor]
        path.append(cursor[0])
    path.reverse()
    return distances[goal_state], goal_state[0], path


def collapsed_room_sequence(
    path_panos: Iterable[str],
    pano_room_mappings: dict[str, str],
) -> list[str]:
    sequence: list[str] = []
    for pano_id in path_panos:
        room_id = pano_room_mappings.get(pano_id)
        if room_id is not None and (not sequence or sequence[-1] != room_id):
            sequence.append(room_id)
    return sequence


def _atomic_write_tasks(path: Path, records: Sequence[dict]) -> None:
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
                fieldnames=TASK_FIELDS,
                delimiter="|",
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(records)
            temporary_path = Path(stream.name)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _atomic_write_global_percentage_matrix(
    path: Path,
    *,
    data,
    percentages: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    fieldnames = [
        "detour_ratio_bin",
        *(str(value) for value in data.room_counts),
        "row_total",
    ]
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
            writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            for row_index, label in enumerate(data.detour_ratio_labels):
                row: dict[str, object] = {"detour_ratio_bin": label}
                for column, room_count in enumerate(data.room_counts):
                    row[str(room_count)] = f"{float(percentages[row_index, column]):.10g}"
                row["row_total"] = f"{float(percentages[row_index, :].sum()):.10g}"
                writer.writerow(row)
            total_row: dict[str, object] = {"detour_ratio_bin": "TOTAL"}
            for column, room_count in enumerate(data.room_counts):
                total_row[str(room_count)] = f"{float(percentages[:, column].sum()):.10g}"
            total_row["row_total"] = f"{float(percentages.sum()):.10g}"
            writer.writerow(total_row)
            temporary_path = Path(stream.name)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def percentile_summary(values: Sequence[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    p50, p90, p95, p99 = np.percentile(array, [50, 90, 95, 99])
    return {
        "count": int(array.size),
        "minimum": float(np.min(array)),
        "median": float(p50),
        "p90": float(p90),
        "p95": float(p95),
        "p99": float(p99),
        "maximum": float(np.max(array)),
    }


def build_cross_floor_tasks(
    *,
    artifacts_dir: Path,
    start_source: Path,
    expected_start_count: int,
    start_floor: str,
    target_floor: str,
    target_room_id: str,
) -> tuple[list[dict], dict]:
    full_pano_graph = load_json(artifacts_dir / "pano_graph.json")
    room_graph = load_json(artifacts_dir / "room_graph.json")
    if not isinstance(full_pano_graph, dict) or not isinstance(room_graph, dict):
        raise ValueError("Normalized panorama and room graphs must be JSON objects.")
    pano_room_mappings = load_pano_room_mappings(
        artifacts_dir / "pano_room_grounding.json"
    )
    canonical_starts = load_canonical_starts(start_source)
    if len(canonical_starts) != expected_start_count:
        raise ValueError(
            f"Expected {expected_start_count} canonical starts, "
            f"found {len(canonical_starts)}."
        )

    pano_graph = select_floor_graph(
        full_pano_graph,
        floors={start_floor, target_floor},
    )
    target_pano_ids = sorted(
        pano_id
        for pano_id, room_id in pano_room_mappings.items()
        if room_id == target_room_id
        and pano_id in pano_graph
        and str(pano_graph[pano_id].get("floor")) == target_floor
    )
    if not target_pano_ids:
        raise ValueError(
            f"No target panoramas found for {target_room_id!r} on floor {target_floor!r}."
        )

    for pano_id, source_room_id in canonical_starts.items():
        if pano_id not in pano_graph:
            raise ValueError(f"Canonical start panorama is absent from the floor graph: {pano_id}")
        if str(pano_graph[pano_id].get("floor")) != start_floor:
            raise ValueError(f"Canonical start panorama is not on floor {start_floor}: {pano_id}")
        grounded_room_id = pano_room_mappings.get(pano_id)
        if grounded_room_id != source_room_id:
            raise ValueError(
                f"Canonical start room mismatch for {pano_id}: "
                f"source={source_room_id!r}, grounding={grounded_room_id!r}."
            )

    route_planner = RoutePlanner(room_graph=room_graph)
    edge_distances = pano_edge_distances(pano_graph)
    records: list[dict] = []
    endpoint_counts: Counter[str] = Counter()
    route_counts: Counter[str] = Counter()

    for index, (start_pano_id, start_room_id) in enumerate(
        canonical_starts.items(),
        start=1,
    ):
        topology_room_route = route_planner.shortest_room_path(
            start_room_id,
            target_room_id,
        )
        if not topology_room_route:
            raise ValueError(
                f"No room-topology route from {start_room_id!r} to {target_room_id!r}."
            )
        result = constrained_shortest_pano_path(
            pano_graph,
            pano_room_mappings,
            start_pano_id=start_pano_id,
            target_pano_ids=target_pano_ids,
            topology_room_route=topology_room_route,
            edge_distances=edge_distances,
        )
        if result is None:
            raise ValueError(
                f"No topology-constrained panorama path from {start_pano_id} "
                f"to {target_room_id}."
            )
        path_distance_m, end_pano_id, shortest_path_panos = result
        collapsed_rooms = collapsed_room_sequence(
            shortest_path_panos,
            pano_room_mappings,
        )
        if collapsed_rooms != topology_room_route:
            raise ValueError(
                f"Mapped panorama route {collapsed_rooms!r} does not match "
                f"the room route {topology_room_route!r}."
            )
        straight_line_distance_m = distance_between_panos_m(
            pano_graph,
            start_pano_id,
            end_pano_id,
        )
        if straight_line_distance_m is None or straight_line_distance_m <= 0.0:
            raise ValueError(
                f"Invalid straight-line distance from {start_pano_id} to {end_pano_id}."
            )
        cross_floor_transition_count = sum(
            str(pano_graph[source].get("floor"))
            != str(pano_graph[target].get("floor"))
            for source, target in zip(
                shortest_path_panos[:-1],
                shortest_path_panos[1:],
                strict=True,
            )
        )
        if cross_floor_transition_count != 1:
            raise ValueError(
                f"Expected exactly one cross-floor transition for {start_pano_id}, "
                f"found {cross_floor_transition_count}."
            )
        route_label = " -> ".join(topology_room_route)
        circulation_node_count = sum(
            room_graph.get(room_id, {}).get("category") == "Circulation"
            or room_graph.get(room_id, {}).get("synthetic") is True
            for room_id in topology_room_route
        )
        visited_room_count = len(topology_room_route) - circulation_node_count
        if visited_room_count < 2:
            raise ValueError(
                f"Expected at least the start and target galleries for {start_pano_id}."
            )
        endpoint_counts[end_pano_id] += 1
        route_counts[route_label] += 1
        records.append(
            {
                "task_id": f"R33TASK{index:05d}",
                "panorama_id_start": start_pano_id,
                "panorama_id_end": end_pano_id,
                "room_start": start_room_id,
                "room_end": target_room_id,
                "start_floor": start_floor,
                "target_floor": target_floor,
                "visited_room_count": visited_room_count,
                "topology_room_node_count": len(topology_room_route),
                "circulation_node_count": circulation_node_count,
                "detour_ratio": path_distance_m / straight_line_distance_m,
                "straight_line_distance_m": straight_line_distance_m,
                "shortest_path_distance_m": path_distance_m,
                "pano_step_count": max(len(shortest_path_panos) - 1, 0),
                "cross_floor_transition_count": cross_floor_transition_count,
                "shortest_path_room_sequence": route_label,
                "shortest_path_panorama_id_sequence": " -> ".join(shortest_path_panos),
            }
        )

    metadata = {
        "canonical_start_count": len(canonical_starts),
        "canonical_start_room_count": len(set(canonical_starts.values())),
        "canonical_start_room_ids": sorted(set(canonical_starts.values())),
        "target_panorama_count": len(target_pano_ids),
        "endpoint_distribution": dict(sorted(endpoint_counts.items())),
        "room_route_distribution": dict(sorted(route_counts.items())),
        "selected_pano_node_count": len(pano_graph),
        "selected_pano_edge_count": sum(
            len(record.get("neighbors", [])) for record in pano_graph.values()
        ),
    }
    return records, metadata


def generate_cross_floor_heatmaps(
    *,
    artifacts_dir: Path,
    start_source: Path,
    output_dir: Path,
    expected_start_count: int = DEFAULT_EXPECTED_START_COUNT,
    start_floor: str = DEFAULT_START_FLOOR,
    target_floor: str = DEFAULT_TARGET_FLOOR,
    target_room_id: str = DEFAULT_TARGET_ROOM_ID,
    bin_width: float = DEFAULT_BIN_WIDTH,
) -> dict:
    records, metadata = build_cross_floor_tasks(
        artifacts_dir=artifacts_dir,
        start_source=start_source,
        expected_start_count=expected_start_count,
        start_floor=start_floor,
        target_floor=target_floor,
        target_room_id=target_room_id,
    )
    task_count = len(records)
    if task_count != expected_start_count:
        raise ValueError(
            f"Expected one cross-floor task per canonical start ({expected_start_count}), "
            f"found {task_count}."
        )
    metrics = [
        CandidateMetric(
            task_id=str(record["task_id"]),
            visited_room_count=int(record["visited_room_count"]),
            detour_ratio=float(record["detour_ratio"]),
        )
        for record in records
    ]
    heatmap_data = build_heatmap_data(metrics, bin_width=bin_width)
    if int(heatmap_data.counts.sum()) != task_count:
        raise ValueError("Heatmap cells do not add up to the cross-floor task count.")
    global_percentages = 100.0 * heatmap_data.counts / task_count
    if not math.isclose(float(global_percentages.sum()), 100.0):
        raise ValueError("Global percentage heatmap cells do not add up to 100%.")

    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"room33_{task_count}"
    tasks_path = output_dir / f"candidate_tasks_{suffix}.psv"
    count_heatmap_path = output_dir / (
        f"detour_ratio_by_visited_room_count_count_heatmap_{suffix}.png"
    )
    percentage_heatmap_path = output_dir / (
        f"detour_ratio_by_visited_room_count_percentage_heatmap_{suffix}.png"
    )
    count_matrix_path = output_dir / (
        f"detour_ratio_by_visited_room_count_count_matrix_{suffix}.csv"
    )
    percentage_matrix_path = output_dir / (
        f"detour_ratio_by_visited_room_count_percentage_matrix_{suffix}.csv"
    )
    summary_path = output_dir / f"detour_visited_room_summary_{suffix}.json"

    _atomic_write_tasks(tasks_path, records)
    _write_heatmap(
        count_heatmap_path,
        data=heatmap_data,
        matrix=heatmap_data.counts,
        title=(
            "Floor 0 Starts to Room 33: Task Count by Visited Room Count "
            f"and Detour Ratio ({task_count:,} Tasks)"
        ),
        colorbar_label="Task Count",
        percentage=False,
    )
    _write_heatmap(
        percentage_heatmap_path,
        data=heatmap_data,
        matrix=global_percentages,
        title=(
            "Floor 0 Starts to Room 33: Share of All Tasks by Visited "
            f"Room Count and Detour Ratio ({task_count:,} Tasks)"
        ),
        colorbar_label="Share of All Tasks (%)",
        percentage=True,
    )
    _atomic_write_matrix_csv(
        count_matrix_path,
        data=heatmap_data,
        matrix=heatmap_data.counts,
        percentage=False,
    )
    _atomic_write_global_percentage_matrix(
        percentage_matrix_path,
        data=heatmap_data,
        percentages=global_percentages,
    )

    ratio_values = [float(record["detour_ratio"]) for record in records]
    room_count_distribution = Counter(
        int(record["visited_room_count"]) for record in records
    )
    output_paths = (
        tasks_path,
        count_heatmap_path,
        percentage_heatmap_path,
        count_matrix_path,
        percentage_matrix_path,
    )
    summary = {
        "schema": "llm_nav_room33_cross_floor_detour_visited_room_heatmaps",
        "schema_version": 1,
        "scope": {
            "start_floor": start_floor,
            "target_floor": target_floor,
            "target_room_id": target_room_id,
            "task_count": task_count,
            "start_panorama_count": metadata["canonical_start_count"],
            "start_room_count": metadata["canonical_start_room_count"],
            "start_room_ids": metadata["canonical_start_room_ids"],
            "selection_rule": (
                "Reuse the 1,083 distinct floor-0 start panoramas from the "
                "thesis-defined 24,909-task candidate corpus and create one "
                "topology-constrained task to Room 33 for each start."
            ),
        },
        "source": {
            "canonical_start_tasks": str(start_source),
            "canonical_start_tasks_sha256": sha256_file(start_source),
            "pano_graph": str(artifacts_dir / "pano_graph.json"),
            "pano_graph_sha256": sha256_file(artifacts_dir / "pano_graph.json"),
            "pano_room_grounding": str(artifacts_dir / "pano_room_grounding.json"),
            "pano_room_grounding_sha256": sha256_file(
                artifacts_dir / "pano_room_grounding.json"
            ),
            "room_graph": str(artifacts_dir / "room_graph.json"),
            "room_graph_sha256": sha256_file(artifacts_dir / "room_graph.json"),
        },
        "definitions": {
            "path_constraint_mode": "room_topology_bfs_constrained_dijkstra",
            "visited_room_count": (
                "Non-circulation gallery rooms on the reference route, including "
                "the start gallery and Room 33; stairs and lift nodes are excluded."
            ),
            "detour_ratio": (
                "topology-constrained panorama-path distance divided by the "
                "2D haversine distance between start and Room 33 endpoint; "
                "floor elevation is not represented in the source graph."
            ),
            "percentage_heatmap": (
                "Each cell is divided by all cross-floor tasks, so every heatmap "
                "cell sums to 100% across the full matrix."
            ),
        },
        "graph": {
            "selected_floors": [start_floor, target_floor],
            "selected_pano_node_count": metadata["selected_pano_node_count"],
            "selected_pano_edge_count": metadata["selected_pano_edge_count"],
            "target_panorama_count": metadata["target_panorama_count"],
        },
        "binning": {
            "bin_width": bin_width,
            "minimum": heatmap_data.minimum,
            "p99": heatmap_data.p99,
            "maximum": heatmap_data.maximum,
            "display_minimum": heatmap_data.display_minimum,
            "display_maximum": heatmap_data.display_maximum,
            "overflow_label": heatmap_data.detour_ratio_labels[-1],
        },
        "visited_room_counts": list(heatmap_data.room_counts),
        "task_count_by_visited_room_count": {
            str(room_count): room_count_distribution[room_count]
            for room_count in sorted(room_count_distribution)
        },
        "detour_ratio_statistics": percentile_summary(ratio_values),
        "endpoint_distribution": metadata["endpoint_distribution"],
        "room_route_distribution": metadata["room_route_distribution"],
        "validation": {
            "one_task_per_start": task_count == metadata["canonical_start_count"],
            "all_paths_have_one_cross_floor_transition": all(
                int(record["cross_floor_transition_count"]) == 1
                for record in records
            ),
            "heatmap_cell_count_total": int(heatmap_data.counts.sum()),
            "percentage_by_visited_room_count": {
                str(room_count): float(global_percentages[:, column].sum())
                for column, room_count in enumerate(heatmap_data.room_counts)
            },
            "global_percentage_sum": float(global_percentages.sum()),
            "status": "valid",
        },
        "output_files": {
            path.name: {"path": str(path), "sha256": sha256_file(path)}
            for path in output_paths
        },
    }
    _atomic_write_json(summary_path, summary)
    summary["output_files"][summary_path.name] = {
        "path": str(summary_path),
        "sha256": sha256_file(summary_path),
    }
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build floor-0 start panorama to Room 33 cross-floor tasks and "
            "generate Detour Ratio x Visited Room Count heatmaps."
        )
    )
    parser.add_argument("--artifacts-dir", default=DEFAULT_ARTIFACTS_DIR)
    parser.add_argument("--start-source", default=DEFAULT_START_SOURCE)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--expected-start-count",
        type=int,
        default=DEFAULT_EXPECTED_START_COUNT,
    )
    parser.add_argument("--start-floor", default=DEFAULT_START_FLOOR)
    parser.add_argument("--target-floor", default=DEFAULT_TARGET_FLOOR)
    parser.add_argument("--target-room-id", default=DEFAULT_TARGET_ROOM_ID)
    parser.add_argument("--bin-width", type=float, default=DEFAULT_BIN_WIDTH)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary = generate_cross_floor_heatmaps(
        artifacts_dir=resolve_project_path(args.artifacts_dir),
        start_source=resolve_project_path(args.start_source),
        output_dir=resolve_project_path(args.output_dir),
        expected_start_count=int(args.expected_start_count),
        start_floor=str(args.start_floor),
        target_floor=str(args.target_floor),
        target_room_id=str(args.target_room_id),
        bin_width=float(args.bin_width),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
