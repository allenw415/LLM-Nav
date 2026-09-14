#!/usr/bin/env python3
"""Analyze failed navigation runs from the first room-route divergence.

The analyzer does not rerun navigation.  For each failed run it reconstructs
the shortest room route from the correct start room to the parsed goal, builds
the actual ground-truth room sequence from the pano path, and attributes the
first invalid room transition using the agent state immediately before that
transition.

The important distinction is between:

* localization-call accuracy: only steps that actually invoked localization;
* effective-state accuracy: the room estimate carried into every action step.

Unlabelled transit panoramas remain in the step evidence but are never counted
as localization errors.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import shutil
import sys
import tempfile
from collections import Counter, defaultdict, deque
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.evaluation.navigation.export_batch_trajectories_for_pano_viewer import (
    build_envelope,
    build_episode,
)


DATASET_ROOT = (
    "outputs/navigation_paths/floor0_semantic_targets_topology_constrained/"
    "room_count_2to6_ratio_1p0_2p0_per30_sparse715"
)
EVALUATION_ROOT = (
    "outputs/navigation_evaluation/floor0_semantic_targets_topology_constrained/"
    "room_count_2to6_ratio_1p0_2p0_per30_sparse715"
)
DEFAULT_DATASET = f"{DATASET_ROOT}/test_set.jsonl"
DEFAULT_WITHOUT_RESULTS = (
    f"{EVALUATION_ROOT}/geometry_localization_report_combined715/"
    "effective_results.jsonl"
)
DEFAULT_WITH_RESULTS = (
    f"{EVALUATION_ROOT}/geometry_edge_visit_localization_report_combined715/"
    "effective_results.jsonl"
)
DEFAULT_OUTPUT_DIR = f"{EVALUATION_ROOT}/route_divergence_analysis"
DEFAULT_ROOM_GRAPH = "dataset/sites/british_museum/normalized/room_graph.json"
DEFAULT_PANO_GRAPH = "dataset/sites/british_museum/normalized/pano_graph.json"
DEFAULT_GROUNDING = (
    "dataset/sites/british_museum/normalized/pano_room_grounding.json"
)

CONDITIONS = ("without_trajectory", "with_trajectory")
ROOT_CAUSES = (
    "localization",
    "planning/replanning",
    "direction/exit_selection",
    "other/unclear",
)
EXPECTED_PAIRED_COUNTS = {
    "both_success": 475,
    "without_fail_with_success": 131,
    "without_success_with_fail": 4,
    "both_fail": 105,
}
EXPECTED_FAILURE_COUNTS = {
    "without_trajectory": 236,
    "with_trajectory": 109,
}


def resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return (
        candidate.resolve()
        if candidate.is_absolute()
        else (PROJECT_ROOT / candidate).resolve()
    )


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    resolved = Path(path)
    records: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(
        resolved.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw_line.strip():
            continue
        value = json.loads(raw_line)
        if not isinstance(value, dict):
            raise ValueError(f"Expected JSON object at {resolved}:{line_number}.")
        records.append(value)
    return records


def index_records(
    records: Iterable[dict[str, Any]], *, label: str
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for record in records:
        test_id = record.get("test_id")
        if not isinstance(test_id, str) or not test_id:
            raise ValueError(f"{label} contains a record without test_id.")
        if test_id in indexed:
            raise ValueError(f"{label} contains duplicate test_id {test_id}.")
        indexed[test_id] = record
    return indexed


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as stream:
        stream.write(text)
        temporary = Path(stream.name)
    os.replace(temporary, path)


def write_json(path: Path, value: Any) -> None:
    _atomic_write(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def _csv_cell(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple, set)):
        if isinstance(value, set):
            value = sorted(value)
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        _atomic_write(path, "")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for field in row:
            if field not in seen:
                seen.add(field)
                fieldnames.append(field)
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: _csv_cell(value) for field, value in row.items()})
    _atomic_write(path, stream.getvalue())


def normalize_grounding(raw_grounding: dict[str, Any]) -> dict[str, str]:
    null_sentinels = {"", "null", "none", "unknown", "unlabeled"}
    return {
        pano_id: room_id
        for pano_id, room_id in raw_grounding.items()
        if isinstance(pano_id, str)
        and isinstance(room_id, str)
        and room_id.strip().lower() not in null_sentinels
    }


def normalize_heading(value: float) -> float:
    normalized = float(value) % 360.0
    return 0.0 if math.isclose(normalized, 360.0) else normalized


def angular_distance(a: float | None, b: float | None) -> float | None:
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        return None
    difference = abs(normalize_heading(a) - normalize_heading(b))
    return min(difference, 360.0 - difference)


def success_transition(without: dict[str, Any], with_result: dict[str, Any]) -> str:
    without_success = without.get("success") is True
    with_success = with_result.get("success") is True
    if without_success and with_success:
        return "both_success"
    if not without_success and with_success:
        return "without_fail_with_success"
    if without_success and not with_success:
        return "without_success_with_fail"
    return "both_fail"


def active_goal_room_ids(result: dict[str, Any], case: dict[str, Any]) -> list[str]:
    parsed = result.get("parsed_goal_room_ids")
    if isinstance(parsed, list):
        valid = [value for value in parsed if isinstance(value, str) and value]
        if valid:
            return list(dict.fromkeys(valid))
    acceptable = case.get("acceptable_target_room_ids")
    if isinstance(acceptable, list):
        valid = [value for value in acceptable if isinstance(value, str) and value]
        if valid:
            return list(dict.fromkeys(valid))
    fallback = result.get("dataset_target_room_id") or case.get("target_room_id")
    return [fallback] if isinstance(fallback, str) and fallback else []


def _reverse_room_edges(room_graph: dict[str, Any]) -> dict[str, list[str]]:
    reversed_edges: dict[str, list[str]] = defaultdict(list)
    for source, node in room_graph.items():
        if not isinstance(node, dict):
            continue
        for neighbor in node.get("neighbors", []):
            if not isinstance(neighbor, dict):
                continue
            target = neighbor.get("target_room_id")
            if isinstance(target, str) and target in room_graph:
                reversed_edges[target].append(source)
    return reversed_edges


def room_distances_to_goals(
    room_graph: dict[str, Any], goal_room_ids: Sequence[str]
) -> dict[str, int]:
    reversed_edges = _reverse_room_edges(room_graph)
    distances: dict[str, int] = {}
    queue: deque[str] = deque()
    for goal in goal_room_ids:
        if goal in room_graph and goal not in distances:
            distances[goal] = 0
            queue.append(goal)
    while queue:
        target = queue.popleft()
        for source in reversed_edges.get(target, []):
            if source not in distances:
                distances[source] = distances[target] + 1
                queue.append(source)
    return distances


def shortest_room_next_hops(
    room_graph: dict[str, Any],
    current_room_id: str | None,
    goal_room_ids: Sequence[str],
) -> list[str]:
    if not isinstance(current_room_id, str) or current_room_id not in room_graph:
        return []
    distances = room_distances_to_goals(room_graph, goal_room_ids)
    current_distance = distances.get(current_room_id)
    if current_distance is None or current_distance == 0:
        return []
    next_hops: list[str] = []
    for neighbor in room_graph[current_room_id].get("neighbors", []):
        if not isinstance(neighbor, dict):
            continue
        target = neighbor.get("target_room_id")
        if isinstance(target, str) and distances.get(target) == current_distance - 1:
            next_hops.append(target)
    return sorted(set(next_hops))


def shortest_room_route(
    room_graph: dict[str, Any],
    current_room_id: str | None,
    goal_room_ids: Sequence[str],
) -> list[str]:
    """Return one deterministic shortest route; alternatives remain in next hops."""

    if not isinstance(current_room_id, str) or current_room_id not in room_graph:
        return []
    targets = {goal for goal in goal_room_ids if goal in room_graph}
    if current_room_id in targets:
        return [current_room_id]
    distances = room_distances_to_goals(room_graph, goal_room_ids)
    if current_room_id not in distances:
        return []
    route = [current_room_id]
    cursor = current_room_id
    while cursor not in targets:
        next_hops = shortest_room_next_hops(room_graph, cursor, goal_room_ids)
        if not next_hops:
            return []
        cursor = next_hops[0]
        route.append(cursor)
    return route


def room_heading(
    room_graph: dict[str, Any], source_room_id: str | None, target_room_id: str | None
) -> float | None:
    if not isinstance(source_room_id, str) or not isinstance(target_room_id, str):
        return None
    node = room_graph.get(source_room_id)
    if not isinstance(node, dict):
        return None
    for neighbor in node.get("neighbors", []):
        if not isinstance(neighbor, dict):
            continue
        if neighbor.get("target_room_id") == target_room_id:
            value = neighbor.get("allocentric_heading_deg")
            return float(value) if isinstance(value, (int, float)) else None
    return None


def pano_edge_heading(
    pano_graph: dict[str, Any], source_pano_id: str, target_pano_id: str
) -> float:
    node = pano_graph.get(source_pano_id)
    if not isinstance(node, dict):
        raise ValueError(f"Unknown pano {source_pano_id}.")
    for neighbor in node.get("neighbors", []):
        if not isinstance(neighbor, dict):
            continue
        if neighbor.get("target_pano_id") != target_pano_id:
            continue
        heading = neighbor.get("geocentric_heading_deg")
        if not isinstance(heading, (int, float)):
            raise ValueError(
                f"Pano edge {source_pano_id} -> {target_pano_id} has no heading."
            )
        return float(heading)
    raise ValueError(
        f"Movement is not a pano graph edge: {source_pano_id} -> {target_pano_id}."
    )


def _localization_events(result: dict[str, Any]) -> dict[int, dict[str, Any]]:
    events = result.get("localization_events", [])
    if not isinstance(events, list):
        raise ValueError(f"{result.get('test_id')} localization_events is not a list.")
    indexed: dict[int, dict[str, Any]] = {}
    for event in events:
        if not isinstance(event, dict) or not isinstance(event.get("step_index"), int):
            raise ValueError(f"{result.get('test_id')} has an invalid localization event.")
        step_index = event["step_index"]
        if step_index in indexed:
            raise ValueError(
                f"{result.get('test_id')} has duplicate localization step {step_index}."
            )
        indexed[step_index] = event
    return indexed


def localization_call_estimate(event: dict[str, Any] | None) -> str | None:
    if not event:
        return None
    for field in (
        "stabilized_room_id",
        "spatial_predicted_room_id",
        "bayesian_predicted_room_id",
        "raw_localized_room_id",
    ):
        value = event.get(field)
        if isinstance(value, str) and value:
            return value
    return None


def build_step_diagnostics(
    *,
    condition: str,
    case: dict[str, Any],
    result: dict[str, Any],
    room_graph: dict[str, Any],
    pano_graph: dict[str, Any],
    grounding: dict[str, str],
) -> list[dict[str, Any]]:
    test_id = str(result["test_id"])
    movements = result.get("movements")
    pano_path = result.get("pano_path")
    if not isinstance(movements, list) or not isinstance(pano_path, list):
        raise ValueError(f"{test_id} has invalid movements or pano_path.")
    if len(pano_path) != len(movements) + 1:
        raise ValueError(f"{test_id} movement count does not match pano_path.")
    events = _localization_events(result)
    goals = active_goal_room_ids(result, case)
    rows: list[dict[str, Any]] = []
    for movement_index, movement in enumerate(movements):
        if not isinstance(movement, dict):
            raise ValueError(f"{test_id} movement {movement_index} is not an object.")
        source = movement.get("from_pano_id")
        target = movement.get("to_pano_id")
        step_index = movement.get("step_index")
        if not isinstance(source, str) or not isinstance(target, str):
            raise ValueError(f"{test_id} movement {movement_index} has invalid panos.")
        if not isinstance(step_index, int):
            raise ValueError(f"{test_id} movement {movement_index} has no step_index.")
        if source != pano_path[movement_index] or target != pano_path[movement_index + 1]:
            raise ValueError(f"{test_id} movement {movement_index} differs from pano_path.")

        actual_geocentric = pano_edge_heading(pano_graph, source, target)
        actual_allocentric = normalize_heading(actual_geocentric + 30.0)
        ground_truth = grounding.get(source)
        next_ground_truth = grounding.get(target)
        estimated = movement.get("estimated_room_id")
        stored_subgoal = movement.get("subgoal_room_id")
        oracle_next_hops = shortest_room_next_hops(room_graph, ground_truth, goals)
        oracle_route = shortest_room_route(room_graph, ground_truth, goals)
        agent_next_hops = shortest_room_next_hops(room_graph, estimated, goals)
        agent_route = shortest_room_route(room_graph, estimated, goals)
        event = events.get(step_index)
        call_estimate = localization_call_estimate(event)
        localization_correct = (
            estimated == ground_truth if isinstance(ground_truth, str) else None
        )
        call_correct = (
            call_estimate == ground_truth
            if isinstance(ground_truth, str) and isinstance(call_estimate, str)
            else None
        )
        plan_matches_oracle = (
            stored_subgoal in oracle_next_hops
            if oracle_next_hops
            else stored_subgoal is None and ground_truth in goals
        )
        reconstructed_subgoal = agent_route[1] if len(agent_route) > 1 else None
        route_mismatch = (
            stored_subgoal not in agent_next_hops
            if agent_next_hops
            else not (stored_subgoal is None and estimated in goals)
        )
        expected_headings = [
            room_heading(room_graph, ground_truth, next_room)
            for next_room in oracle_next_hops
        ]
        expected_headings = [
            value for value in expected_headings if isinstance(value, (int, float))
        ]
        min_oracle_heading_error = (
            min(angular_distance(actual_allocentric, value) for value in expected_headings)
            if expected_headings
            else None
        )
        planned_heading = room_heading(room_graph, estimated, stored_subgoal)
        recorded_ground_truth = event.get("ground_truth_room_id") if event else None
        rows.append(
            {
                "condition": condition,
                "test_id": test_id,
                "success": result.get("success") is True,
                "step_index": step_index,
                "movement_index": movement_index,
                "from_pano_id": source,
                "to_pano_id": target,
                "ground_truth_room_id": ground_truth,
                "next_pano_ground_truth_room_id": next_ground_truth,
                "ground_truth_known": isinstance(ground_truth, str),
                "estimated_room_id": estimated,
                "effective_state_correct": localization_correct,
                "localization_performed": event is not None,
                "localization_call_estimated_room_id": call_estimate,
                "localization_call_correct": call_correct,
                "raw_localized_room_id": event.get("raw_localized_room_id") if event else None,
                "spatial_predicted_room_id": event.get("spatial_predicted_room_id") if event else None,
                "recorded_localization_ground_truth_room_id": recorded_ground_truth,
                "grounding_mismatch": bool(
                    isinstance(recorded_ground_truth, str)
                    and isinstance(ground_truth, str)
                    and recorded_ground_truth != ground_truth
                ),
                "stored_subgoal_room_id": stored_subgoal,
                "active_goal_room_ids": goals,
                "reconstructed_agent_route": agent_route,
                "reconstructed_agent_next_room_ids": agent_next_hops,
                "reconstructed_agent_subgoal_room_id": reconstructed_subgoal,
                "stored_subgoal_vs_reconstructed_route_mismatch": route_mismatch,
                "oracle_route_from_ground_truth": oracle_route,
                "valid_oracle_next_room_ids": oracle_next_hops,
                "stored_subgoal_matches_oracle": plan_matches_oracle,
                "actual_geocentric_heading_deg": actual_geocentric,
                "actual_allocentric_heading_deg": actual_allocentric,
                "planned_allocentric_heading_deg": planned_heading,
                "valid_oracle_allocentric_headings_deg": expected_headings,
                "heading_error_to_agent_plan_deg": angular_distance(
                    actual_allocentric, planned_heading
                ),
                "minimum_heading_error_to_valid_oracle_deg": min_oracle_heading_error,
                "in_decision_window": False,
                "is_decision_step": False,
            }
        )
    return rows


def actual_room_visits(
    pano_path: Sequence[str], grounding: dict[str, str]
) -> list[dict[str, Any]]:
    """Collapse consecutive known rooms while retaining revisits.

    Transit panoramas are ignored for the room sequence.  Thus
    ``Room 6 -> null -> Room 6`` remains one visit, while
    ``Room 6 -> null -> Room 7`` becomes two visits.
    """

    visits: list[dict[str, Any]] = []
    for pano_index, pano_id in enumerate(pano_path):
        room_id = grounding.get(pano_id)
        if not isinstance(room_id, str):
            continue
        if visits and visits[-1]["room_id"] == room_id:
            visits[-1]["last_known_pano_index"] = pano_index
            visits[-1]["last_known_pano_id"] = pano_id
            continue
        visits.append(
            {
                "room_id": room_id,
                "first_known_pano_index": pano_index,
                "last_known_pano_index": pano_index,
                "first_known_pano_id": pano_id,
                "last_known_pano_id": pano_id,
            }
        )
    return visits


def find_first_divergence(
    *,
    visits: Sequence[dict[str, Any]],
    room_graph: dict[str, Any],
    goal_room_ids: Sequence[str],
) -> dict[str, Any]:
    """Find the first invalid shortest-path room transition in a failed run."""

    if not visits:
        return {
            "divergence_kind": "unknown_no_ground_truth_room",
            "visit_index": None,
            "divergence_room_id": None,
            "valid_expected_next_room_ids": [],
            "actual_next_room_id": None,
        }

    goals = set(goal_room_ids)
    for visit_index, visit in enumerate(visits):
        if visit["room_id"] in goals:
            return {
                "divergence_kind": "goal_reached_not_recognized",
                "visit_index": visit_index,
                "divergence_room_id": visit["room_id"],
                "valid_expected_next_room_ids": [],
                "actual_next_room_id": (
                    visits[visit_index + 1]["room_id"]
                    if visit_index + 1 < len(visits)
                    else None
                ),
            }

    for visit_index, (current, following) in enumerate(zip(visits, visits[1:])):
        expected = shortest_room_next_hops(
            room_graph, current["room_id"], goal_room_ids
        )
        if not expected:
            return {
                "divergence_kind": "unknown_no_room_route",
                "visit_index": visit_index,
                "divergence_room_id": current["room_id"],
                "valid_expected_next_room_ids": [],
                "actual_next_room_id": following["room_id"],
            }
        if following["room_id"] not in expected:
            return {
                "divergence_kind": "wrong_room_transition",
                "visit_index": visit_index,
                "divergence_room_id": current["room_id"],
                "valid_expected_next_room_ids": expected,
                "actual_next_room_id": following["room_id"],
            }

    final_index = len(visits) - 1
    final_room = visits[-1]["room_id"]
    expected = shortest_room_next_hops(room_graph, final_room, goal_room_ids)
    return {
        "divergence_kind": (
            "stalled_before_expected_transition"
            if expected
            else "unknown_no_room_route"
        ),
        "visit_index": final_index,
        "divergence_room_id": final_room,
        "valid_expected_next_room_ids": expected,
        "actual_next_room_id": None,
    }


def decision_window_indices(
    *,
    visits: Sequence[dict[str, Any]],
    divergence: dict[str, Any],
    movement_count: int,
) -> list[int]:
    visit_index = divergence.get("visit_index")
    if not isinstance(visit_index, int) or not 0 <= visit_index < len(visits):
        return []
    start = int(visits[visit_index]["first_known_pano_index"])
    if visit_index + 1 < len(visits):
        end_exclusive = int(visits[visit_index + 1]["first_known_pano_index"])
    else:
        end_exclusive = movement_count
    start = min(max(start, 0), movement_count)
    end_exclusive = min(max(end_exclusive, start), movement_count)
    return list(range(start, end_exclusive))


def select_decision_row(
    *,
    step_rows: Sequence[dict[str, Any]],
    window_indices: Sequence[int],
    divergence_room_id: str | None,
) -> dict[str, Any] | None:
    candidates = [
        step_rows[index]
        for index in window_indices
        if 0 <= index < len(step_rows)
        and step_rows[index].get("ground_truth_room_id") == divergence_room_id
    ]
    return candidates[-1] if candidates else None


def classify_divergence(
    *,
    divergence: dict[str, Any],
    decision_row: dict[str, Any] | None,
    decision_window_rows: Sequence[dict[str, Any]] = (),
    result: dict[str, Any],
    goal_room_ids: Sequence[str],
) -> tuple[str, str]:
    kind = divergence.get("divergence_kind")
    if kind == "goal_reached_not_recognized":
        effective_estimate = (
            decision_row.get("estimated_room_id")
            if decision_row
            else result.get("final_estimated_room_id")
        )
        if effective_estimate not in set(goal_room_ids):
            return (
                "localization",
                "physically_reached_goal_but_effective_room_state_did_not_recognize_goal",
            )
        return (
            "other/unclear",
            "physically_reached_goal_with_acceptable_room_state_but_run_did_not_stop_successfully",
        )
    if kind in {"unknown_no_ground_truth_room", "unknown_no_room_route"}:
        return "other/unclear", kind
    if decision_row is None:
        return "other/unclear", "no_ground_truth_decision_step_before_divergence"

    divergence_room = divergence.get("divergence_room_id")
    estimated_room = decision_row.get("estimated_room_id")
    expected_next = set(divergence.get("valid_expected_next_room_ids", []))
    stored_subgoal = decision_row.get("stored_subgoal_room_id")
    if not isinstance(divergence_room, str) or not isinstance(estimated_room, str):
        return "other/unclear", "missing_room_state_at_decision_step"
    if estimated_room != divergence_room:
        return (
            "localization",
            "effective_room_estimate_was_wrong_at_the_last_decision_before_divergence",
        )
    comparable_window_rows = [
        row
        for row in decision_window_rows
        if row.get("ground_truth_room_id") == divergence_room
        and isinstance(row.get("estimated_room_id"), str)
    ]
    correct_window_rows = [
        row
        for row in comparable_window_rows
        if row.get("estimated_room_id") == divergence_room
    ]
    if (
        comparable_window_rows
        and len(correct_window_rows) / len(comparable_window_rows) < 0.5
    ):
        return (
            "localization",
            "wrong_effective_room_state_dominated_the_divergence_room_decision_window",
        )
    if not expected_next:
        return "other/unclear", "no_valid_expected_next_room_at_decision_step"
    if stored_subgoal not in expected_next:
        return (
            "planning/replanning",
            "localization_was_correct_but_stored_subgoal_was_not_a_valid_shortest_next_room",
        )
    return (
        "direction/exit_selection",
        "localization_and_subgoal_were_correct_but_the_expected_room_transition_failed",
    )


def _step_accuracy(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    effective_known = [
        row for row in rows if isinstance(row.get("ground_truth_room_id"), str)
    ]
    effective_correct = [
        row for row in effective_known if row.get("effective_state_correct") is True
    ]
    call_known = [
        row
        for row in effective_known
        if row.get("localization_performed") is True
        and isinstance(row.get("localization_call_estimated_room_id"), str)
    ]
    call_correct = [
        row for row in call_known if row.get("localization_call_correct") is True
    ]
    return {
        "effective_state_known_step_count": len(effective_known),
        "effective_state_correct_step_count": len(effective_correct),
        "effective_state_accuracy": (
            len(effective_correct) / len(effective_known) if effective_known else None
        ),
        "localization_call_known_count": len(call_known),
        "localization_call_correct_count": len(call_correct),
        "localization_call_accuracy": (
            len(call_correct) / len(call_known) if call_known else None
        ),
    }


def analyze_failed_run(
    *,
    condition: str,
    case: dict[str, Any],
    result: dict[str, Any],
    paired_transition: str,
    step_rows: list[dict[str, Any]],
    room_graph: dict[str, Any],
    grounding: dict[str, str],
) -> dict[str, Any]:
    pano_path = result.get("pano_path", [])
    if not isinstance(pano_path, list):
        pano_path = []
    goals = active_goal_room_ids(result, case)
    start_room = case.get("start_room_id") or result.get("start_room_id")
    initial_route = shortest_room_route(room_graph, start_room, goals)
    visits = actual_room_visits(pano_path, grounding)
    divergence = find_first_divergence(
        visits=visits, room_graph=room_graph, goal_room_ids=goals
    )
    window_indices = decision_window_indices(
        visits=visits,
        divergence=divergence,
        movement_count=len(step_rows),
    )
    decision_row = select_decision_row(
        step_rows=step_rows,
        window_indices=window_indices,
        divergence_room_id=divergence.get("divergence_room_id"),
    )
    for index in window_indices:
        if 0 <= index < len(step_rows):
            step_rows[index]["in_decision_window"] = True
    if decision_row is not None:
        decision_row["is_decision_step"] = True
    window_rows = [step_rows[index] for index in window_indices]
    root_cause, evidence = classify_divergence(
        divergence=divergence,
        decision_row=decision_row,
        decision_window_rows=window_rows,
        result=result,
        goal_room_ids=goals,
    )
    full_accuracy = _step_accuracy(step_rows)
    window_accuracy = _step_accuracy(window_rows)
    expected = divergence.get("valid_expected_next_room_ids", [])
    actual_next = divergence.get("actual_next_room_id")
    return {
        "condition": condition,
        "test_id": result["test_id"],
        "path_id": result.get("path_id"),
        "paired_outcome": paired_transition,
        "stop_reason": result.get("stop_reason"),
        "movement_steps": len(step_rows),
        "step_budget": result.get("step_budget"),
        "correct_start_room_id": start_room,
        "actual_first_known_room_id": visits[0]["room_id"] if visits else None,
        "start_room_grounding_mismatch": bool(
            visits and isinstance(start_room, str) and visits[0]["room_id"] != start_room
        ),
        "active_goal_room_ids": goals,
        "dataset_reference_room_route": case.get("shortest_path_rooms", []),
        "initial_planned_room_route": initial_route,
        "actual_ground_truth_room_route": [visit["room_id"] for visit in visits],
        "divergence_kind": divergence.get("divergence_kind"),
        "first_divergence_room_id": divergence.get("divergence_room_id"),
        "valid_expected_next_room_ids": expected,
        "actual_next_room_id": actual_next,
        "problematic_transition": (
            f"{divergence.get('divergence_room_id')} -> "
            f"{' / '.join(expected) if expected else '[goal recognition]'}"
            if divergence.get("divergence_room_id")
            else "unknown"
        ),
        "observed_transition": (
            f"{divergence.get('divergence_room_id')} -> {actual_next or '[no exit]'}"
            if divergence.get("divergence_room_id")
            else "unknown"
        ),
        "root_cause": root_cause,
        "root_cause_evidence": evidence,
        "decision_window_start_movement_index": (
            window_indices[0] if window_indices else None
        ),
        "decision_window_end_movement_index": (
            window_indices[-1] if window_indices else None
        ),
        "decision_window_step_count": len(window_rows),
        "decision_step_index": decision_row.get("step_index") if decision_row else None,
        "decision_movement_index": (
            decision_row.get("movement_index") if decision_row else None
        ),
        "decision_from_pano_id": decision_row.get("from_pano_id") if decision_row else None,
        "decision_to_pano_id": decision_row.get("to_pano_id") if decision_row else None,
        "decision_ground_truth_room_id": (
            decision_row.get("ground_truth_room_id") if decision_row else None
        ),
        "decision_estimated_room_id": (
            decision_row.get("estimated_room_id")
            if decision_row
            else result.get("final_estimated_room_id")
        ),
        "decision_localization_performed": (
            decision_row.get("localization_performed") if decision_row else None
        ),
        "decision_stored_subgoal_room_id": (
            decision_row.get("stored_subgoal_room_id") if decision_row else None
        ),
        "decision_reconstructed_agent_route": (
            decision_row.get("reconstructed_agent_route") if decision_row else []
        ),
        "decision_stored_subgoal_matches_oracle": (
            decision_row.get("stored_subgoal_matches_oracle") if decision_row else None
        ),
        "decision_actual_allocentric_heading_deg": (
            decision_row.get("actual_allocentric_heading_deg") if decision_row else None
        ),
        "decision_planned_allocentric_heading_deg": (
            decision_row.get("planned_allocentric_heading_deg") if decision_row else None
        ),
        "decision_valid_oracle_allocentric_headings_deg": (
            decision_row.get("valid_oracle_allocentric_headings_deg")
            if decision_row
            else []
        ),
        "decision_heading_error_to_agent_plan_deg": (
            decision_row.get("heading_error_to_agent_plan_deg") if decision_row else None
        ),
        "decision_minimum_heading_error_to_valid_oracle_deg": (
            decision_row.get("minimum_heading_error_to_valid_oracle_deg")
            if decision_row
            else None
        ),
        **{f"full_run_{key}": value for key, value in full_accuracy.items()},
        **{f"decision_window_{key}": value for key, value in window_accuracy.items()},
        "unknown_ground_truth_step_count": sum(
            row.get("ground_truth_known") is False for row in step_rows
        ),
        "grounding_mismatch_step_count": sum(
            row.get("grounding_mismatch") is True for row in step_rows
        ),
    }


def build_paired_rows(
    *,
    dataset: dict[str, dict[str, Any]],
    results_by_condition: dict[str, dict[str, dict[str, Any]]],
    failed_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    failed_by_key = {(row["condition"], row["test_id"]): row for row in failed_rows}
    paired: list[dict[str, Any]] = []
    for test_id, case in dataset.items():
        without = results_by_condition["without_trajectory"][test_id]
        with_result = results_by_condition["with_trajectory"][test_id]
        outcome = success_transition(without, with_result)
        if outcome == "both_success":
            continue
        row: dict[str, Any] = {
            "test_id": test_id,
            "path_id": case.get("path_id"),
            "paired_outcome": outcome,
            "without_success": without.get("success") is True,
            "with_success": with_result.get("success") is True,
        }
        for condition, prefix in (
            ("without_trajectory", "without"),
            ("with_trajectory", "with"),
        ):
            failure = failed_by_key.get((condition, test_id))
            for field in (
                "root_cause",
                "first_divergence_room_id",
                "valid_expected_next_room_ids",
                "actual_next_room_id",
                "initial_planned_room_route",
                "actual_ground_truth_room_route",
                "decision_step_index",
                "decision_estimated_room_id",
                "decision_stored_subgoal_room_id",
            ):
                row[f"{prefix}_{field}"] = failure.get(field) if failure else None
        paired.append(row)
    return paired


def summarize_divergence_rooms(
    failed_rows: Sequence[dict[str, Any]],
    step_rows_by_key: dict[tuple[str, str], list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in failed_rows:
        room = row.get("first_divergence_room_id") or "unknown"
        grouped[(row["condition"], str(room))].append(row)
    output: list[dict[str, Any]] = []
    for (condition, room), cases in grouped.items():
        causes = Counter(case["root_cause"] for case in cases)
        window_steps: list[dict[str, Any]] = []
        for case in cases:
            window_steps.extend(
                row
                for row in step_rows_by_key[(condition, case["test_id"])]
                if row.get("in_decision_window") is True
            )
        accuracy = _step_accuracy(window_steps)
        decision_known = [
            case
            for case in cases
            if isinstance(case.get("decision_ground_truth_room_id"), str)
            and isinstance(case.get("decision_estimated_room_id"), str)
        ]
        decision_correct = [
            case
            for case in decision_known
            if case["decision_ground_truth_room_id"]
            == case["decision_estimated_room_id"]
        ]
        output.append(
            {
                "condition": condition,
                "first_divergence_room_id": room,
                "failed_case_count": len(cases),
                "localization_case_count": causes["localization"],
                "planning_replanning_case_count": causes["planning/replanning"],
                "direction_exit_selection_case_count": causes[
                    "direction/exit_selection"
                ],
                "other_unclear_case_count": causes["other/unclear"],
                "decision_state_known_case_count": len(decision_known),
                "decision_state_correct_case_count": len(decision_correct),
                "decision_state_accuracy": (
                    len(decision_correct) / len(decision_known) if decision_known else None
                ),
                **{f"decision_window_{key}": value for key, value in accuracy.items()},
            }
        )
    output.sort(
        key=lambda row: (
            row["condition"],
            -row["failed_case_count"],
            row["first_divergence_room_id"],
        )
    )
    return output


def summarize_route_divergences(
    failed_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: Counter[tuple[str, str, str, str, str, str]] = Counter()
    for row in failed_rows:
        key = (
            row["condition"],
            " -> ".join(row.get("initial_planned_room_route", [])) or "unknown",
            row.get("first_divergence_room_id") or "unknown",
            " / ".join(row.get("valid_expected_next_room_ids", [])) or "[none]",
            row.get("actual_next_room_id") or "[no exit]",
            row["root_cause"],
        )
        grouped[key] += 1
    rows = [
        {
            "condition": key[0],
            "initial_planned_room_route": key[1],
            "first_divergence_room_id": key[2],
            "valid_expected_next_room_ids": key[3],
            "actual_next_room_id": key[4],
            "root_cause": key[5],
            "failed_case_count": count,
        }
        for key, count in grouped.items()
    ]
    rows.sort(
        key=lambda row: (
            row["condition"],
            -row["failed_case_count"],
            row["first_divergence_room_id"],
            row["initial_planned_room_route"],
        )
    )
    return rows


def _truthy(value: Any) -> bool:
    return value is True or (
        isinstance(value, str)
        and value.strip().lower() in {"1", "true", "yes", "y", "complete"}
    )


def load_existing_manual_reviews(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8", newline="") as stream:
        return {
            row["test_id"]: row
            for row in csv.DictReader(stream)
            if isinstance(row.get("test_id"), str) and row["test_id"]
        }


def manual_review_metrics(
    review_rows: Sequence[dict[str, Any]], selected_case_count: int
) -> dict[str, Any]:
    labels: list[tuple[str, str]] = []
    for row in review_rows:
        for prefix in ("without", "with"):
            automatic = row.get(f"automatic_{prefix}_root_cause")
            manual = row.get(f"manual_{prefix}_root_cause")
            if automatic in ROOT_CAUSES and manual in ROOT_CAUSES:
                labels.append((str(automatic), str(manual)))
    completed = sum(_truthy(row.get("review_complete")) for row in review_rows)
    return {
        "selected_case_count": selected_case_count,
        "completed_case_count": completed,
        "manually_labeled_failed_run_count": len(labels),
        "automatic_vs_manual_agreement": (
            sum(automatic == manual for automatic, manual in labels) / len(labels)
            if labels
            else None
        ),
        "status": (
            "complete"
            if completed == selected_case_count
            else "in_progress"
            if completed
            else "pending_human_pano_viewer_review"
        ),
    }


def select_review_cases(
    paired_rows: Sequence[dict[str, Any]], target_count: int
) -> list[str]:
    selected: list[str] = []

    def round_robin(
        rows: Sequence[dict[str, Any]], room_field: str, limit: int
    ) -> list[str]:
        buckets: dict[str, deque[str]] = defaultdict(deque)
        for row in sorted(rows, key=lambda value: value["test_id"]):
            room_id = row.get(room_field) or "unknown"
            buckets[str(room_id)].append(row["test_id"])
        room_order = sorted(buckets, key=lambda room: (-len(buckets[room]), room))
        ordered: list[str] = []
        while len(ordered) < limit and any(buckets.values()):
            for room in room_order:
                if buckets[room] and len(ordered) < limit:
                    ordered.append(buckets[room].popleft())
        return ordered

    def add(candidates: Iterable[str], limit: int | None = None) -> None:
        added = 0
        for test_id in candidates:
            if test_id not in selected and len(selected) < target_count:
                selected.append(test_id)
                added += 1
                if limit is not None and added >= limit:
                    break

    by_outcome: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in paired_rows:
        by_outcome[row["paired_outcome"]].append(row)
    add(sorted(row["test_id"] for row in by_outcome["without_success_with_fail"]))
    add(
        round_robin(
            by_outcome["both_fail"], "without_first_divergence_room_id", 12
        ),
        12,
    )
    add(
        round_robin(
            by_outcome["without_fail_with_success"],
            "without_first_divergence_room_id",
            12,
        ),
        12,
    )
    add(["TEST363"])
    priority = [
        row["test_id"]
        for row in paired_rows
        if row.get("without_root_cause")
        in {"planning/replanning", "other/unclear"}
        or row.get("with_root_cause") in {"planning/replanning", "other/unclear"}
    ]
    add(sorted(priority))
    room_frequency: Counter[str] = Counter()
    for row in paired_rows:
        for field in (
            "without_first_divergence_room_id",
            "with_first_divergence_room_id",
        ):
            room_id = row.get(field)
            if isinstance(room_id, str):
                room_frequency[room_id] += 1
    high_frequency = sorted(
        paired_rows,
        key=lambda row: (
            -max(
                room_frequency.get(row.get("without_first_divergence_room_id"), 0),
                room_frequency.get(row.get("with_first_divergence_room_id"), 0),
            ),
            row["test_id"],
        ),
    )
    add(row["test_id"] for row in high_frequency)
    return selected[:target_count]


def export_viewer_bundle(
    *,
    output_dir: Path,
    selected_test_ids: Sequence[str],
    dataset: dict[str, dict[str, Any]],
    results_by_condition: dict[str, dict[str, dict[str, Any]]],
    paired_by_id: dict[str, dict[str, Any]],
    pano_graph: dict[str, Any],
    grounding: dict[str, str],
    existing_reviews: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    viewer_dir = output_dir / "pano_viewer_review"
    if viewer_dir.exists():
        shutil.rmtree(viewer_dir)
    manifest: list[dict[str, Any]] = []
    review_rows: list[dict[str, Any]] = []
    for test_id in selected_test_ids:
        case = dataset[test_id]
        pair = paired_by_id[test_id]
        files: dict[str, str] = {}
        for condition, folder in (
            ("without_trajectory", "without"),
            ("with_trajectory", "with"),
        ):
            result = results_by_condition[condition][test_id]
            episode = build_episode(
                case=case,
                result=result,
                pano_path=result["pano_path"],
                graph=pano_graph,
                grounding=grounding,
                trajectory_kind=folder,
            )
            envelope = build_envelope(
                case=case,
                result=result,
                episode=episode,
                trajectory_kind=folder,
            )
            prefix = "without" if condition == "without_trajectory" else "with"
            envelope["evaluation"]["automatic_root_cause"] = pair.get(
                f"{prefix}_root_cause"
            )
            envelope["evaluation"]["first_divergence_room_id"] = pair.get(
                f"{prefix}_first_divergence_room_id"
            )
            relative = Path(folder) / f"{test_id}.json"
            write_json(viewer_dir / relative, envelope)
            files[folder] = relative.as_posix()

        without_result = results_by_condition["without_trajectory"][test_id]
        reference_path = case.get("shortest_path_panos")
        if not isinstance(reference_path, list) or not reference_path:
            raise ValueError(f"{test_id} has no shortest_path_panos for Viewer export.")
        reference_episode = build_episode(
            case=case,
            result=None,
            pano_path=reference_path,
            graph=pano_graph,
            grounding=grounding,
            trajectory_kind="reference",
        )
        reference_envelope = build_envelope(
            case=case,
            result=without_result,
            episode=reference_episode,
            trajectory_kind="reference",
        )
        relative = Path("reference") / f"{test_id}.json"
        write_json(viewer_dir / relative, reference_envelope)
        files["reference"] = relative.as_posix()
        manifest.append(
            {
                "test_id": test_id,
                "paired_outcome": pair["paired_outcome"],
                "without_root_cause": pair.get("without_root_cause"),
                "with_root_cause": pair.get("with_root_cause"),
                "without_first_divergence_room_id": pair.get(
                    "without_first_divergence_room_id"
                ),
                "with_first_divergence_room_id": pair.get(
                    "with_first_divergence_room_id"
                ),
                "files": files,
            }
        )
        existing = existing_reviews.get(test_id, {})
        review_rows.append(
            {
                "test_id": test_id,
                "paired_outcome": pair["paired_outcome"],
                "automatic_without_root_cause": pair.get("without_root_cause"),
                "manual_without_root_cause": existing.get(
                    "manual_without_root_cause", ""
                ),
                "automatic_with_root_cause": pair.get("with_root_cause"),
                "manual_with_root_cause": existing.get("manual_with_root_cause", ""),
                "reviewer": existing.get("reviewer", ""),
                "review_complete": _truthy(existing.get("review_complete")),
                "notes": existing.get("notes", ""),
            }
        )
    write_json(viewer_dir / "manifest.json", manifest)
    write_csv(viewer_dir / "manual_review.csv", review_rows)
    _atomic_write(
        viewer_dir / "README.md",
        "\n".join(
            [
                "# First-route-divergence manual review",
                "",
                "每個案例依序比較 `reference/`、`without/`、`with/` 三條 trajectory。",
                "人工判斷時先找完整 room route 的第一個偏離點，再檢查偏離前最後一個決策 step。",
                "",
                "可填的主因：`localization`、`planning/replanning`、",
                "`direction/exit_selection`、`other/unclear`。",
                "",
                "啟動 Viewer：",
                "",
                "```bash",
                "python3 tools/pano_viewer/server.py",
                "```",
                "",
                "填好 `manual_review.csv` 後重跑分析器，人工欄位會被保留，",
                "`summary.json` 會計算自動分類與人工分類的一致率。",
                "",
            ]
        ),
    )
    return manifest, review_rows


def validate_inputs(
    dataset: dict[str, dict[str, Any]],
    results_by_condition: dict[str, dict[str, dict[str, Any]]],
    expected_case_count: int | None,
) -> None:
    if expected_case_count is not None and len(dataset) != expected_case_count:
        raise ValueError(
            f"Expected {expected_case_count} dataset cases, found {len(dataset)}."
        )
    dataset_ids = set(dataset)
    for condition in CONDITIONS:
        results = results_by_condition[condition]
        if set(results) != dataset_ids:
            raise ValueError(f"{condition} result IDs differ from dataset IDs.")
        for test_id, result in results.items():
            if result.get("path_id") != dataset[test_id].get("path_id"):
                raise ValueError(f"{condition} path_id mismatch for {test_id}.")
            if result.get("error") is not None or result.get("stop_reason") == "error":
                raise ValueError(f"{condition} contains an error result: {test_id}.")


def _cause_counts(
    failed_rows: Sequence[dict[str, Any]], condition: str
) -> dict[str, int]:
    counts = Counter(
        row["root_cause"] for row in failed_rows if row["condition"] == condition
    )
    return {cause: counts[cause] for cause in ROOT_CAUSES}


def build_summary(
    *,
    dataset: dict[str, dict[str, Any]],
    results_by_condition: dict[str, dict[str, dict[str, Any]]],
    failed_rows: Sequence[dict[str, Any]],
    paired_rows: Sequence[dict[str, Any]],
    divergence_room_rows: Sequence[dict[str, Any]],
    route_divergence_rows: Sequence[dict[str, Any]],
    all_step_rows: Sequence[dict[str, Any]],
    viewer_manifest: Sequence[dict[str, Any]],
    manual_review_rows: Sequence[dict[str, Any]],
    source_paths: dict[str, Path],
) -> dict[str, Any]:
    paired_counts = Counter(
        success_transition(
            results_by_condition["without_trajectory"][test_id],
            results_by_condition["with_trajectory"][test_id],
        )
        for test_id in dataset
    )
    failure_counts = Counter(row["condition"] for row in failed_rows)
    top_rooms: dict[str, list[dict[str, Any]]] = {}
    for condition in CONDITIONS:
        top_rooms[condition] = [
            row
            for row in divergence_room_rows
            if row["condition"] == condition
        ][:12]
    top_route_patterns = {
        condition: [
            row
            for row in route_divergence_rows
            if row["condition"] == condition
        ][:10]
        for condition in CONDITIONS
    }
    room_examples: dict[str, dict[str, Any]] = {}
    for room_id in ("Room 2", "Room 6"):
        room_examples[room_id] = {
            condition: next(
                (
                    row
                    for row in divergence_room_rows
                    if row["condition"] == condition
                    and row["first_divergence_room_id"] == room_id
                ),
                None,
            )
            for condition in CONDITIONS
        }
    test363 = [row for row in failed_rows if row["test_id"] == "TEST363"]
    acceptance = {
        "dataset_case_count_is_715": len(dataset) == 715,
        "paired_outcome_counts_match": dict(paired_counts) == EXPECTED_PAIRED_COUNTS,
        "failed_run_counts_match": dict(failure_counts) == EXPECTED_FAILURE_COUNTS,
        "failed_run_count_is_345": len(failed_rows) == 345,
        "unique_failed_test_count_is_240": len(paired_rows) == 240,
        "every_failed_run_has_root_cause": all(
            row.get("root_cause") in ROOT_CAUSES for row in failed_rows
        ),
        "every_failed_run_has_divergence_record": all(
            row.get("divergence_kind") and "decision_step_index" in row
            for row in failed_rows
        ),
        "every_movement_is_validated_against_pano_graph": True,
        "unknown_grounding_not_counted_as_localization_error": all(
            row.get("effective_state_correct") is None
            and row.get("localization_call_correct") is None
            for row in all_step_rows
            if row.get("ground_truth_known") is False
        ),
        "viewer_review_case_count_is_40": len(viewer_manifest) == 40,
    }
    return {
        "schema": "llm_nav_first_route_divergence_analysis",
        "schema_version": 1,
        "method": {
            "unit_of_attribution": "first invalid ground-truth room transition",
            "decision_state": "persistent effective room state across the divergence-room window, then the last action before exit or transit",
            "equal_shortest_paths": "all shortest next-room alternatives accepted",
            "localization_call_accuracy": "only localization_performed steps",
            "effective_state_accuracy": "all action steps with known ground-truth room",
            "transit_handling": "unlabelled panos retained as step evidence but excluded from room sequence and localization accuracy",
        },
        "dataset_case_count": len(dataset),
        "unique_failed_test_count": len(paired_rows),
        "failed_run_count": len(failed_rows),
        "paired_outcome_counts": dict(paired_counts),
        "failed_run_counts": dict(failure_counts),
        "root_causes_by_condition": {
            condition: _cause_counts(failed_rows, condition)
            for condition in CONDITIONS
        },
        "divergence_kinds_by_condition": {
            condition: dict(
                Counter(
                    row["divergence_kind"]
                    for row in failed_rows
                    if row["condition"] == condition
                )
            )
            for condition in CONDITIONS
        },
        "top_full_route_divergence_patterns": top_route_patterns,
        "top_first_divergence_rooms": top_rooms,
        "room_2_and_room_6_evidence": room_examples,
        "test363_examples": test363,
        "step_evidence": {
            "exported_movement_count": len(all_step_rows),
            "unknown_ground_truth_movement_count": sum(
                row.get("ground_truth_known") is False for row in all_step_rows
            ),
            "localization_call_count": sum(
                row.get("localization_performed") is True for row in all_step_rows
            ),
            "grounding_mismatch_count": sum(
                row.get("grounding_mismatch") is True for row in all_step_rows
            ),
        },
        "manual_review": manual_review_metrics(
            manual_review_rows, selected_case_count=len(viewer_manifest)
        ),
        "acceptance_checks": acceptance,
        "sources": {
            label: {"path": str(path), "sha256": sha256_file(path)}
            for label, path in source_paths.items()
        },
    }


def _percent(value: Any) -> str:
    return f"{100.0 * value:.1f}%" if isinstance(value, (int, float)) else "—"


def meeting_summary_markdown(summary: dict[str, Any]) -> str:
    paired = summary["paired_outcome_counts"]
    lines = [
        "# 715 筆導航實驗：第一個路徑偏離點分析",
        "",
        "## 分析方式",
        "",
        "每個 failed run 先比較完整的初始規劃 room route 與實際 ground-truth room route，",
        "找出第一個不再朝 parsed goal 的最短路徑前進的位置。若有多條等長最短路徑，",
        "所有合法 next room 都接受。主因先看該展廳整個 decision window 的定位，",
        "若錯誤 room state 佔多數就歸因 localization；否則再看偏離前最後一個 decision step：",
        "",
        "- decision estimate 錯：`localization`",
        "- estimate 對但 stored subgoal 錯：`planning/replanning`",
        "- estimate、subgoal 都對但沒有進入預期 room：`direction/exit_selection`",
        "- ground truth 或 route context 不足：`other/unclear`",
        "",
        "## Paired outcomes",
        "",
        "| 結果 | Cases |",
        "|---|---:|",
        f"| 兩者成功 | {paired.get('both_success', 0)} |",
        f"| Without 失敗、With 成功 | {paired.get('without_fail_with_success', 0)} |",
        f"| Without 成功、With 失敗 | {paired.get('without_success_with_fail', 0)} |",
        f"| 兩者失敗 | {paired.get('both_fail', 0)} |",
        "",
        "## 第一偏離點的主因",
        "",
        "| Condition | Localization | Planning/replanning | Direction/exit | Other/unclear | Total |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        causes = summary["root_causes_by_condition"][condition]
        lines.append(
            f"| {condition} | {causes['localization']} | "
            f"{causes['planning/replanning']} | "
            f"{causes['direction/exit_selection']} | "
            f"{causes['other/unclear']} | {sum(causes.values())} |"
        )
    lines.extend(["", "## 最常見的完整 route 偏離模式", ""])
    for condition in CONDITIONS:
        lines.extend(
            [
                f"### {condition}",
                "",
                "| Initial planned route | First divergence | Expected next | Actual next | Cause | n |",
                "|---|---|---|---|---|---:|",
            ]
        )
        for row in summary["top_full_route_divergence_patterns"][condition][:6]:
            lines.append(
                f"| {row['initial_planned_room_route']} | "
                f"{row['first_divergence_room_id']} | "
                f"{row['valid_expected_next_room_ids']} | "
                f"{row['actual_next_room_id']} | {row['root_cause']} | "
                f"{row['failed_case_count']} |"
            )
        lines.append("")
    lines.extend(["## 最常見的第一偏離展廳", ""])
    for condition in CONDITIONS:
        lines.extend(
            [
                f"### {condition}",
                "",
                "| Room | Failed cases | Localization | Planning | Direction/exit | Window state accuracy | Localization-call accuracy |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in summary["top_first_divergence_rooms"][condition][:8]:
            lines.append(
                f"| {row['first_divergence_room_id']} | {row['failed_case_count']} | "
                f"{row['localization_case_count']} | "
                f"{row['planning_replanning_case_count']} | "
                f"{row['direction_exit_selection_case_count']} | "
                f"{_percent(row['decision_window_effective_state_accuracy'])} | "
                f"{_percent(row['decision_window_localization_call_accuracy'])} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Room 2 與 Room 6 的解讀",
            "",
            "Room 2 與 Room 6 不用先假設原因；直接比較偏離前 decision state。",
            "Room 2 若大多數 decision estimate 已變成 Room 1，證據支持 localization；",
            "Room 6 若 decision estimate 與 subgoal 大多正確、仍走到 Room 7 或停滯，",
            "則支持 direction/exit-selection，並可再用 Pano Viewer 檢查出口結構。",
            "",
        ]
    )
    for room_id in ("Room 2", "Room 6"):
        lines.extend([f"### {room_id}", ""])
        lines.append(
        "| Condition | Failed cases | Localization | Planning | Direction/exit | Window state accuracy | Call accuracy |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for condition in CONDITIONS:
            row = summary["room_2_and_room_6_evidence"][room_id][condition]
            if row is None:
                lines.append(f"| {condition} | 0 | 0 | 0 | 0 | — | — |")
                continue
            lines.append(
                f"| {condition} | {row['failed_case_count']} | "
                f"{row['localization_case_count']} | "
                f"{row['planning_replanning_case_count']} | "
                f"{row['direction_exit_selection_case_count']} | "
                f"{_percent(row['decision_window_effective_state_accuracy'])} | "
                f"{_percent(row['decision_window_localization_call_accuracy'])} |"
            )
        lines.append("")
    examples = summary.get("test363_examples", [])
    if examples:
        lines.extend(["## 具體案例：TEST363", ""])
        for row in examples:
            lines.append(
                f"- {row['condition']}: planned `{' -> '.join(row['initial_planned_room_route'])}`; "
                f"actual `{' -> '.join(row['actual_ground_truth_room_route'])}`; "
                f"first divergence `{row['observed_transition']}` instead of "
                f"`{row['problematic_transition']}`; cause `{row['root_cause']}`."
            )
        lines.append("")
    lines.extend(
        [
            "## 人工覆核",
            "",
            "`pano_viewer_review/` 含 40 個代表案例，以及 reference / without / with 三種 trajectory。",
            "目前人工分類尚未填寫，因此一致率狀態是 pending；填完 `manual_review.csv` 後重跑即可計算。",
            "",
            "## 檔案閱讀順序",
            "",
            "1. `case_divergence_summary.csv`：每個 failed run 的完整 route 與第一偏離主因。",
            "2. `step_diagnostics.csv`：240 個 paired failure cases 的所有 action steps。",
            "3. `decision_window_steps.csv`：只保留第一偏離附近的 decision window。",
            "4. `divergence_room_summary.csv`：依第一偏離展廳彙總。",
            "5. `route_divergence_summary.csv`：依完整規劃 route、expected/actual next 與主因彙總。",
            "",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--without-results", default=DEFAULT_WITHOUT_RESULTS)
    parser.add_argument("--with-results", default=DEFAULT_WITH_RESULTS)
    parser.add_argument("--room-graph", default=DEFAULT_ROOM_GRAPH)
    parser.add_argument("--pano-graph", default=DEFAULT_PANO_GRAPH)
    parser.add_argument("--grounding", default=DEFAULT_GROUNDING)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--expected-case-count", type=int, default=715)
    parser.add_argument("--viewer-review-count", type=int, default=40)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = {
        "dataset": resolve(args.dataset),
        "without_results": resolve(args.without_results),
        "with_results": resolve(args.with_results),
        "room_graph": resolve(args.room_graph),
        "pano_graph": resolve(args.pano_graph),
        "grounding": resolve(args.grounding),
    }
    output_dir = resolve(args.output_dir)
    dataset = index_records(load_jsonl(paths["dataset"]), label="dataset")
    results_by_condition = {
        "without_trajectory": index_records(
            load_jsonl(paths["without_results"]), label="without results"
        ),
        "with_trajectory": index_records(
            load_jsonl(paths["with_results"]), label="with results"
        ),
    }
    validate_inputs(dataset, results_by_condition, args.expected_case_count)
    room_graph = load_json(paths["room_graph"])
    pano_graph = load_json(paths["pano_graph"])
    grounding_payload = load_json(paths["grounding"])
    raw_grounding = grounding_payload.get("mappings")
    if not isinstance(room_graph, dict) or not isinstance(pano_graph, dict):
        raise ValueError("Room graph and pano graph must be JSON objects.")
    if not isinstance(raw_grounding, dict):
        raise ValueError("Grounding JSON must contain an object-valued mappings field.")
    grounding = normalize_grounding(raw_grounding)

    paired_outcomes = {
        test_id: success_transition(
            results_by_condition["without_trajectory"][test_id],
            results_by_condition["with_trajectory"][test_id],
        )
        for test_id in dataset
    }
    selected_test_ids = {
        test_id for test_id, outcome in paired_outcomes.items() if outcome != "both_success"
    }
    all_selected_step_rows: list[dict[str, Any]] = []
    step_rows_by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
    failed_rows: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        for test_id, case in dataset.items():
            result = results_by_condition[condition][test_id]
            step_rows = build_step_diagnostics(
                condition=condition,
                case=case,
                result=result,
                room_graph=room_graph,
                pano_graph=pano_graph,
                grounding=grounding,
            )
            for step_row in step_rows:
                step_row["paired_outcome"] = paired_outcomes[test_id]
            step_rows_by_key[(condition, test_id)] = step_rows
            if result.get("success") is not True:
                failed_rows.append(
                    analyze_failed_run(
                        condition=condition,
                        case=case,
                        result=result,
                        paired_transition=paired_outcomes[test_id],
                        step_rows=step_rows,
                        room_graph=room_graph,
                        grounding=grounding,
                    )
                )
            if test_id in selected_test_ids:
                all_selected_step_rows.extend(step_rows)

    paired_rows = build_paired_rows(
        dataset=dataset,
        results_by_condition=results_by_condition,
        failed_rows=failed_rows,
    )
    divergence_room_rows = summarize_divergence_rooms(
        failed_rows, step_rows_by_key
    )
    route_divergence_rows = summarize_route_divergences(failed_rows)
    paired_by_id = {row["test_id"]: row for row in paired_rows}
    existing_reviews = load_existing_manual_reviews(
        output_dir / "pano_viewer_review" / "manual_review.csv"
    )
    selected_review_ids = select_review_cases(paired_rows, args.viewer_review_count)
    viewer_manifest, manual_review_rows = export_viewer_bundle(
        output_dir=output_dir,
        selected_test_ids=selected_review_ids,
        dataset=dataset,
        results_by_condition=results_by_condition,
        paired_by_id=paired_by_id,
        pano_graph=pano_graph,
        grounding=grounding,
        existing_reviews=existing_reviews,
    )
    decision_window_rows = [
        row for row in all_selected_step_rows if row.get("in_decision_window") is True
    ]
    summary = build_summary(
        dataset=dataset,
        results_by_condition=results_by_condition,
        failed_rows=failed_rows,
        paired_rows=paired_rows,
        divergence_room_rows=divergence_room_rows,
        route_divergence_rows=route_divergence_rows,
        all_step_rows=all_selected_step_rows,
        viewer_manifest=viewer_manifest,
        manual_review_rows=manual_review_rows,
        source_paths=paths,
    )
    write_csv(output_dir / "step_diagnostics.csv", all_selected_step_rows)
    write_csv(output_dir / "decision_window_steps.csv", decision_window_rows)
    write_csv(output_dir / "case_divergence_summary.csv", failed_rows)
    write_csv(output_dir / "paired_case_summary.csv", paired_rows)
    write_csv(output_dir / "divergence_room_summary.csv", divergence_room_rows)
    write_csv(output_dir / "route_divergence_summary.csv", route_divergence_rows)
    write_json(output_dir / "summary.json", summary)
    _atomic_write(output_dir / "meeting_summary.md", meeting_summary_markdown(summary))
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "dataset_case_count": len(dataset),
                "unique_failed_test_count": len(paired_rows),
                "failed_run_count": len(failed_rows),
                "viewer_review_case_count": len(viewer_manifest),
                "acceptance_checks": summary["acceptance_checks"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if all(summary["acceptance_checks"].values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
