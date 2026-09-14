#!/usr/bin/env python3
"""Report final navigation error and runtime room-state accuracy."""

from __future__ import annotations

import argparse
import heapq
import json
import math
import os
import statistics
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_EVALUATION_DIR = PROJECT_ROOT / (
    "outputs/navigation_evaluation/floor0_semantic_targets_topology_constrained/"
    "room_count_2to6_ratio_1p0_2p0_per20_sparse485/llm_nav_combined485"
)
DEFAULT_RESULTS = DEFAULT_EVALUATION_DIR / "results.jsonl"
DEFAULT_GROUNDING = (
    PROJECT_ROOT
    / "dataset/sites/british_museum/normalized/pano_room_grounding.json"
)
DEFAULT_GRAPH = (
    PROJECT_ROOT / "dataset/sites/british_museum/normalized/pano_graph.json"
)
DEFAULT_OUTPUT = DEFAULT_EVALUATION_DIR / "final_error_localization_report"
EARTH_RADIUS_M = 6_371_000.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--grounding", type=Path, default=DEFAULT_GROUNDING)
    parser.add_argument("--pano-graph", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def haversine_m(source: dict[str, Any], target: dict[str, Any]) -> float:
    lat_a = math.radians(float(source["lat"]))
    lat_b = math.radians(float(target["lat"]))
    delta_lat = lat_b - lat_a
    delta_lng = math.radians(float(target["lng"]) - float(source["lng"]))
    value = (
        math.sin(delta_lat / 2.0) ** 2
        + math.cos(lat_a) * math.cos(lat_b) * math.sin(delta_lng / 2.0) ** 2
    )
    return 2.0 * EARTH_RADIUS_M * math.asin(math.sqrt(value))


def reverse_adjacency(
    graph: dict[str, dict[str, Any]],
) -> dict[str, list[tuple[str, float]]]:
    reversed_edges: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for source_id, source in graph.items():
        for neighbor in source.get("neighbors", []):
            target_id = neighbor.get("target_pano_id")
            target = graph.get(target_id)
            if target is None:
                continue
            reversed_edges[target_id].append(
                (source_id, haversine_m(source, target))
            )
    return reversed_edges


def distances_to_targets(
    reversed_edges: dict[str, list[tuple[str, float]]],
    targets: set[str],
) -> dict[str, float]:
    distances = {target_id: 0.0 for target_id in targets}
    queue = [(0.0, target_id) for target_id in targets]
    heapq.heapify(queue)
    while queue:
        distance, pano_id = heapq.heappop(queue)
        if distance > distances[pano_id] + 1e-12:
            continue
        for predecessor_id, edge_distance in reversed_edges.get(pano_id, []):
            candidate = distance + edge_distance
            if candidate + 1e-12 < distances.get(predecessor_id, math.inf):
                distances[predecessor_id] = candidate
                heapq.heappush(queue, (candidate, predecessor_id))
    return distances


def percentile(values: list[float], proportion: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * proportion
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def distance_summary(values: list[float]) -> dict[str, Any]:
    return {
        "case_count": len(values),
        "mean_m": statistics.fmean(values),
        "median_m": statistics.median(values),
        "rmse_m": math.sqrt(statistics.fmean(value * value for value in values)),
        "p75_m": percentile(values, 0.75),
        "p90_m": percentile(values, 0.90),
        "max_m": max(values),
        "within_5m_count": sum(value <= 5.0 for value in values),
        "within_10m_count": sum(value <= 10.0 for value in values),
        "within_20m_count": sum(value <= 20.0 for value in values),
    }


def accuracy_summary(pairs: list[tuple[str, str]]) -> dict[str, Any]:
    correct = sum(estimated == ground_truth for estimated, ground_truth in pairs)
    return {
        "coverage_count": len(pairs),
        "correct_count": correct,
        "error_count": len(pairs) - correct,
        "accuracy": correct / len(pairs) if pairs else None,
        "error_rate": (len(pairs) - correct) / len(pairs) if pairs else None,
    }


def build_report(
    results: list[dict[str, Any]],
    grounding: dict[str, str],
    graph: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    room_panos: dict[str, set[str]] = defaultdict(set)
    for pano_id, room_id in grounding.items():
        if isinstance(room_id, str) and room_id != "null" and pano_id in graph:
            room_panos[room_id].add(pano_id)

    reversed_edges = reverse_adjacency(graph)
    distance_cache: dict[tuple[str, ...], dict[str, float]] = {}
    navigation_errors: list[dict[str, Any]] = []
    for result in results:
        acceptable = tuple(sorted(set(result.get("acceptable_target_room_ids") or [])))
        if acceptable not in distance_cache:
            target_panos = set().union(
                *(room_panos.get(room_id, set()) for room_id in acceptable)
            )
            if not target_panos:
                raise ValueError(f"No target panos for acceptable rooms {acceptable}")
            distance_cache[acceptable] = distances_to_targets(
                reversed_edges, target_panos
            )
        final_pano_id = result.get("final_pano_id")
        distance = distance_cache[acceptable].get(final_pano_id)
        if distance is None:
            raise ValueError(
                f"No graph path from {final_pano_id} to acceptable rooms {acceptable}"
            )
        navigation_errors.append(
            {
                "test_id": result["test_id"],
                "success": bool(result.get("success")),
                "final_pano_id": final_pano_id,
                "acceptable_target_room_ids": list(acceptable),
                "final_navigation_error_m": distance,
            }
        )

    final_pairs: list[tuple[str, str]] = []
    final_unknown = 0
    runtime_pairs: list[tuple[str, str]] = []
    runtime_unknown = 0
    per_case_accuracies: list[float] = []
    for result in results:
        estimated = result.get("final_estimated_room_id")
        ground_truth = result.get("final_ground_truth_room_id")
        if isinstance(estimated, str) and isinstance(ground_truth, str):
            final_pairs.append((estimated, ground_truth))
        else:
            final_unknown += 1

        case_pairs: list[tuple[str, str]] = []
        for movement in result.get("movements", []):
            estimated = movement.get("estimated_room_id")
            ground_truth = grounding.get(movement.get("from_pano_id"))
            if (
                isinstance(estimated, str)
                and isinstance(ground_truth, str)
                and ground_truth != "null"
            ):
                pair = (estimated, ground_truth)
                runtime_pairs.append(pair)
                case_pairs.append(pair)
            else:
                runtime_unknown += 1
        if case_pairs:
            per_case_accuracies.append(
                sum(a == b for a, b in case_pairs) / len(case_pairs)
            )

    all_errors = [item["final_navigation_error_m"] for item in navigation_errors]
    success_errors = [
        item["final_navigation_error_m"]
        for item in navigation_errors
        if item["success"]
    ]
    failure_errors = [
        item["final_navigation_error_m"]
        for item in navigation_errors
        if not item["success"]
    ]
    runtime = accuracy_summary(runtime_pairs)
    runtime.update(
        {
            "decision_step_count": len(runtime_pairs) + runtime_unknown,
            "unknown_ground_truth_count": runtime_unknown,
            "coverage_rate": len(runtime_pairs)
            / (len(runtime_pairs) + runtime_unknown),
            "macro_case_accuracy": statistics.fmean(per_case_accuracies),
            "median_case_accuracy": statistics.median(per_case_accuracies),
            "case_coverage_count": len(per_case_accuracies),
        }
    )
    final = accuracy_summary(final_pairs)
    final.update(
        {
            "case_count": len(results),
            "unknown_ground_truth_or_estimate_count": final_unknown,
            "coverage_rate": len(final_pairs) / len(results),
            "interpretation_caveat": (
                "The final estimated room is the state carried through the final "
                "movement; the batch does not force a new localization at the final pano."
            ),
        }
    )
    return {
        "schema": "llm_nav_final_error_and_runtime_localization_report",
        "schema_version": 1,
        "case_count": len(results),
        "final_navigation_error": {
            "definition": (
                "Shortest directed pano-graph distance in metres from final pano "
                "to any pano grounded in an acceptable target room"
            ),
            "overall": distance_summary(all_errors),
            "success_cases": distance_summary(success_errors),
            "failure_cases": distance_summary(failure_errors),
        },
        "runtime_room_state_accuracy": runtime,
        "final_room_state_accuracy": final,
        "limitations": [
            (
                "Runtime accuracy compares the maintained estimated room at every "
                "movement decision with the ground-truth room of from_pano_id."
            ),
            (
                "The batch output does not preserve localization_performed or raw "
                "localization predictions, so localization-call-only accuracy cannot "
                "be recovered retrospectively."
            ),
        ],
        "cases": navigation_errors,
    }


def markdown(report: dict[str, Any]) -> str:
    error = report["final_navigation_error"]
    runtime = report["runtime_room_state_accuracy"]
    final = report["final_room_state_accuracy"]
    lines = [
        "# Final Navigation Error and Runtime Localization",
        "",
        "## Final Navigation Error",
        "",
        f"Definition: {error['definition']}.",
        "",
        "| Scope | Cases | Mean | Median | RMSE | P90 | Max |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, key in (
        ("Overall", "overall"),
        ("Success", "success_cases"),
        ("Failure", "failure_cases"),
    ):
        item = error[key]
        lines.append(
            f"| {label} | {item['case_count']} | {item['mean_m']:.2f} m | "
            f"{item['median_m']:.2f} m | {item['rmse_m']:.2f} m | "
            f"{item['p90_m']:.2f} m | {item['max_m']:.2f} m |"
        )
    lines.extend(
        [
            "",
            "## Runtime Room-State Accuracy",
            "",
            "| Metric | Result |",
            "|---|---:|",
            f"| Decision steps | {runtime['decision_step_count']} |",
            f"| Comparable steps | {runtime['coverage_count']} |",
            f"| Correct steps | {runtime['correct_count']} |",
            f"| Micro accuracy | {100.0 * runtime['accuracy']:.2f}% |",
            f"| Macro case accuracy | {100.0 * runtime['macro_case_accuracy']:.2f}% |",
            f"| Median case accuracy | {100.0 * runtime['median_case_accuracy']:.2f}% |",
            f"| Unknown-ground-truth steps | {runtime['unknown_ground_truth_count']} |",
            "",
            "## Final Room-State Comparison",
            "",
            "| Metric | Result |",
            "|---|---:|",
            f"| Cases | {final['case_count']} |",
            f"| Comparable final states | {final['coverage_count']} |",
            f"| Exact matches | {final['correct_count']} |",
            f"| Exact-match accuracy | {100.0 * final['accuracy']:.2f}% |",
            f"| Mismatches | {final['error_count']} |",
            f"| Unknown final ground truth/estimate | {final['unknown_ground_truth_or_estimate_count']} |",
            "",
            "> Caveat: the final estimated room is carried through the final movement; "
            "the evaluator does not force localization at the final pano.",
            "",
            "## Limitation",
            "",
            "The saved batch output does not include `localization_performed` or raw "
            "localization predictions. Therefore this report measures runtime maintained "
            "room-state accuracy, not localization-call-only accuracy.",
            "",
        ]
    )
    return "\n".join(lines)


def atomic_write(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        stream.write(contents)
        temporary = Path(stream.name)
    os.replace(temporary, path)


def main() -> int:
    args = parse_args()
    results = load_jsonl(args.results)
    grounding_payload = load_json(args.grounding)
    grounding = grounding_payload.get("mappings")
    graph = load_json(args.pano_graph)
    if not isinstance(grounding, dict) or not isinstance(graph, dict):
        raise ValueError("Invalid grounding or pano graph")
    report = build_report(results, grounding, graph)
    atomic_write(
        args.output_dir / "summary.json",
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
    )
    atomic_write(args.output_dir / "meeting_summary.md", markdown(report))
    print(
        f"Wrote report for {len(results)} cases to {args.output_dir}; "
        f"runtime_accuracy={report['runtime_room_state_accuracy']['accuracy']:.6f}; "
        f"final_navigation_error_m="
        f"{report['final_navigation_error']['overall']['mean_m']:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
