#!/usr/bin/env python3
"""Export batch-evaluation trajectories as pano-viewer compatible JSON files.

The pano viewer accepts one episode object at a time.  Batch result JSONL files
only contain a flat pano path and movement list, so this tool joins a dataset
record with its result and reconstructs the episode envelope expected by
``tools/pano_viewer/web/trajectory.js``.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET = REPO_ROOT / (
    "outputs/navigation_paths/floor0_semantic_targets_topology_constrained/"
    "room_count_2to6_ratio_1p0_2p0_per10_sparse249/test_set.jsonl"
)
DEFAULT_RESULTS = REPO_ROOT / (
    "outputs/navigation_evaluation/floor0_semantic_targets_topology_constrained/"
    "room_count_2to6_ratio_1p0_2p0_per10_sparse249/"
    "llm_nav_combined249/results.jsonl"
)
DEFAULT_OUTPUT = REPO_ROOT / (
    "outputs/navigation_evaluation/floor0_semantic_targets_topology_constrained/"
    "room_count_2to6_ratio_1p0_2p0_per10_sparse249/"
    "pano_viewer_2rooms_high_detour"
)
DEFAULT_PANO_GRAPH = (
    REPO_ROOT / "dataset/sites/british_museum/normalized/pano_graph.json"
)
DEFAULT_GROUNDING = REPO_ROOT / (
    "dataset/sites/british_museum/normalized/pano_room_grounding.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pano-graph", type=Path, default=DEFAULT_PANO_GRAPH)
    parser.add_argument("--grounding", type=Path, default=DEFAULT_GROUNDING)
    parser.add_argument("--visited-room-count", type=int, default=2)
    parser.add_argument("--min-detour-ratio", type=float, default=1.6)
    parser.add_argument("--max-detour-ratio", type=float, default=2.0)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            records.append(value)
    return records


def index_unique(
    records: Iterable[dict[str, Any]], *, key: str, source: Path
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for record in records:
        value = record.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"Record in {source} has no non-empty {key}")
        if value in indexed:
            raise ValueError(f"Duplicate {key}={value!r} in {source}")
        indexed[value] = record
    return indexed


def edge_heading(
    graph: dict[str, Any], source: str, target: str
) -> float:
    node = graph.get(source)
    if not isinstance(node, dict):
        raise ValueError(f"Unknown pano in trajectory: {source}")
    for neighbor in node.get("neighbors", []):
        if not isinstance(neighbor, dict):
            continue
        if neighbor.get("target_pano_id") != target:
            continue
        heading = neighbor.get("geocentric_heading_deg")
        if not isinstance(heading, (int, float)):
            raise ValueError(f"Graph edge {source} -> {target} has no heading")
        return float(heading)
    raise ValueError(f"Trajectory move is not a legal graph edge: {source} -> {target}")


def room_sequence(
    pano_path: Sequence[str], grounding: dict[str, str]
) -> list[str]:
    sequence: list[str] = []
    for pano_id in pano_path:
        room_id = grounding.get(pano_id)
        if room_id and (not sequence or sequence[-1] != room_id):
            sequence.append(room_id)
    return sequence


def movement_rounds(
    *,
    pano_path: Sequence[str],
    graph: dict[str, Any],
    grounding: dict[str, str],
    target_room_id: str,
    raw_movements: Sequence[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    expected_count = max(0, len(pano_path) - 1)
    if raw_movements is not None and len(raw_movements) != expected_count:
        raise ValueError(
            f"pano_path has {expected_count} moves but result has "
            f"{len(raw_movements)} movement records"
        )

    rounds: list[dict[str, Any]] = []
    for index, (source, target) in enumerate(zip(pano_path, pano_path[1:])):
        raw = raw_movements[index] if raw_movements is not None else {}
        if raw_movements is not None and (
            raw.get("from_pano_id") != source or raw.get("to_pano_id") != target
        ):
            raise ValueError(
                f"Movement {index} does not match pano_path: {source} -> {target}"
            )
        current_room_id = grounding.get(source)
        next_room_id = grounding.get(target)
        movement = {
            "local_step_index": raw.get("step_index", index),
            "current_pano_id": source,
            "next_pano_id": target,
            "current_room_id": current_room_id,
            "next_room_id": next_room_id,
            "selected_action_heading": edge_heading(graph, source, target),
            "subgoal_room_id": raw.get("subgoal_room_id"),
            "active_target_room_id": target_room_id,
        }
        if raw_movements is not None:
            movement["estimated_room_id"] = raw.get("estimated_room_id")

        round_payload: dict[str, Any] = {
            "round_index": index,
            "start_pano_id": source,
            "active_target_room_id": target_room_id,
            "subgoal_room_id": raw.get("subgoal_room_id"),
            "completed_waypoints": [],
            "movement_steps": [movement],
        }
        if current_room_id and next_room_id and current_room_id != next_room_id:
            round_payload["room_boundary"] = {
                "from_room_id": current_room_id,
                "to_room_id": next_room_id,
                "at_pano_id": target,
            }
        rounds.append(round_payload)
    return rounds


def build_episode(
    *,
    case: dict[str, Any],
    result: dict[str, Any] | None,
    pano_path: Sequence[str],
    graph: dict[str, Any],
    grounding: dict[str, str],
    trajectory_kind: str,
) -> dict[str, Any]:
    if not pano_path:
        raise ValueError(f"{case['test_id']} {trajectory_kind} pano_path is empty")
    target_room_id = str(case["target_room_id"])
    raw_movements = None if result is None else result.get("movements")
    if raw_movements is not None and not isinstance(raw_movements, list):
        raise ValueError(f"{case['test_id']} movements must be a list")
    parsed_waypoints = [] if result is None else result.get("parsed_waypoint_room_ids", [])
    if not isinstance(parsed_waypoints, list):
        parsed_waypoints = []

    is_reference = trajectory_kind == "reference"
    success = True if is_reference else bool(result and result.get("success"))
    reason = "reference_shortest_path" if is_reference else result.get("stop_reason")
    return {
        "schema": "st_nav_navigation_episode",
        "schema_version": 1,
        "trajectory_kind": trajectory_kind,
        "test_id": case["test_id"],
        "path_id": case["path_id"],
        "start_pano_id": pano_path[0],
        "final_pano_id": pano_path[-1],
        "final_ground_truth_room_id": grounding.get(pano_path[-1]),
        "target_room_id": target_room_id,
        "acceptable_target_room_ids": case.get("acceptable_target_room_ids", []),
        "pano_path": list(pano_path),
        "waypoint_room_ids": parsed_waypoints,
        "completed_waypoints": parsed_waypoints if success else [],
        "remaining_waypoints": [] if success else parsed_waypoints,
        "ordered_targets": [*parsed_waypoints, target_room_id],
        "success": success,
        "reason": reason,
        "step_budget": None if is_reference else result.get("step_budget"),
        "ground_truth_room_sequence": room_sequence(pano_path, grounding),
        "rounds": movement_rounds(
            pano_path=pano_path,
            graph=graph,
            grounding=grounding,
            target_room_id=target_room_id,
            raw_movements=raw_movements,
        ),
    }


def build_envelope(
    *,
    case: dict[str, Any],
    result: dict[str, Any],
    episode: dict[str, Any],
    trajectory_kind: str,
) -> dict[str, Any]:
    evaluation = {
        "test_id": case["test_id"],
        "path_id": case["path_id"],
        "query": case["query"],
        "difficulty": case.get("difficulty"),
        "ratio_stratum": case.get("ratio_stratum"),
        "detour_ratio": case.get("detour_ratio"),
        "visited_room_count": case.get("visited_room_count"),
        "target_group_theme": case.get("target_group_theme"),
        "trajectory_kind": trajectory_kind,
        "success": True if trajectory_kind == "reference" else result.get("success"),
        "stop_reason": (
            "reference_shortest_path"
            if trajectory_kind == "reference"
            else result.get("stop_reason")
        ),
        "movement_steps": len(episode["pano_path"]) - 1,
        "step_budget": result.get("step_budget"),
    }
    return {
        "schema": "llm_nav_pano_viewer_evaluation",
        "schema_version": 1,
        "test_case": case,
        "evaluation": evaluation,
        "episode": episode,
    }


def failure_category(
    *, case: dict[str, Any], result: dict[str, Any], rooms: Sequence[str]
) -> str:
    if result.get("success") is True:
        return "success"
    start_room = case.get("start_room_id")
    if not rooms or all(room_id == start_room for room_id in rooms):
        return "stayed_in_start_room"
    acceptable = set(case.get("acceptable_target_room_ids", []))
    if acceptable.isdisjoint(rooms):
        return "left_start_but_never_reached_target"
    return "reached_target_room_but_not_counted_success"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def markdown_table(entries: Sequence[dict[str, Any]]) -> str:
    lines = [
        "# Pano Viewer：2 展廳、高 detour ratio 測資",
        "",
        "此目錄包含 19 筆 `visited_room_count = 2`、`1.6 <= detour_ratio < 2.0` 的案例。",
        "`actual/` 是 LLM-Nav 實際導航路徑；`reference/` 是資料集參考最短路徑。",
        "兩種 JSON 都能逐檔放進 Pano Viewer 的 Trajectory 面板。",
        "",
        "| 測資 | Ratio | 起點 → 目標 | 結果 | 步數 | 分類 |",
        "|---|---:|---|---:|---:|---|",
    ]
    for entry in entries:
        result_label = "成功" if entry["success"] else "失敗"
        lines.append(
            f"| {entry['test_id']} | {entry['detour_ratio']:.3f} "
            f"({entry['ratio_stratum']}) | {entry['start_room_id']} → "
            f"{entry['target_room_id']} | {result_label} | "
            f"{entry['movement_steps']}/{entry['step_budget']} | "
            f"{entry['failure_category']} |"
        )
    lines.extend(
        [
            "",
            "## 使用方式",
            "",
            "```bash",
            "python3 tools/pano_viewer/export.py",
            "python3 tools/pano_viewer/server.py",
            "```",
            "",
            "開啟 <http://127.0.0.1:8000>，先載入 `actual/TESTxxx.json`，",
            "再載入同編號的 `reference/TESTxxx.json` 對照。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    dataset_records = load_jsonl(args.dataset)
    result_records = load_jsonl(args.results)
    results_by_id = index_unique(result_records, key="test_id", source=args.results)
    graph = load_json(args.pano_graph)
    grounding_payload = load_json(args.grounding)
    grounding = grounding_payload.get("mappings", {})
    if not isinstance(graph, dict) or not isinstance(grounding, dict):
        raise ValueError("Pano graph and grounding must be JSON objects")

    selected = [
        case
        for case in dataset_records
        if case.get("visited_room_count") == args.visited_room_count
        and isinstance(case.get("detour_ratio"), (int, float))
        and args.min_detour_ratio <= float(case["detour_ratio"])
        < args.max_detour_ratio
    ]
    selected.sort(key=lambda case: case["test_id"])
    if not selected:
        raise ValueError("No dataset records match the requested filter")

    entries: list[dict[str, Any]] = []
    for case in selected:
        test_id = case["test_id"]
        if test_id not in results_by_id:
            raise ValueError(f"No batch result found for {test_id}")
        result = results_by_id[test_id]
        if result.get("path_id") != case.get("path_id"):
            raise ValueError(f"Dataset/result path_id mismatch for {test_id}")

        actual_path = result.get("pano_path")
        reference_path = case.get("shortest_path_panos")
        if not isinstance(actual_path, list) or not all(
            isinstance(value, str) and value for value in actual_path
        ):
            raise ValueError(f"{test_id} result has no valid pano_path")
        if not isinstance(reference_path, list) or not all(
            isinstance(value, str) and value for value in reference_path
        ):
            raise ValueError(f"{test_id} dataset has no valid shortest_path_panos")

        actual_episode = build_episode(
            case=case,
            result=result,
            pano_path=actual_path,
            graph=graph,
            grounding=grounding,
            trajectory_kind="actual",
        )
        reference_episode = build_episode(
            case=case,
            result=None,
            pano_path=reference_path,
            graph=graph,
            grounding=grounding,
            trajectory_kind="reference",
        )
        actual_relative = Path("actual") / f"{test_id}.json"
        reference_relative = Path("reference") / f"{test_id}.json"
        write_json(
            args.output_dir / actual_relative,
            build_envelope(
                case=case,
                result=result,
                episode=actual_episode,
                trajectory_kind="actual",
            ),
        )
        write_json(
            args.output_dir / reference_relative,
            build_envelope(
                case=case,
                result=result,
                episode=reference_episode,
                trajectory_kind="reference",
            ),
        )

        actual_rooms = actual_episode["ground_truth_room_sequence"]
        entries.append(
            {
                "test_id": test_id,
                "path_id": case["path_id"],
                "query": case["query"],
                "ratio_stratum": case["ratio_stratum"],
                "detour_ratio": case["detour_ratio"],
                "start_room_id": case["start_room_id"],
                "target_room_id": case["target_room_id"],
                "acceptable_target_room_ids": case.get(
                    "acceptable_target_room_ids", []
                ),
                "success": bool(result.get("success")),
                "stop_reason": result.get("stop_reason"),
                "movement_steps": result.get("movement_steps"),
                "step_budget": result.get("step_budget"),
                "reference_steps": len(reference_path) - 1,
                "actual_ground_truth_room_sequence": actual_rooms,
                "reference_room_sequence": reference_episode[
                    "ground_truth_room_sequence"
                ],
                "failure_category": failure_category(
                    case=case, result=result, rooms=actual_rooms
                ),
                "actual_file": actual_relative.as_posix(),
                "reference_file": reference_relative.as_posix(),
            }
        )

    failures = [entry for entry in entries if not entry["success"]]
    successes = [entry for entry in entries if entry["success"]]
    manifest = {
        "schema": "llm_nav_pano_viewer_case_manifest",
        "schema_version": 1,
        "filter": {
            "visited_room_count": args.visited_room_count,
            "min_detour_ratio_inclusive": args.min_detour_ratio,
            "max_detour_ratio_exclusive": args.max_detour_ratio,
        },
        "source_dataset": str(args.dataset.resolve()),
        "source_results": str(args.results.resolve()),
        "case_count": len(entries),
        "success_count": len(successes),
        "failure_count": len(failures),
        "success_test_ids": [entry["test_id"] for entry in successes],
        "failure_test_ids": [entry["test_id"] for entry in failures],
        "cases": entries,
    }
    write_json(args.output_dir / "manifest.json", manifest)

    csv_fields = [
        "test_id",
        "path_id",
        "ratio_stratum",
        "detour_ratio",
        "start_room_id",
        "target_room_id",
        "success",
        "stop_reason",
        "movement_steps",
        "step_budget",
        "reference_steps",
        "failure_category",
        "actual_ground_truth_room_sequence",
        "actual_file",
        "reference_file",
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "manifest.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        for entry in entries:
            csv_entry = dict(entry)
            csv_entry["actual_ground_truth_room_sequence"] = " -> ".join(
                entry["actual_ground_truth_room_sequence"]
            )
            writer.writerow(csv_entry)

    with (args.output_dir / "README.md").open("w", encoding="utf-8") as handle:
        handle.write(markdown_table(entries))

    print(
        f"Exported {len(entries)} cases ({len(failures)} failures, "
        f"{len(successes)} successes) to {args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
