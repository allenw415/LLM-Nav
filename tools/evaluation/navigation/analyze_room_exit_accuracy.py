#!/usr/bin/env python3
"""Report room-level first-exit accuracy for paired navigation results.

The unit is one test-condition-room opportunity. For each run, only the first
eligible visit to a room is counted. The expected next room is reconstructed
from the ground-truth current room, the parsed goal, and the room graph; every
equal-length shortest-path next hop is accepted.

An attempt is correct when the first subsequently observed room is one of the
valid next hops. Entering another known room and never leaving the current
room are both counted as wrong in the main table. They remain separate audit
fields, and a wrong-room attempt records whether the run later recovered to a
valid next room.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.evaluation.navigation.analyze_route_divergence import (
    CONDITIONS,
    DEFAULT_DATASET,
    DEFAULT_GROUNDING,
    DEFAULT_ROOM_GRAPH,
    DEFAULT_WITHOUT_RESULTS,
    DEFAULT_WITH_RESULTS,
    _atomic_write,
    active_goal_room_ids,
    actual_room_visits,
    index_records,
    load_json,
    load_jsonl,
    normalize_grounding,
    resolve,
    sha256_file,
    shortest_room_next_hops,
    success_transition,
    validate_inputs,
    write_csv,
    write_json,
)


EVALUATION_ROOT = (
    "outputs/navigation_evaluation/floor0_semantic_targets_topology_constrained/"
    "room_count_2to6_ratio_1p0_2p0_per30_sparse715"
)
DEFAULT_OUTPUT_DIR = f"{EVALUATION_ROOT}/room_level_exit_accuracy"


def _room_sort_key(room_id: str) -> tuple[int, int, str]:
    parts = room_id.split()
    if len(parts) >= 2:
        numeric = ""
        suffix = ""
        for character in parts[1]:
            if character.isdigit() and not suffix:
                numeric += character
            else:
                suffix += character
        if numeric:
            return (0, int(numeric), suffix.lower())
    return (1, 10**9, room_id.lower())


def build_first_room_attempts(
    *,
    condition: str,
    case: dict[str, Any],
    result: dict[str, Any],
    room_graph: dict[str, Any],
    grounding: dict[str, str],
    paired_outcome: str,
) -> list[dict[str, Any]]:
    """Build at most one first-exit observation per room in one run."""

    pano_path = result.get("pano_path")
    if not isinstance(pano_path, list) or not all(
        isinstance(pano_id, str) for pano_id in pano_path
    ):
        raise ValueError(f"{result.get('test_id')} has an invalid pano_path.")
    visits = actual_room_visits(pano_path, grounding)
    goals = active_goal_room_ids(result, case)
    goal_set = set(goals)
    seen_rooms: set[str] = set()
    attempts: list[dict[str, Any]] = []

    for visit_index, visit in enumerate(visits):
        current_room = visit["room_id"]
        if current_room in goal_set:
            break
        if current_room in seen_rooms:
            continue

        expected = shortest_room_next_hops(room_graph, current_room, goals)
        if not expected:
            continue
        seen_rooms.add(current_room)

        following = visits[visit_index + 1] if visit_index + 1 < len(visits) else None
        actual_next = following["room_id"] if following else None
        correct = actual_next in expected if actual_next is not None else False
        no_exit = actual_next is None
        wrong_room = actual_next is not None and not correct

        recovery_visit_index: int | None = None
        recovered_room: str | None = None
        if wrong_room:
            for later_index in range(visit_index + 2, len(visits)):
                later_room = visits[later_index]["room_id"]
                if later_room in expected:
                    recovery_visit_index = later_index
                    recovered_room = later_room
                    break

        recovered = recovery_visit_index is not None
        recovery_pano_steps = None
        recovery_room_hops = None
        if recovery_visit_index is not None:
            recovery_pano_steps = (
                visits[recovery_visit_index]["first_known_pano_index"]
                - visit["last_known_pano_index"]
            )
            recovery_room_hops = recovery_visit_index - visit_index

        attempts.append(
            {
                "condition": condition,
                "test_id": result["test_id"],
                "path_id": result.get("path_id") or case.get("path_id"),
                "paired_outcome": paired_outcome,
                "run_success": result.get("success") is True,
                "current_room_id": current_room,
                "active_goal_room_ids": goals,
                "valid_expected_next_room_ids": expected,
                "actual_next_room_id": actual_next,
                "outcome": "correct" if correct else "wrong",
                "correct": correct,
                "wrong": not correct,
                "wrong_room": wrong_room,
                "no_exit": no_exit,
                "recovered_after_wrong": recovered,
                "recovered_expected_room_id": recovered_room,
                "recovery_room_hops": recovery_room_hops,
                "recovery_pano_steps": recovery_pano_steps,
                "room_visit_index": visit_index,
                "room_first_known_pano_index": visit["first_known_pano_index"],
                "room_last_known_pano_index": visit["last_known_pano_index"],
                "actual_next_first_known_pano_index": (
                    following["first_known_pano_index"] if following else None
                ),
                "dataset_reference_room_route": case.get("shortest_path_rooms", []),
            }
        )
    return attempts


def summarize_room_attempts(
    attempts: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for attempt in attempts:
        grouped[(attempt["condition"], attempt["current_room_id"])].append(attempt)

    rows: list[dict[str, Any]] = []
    for (condition, room_id), group in grouped.items():
        attempt_count = len(group)
        correct_count = sum(row["correct"] is True for row in group)
        wrong_count = attempt_count - correct_count
        wrong_room_count = sum(row["wrong_room"] is True for row in group)
        no_exit_count = sum(row["no_exit"] is True for row in group)
        recovered_count = sum(row["recovered_after_wrong"] is True for row in group)
        rows.append(
            {
                "condition": condition,
                "room_id": room_id,
                "attempt_case_count": attempt_count,
                "correct_case_count": correct_count,
                "wrong_case_count": wrong_count,
                "wrong_room_case_count": wrong_room_count,
                "no_exit_case_count": no_exit_count,
                "recovered_after_wrong_case_count": recovered_count,
                "unrecovered_wrong_case_count": wrong_count - recovered_count,
                "correct_rate": correct_count / attempt_count,
                "wrong_rate": wrong_count / attempt_count,
                "wrong_to_correct_ratio": (
                    wrong_count / correct_count if correct_count else None
                ),
            }
        )
    condition_order = {condition: index for index, condition in enumerate(CONDITIONS)}
    return sorted(
        rows,
        key=lambda row: (
            _room_sort_key(row["room_id"]),
            condition_order[row["condition"]],
        ),
    )


def build_condition_comparison(
    summary_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_room: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in summary_rows:
        by_room[row["room_id"]][row["condition"]] = row

    comparison: list[dict[str, Any]] = []
    for room_id in sorted(by_room, key=_room_sort_key):
        without = by_room[room_id].get("without_trajectory", {})
        with_result = by_room[room_id].get("with_trajectory", {})
        without_rate = without.get("correct_rate")
        with_rate = with_result.get("correct_rate")
        comparison.append(
            {
                "room_id": room_id,
                "without_attempt_case_count": without.get("attempt_case_count", 0),
                "without_correct_case_count": without.get("correct_case_count", 0),
                "without_wrong_case_count": without.get("wrong_case_count", 0),
                "without_correct_rate": without_rate,
                "without_wrong_rate": without.get("wrong_rate"),
                "without_wrong_to_correct_ratio": without.get(
                    "wrong_to_correct_ratio"
                ),
                "with_attempt_case_count": with_result.get("attempt_case_count", 0),
                "with_correct_case_count": with_result.get("correct_case_count", 0),
                "with_wrong_case_count": with_result.get("wrong_case_count", 0),
                "with_correct_rate": with_rate,
                "with_wrong_rate": with_result.get("wrong_rate"),
                "with_wrong_to_correct_ratio": with_result.get(
                    "wrong_to_correct_ratio"
                ),
                "with_minus_without_correct_rate_pp": (
                    (with_rate - without_rate) * 100.0
                    if isinstance(with_rate, (int, float))
                    and isinstance(without_rate, (int, float))
                    else None
                ),
            }
        )
    return comparison


def build_condition_main_table(
    summary_rows: Sequence[dict[str, Any]], condition: str
) -> list[dict[str, Any]]:
    return [
        {
            "room_id": row["room_id"],
            "total_case_count": row["attempt_case_count"],
            "correct_case_count": row["correct_case_count"],
            "wrong_case_count": row["wrong_case_count"],
            "correct_rate": row["correct_rate"],
            "wrong_rate": row["wrong_rate"],
            "wrong_to_correct_ratio": row["wrong_to_correct_ratio"],
        }
        for row in summary_rows
        if row["condition"] == condition
    ]


def _aggregate_condition(
    attempts: Sequence[dict[str, Any]], condition: str
) -> dict[str, Any]:
    selected = [row for row in attempts if row["condition"] == condition]
    correct = sum(row["correct"] is True for row in selected)
    wrong_room = sum(row["wrong_room"] is True for row in selected)
    no_exit = sum(row["no_exit"] is True for row in selected)
    recovered = sum(row["recovered_after_wrong"] is True for row in selected)
    total = len(selected)
    wrong = wrong_room + no_exit
    return {
        "attempt_case_count": total,
        "correct_case_count": correct,
        "wrong_case_count": wrong,
        "wrong_room_case_count": wrong_room,
        "no_exit_case_count": no_exit,
        "recovered_after_wrong_case_count": recovered,
        "correct_rate": correct / total if total else None,
        "wrong_rate": wrong / total if total else None,
        "wrong_to_correct_ratio": wrong / correct if correct else None,
    }


def build_meeting_summary(
    *,
    attempts: Sequence[dict[str, Any]],
    comparison_rows: Sequence[dict[str, Any]],
    minimum_n: int,
) -> str:
    lines = [
        "# 715 筆導航實驗：Room-level 第一次出口正確率",
        "",
        "## 定義",
        "",
        "- 單位：每個 condition、test、room 的第一次有效出口嘗試。",
        "- Correct：第一個後續已知展廳屬於 ground-truth room 到目標的任一合法最短 next room。",
        "- Wrong：進入其他展廳，或直到結束都沒有離開；兩個細項仍保留在 CSV。",
        "- Transit pano 不算展廳；走錯後再回到合法 next room 另記 recovered_after_wrong。",
        "- 同一 run 重複回到同一 room 不重複計數，避免循環案例放大分母。",
        "",
        "## 全部 Room opportunities",
        "",
        "| Condition | Attempts | Correct | Wrong | Correct rate | Wrong rate |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        aggregate = _aggregate_condition(attempts, condition)
        lines.append(
            f"| {condition} | {aggregate['attempt_case_count']} | "
            f"{aggregate['correct_case_count']} | {aggregate['wrong_case_count']} | "
            f"{aggregate['correct_rate']:.1%} | {aggregate['wrong_rate']:.1%} |"
        )

    lines.extend(
        [
            "",
            f"## Room-level 主表（兩個 condition 的 n 均至少 {minimum_n}）",
            "",
            "| Room | Without n | Without correct | Without wrong | Without correct rate | With n | With correct | With wrong | With correct rate | Difference |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    eligible = [
        row
        for row in comparison_rows
        if row["without_attempt_case_count"] >= minimum_n
        and row["with_attempt_case_count"] >= minimum_n
    ]
    eligible.sort(
        key=lambda row: (
            -(row["without_wrong_rate"] or 0.0),
            _room_sort_key(row["room_id"]),
        )
    )
    for row in eligible:
        lines.append(
            f"| {row['room_id']} | {row['without_attempt_case_count']} | "
            f"{row['without_correct_case_count']} | {row['without_wrong_case_count']} | "
            f"{row['without_correct_rate']:.1%} | {row['with_attempt_case_count']} | "
            f"{row['with_correct_case_count']} | {row['with_wrong_case_count']} | "
            f"{row['with_correct_rate']:.1%} | "
            f"{row['with_minus_without_correct_rate_pp']:+.1f} pp |"
        )
    lines.extend(
        [
            "",
            "完整低樣本 room 與 Wrong 的 wrong-room / no-exit / recovered 細項請看 `room_exit_summary.csv`；逐 case 證據請看 `room_exit_attempts.csv`。",
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
    parser.add_argument("--grounding", default=DEFAULT_GROUNDING)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--expected-case-count", type=int, default=715)
    parser.add_argument("--minimum-meeting-summary-n", type=int, default=10)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = {
        "dataset": resolve(args.dataset),
        "without_results": resolve(args.without_results),
        "with_results": resolve(args.with_results),
        "room_graph": resolve(args.room_graph),
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
    grounding_payload = load_json(paths["grounding"])
    raw_grounding = grounding_payload.get("mappings")
    if not isinstance(room_graph, dict):
        raise ValueError("Room graph must be a JSON object.")
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
    attempts: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        for test_id, case in dataset.items():
            attempts.extend(
                build_first_room_attempts(
                    condition=condition,
                    case=case,
                    result=results_by_condition[condition][test_id],
                    room_graph=room_graph,
                    grounding=grounding,
                    paired_outcome=paired_outcomes[test_id],
                )
            )

    summary_rows = summarize_room_attempts(attempts)
    comparison_rows = build_condition_comparison(summary_rows)
    condition_totals = {
        condition: _aggregate_condition(attempts, condition)
        for condition in CONDITIONS
    }
    summary = {
        "schema": "llm_nav_room_level_first_exit_accuracy",
        "schema_version": 1,
        "method": {
            "unit": "first eligible exit attempt per condition-test-room",
            "expected_next": "all equal-length shortest next rooms from ground-truth room to active goal",
            "wrong": "wrong known room or no exit",
            "transit": "ignored in room sequence",
            "repeat_room_visits": "only first eligible visit counted",
            "recovery": "later entry into any originally valid next room after an immediate wrong-room exit",
        },
        "dataset_case_count": len(dataset),
        "attempt_row_count": len(attempts),
        "condition_totals": condition_totals,
        "room_summary": summary_rows,
        "validation": {
            "dataset_case_count_matches_expected": len(dataset)
            == args.expected_case_count,
            "every_attempt_is_binary_correct_or_wrong": all(
                row["correct"] is not row["wrong"] for row in attempts
            ),
            "wrong_equals_wrong_room_plus_no_exit": all(
                row["wrong"] == (row["wrong_room"] or row["no_exit"])
                for row in attempts
            ),
            "one_attempt_per_condition_test_room": len(attempts)
            == len(
                {
                    (row["condition"], row["test_id"], row["current_room_id"])
                    for row in attempts
                }
            ),
        },
        "inputs": {
            label: {"path": str(path), "sha256": sha256_file(path)}
            for label, path in paths.items()
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "room_exit_attempts.csv", attempts)
    write_csv(output_dir / "room_exit_summary.csv", summary_rows)
    write_csv(output_dir / "room_exit_comparison.csv", comparison_rows)
    for condition in CONDITIONS:
        write_csv(
            output_dir / f"{condition}_room_exit_summary.csv",
            build_condition_main_table(summary_rows, condition),
        )
    write_json(output_dir / "summary.json", summary)
    _atomic_write(
        output_dir / "meeting_summary.md",
        build_meeting_summary(
            attempts=attempts,
            comparison_rows=comparison_rows,
            minimum_n=args.minimum_meeting_summary_n,
        ),
    )
    print(f"Wrote room-level exit analysis to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
