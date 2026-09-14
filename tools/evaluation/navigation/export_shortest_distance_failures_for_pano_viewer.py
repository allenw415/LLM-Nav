#!/usr/bin/env python3
"""Export failed navigation cases in a shortest-distance range for Pano Viewer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from export_batch_trajectories_for_pano_viewer import (
    REPO_ROOT,
    build_envelope,
    build_episode,
    failure_category,
    index_unique,
    load_json,
    load_jsonl,
    write_json,
)


DEFAULT_DATASET = REPO_ROOT / (
    "outputs/navigation_paths/floor0_semantic_targets_topology_constrained/"
    "room_count_2to6_ratio_1p0_2p0_per20_sparse485/test_set.jsonl"
)
DEFAULT_RESULTS = REPO_ROOT / (
    "outputs/navigation_evaluation/floor0_semantic_targets_topology_constrained/"
    "room_count_2to6_ratio_1p0_2p0_per20_sparse485/"
    "llm_nav_combined485/results.jsonl"
)
DEFAULT_OUTPUT = REPO_ROOT / (
    "outputs/navigation_evaluation/floor0_semantic_targets_topology_constrained/"
    "room_count_2to6_ratio_1p0_2p0_per20_sparse485/"
    "llm_nav_combined485/pano_viewer_failures_140_180m"
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
    parser.add_argument("--min-distance-m", type=float, default=140.0)
    parser.add_argument("--max-distance-m", type=float, default=180.0)
    parser.add_argument("--bin-width-m", type=float, default=20.0)
    return parser.parse_args()


def distance_bin_label(distance_m: float, width_m: float) -> str:
    lower = int(distance_m // width_m * width_m)
    upper = int(lower + width_m)
    return f"{lower}-{upper} m"


def add_distance_metadata(
    envelope: dict[str, Any], result: dict[str, Any], label: str
) -> None:
    evaluation = envelope["evaluation"]
    evaluation.update(
        {
            "shortest_distance_bin": label,
            "shortest_path_distance_m": result.get("shortest_path_distance_m"),
            "actual_path_distance_m": result.get("actual_path_distance_m"),
            "actual_over_shortest_ratio": result.get("actual_over_shortest_ratio"),
        }
    )


def build_readme(entries: list[dict[str, Any]]) -> str:
    lines = [
        "# Pano Viewer：140–180 m 失敗案例",
        "",
        f"共 {len(entries)} 筆失敗案例。`actual/` 是模型實際路徑；",
        "`reference/` 是資料集最短路徑。請在 Pano Viewer 的 Trajectory 面板逐檔載入。",
        "",
        "| 測資 | 距離區間 | 最短距離 | 起點 → 目標 | 最後房間 | 目標曾到達 | 步數 | 重訪步數 |",
        "|---|---|---:|---|---|---|---:|---:|",
    ]
    for entry in entries:
        target_visited = "是" if entry["target_room_visited"] else "否"
        final_room = entry["final_ground_truth_room_id"] or "—"
        lines.append(
            f"| {entry['test_id']} | {entry['shortest_distance_bin']} | "
            f"{entry['shortest_path_distance_m']:.2f} m | "
            f"{entry['start_room_id']} → {entry['target_room_id']} | "
            f"{final_room} | {target_visited} | "
            f"{entry['movement_steps']}/{entry['step_budget']} | "
            f"{entry['revisited_pano_steps']} |"
        )
    lines.extend(
        [
            "",
            "所有案例的停止原因皆可在 `manifest.json` 與各 actual JSON 的 "
            "`evaluation.stop_reason` 查看。",
            "",
            "## 使用方式",
            "",
            "```bash",
            "python3 tools/pano_viewer/export.py",
            "python3 tools/pano_viewer/server.py",
            "```",
            "",
            "開啟 <http://127.0.0.1:8000> 後，先看 `actual/TESTxxx.json`，",
            "再載入同編號的 `reference/TESTxxx.json` 對照。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    if args.min_distance_m >= args.max_distance_m:
        raise ValueError("min-distance-m must be smaller than max-distance-m")
    if args.bin_width_m <= 0:
        raise ValueError("bin-width-m must be positive")

    cases = index_unique(load_jsonl(args.dataset), key="test_id", source=args.dataset)
    results = load_jsonl(args.results)
    graph = load_json(args.pano_graph)
    grounding_payload = load_json(args.grounding)
    grounding = grounding_payload.get("mappings", {})
    if not isinstance(graph, dict) or not isinstance(grounding, dict):
        raise ValueError("Pano graph and grounding must be JSON objects")

    selected: list[dict[str, Any]] = []
    for result in results:
        distance = result.get("shortest_path_distance_m")
        if (
            result.get("success") is False
            and isinstance(distance, (int, float))
            and args.min_distance_m <= float(distance) < args.max_distance_m
        ):
            selected.append(result)
    selected.sort(key=lambda result: result["test_id"])
    if not selected:
        raise ValueError("No failed results match the requested distance range")

    entries: list[dict[str, Any]] = []
    for result in selected:
        test_id = result["test_id"]
        case = cases.get(test_id)
        if case is None:
            raise ValueError(f"No dataset case found for {test_id}")
        if result.get("path_id") != case.get("path_id"):
            raise ValueError(f"Dataset/result path_id mismatch for {test_id}")

        actual_path = result.get("pano_path")
        reference_path = case.get("shortest_path_panos")
        if not isinstance(actual_path, list) or not actual_path:
            raise ValueError(f"{test_id} result has no valid pano_path")
        if not isinstance(reference_path, list) or not reference_path:
            raise ValueError(f"{test_id} dataset has no valid shortest_path_panos")

        distance = float(result["shortest_path_distance_m"])
        bin_label = distance_bin_label(distance, args.bin_width_m)
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
        actual_envelope = build_envelope(
            case=case,
            result=result,
            episode=actual_episode,
            trajectory_kind="actual",
        )
        reference_envelope = build_envelope(
            case=case,
            result=result,
            episode=reference_episode,
            trajectory_kind="reference",
        )
        add_distance_metadata(actual_envelope, result, bin_label)
        add_distance_metadata(reference_envelope, result, bin_label)

        actual_relative = Path("actual") / f"{test_id}.json"
        reference_relative = Path("reference") / f"{test_id}.json"
        write_json(args.output_dir / actual_relative, actual_envelope)
        write_json(args.output_dir / reference_relative, reference_envelope)

        actual_rooms = actual_episode["ground_truth_room_sequence"]
        acceptable = set(case.get("acceptable_target_room_ids", []))
        entries.append(
            {
                "test_id": test_id,
                "path_id": case["path_id"],
                "query": case["query"],
                "shortest_distance_bin": bin_label,
                "shortest_path_distance_m": distance,
                "actual_path_distance_m": result.get("actual_path_distance_m"),
                "start_room_id": case["start_room_id"],
                "target_room_id": case["target_room_id"],
                "acceptable_target_room_ids": sorted(acceptable),
                "final_ground_truth_room_id": result.get("final_ground_truth_room_id"),
                "final_estimated_room_id": result.get("final_estimated_room_id"),
                "stop_reason": result.get("stop_reason"),
                "movement_steps": result.get("movement_steps"),
                "step_budget": result.get("step_budget"),
                "unique_pano_count": len(set(actual_path)),
                "revisited_pano_steps": len(actual_path) - len(set(actual_path)),
                "actual_ground_truth_room_sequence": actual_rooms,
                "reference_room_sequence": reference_episode["ground_truth_room_sequence"],
                "target_room_visited": not acceptable.isdisjoint(actual_rooms),
                "failure_category": failure_category(
                    case=case, result=result, rooms=actual_rooms
                ),
                "actual_file": actual_relative.as_posix(),
                "reference_file": reference_relative.as_posix(),
            }
        )

    by_bin: dict[str, int] = {}
    for entry in entries:
        label = entry["shortest_distance_bin"]
        by_bin[label] = by_bin.get(label, 0) + 1
    manifest = {
        "schema": "llm_nav_pano_viewer_failure_manifest",
        "schema_version": 1,
        "filter": {
            "success": False,
            "min_shortest_path_distance_m_inclusive": args.min_distance_m,
            "max_shortest_path_distance_m_exclusive": args.max_distance_m,
        },
        "source_dataset": str(args.dataset.resolve()),
        "source_results": str(args.results.resolve()),
        "case_count": len(entries),
        "case_count_by_distance_bin": by_bin,
        "cases": entries,
    }
    write_json(args.output_dir / "manifest.json", manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "README.md").write_text(
        build_readme(entries), encoding="utf-8"
    )
    print(f"Exported {len(entries)} failed cases to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
