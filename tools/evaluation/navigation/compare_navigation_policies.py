from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_DIR = (
    "outputs/navigation_paths/floor0_semantic_targets_topology_constrained/"
    "room_count_2to6_ratio_1p0_2p0_per20_sparse485"
)
EVALUATION_DIR = (
    "outputs/navigation_evaluation/floor0_semantic_targets_topology_constrained/"
    "room_count_2to6_ratio_1p0_2p0_per20_sparse485"
)
DEFAULT_DATASET = f"{DATASET_DIR}/test_set.jsonl"
DEFAULT_BASELINE_RESULTS = (
    f"{EVALUATION_DIR}/llm_nav_combined485_room4_room9_corrected/results.jsonl"
)
DEFAULT_CANDIDATE_RESULTS = f"{EVALUATION_DIR}/geometry_edge_visit_combined485/results.jsonl"
DEFAULT_OUTPUT_DIR = f"{EVALUATION_DIR}/geometry_edge_visit_vs_geometry"
ROOM_COUNTS = (2, 3, 4, 5, 6)
RATIO_BINS = ("1.0-1.2", "1.2-1.4", "1.4-1.6", "1.6-1.8", "1.8-2.0")


def resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate.resolve() if candidate.is_absolute() else (PROJECT_ROOT / candidate).resolve()


def load_jsonl(path: str | Path) -> list[dict]:
    records: list[dict] = []
    resolved = Path(path)
    for line_number, raw_line in enumerate(
        resolved.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw_line.strip():
            continue
        payload = json.loads(raw_line)
        if not isinstance(payload, dict):
            raise ValueError(f"Expected a JSON object at {resolved}:{line_number}.")
        records.append(payload)
    return records


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def records_by_test_id(records: Iterable[dict], *, label: str) -> dict[str, dict]:
    indexed: dict[str, dict] = {}
    for record in records:
        test_id = record.get("test_id")
        if not isinstance(test_id, str) or not test_id:
            raise ValueError(f"{label} contains a record without test_id.")
        if test_id in indexed:
            raise ValueError(f"{label} contains duplicate test_id {test_id}.")
        indexed[test_id] = record
    return indexed


def validate_result_identity(
    dataset: dict[str, dict],
    results: dict[str, dict],
    *,
    label: str,
) -> None:
    unknown = sorted(set(results) - set(dataset))
    if unknown:
        raise ValueError(f"{label} contains unknown test IDs: {unknown}")
    for test_id, result in results.items():
        expected_path_id = dataset[test_id].get("path_id")
        if result.get("path_id") != expected_path_id:
            raise ValueError(
                f"{label} path_id differs for {test_id}: "
                f"{result.get('path_id')} != {expected_path_id}"
            )


def apply_error_replacements(
    candidate_results: dict[str, dict],
    replacement_groups: Iterable[tuple[str, Iterable[dict]]],
) -> tuple[dict[str, dict], list[dict]]:
    effective = dict(candidate_results)
    audit: list[dict] = []
    seen: set[str] = set()
    for source, replacements in replacement_groups:
        for replacement in replacements:
            test_id = replacement.get("test_id")
            if not isinstance(test_id, str) or test_id not in candidate_results:
                raise ValueError(f"Replacement contains unknown candidate test ID: {test_id}")
            if test_id in seen:
                raise ValueError(f"Duplicate replacement for {test_id}.")
            seen.add(test_id)
            original = candidate_results[test_id]
            if not original.get("error"):
                raise ValueError(
                    f"Replacement is only allowed for an original error result: {test_id}"
                )
            if replacement.get("path_id") != original.get("path_id"):
                raise ValueError(f"Replacement path_id differs for {test_id}.")
            if replacement.get("error"):
                raise ValueError(f"Replacement rerun did not complete successfully: {test_id}")
            effective[test_id] = replacement
            audit.append(
                {
                    "test_id": test_id,
                    "replacement_source": source,
                    "original_error": original.get("error"),
                    "replacement_success": bool(replacement.get("success")),
                    "accepted": True,
                }
            )
    return effective, audit


def trajectory_metrics(pano_path: object) -> dict:
    if not isinstance(pano_path, list) or not all(
        isinstance(pano_id, str) and pano_id for pano_id in pano_path
    ):
        raise ValueError("Result has no valid pano_path.")
    edge_counts = Counter(
        tuple(sorted((source, target)))
        for source, target in zip(pano_path, pano_path[1:])
    )
    repeated_edge_traversals = sum(max(count - 1, 0) for count in edge_counts.values())
    repeated_pano_visits = len(pano_path) - len(set(pano_path))
    return {
        "repeated_pano_visits": repeated_pano_visits,
        "repeated_undirected_edge_traversals": repeated_edge_traversals,
        "has_repeated_undirected_edge": repeated_edge_traversals > 0,
    }


def _number(record: dict, key: str) -> float:
    value = record.get(key)
    if not isinstance(value, (int, float)):
        raise ValueError(f"{record.get('test_id')} has no numeric {key}.")
    return float(value)


def paired_case(dataset_record: dict, baseline: dict, candidate: dict) -> dict:
    baseline_trajectory = trajectory_metrics(baseline.get("pano_path"))
    candidate_trajectory = trajectory_metrics(candidate.get("pano_path"))
    baseline_success = bool(baseline.get("success"))
    candidate_success = bool(candidate.get("success"))
    if baseline_success and candidate_success:
        transition = "both_success"
    elif baseline_success:
        transition = "baseline_success_candidate_fail"
    elif candidate_success:
        transition = "baseline_fail_candidate_success"
    else:
        transition = "both_fail"
    row = {
        "test_id": dataset_record["test_id"],
        "path_id": dataset_record["path_id"],
        "visited_room_count": int(dataset_record["visited_room_count"]),
        "ratio_stratum": str(dataset_record["ratio_stratum"]),
        "baseline_success": baseline_success,
        "candidate_success": candidate_success,
        "success_transition": transition,
        "baseline_spl": _number(baseline, "spl"),
        "candidate_spl": _number(candidate, "spl"),
        "baseline_movement_steps": _number(baseline, "movement_steps"),
        "candidate_movement_steps": _number(candidate, "movement_steps"),
        "baseline_actual_path_distance_m": _number(baseline, "actual_path_distance_m"),
        "candidate_actual_path_distance_m": _number(candidate, "actual_path_distance_m"),
    }
    for key, value in baseline_trajectory.items():
        row[f"baseline_{key}"] = value
    for key, value in candidate_trajectory.items():
        row[f"candidate_{key}"] = value
    for key in (
        "spl",
        "movement_steps",
        "actual_path_distance_m",
        "repeated_pano_visits",
        "repeated_undirected_edge_traversals",
    ):
        row[f"delta_{key}"] = row[f"candidate_{key}"] - row[f"baseline_{key}"]
    return row


def build_paired_cases(
    dataset: dict[str, dict],
    baseline: dict[str, dict],
    candidate: dict[str, dict],
) -> list[dict]:
    missing_baseline = sorted(set(candidate) - set(baseline))
    if missing_baseline:
        raise ValueError(f"Baseline is missing candidate test IDs: {missing_baseline}")
    return [
        paired_case(dataset[test_id], baseline[test_id], candidate[test_id])
        for test_id in dataset
        if test_id in candidate
    ]


def _average(records: list[dict], key: str) -> float | None:
    return sum(float(record[key]) for record in records) / len(records) if records else None


def _method_summary(records: list[dict], prefix: str) -> dict:
    count = len(records)
    success_count = sum(bool(record[f"{prefix}_success"]) for record in records)
    repeated_edge_case_count = sum(
        bool(record[f"{prefix}_has_repeated_undirected_edge"]) for record in records
    )
    return {
        "case_count": count,
        "success_count": success_count,
        "success_rate": success_count / count if count else None,
        "average_spl": _average(records, f"{prefix}_spl"),
        "average_movement_steps": _average(records, f"{prefix}_movement_steps"),
        "average_actual_path_distance_m": _average(
            records, f"{prefix}_actual_path_distance_m"
        ),
        "average_repeated_pano_visits": _average(
            records, f"{prefix}_repeated_pano_visits"
        ),
        "average_repeated_undirected_edge_traversals": _average(
            records, f"{prefix}_repeated_undirected_edge_traversals"
        ),
        "repeated_undirected_edge_case_count": repeated_edge_case_count,
        "repeated_undirected_edge_case_rate": (
            repeated_edge_case_count / count if count else None
        ),
    }


def _difference(candidate: float | None, baseline: float | None) -> float | None:
    if candidate is None or baseline is None:
        return None
    return candidate - baseline


def group_summary(records: list[dict]) -> dict:
    baseline = _method_summary(records, "baseline")
    candidate = _method_summary(records, "candidate")
    transitions = Counter(record["success_transition"] for record in records)
    delta = {
        "success_rate": _difference(candidate["success_rate"], baseline["success_rate"]),
        "success_rate_percentage_points": (
            100.0 * (candidate["success_rate"] - baseline["success_rate"])
            if records
            else None
        ),
    }
    for key in (
        "average_spl",
        "average_movement_steps",
        "average_actual_path_distance_m",
        "average_repeated_pano_visits",
        "average_repeated_undirected_edge_traversals",
        "repeated_undirected_edge_case_rate",
    ):
        delta[key] = _difference(candidate[key], baseline[key])
    return {
        "case_count": len(records),
        "baseline": baseline,
        "candidate": candidate,
        "delta_candidate_minus_baseline": delta,
        "success_transitions": {
            key: transitions.get(key, 0)
            for key in (
                "both_success",
                "baseline_fail_candidate_success",
                "baseline_success_candidate_fail",
                "both_fail",
            )
        },
    }


def build_summary(
    dataset: dict[str, dict],
    paired_cases: list[dict],
    *,
    source_paths: dict[str, Path] | None = None,
    replacement_count: int = 0,
) -> dict:
    cells = {
        str(room_count): {
            ratio_bin: group_summary(
                [
                    row
                    for row in paired_cases
                    if row["visited_room_count"] == room_count
                    and row["ratio_stratum"] == ratio_bin
                ]
            )
            for ratio_bin in RATIO_BINS
        }
        for room_count in ROOM_COUNTS
    }
    payload = {
        "schema": "llm_nav_paired_policy_comparison",
        "schema_version": 1,
        "dataset_case_count": len(dataset),
        "completed_candidate_case_count": len(paired_cases),
        "pending_candidate_case_count": len(dataset) - len(paired_cases),
        "replacement_count": replacement_count,
        "overall": group_summary(paired_cases),
        "grid": {
            "room_counts": list(ROOM_COUNTS),
            "ratio_bins": list(RATIO_BINS),
            "cells": cells,
        },
    }
    if source_paths:
        payload["provenance"] = {
            label: {"path": str(path), "sha256": sha256_file(path)}
            for label, path in source_paths.items()
        }
    return payload


def _atomic_write(path: Path, text: str) -> None:
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
    _atomic_write(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    _atomic_write(
        path,
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
    )


def write_csv(path: Path, records: list[dict]) -> None:
    if not records:
        _atomic_write(path, "")
        return
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(records[0]))
    writer.writeheader()
    writer.writerows(records)
    _atomic_write(path, stream.getvalue())


def _rate(value: float | None) -> str:
    return "—" if value is None else f"{100.0 * value:.1f}%"


def _mean(value: float | None, digits: int = 3) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def build_meeting_markdown(summary: dict) -> str:
    overall = summary["overall"]
    baseline = overall["baseline"]
    candidate = overall["candidate"]
    delta = overall["delta_candidate_minus_baseline"]
    transitions = overall["success_transitions"]
    lines = [
        "# Geometry vs Geometry Edge-Visit",
        "",
        f"Completed: {summary['completed_candidate_case_count']}/{summary['dataset_case_count']}; "
        f"pending: {summary['pending_candidate_case_count']}.",
        "",
        "## Overall paired comparison",
        "",
        "| Metric | Geometry | Geometry edge-visit | Delta |",
        "|---|---:|---:|---:|",
        f"| Success | {baseline['success_count']}/{baseline['case_count']} ({_rate(baseline['success_rate'])}) | "
        f"{candidate['success_count']}/{candidate['case_count']} ({_rate(candidate['success_rate'])}) | "
        f"{_mean(delta['success_rate_percentage_points'], 1)} pp |",
        f"| SPL | {_mean(baseline['average_spl'])} | {_mean(candidate['average_spl'])} | {_mean(delta['average_spl'])} |",
        f"| Movement steps | {_mean(baseline['average_movement_steps'], 1)} | {_mean(candidate['average_movement_steps'], 1)} | {_mean(delta['average_movement_steps'], 1)} |",
        f"| Actual path distance (m) | {_mean(baseline['average_actual_path_distance_m'], 1)} | {_mean(candidate['average_actual_path_distance_m'], 1)} | {_mean(delta['average_actual_path_distance_m'], 1)} |",
        f"| Repeated pano visits | {_mean(baseline['average_repeated_pano_visits'], 2)} | {_mean(candidate['average_repeated_pano_visits'], 2)} | {_mean(delta['average_repeated_pano_visits'], 2)} |",
        f"| Repeated undirected-edge traversals | {_mean(baseline['average_repeated_undirected_edge_traversals'], 2)} | {_mean(candidate['average_repeated_undirected_edge_traversals'], 2)} | {_mean(delta['average_repeated_undirected_edge_traversals'], 2)} |",
        f"| Cases with a repeated edge | {baseline['repeated_undirected_edge_case_count']}/{baseline['case_count']} ({_rate(baseline['repeated_undirected_edge_case_rate'])}) | "
        f"{candidate['repeated_undirected_edge_case_count']}/{candidate['case_count']} ({_rate(candidate['repeated_undirected_edge_case_rate'])}) | "
        f"{_mean(100.0 * delta['repeated_undirected_edge_case_rate'], 1)} pp |",
        "",
        "## Success transitions",
        "",
        "| Transition | Cases |",
        "|---|---:|",
        f"| Both success | {transitions['both_success']} |",
        f"| Baseline fail → edge-visit success | {transitions['baseline_fail_candidate_success']} |",
        f"| Baseline success → edge-visit fail | {transitions['baseline_success_candidate_fail']} |",
        f"| Both fail | {transitions['both_fail']} |",
        "",
        "## Per-cell comparison",
        "",
        "| Rooms / Detour ratio | Baseline SR | Edge-visit SR | Δ SR | Baseline SPL | Edge-visit SPL | Δ repeated edges |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for room_count in ROOM_COUNTS:
        for ratio_bin in RATIO_BINS:
            cell = summary["grid"]["cells"][str(room_count)][ratio_bin]
            base = cell["baseline"]
            cand = cell["candidate"]
            change = cell["delta_candidate_minus_baseline"]
            lines.append(
                f"| {room_count} / {ratio_bin} | "
                f"{base['success_count']}/{base['case_count']} ({_rate(base['success_rate'])}) | "
                f"{cand['success_count']}/{cand['case_count']} ({_rate(cand['success_rate'])}) | "
                f"{_mean(change['success_rate_percentage_points'], 1)} pp | "
                f"{_mean(base['average_spl'])} | {_mean(cand['average_spl'])} | "
                f"{_mean(change['average_repeated_undirected_edge_traversals'], 2)} |"
            )
    lines.extend(
        [
            "",
            "Negative trajectory deltas mean the edge-visit policy repeated less movement than the geometry baseline.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a partial or full paired geometry-policy comparison report."
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--baseline-results", default=DEFAULT_BASELINE_RESULTS)
    parser.add_argument("--candidate-results", default=DEFAULT_CANDIDATE_RESULTS)
    parser.add_argument("--replacement-results", action="append", default=[])
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    dataset_path = resolve(args.dataset)
    baseline_path = resolve(args.baseline_results)
    candidate_path = resolve(args.candidate_results)
    replacement_paths = [resolve(path) for path in args.replacement_results]
    output_dir = resolve(args.output_dir)

    dataset = records_by_test_id(load_jsonl(dataset_path), label="dataset")
    baseline = records_by_test_id(load_jsonl(baseline_path), label="baseline results")
    candidate = records_by_test_id(load_jsonl(candidate_path), label="candidate results")
    validate_result_identity(dataset, baseline, label="baseline results")
    validate_result_identity(dataset, candidate, label="candidate results")
    replacement_groups = [
        (str(path), load_jsonl(path)) for path in replacement_paths
    ]
    effective_candidate, replacement_audit = apply_error_replacements(
        candidate, replacement_groups
    )
    validate_result_identity(dataset, effective_candidate, label="effective candidate results")
    paired_cases = build_paired_cases(dataset, baseline, effective_candidate)
    effective_records = [
        effective_candidate[test_id] for test_id in dataset if test_id in effective_candidate
    ]
    summary = build_summary(
        dataset,
        paired_cases,
        source_paths={
            "dataset": dataset_path,
            "baseline_results": baseline_path,
            "candidate_results": candidate_path,
            **{
                f"replacement_results_{index}": path
                for index, path in enumerate(replacement_paths, start=1)
            },
        },
        replacement_count=len(replacement_audit),
    )

    write_json(output_dir / "summary.json", summary)
    write_csv(output_dir / "paired_cases.csv", paired_cases)
    write_jsonl(output_dir / "effective_candidate_results.jsonl", effective_records)
    write_json(output_dir / "replacement_audit.json", replacement_audit)
    _atomic_write(output_dir / "meeting_summary.md", build_meeting_markdown(summary))
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "completed_candidate_case_count": len(paired_cases),
                "pending_candidate_case_count": len(dataset) - len(paired_cases),
                "replacement_count": len(replacement_audit),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
