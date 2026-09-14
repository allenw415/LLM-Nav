from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter
from pathlib import Path


def load_results(path: str | Path) -> list[dict]:
    resolved = Path(path).resolve()
    if resolved.is_dir():
        resolved = resolved / "results.jsonl"
    records = [
        json.loads(line)
        for line in resolved.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ids = [record.get("test_id") for record in records]
    if any(not isinstance(test_id, str) or not test_id for test_id in ids):
        raise ValueError(f"Invalid test_id in {resolved}")
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate test_id in {resolved}")
    return records


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def stats(values: list[float]) -> dict:
    return {
        "count": len(values),
        "mean": statistics.fmean(values) if values else None,
        "median": statistics.median(values) if values else None,
        "p95": percentile(values, 0.95),
        "total": sum(values) if values else None,
    }


def condition_summary(records: list[dict]) -> dict:
    completed = [record for record in records if record.get("error") is None]
    successes = [record for record in completed if record.get("success") is True]
    alignments = [
        record.get("initial_heading_alignment", {})
        for record in completed
        if isinstance(record.get("initial_heading_alignment"), dict)
    ]
    offset_errors = [
        float(alignment["circular_offset_error_deg"])
        for alignment in alignments
        if isinstance(alignment.get("circular_offset_error_deg"), (int, float))
    ]
    timing_records = [
        record.get("episode_timing", {})
        for record in completed
        if isinstance(record.get("episode_timing"), dict)
    ]
    module_names = sorted(
        {
            module
            for timing in timing_records
            for module in timing.get("modules", {})
            if isinstance(module, str)
        }
    )
    module_timing = {}
    for module in module_names:
        module_timing[module] = {}
        for metric in ("total_wall_ms", "vlm_response_ms", "program_module_ms"):
            values = []
            for timing in timing_records:
                module_record = timing.get("modules", {}).get(module, {})
                value = module_record.get(metric) if isinstance(module_record, dict) else None
                if isinstance(value, (int, float)):
                    values.append(float(value))
            module_timing[module][metric] = stats(values)
    return {
        "case_count": len(records),
        "completed_count": len(completed),
        "error_count": len(records) - len(completed),
        "success_count": len(successes),
        "success_rate": len(successes) / len(completed) if completed else None,
        "average_spl": statistics.fmean(
            float(record.get("spl", 0.0)) for record in completed
        )
        if completed
        else None,
        "offset": {
            "estimated_count": len(offset_errors),
            "coverage": len(offset_errors) / len(completed) if completed else None,
            "correct_count": sum(math.isclose(error, 0.0, abs_tol=1e-6) for error in offset_errors),
            "accuracy": (
                sum(math.isclose(error, 0.0, abs_tol=1e-6) for error in offset_errors)
                / len(offset_errors)
                if offset_errors
                else None
            ),
            "error_counts": dict(Counter(str(error) for error in offset_errors)),
        },
        "episode_timing": {
            metric: stats(
                [
                    float(timing[metric])
                    for timing in timing_records
                    if isinstance(timing.get(metric), (int, float))
                ]
            )
            for metric in (
                "end_to_end_wall_ms",
                "episode_runner_wall_ms",
                "total_vlm_response_ms",
                "total_program_module_ms",
            )
        },
        "module_timing": module_timing,
    }


def paired_summary(without: list[dict], with_trajectory: list[dict]) -> tuple[dict, list[dict]]:
    without_by_id = {record["test_id"]: record for record in without}
    with_by_id = {record["test_id"]: record for record in with_trajectory}
    common_ids = sorted(set(without_by_id) & set(with_by_id))
    rows = []
    transitions = Counter()
    strata = Counter()
    for test_id in common_ids:
        baseline = without_by_id[test_id]
        candidate = with_by_id[test_id]
        baseline_success = baseline.get("success") is True
        candidate_success = candidate.get("success") is True
        if baseline_success and candidate_success:
            transitions["both_success"] += 1
        elif baseline_success:
            transitions["only_without_success"] += 1
        elif candidate_success:
            transitions["only_with_success"] += 1
        else:
            transitions["both_fail"] += 1
        baseline_offset = baseline.get("initial_heading_alignment", {})
        candidate_offset = candidate.get("initial_heading_alignment", {})
        baseline_correct = baseline_offset.get("offset_correct") is True
        candidate_correct = candidate_offset.get("offset_correct") is True
        stratum = (
            "both_offset_correct"
            if baseline_correct and candidate_correct
            else "only_without_offset_correct"
            if baseline_correct
            else "only_with_offset_correct"
            if candidate_correct
            else "both_offset_wrong_or_failed"
        )
        strata[stratum] += 1
        rows.append(
            {
                "test_id": test_id,
                "without_success": baseline_success,
                "with_success": candidate_success,
                "without_offset_deg": baseline_offset.get("estimated_offset_deg"),
                "with_offset_deg": candidate_offset.get("estimated_offset_deg"),
                "offset_stratum": stratum,
            }
        )
    return {
        "common_case_count": len(common_ids),
        "success_transitions": dict(transitions),
        "offset_strata": dict(strata),
    }, rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--without-trajectory", required=True)
    parser.add_argument("--with-trajectory", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    without = load_results(args.without_trajectory)
    with_trajectory = load_results(args.with_trajectory)
    paired, paired_rows = paired_summary(without, with_trajectory)
    payload = {
        "schema": "initial_offset_navigation_experiment",
        "schema_version": 1,
        "without_trajectory": condition_summary(without),
        "with_trajectory": condition_summary(with_trajectory),
        "paired": paired,
    }
    destination = Path(args.output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_csv(destination / "paired_cases.csv", paired_rows)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
