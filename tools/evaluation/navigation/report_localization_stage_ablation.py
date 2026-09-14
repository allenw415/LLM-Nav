from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


CONDITIONS = ("visual", "bayesian", "spatial")


def _results_path(path: str | Path) -> Path:
    resolved = Path(path).resolve()
    return resolved / "results.jsonl" if resolved.is_dir() else resolved


def load_results(path: str | Path) -> list[dict]:
    resolved = _results_path(path)
    records = [
        json.loads(line)
        for line in resolved.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    seen: set[str] = set()
    for index, record in enumerate(records, start=1):
        test_id = record.get("test_id")
        if not isinstance(test_id, str) or not test_id:
            raise ValueError(f"{resolved}:{index} has no valid test_id")
        if test_id in seen:
            raise ValueError(f"Duplicate test_id in {resolved}: {test_id}")
        seen.add(test_id)
    return records


def condition_summary(records: list[dict], *, expected_case_count: int) -> dict:
    valid = [record for record in records if record.get("error") is None]
    successes = [record for record in valid if record.get("success") is True]
    valid_spl = [float(record.get("spl", 0.0)) for record in valid]
    alignments = [
        record["initial_heading_alignment"]
        for record in valid
        if isinstance(record.get("initial_heading_alignment"), dict)
    ]
    estimated_alignments = [
        alignment
        for alignment in alignments
        if isinstance(alignment.get("estimated_offset_deg"), (int, float))
    ]
    correct_alignments = [
        alignment
        for alignment in estimated_alignments
        if alignment.get("offset_correct") is True
    ]
    return {
        "expected_case_count": expected_case_count,
        "case_count": len(records),
        "valid_count": len(valid),
        "error_count": len(records) - len(valid),
        "success_count": len(successes),
        "fixed_denominator": {
            "success_rate": (
                len(successes) / expected_case_count if expected_case_count else None
            ),
            "average_spl": (
                sum(valid_spl) / expected_case_count if expected_case_count else None
            ),
        },
        "valid_only": {
            "success_rate": len(successes) / len(valid) if valid else None,
            "average_spl": statistics.fmean(valid_spl) if valid_spl else None,
        },
        "initial_offset": {
            "alignment_record_count": len(alignments),
            "estimated_count": len(estimated_alignments),
            "correct_count": len(correct_alignments),
            "accuracy": (
                len(correct_alignments) / len(estimated_alignments)
                if estimated_alignments
                else None
            ),
        },
    }


def paired_summary(results_by_condition: dict[str, list[dict]]) -> dict:
    valid_by_condition = {
        condition: {
            record["test_id"]: record
            for record in records
            if record.get("error") is None
        }
        for condition, records in results_by_condition.items()
    }
    common_ids = set.intersection(
        *(set(records) for records in valid_by_condition.values())
    )
    ordered_ids = sorted(common_ids)
    metrics = {}
    for condition in CONDITIONS:
        records = [valid_by_condition[condition][test_id] for test_id in ordered_ids]
        metrics[condition] = {
            "success_count": sum(record.get("success") is True for record in records),
            "success_rate": (
                sum(record.get("success") is True for record in records) / len(records)
                if records
                else None
            ),
            "average_spl": (
                statistics.fmean(float(record.get("spl", 0.0)) for record in records)
                if records
                else None
            ),
        }
    return {
        "common_valid_count": len(ordered_ids),
        "test_ids": ordered_ids,
        "conditions": metrics,
    }


def write_metrics_csv(path: Path, summaries: dict[str, dict]) -> None:
    rows = []
    for condition in CONDITIONS:
        summary = summaries[condition]
        rows.append(
            {
                "condition": condition,
                "case_count": summary["case_count"],
                "valid_count": summary["valid_count"],
                "error_count": summary["error_count"],
                "success_count": summary["success_count"],
                "fixed_success_rate": summary["fixed_denominator"]["success_rate"],
                "fixed_average_spl": summary["fixed_denominator"]["average_spl"],
                "valid_success_rate": summary["valid_only"]["success_rate"],
                "valid_average_spl": summary["valid_only"]["average_spl"],
                "offset_accuracy": summary["initial_offset"]["accuracy"],
            }
        )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_report(
    *,
    visual: str | Path,
    bayesian: str | Path,
    spatial: str | Path,
    output_dir: str | Path,
    expected_case_count: int = 715,
) -> dict:
    results_by_condition = {
        "visual": load_results(visual),
        "bayesian": load_results(bayesian),
        "spatial": load_results(spatial),
    }
    summaries = {
        condition: condition_summary(
            results_by_condition[condition],
            expected_case_count=expected_case_count,
        )
        for condition in CONDITIONS
    }
    payload = {
        "schema": "localization_stage_navigation_ablation",
        "schema_version": 1,
        "expected_case_count": expected_case_count,
        "conditions": summaries,
        "paired_common_valid": paired_summary(results_by_condition),
    }
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_metrics_csv(destination / "condition_metrics.csv", summaries)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--visual", required=True)
    parser.add_argument("--bayesian", required=True)
    parser.add_argument("--spatial", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-case-count", type=int, default=715)
    args = parser.parse_args()
    payload = build_report(
        visual=args.visual,
        bayesian=args.bayesian,
        spatial=args.spatial,
        output_dir=args.output_dir,
        expected_case_count=args.expected_case_count,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
