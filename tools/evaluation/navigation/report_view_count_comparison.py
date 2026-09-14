from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from st_nav.evaluation.navigation_batch import load_jsonl


CONDITIONS = ("without_trajectory", "with_trajectory")
STAGES = ("visual", "bayesian", "spatial")


def resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate.resolve() if candidate.is_absolute() else (PROJECT_ROOT / candidate).resolve()


def _results_by_id(root: Path, condition: str) -> dict[str, dict]:
    path = root / condition / "results.jsonl"
    records = load_jsonl(path)
    by_id: dict[str, dict] = {}
    for record in records:
        test_id = record.get("test_id")
        if not isinstance(test_id, str) or not test_id:
            raise ValueError(f"Invalid test_id in {path}")
        if test_id in by_id:
            raise ValueError(f"Duplicate test_id in {path}: {test_id}")
        by_id[test_id] = record
    return by_id


def _valid(record: dict) -> bool:
    return record.get("error") is None and record.get("stop_reason") != "error"


def _initial_event(record: dict) -> dict:
    events = record.get("localization_events")
    if not isinstance(events, list):
        return {}
    return next(
        (
            event
            for event in events
            if isinstance(event, dict) and event.get("step_index") == 0
        ),
        {},
    )


def _stage_correct(record: dict, stage: str) -> bool | None:
    event = _initial_event(record)
    ground_truth = event.get("ground_truth_room_id")
    predicted = event.get(f"{stage}_predicted_room_id")
    if not isinstance(ground_truth, str) or not ground_truth:
        return None
    if not isinstance(predicted, str) or not predicted:
        return None
    return predicted == ground_truth


def _offset_values(record: dict) -> tuple[float | None, float | None, float | None]:
    alignment = record.get("initial_heading_alignment")
    alignment = alignment if isinstance(alignment, dict) else {}
    predicted = alignment.get("estimated_offset_deg")
    expected = alignment.get("ground_truth_offset_deg", 30.0)
    if not isinstance(predicted, (int, float)) or not isinstance(expected, (int, float)):
        return None, None, None
    difference = abs((float(predicted) - float(expected)) % 360.0)
    error = min(difference, 360.0 - difference)
    return float(predicted), float(expected), error


def _metrics(record: dict) -> dict:
    predicted_offset, expected_offset, offset_error = _offset_values(record)
    return {
        "success": record.get("success") if isinstance(record.get("success"), bool) else None,
        "spl": float(record["spl"]) if isinstance(record.get("spl"), (int, float)) else None,
        "detour_ratio": (
            float(record["actual_over_shortest_ratio"])
            if isinstance(record.get("actual_over_shortest_ratio"), (int, float))
            else None
        ),
        **{f"initial_{stage}_correct": _stage_correct(record, stage) for stage in STAGES},
        "offset_available": predicted_offset is not None,
        "offset_exact": (
            math.isclose(offset_error, 0.0, abs_tol=1e-6)
            if offset_error is not None
            else None
        ),
        "predicted_offset_deg": predicted_offset,
        "ground_truth_offset_deg": expected_offset,
        "offset_error_deg": offset_error,
    }


def _boolean_summary(rows: list[dict], metric: str) -> dict:
    comparable = [
        row
        for row in rows
        if isinstance(row["eight"].get(metric), bool)
        and isinstance(row["four"].get(metric), bool)
    ]
    eight_correct = sum(row["eight"][metric] for row in comparable)
    four_correct = sum(row["four"][metric] for row in comparable)
    transitions = {
        "both_true": sum(row["eight"][metric] and row["four"][metric] for row in comparable),
        "eight_only_true": sum(row["eight"][metric] and not row["four"][metric] for row in comparable),
        "four_only_true": sum(not row["eight"][metric] and row["four"][metric] for row in comparable),
        "both_false": sum(not row["eight"][metric] and not row["four"][metric] for row in comparable),
    }
    count = len(comparable)
    eight_rate = eight_correct / count if count else None
    four_rate = four_correct / count if count else None
    return {
        "paired_comparable_count": count,
        "eight_view_true_count": eight_correct,
        "eight_view_rate": eight_rate,
        "four_view_true_count": four_correct,
        "four_view_rate": four_rate,
        "four_minus_eight_rate": (
            four_rate - eight_rate
            if four_rate is not None and eight_rate is not None
            else None
        ),
        "paired_transitions": transitions,
    }


def _continuous_summary(rows: list[dict], metric: str) -> dict:
    comparable = [
        row
        for row in rows
        if isinstance(row["eight"].get(metric), (int, float))
        and isinstance(row["four"].get(metric), (int, float))
    ]
    eight_values = [float(row["eight"][metric]) for row in comparable]
    four_values = [float(row["four"][metric]) for row in comparable]
    differences = [four - eight for four, eight in zip(four_values, eight_values)]
    return {
        "paired_comparable_count": len(comparable),
        "eight_view_mean": statistics.fmean(eight_values) if eight_values else None,
        "four_view_mean": statistics.fmean(four_values) if four_values else None,
        "mean_paired_difference_four_minus_eight": (
            statistics.fmean(differences) if differences else None
        ),
    }


def build_report(*, eight_root: Path, four_root: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema": "view_count_comparison",
        "schema_version": 1,
        "eight_view_root": str(eight_root),
        "four_view_root": str(four_root),
        "comparison_policy": "Per condition, metrics use only test ids with non-error results in both view settings.",
        "conditions": {},
    }
    csv_rows: list[dict] = []
    for condition in CONDITIONS:
        eight_by_id = _results_by_id(eight_root, condition)
        four_by_id = _results_by_id(four_root, condition)
        common_ids = sorted(set(eight_by_id) & set(four_by_id))
        paired_ids = [
            test_id
            for test_id in common_ids
            if _valid(eight_by_id[test_id]) and _valid(four_by_id[test_id])
        ]
        paired_rows = []
        for test_id in paired_ids:
            eight_metrics = _metrics(eight_by_id[test_id])
            four_metrics = _metrics(four_by_id[test_id])
            paired_rows.append(
                {"test_id": test_id, "eight": eight_metrics, "four": four_metrics}
            )
            csv_record = {"condition": condition, "test_id": test_id}
            for metric, value in eight_metrics.items():
                csv_record[f"eight_{metric}"] = value
            for metric, value in four_metrics.items():
                csv_record[f"four_{metric}"] = value
            csv_rows.append(csv_record)

        boolean_metrics = (
            "success",
            "initial_visual_correct",
            "initial_bayesian_correct",
            "initial_spatial_correct",
            "offset_available",
            "offset_exact",
        )
        continuous_metrics = ("spl", "detour_ratio", "offset_error_deg")
        summary["conditions"][condition] = {
            "eight_view_case_count": len(eight_by_id),
            "four_view_case_count": len(four_by_id),
            "common_test_id_count": len(common_ids),
            "eight_view_error_count": sum(not _valid(record) for record in eight_by_id.values()),
            "four_view_error_count": sum(not _valid(record) for record in four_by_id.values()),
            "paired_api_valid_count": len(paired_ids),
            "boolean_metrics": {
                metric: _boolean_summary(paired_rows, metric)
                for metric in boolean_metrics
            },
            "continuous_metrics": {
                metric: _continuous_summary(paired_rows, metric)
                for metric in continuous_metrics
            },
        }

    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    fieldnames = list(csv_rows[0]) if csv_rows else []
    with (output_dir / "per_case.csv").open("w", encoding="utf-8", newline="") as stream:
        if fieldnames:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare paired 8-view and 4-view navigation results.")
    parser.add_argument("--eight-view-root", required=True)
    parser.add_argument("--four-view-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    summary = build_report(
        eight_root=resolve(args.eight_view_root),
        four_root=resolve(args.four_view_root),
        output_dir=resolve(args.output_dir),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
