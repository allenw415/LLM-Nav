from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sa_nav.evaluation.navigation_batch import load_jsonl


STAGES = ("visual", "bayesian", "spatial")
TRANSITION_PAIRS = (("visual", "bayesian"), ("bayesian", "spatial"))
GROUND_TRUTH_OFFSET_DEG = 30.0


def resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate.resolve() if candidate.is_absolute() else (PROJECT_ROOT / candidate).resolve()


def _results_path(path: str | Path) -> Path:
    resolved = resolve(path)
    return resolved / "results.jsonl" if resolved.is_dir() else resolved


def _load_unique_results(path: str | Path) -> list[dict]:
    resolved = _results_path(path)
    records = load_jsonl(resolved)
    seen: set[str] = set()
    for index, record in enumerate(records, start=1):
        test_id = record.get("test_id")
        if not isinstance(test_id, str) or not test_id:
            raise ValueError(f"{resolved}:{index} has no valid test_id")
        if test_id in seen:
            raise ValueError(f"Duplicate test_id in {resolved}: {test_id}")
        seen.add(test_id)
    return records


def apply_error_replacements(
    original_results: list[dict],
    replacement_paths: list[str | Path],
) -> tuple[list[dict], dict]:
    original_by_id = {record["test_id"]: record for record in original_results}
    replacements: dict[str, dict] = {}
    replacement_sources: dict[str, str] = {}
    for path in replacement_paths:
        resolved = _results_path(path)
        for record in _load_unique_results(resolved):
            test_id = record["test_id"]
            if test_id not in original_by_id:
                raise ValueError(f"Unknown replacement test_id: {test_id}")
            if test_id in replacements:
                raise ValueError(f"Duplicate replacement test_id: {test_id}")
            if original_by_id[test_id].get("error") is None:
                raise ValueError(f"Replacement is only allowed for an original error: {test_id}")
            if record.get("error") is not None or record.get("stop_reason") == "error":
                raise ValueError(f"Replacement must be a completed non-error rerun: {test_id}")
            replacements[test_id] = record
            replacement_sources[test_id] = str(resolved)

    effective = [replacements.get(record["test_id"], record) for record in original_results]
    audit = {
        "original_case_count": len(original_results),
        "original_error_count": sum(record.get("error") is not None for record in original_results),
        "replacement_count": len(replacements),
        "remaining_error_count": sum(record.get("error") is not None for record in effective),
        "replacements": [
            {"test_id": test_id, "source": replacement_sources[test_id]}
            for test_id in sorted(replacements)
        ],
    }
    return effective, audit


def flatten_localization_events(results: list[dict]) -> list[dict]:
    rows: list[dict] = []
    seen: set[tuple[str, int, str]] = set()
    for result in results:
        test_id = result["test_id"]
        initial_alignment = result.get("initial_heading_alignment")
        initial_alignment = initial_alignment if isinstance(initial_alignment, dict) else {}
        result_ground_truth_offset = initial_alignment.get("ground_truth_offset_deg")
        if not isinstance(result_ground_truth_offset, (int, float)):
            result_ground_truth_offset = GROUND_TRUTH_OFFSET_DEG
        events = result.get("localization_events", [])
        if not isinstance(events, list):
            raise ValueError(f"localization_events must be a list: {test_id}")
        declared_count = result.get("localization_event_count")
        if isinstance(declared_count, int) and declared_count != len(events):
            raise ValueError(f"Localization event count mismatch: {test_id}")
        for event in events:
            if not isinstance(event, dict):
                raise ValueError(f"Invalid localization event: {test_id}")
            step_index = event.get("step_index")
            pano_id = event.get("pano_id")
            if not isinstance(step_index, int) or not isinstance(pano_id, str) or not pano_id:
                raise ValueError(f"Localization event identity is invalid: {test_id}")
            identity = (test_id, step_index, pano_id)
            if identity in seen:
                raise ValueError(f"Duplicate localization event: {identity}")
            seen.add(identity)
            row = dict(event)
            row["test_id"] = test_id
            row["case_error"] = result.get("error") is not None
            row.setdefault(
                "ground_truth_offset_deg",
                float(result_ground_truth_offset),
            )
            if not row.get("spatial_predicted_room_id"):
                row["spatial_predicted_room_id"] = row.get("bayesian_predicted_room_id")
            rows.append(row)
    return rows


def _correct(row: dict, stage: str) -> bool | None:
    ground_truth = row.get("ground_truth_room_id")
    prediction = row.get(f"{stage}_predicted_room_id")
    if not isinstance(ground_truth, str) or not ground_truth:
        return None
    if not isinstance(prediction, str) or not prediction:
        return None
    return prediction == ground_truth


def _stage_summary(rows: list[dict]) -> list[dict]:
    summary = []
    for stage in STAGES:
        comparable = [value for row in rows if (value := _correct(row, stage)) is not None]
        correct = sum(comparable)
        summary.append(
            {
                "stage": stage,
                "event_count": len(rows),
                "comparable_count": len(comparable),
                "correct_count": correct,
                "accuracy": correct / len(comparable) if comparable else None,
                "coverage": len(comparable) / len(rows) if rows else None,
            }
        )
    return summary


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _timing_summary(rows: list[dict]) -> list[dict]:
    output = []
    for stage in STAGES:
        for metric in ("program_module_ms", "vlm_response_ms", "stage_total_ms"):
            values = []
            for row in rows:
                timing = row.get("localization_timing")
                timing = timing if isinstance(timing, dict) else {}
                stage_timing = timing.get(stage)
                stage_timing = stage_timing if isinstance(stage_timing, dict) else {}
                value = stage_timing.get(metric)
                if isinstance(value, (int, float)):
                    values.append(float(value))
            output.append(
                {
                    "stage": stage,
                    "metric": metric,
                    "count": len(values),
                    "mean_ms": statistics.fmean(values) if values else None,
                    "median_ms": statistics.median(values) if values else None,
                    "p95_ms": _percentile(values, 0.95),
                    "total_ms": sum(values) if values else None,
                }
            )
    return output


def _circular_error_deg(predicted: object, expected: float) -> float | None:
    if not isinstance(predicted, (int, float)):
        return None
    difference = abs((float(predicted) - expected) % 360.0)
    return min(difference, 360.0 - difference)


def _initial_alignment_summary(rows: list[dict]) -> dict:
    initial = [row for row in rows if row.get("step_index") == 0]
    errors = [
        error
        for row in initial
        if (
            error := _circular_error_deg(
                row.get("estimated_offset_deg"),
                float(row.get("ground_truth_offset_deg", GROUND_TRUTH_OFFSET_DEG)),
            )
        )
        is not None
    ]
    ground_truth_offsets = sorted(
        {
            float(row.get("ground_truth_offset_deg", GROUND_TRUTH_OFFSET_DEG))
            for row in initial
            if isinstance(
                row.get("ground_truth_offset_deg", GROUND_TRUTH_OFFSET_DEG),
                (int, float),
            )
        }
    )
    correct_count = sum(math.isclose(error, 0.0, abs_tol=1e-6) for error in errors)
    return {
        "event_count": len(initial),
        "estimated_count": len(errors),
        "coverage": len(errors) / len(initial) if initial else None,
        "correct_count": correct_count,
        "accuracy": correct_count / len(errors) if errors else None,
        "ground_truth_offset_deg": (
            ground_truth_offsets[0] if len(ground_truth_offsets) == 1 else None
        ),
        "ground_truth_offset_values_deg": ground_truth_offsets,
        "error_counts": {
            str(int(error) if float(error).is_integer() else error): sum(
                math.isclose(value, error, abs_tol=1e-6) for value in errors
            )
            for error in sorted(set(errors))
        },
        "failed_count": len(initial) - len(errors),
    }


def _transition_summary(rows: list[dict], source: str, target: str) -> dict:
    counts = Counter()
    for row in rows:
        source_correct = _correct(row, source)
        target_correct = _correct(row, target)
        if source_correct is None or target_correct is None:
            continue
        if not source_correct and target_correct:
            counts["fail_to_success"] += 1
        elif source_correct and not target_correct:
            counts["success_to_fail"] += 1
        elif source_correct:
            counts["both_success"] += 1
        else:
            counts["both_fail"] += 1
    return {
        "source_stage": source,
        "target_stage": target,
        "comparable_count": sum(counts.values()),
        **{key: counts[key] for key in ("fail_to_success", "success_to_fail", "both_success", "both_fail")},
    }


def summarize_events(rows: list[dict], results: list[dict]) -> dict:
    cache_counts = Counter(
        "hit" if row.get("visual_cache_hit") is True
        else "miss" if row.get("visual_cache_hit") is False
        else "unknown"
        for row in rows
    )
    alignment_counts = Counter(
        "applied" if row.get("alignment_applied") is True else "skipped"
        for row in rows
    )
    ground_truth_count = sum(
        isinstance(row.get("ground_truth_room_id"), str) and bool(row.get("ground_truth_room_id"))
        for row in rows
    )
    return {
        "case_count": len(results),
        "error_case_count": sum(result.get("error") is not None for result in results),
        "localization_event_count": len(rows),
        "ground_truth_event_count": ground_truth_count,
        "ground_truth_coverage": ground_truth_count / len(rows) if rows else None,
        "stages": _stage_summary(rows),
        "initial_stages": _stage_summary(
            [row for row in rows if row.get("step_index") == 0]
        ),
        "timing": _timing_summary(rows),
        "initial_timing": _timing_summary(
            [row for row in rows if row.get("step_index") == 0]
        ),
        "initial_alignment": _initial_alignment_summary(rows),
        "transitions": [
            _transition_summary(rows, source, target)
            for source, target in TRANSITION_PAIRS
        ],
        "visual_cache": dict(cache_counts),
        "alignment": dict(alignment_counts),
        "cache_provenance_note": (
            "Cache validation covers schema and candidate room IDs; the legacy cache has no "
            "model, prompt, visual-profile, or image fingerprint."
        ),
    }


def _by_ground_truth_room(rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        room_id = row.get("ground_truth_room_id")
        if isinstance(room_id, str) and room_id:
            grouped[room_id].append(row)
    output = []
    for room_id in sorted(grouped):
        room_rows = grouped[room_id]
        record: dict = {"ground_truth_room_id": room_id, "event_count": len(room_rows)}
        for stage in STAGES:
            values = [value for row in room_rows if (value := _correct(row, stage)) is not None]
            record[f"{stage}_comparable_count"] = len(values)
            record[f"{stage}_correct_count"] = sum(values)
            record[f"{stage}_accuracy"] = sum(values) / len(values) if values else None
        output.append(record)
    return output


def _json_cell(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, (dict, list)) else ""


def _event_csv_rows(rows: list[dict]) -> list[dict]:
    output = []
    for row in rows:
        record = {
            "test_id": row.get("test_id"),
            "step_index": row.get("step_index"),
            "pano_id": row.get("pano_id"),
            "ground_truth_room_id": row.get("ground_truth_room_id"),
            "visual_cache_hit": row.get("visual_cache_hit"),
            "visual_cache_version": row.get("visual_cache_version"),
            "visual_predicted_room_id": row.get("visual_predicted_room_id"),
            "visual_correct": _correct(row, "visual"),
            "bayesian_predicted_room_id": row.get("bayesian_predicted_room_id"),
            "bayesian_correct": _correct(row, "bayesian"),
            "alignment_applied": row.get("alignment_applied"),
            "alignment_skipped_reason": row.get("alignment_skipped_reason"),
            "alignment_predicted_room_id": row.get("alignment_predicted_room_id"),
            "spatial_predicted_room_id": row.get("spatial_predicted_room_id"),
            "spatial_correct": _correct(row, "spatial"),
            "stabilized_room_id": row.get("stabilized_room_id"),
            "pending_room_id": row.get("pending_room_id"),
            "case_error": row.get("case_error"),
            "visual_room_scores": _json_cell(row.get("visual_room_scores")),
            "visual_room_belief": _json_cell(row.get("visual_room_belief")),
            "prior_room_belief": _json_cell(row.get("prior_room_belief")),
            "transition_support": _json_cell(row.get("transition_support")),
            "bayesian_room_belief": _json_cell(row.get("bayesian_room_belief")),
            "alignment_candidate_room_ids": _json_cell(row.get("alignment_candidate_room_ids")),
            "alignment_top_k": _json_cell(row.get("alignment_top_k")),
            "initial_heading_alignment_status": row.get(
                "initial_heading_alignment_status"
            ),
            "heading_alignment_status": row.get("heading_alignment_status"),
            "view_0_geocentric_heading_deg": row.get(
                "view_0_geocentric_heading_deg"
            ),
            "view_0_allocentric_direction": row.get(
                "view_0_allocentric_direction"
            ),
            "estimated_offset_deg": row.get("estimated_offset_deg"),
            "ground_truth_offset_deg": row.get(
                "ground_truth_offset_deg",
                GROUND_TRUTH_OFFSET_DEG,
            ),
            "offset_error_deg": _circular_error_deg(
                row.get("estimated_offset_deg"),
                float(
                    row.get(
                        "ground_truth_offset_deg",
                        GROUND_TRUTH_OFFSET_DEG,
                    )
                ),
            ),
            "localization_timing": _json_cell(row.get("localization_timing")),
        }
        output.append(record)
    return output


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as stream:
        if not fieldnames:
            return
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_report(
    *,
    results_path: str | Path,
    output_dir: str | Path,
    replacement_paths: list[str | Path] | None = None,
) -> dict:
    original = _load_unique_results(results_path)
    effective, replacement_audit = apply_error_replacements(original, replacement_paths or [])
    rows = flatten_localization_events(effective)
    summary = summarize_events(rows, effective)
    summary["replacement_audit"] = replacement_audit
    destination = resolve(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    _write_csv(destination / "localization_events.csv", _event_csv_rows(rows))
    _write_csv(destination / "stage_summary.csv", summary["stages"])
    _write_csv(destination / "initial_stage_summary.csv", summary["initial_stages"])
    _write_csv(destination / "timing_summary.csv", summary["timing"])
    _write_csv(destination / "initial_timing_summary.csv", summary["initial_timing"])
    _write_csv(
        destination / "initial_alignment_events.csv",
        _event_csv_rows([row for row in rows if row.get("step_index") == 0]),
    )
    _write_csv(destination / "by_ground_truth_room.csv", _by_ground_truth_room(rows))
    _write_json(destination / "summary.json", summary)
    _write_json(destination / "replacement_audit.json", replacement_audit)
    with (destination / "effective_results.jsonl").open("w", encoding="utf-8") as stream:
        for record in effective:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize Visual, Bayesian, and Spatial localization stages.")
    parser.add_argument("--results", required=True, help="Candidate results.jsonl or its containing directory.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--replacement-results", action="append", default=[])
    args = parser.parse_args()
    summary = build_report(
        results_path=args.results,
        output_dir=args.output_dir,
        replacement_paths=args.replacement_results,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
