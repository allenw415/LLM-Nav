from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sa_nav import resolve_model_environment, resolve_task_num_ctx  # noqa: E402
from sa_nav.cli._common import load_json, render_json, resolve_project_path  # noqa: E402
from sa_nav.cli.run_spatial_alignment import (  # noqa: E402
    angular_error_deg,
    expected_orientation_from_manifest,
    load_manifest_observation,
    response_diagnostics,
)
from sa_nav.common.room_profiles import preferred_room_graph_path  # noqa: E402
from sa_nav.experiments.map_grounded_alignment import (  # noqa: E402
    MapGroundedSpatialAlignmentExperiment,
    render_semantic_map,
    validate_map_regions,
)
from sa_nav.perception.renderer import sanitize_name  # noqa: E402


MODEL_ENV = resolve_model_environment(
    default_model="gemma-4-31b-it",
    default_api_base="https://generativelanguage.googleapis.com/v1beta/openai",
    default_api_kind="chat_completions",
)
DEFAULT_LLM_NUM_CTX = resolve_task_num_ctx(
    "localization",
    fallback_num_ctx=MODEL_ENV.num_ctx,
    default_num_ctx=16384,
)
SOURCE_SUMMARY_DEFAULT = (
    "outputs/localization_evaluation/proportional_gallery_three_stage_gemini/summary.json"
)
SAMPLING_MODES = ("change-balanced-smoke", "source-all")
SMOKE_TARGETS = {
    "regression": 4,
    "correction": 8,
    "unresolved": 4,
    "stable_correct": 4,
}
SPECIFICITIES = {"specific", "exact"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one north-up map plus eight panorama views using Bayesian Top-5 "
            "candidates from an existing three-stage localization summary."
        )
    )
    parser.add_argument("--source-summary", default=SOURCE_SUMMARY_DEFAULT)
    parser.add_argument("--sampling-mode", choices=SAMPLING_MODES, default="change-balanced-smoke")
    parser.add_argument("--auto-source-all-on-pass", action="store_true")
    parser.add_argument("--artifacts-dir", default="dataset/sites/british_museum/normalized")
    parser.add_argument("--render-output-dir", default="renders/room_grounding_fov90")
    parser.add_argument("--map-image", default="dataset/sites/british_museum/maps/level_0_allocentric.png")
    parser.add_argument("--map-regions", default="dataset/sites/british_museum/maps/level_0_room_regions.json")
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--delay", type=float, default=0.0)
    parser.add_argument("--quota-retry-delay", type=float, default=0.0)
    parser.add_argument("--max-quota-retries", type=int, default=0)
    parser.add_argument("--max-http-retries", type=int, default=1)
    parser.add_argument("--validation-retries", type=int, default=0)
    parser.add_argument("--llm-model", default=MODEL_ENV.model_name or "gemma-4-31b-it")
    parser.add_argument("--llm-api-key")
    parser.add_argument("--llm-api-kind", default=MODEL_ENV.api_kind or "chat_completions")
    parser.add_argument("--llm-api-base", default=MODEL_ENV.api_base)
    parser.add_argument("--llm-timeout", type=float, default=MODEL_ENV.request_timeout or 300.0)
    parser.add_argument("--llm-num-ctx", type=int, default=DEFAULT_LLM_NUM_CTX)
    parser.add_argument("--llm-max-output-tokens", type=int, default=1024)
    parser.add_argument(
        "--output-dir",
        default="outputs/spatial_alignment_map_grounded_bayesian_top5",
    )
    parser.add_argument("--json", action="store_true")
    return parser


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_json(payload), encoding="utf-8")


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def _stage_top1(case: dict, stage: str) -> str | None:
    payload = case.get(stage)
    value = payload.get("top1") if isinstance(payload, dict) else None
    return str(value) if isinstance(value, str) and value else None


def source_change_category(case: dict) -> str:
    expected = str(case.get("expected_room_id"))
    bayesian_correct = _stage_top1(case, "prior_fused") == expected
    spatial_correct = _stage_top1(case, "spatial_aligned") == expected
    if bayesian_correct and spatial_correct:
        return "stable_correct"
    if bayesian_correct:
        return "regression"
    if spatial_correct:
        return "correction"
    return "unresolved"


def _sorted_cases(cases: list[dict]) -> list[dict]:
    return sorted(cases, key=lambda case: (str(case.get("expected_room_id")), str(case.get("pano_id"))))


def _diverse_take(cases: list[dict], count: int) -> list[dict]:
    ordered = _sorted_cases(cases)
    selected: list[dict] = []
    used_rooms: set[str] = set()
    for case in ordered:
        room_id = str(case.get("expected_room_id"))
        if room_id in used_rooms:
            continue
        selected.append(case)
        used_rooms.add(room_id)
        if len(selected) == count:
            return selected
    selected_ids = {str(case.get("pano_id")) for case in selected}
    for case in ordered:
        if str(case.get("pano_id")) in selected_ids:
            continue
        selected.append(case)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError(f"Requested {count} cases but only {len(selected)} were available.")
    return selected


def select_source_cases(source_results: list[dict], sampling_mode: str) -> list[dict]:
    usable = [
        dict(case)
        for case in source_results
        if isinstance(case, dict)
        and isinstance(case.get("pano_id"), str)
        and isinstance(case.get("expected_room_id"), str)
    ]
    if sampling_mode == "source-all":
        return _sorted_cases(usable)
    if sampling_mode != "change-balanced-smoke":
        raise ValueError(f"Unsupported sampling mode: {sampling_mode}")
    buckets: dict[str, list[dict]] = defaultdict(list)
    for case in usable:
        buckets[source_change_category(case)].append(case)
    if len(buckets["regression"]) != SMOKE_TARGETS["regression"]:
        raise ValueError(
            "The approved Level 0 smoke set expects exactly four source regressions; "
            f"found {len(buckets['regression'])}."
        )
    selected = _sorted_cases(buckets["regression"])
    for category in ("correction", "unresolved", "stable_correct"):
        selected.extend(_diverse_take(buckets[category], SMOKE_TARGETS[category]))
    return _sorted_cases(selected)


def bayesian_top5(case: dict) -> list[str]:
    prior = case.get("prior_fused")
    raw = prior.get("top5") if isinstance(prior, dict) else None
    values: list[str] = []
    for value in raw if isinstance(raw, list) else []:
        if isinstance(value, str) and value and value not in values:
            values.append(value)
    if len(values) != 5:
        raise ValueError(f"{case.get('pano_id')} does not contain five unique Bayesian candidates.")
    return sorted(values)


def source_inventory(cases: list[dict], region_payload: dict) -> dict:
    map_rooms = region_payload.get("rooms") if isinstance(region_payload, dict) else None
    map_room_ids = set(map_rooms) if isinstance(map_rooms, dict) else set()
    invalid_candidate_count: list[dict] = []
    candidates_missing_from_map: list[dict] = []
    missing_manifests: list[str] = []
    recall_count = 0
    ready_count = 0
    for case in cases:
        pano_id = str(case["pano_id"])
        prior = case.get("prior_fused")
        raw = prior.get("top5") if isinstance(prior, dict) else None
        candidates = list(dict.fromkeys(value for value in raw if isinstance(value, str))) if isinstance(raw, list) else []
        if str(case["expected_room_id"]) in candidates:
            recall_count += 1
        missing_rooms = sorted(set(candidates) - map_room_ids)
        if len(candidates) != 5:
            invalid_candidate_count.append(
                {"pano_id": pano_id, "expected_room_id": case["expected_room_id"], "candidate_count": len(candidates)}
            )
        if missing_rooms:
            candidates_missing_from_map.append(
                {"pano_id": pano_id, "expected_room_id": case["expected_room_id"], "room_ids": missing_rooms}
            )
        manifest_value = case.get("manifest_path")
        if not isinstance(manifest_value, str) or not resolve_project_path(Path(manifest_value)).is_file():
            missing_manifests.append(pano_id)
        if len(candidates) == 5 and not missing_rooms and pano_id not in missing_manifests:
            ready_count += 1
    return {
        "selected_count": len(cases),
        "ready_count": ready_count,
        "candidate_recall_at_5": recall_count / len(cases) if cases else None,
        "invalid_candidate_count_cases": invalid_candidate_count,
        "candidates_missing_from_map_cases": candidates_missing_from_map,
        "missing_manifest_pano_ids": missing_manifests,
    }


def map_eligible_source_cases(source_results: list[dict], region_payload: dict) -> list[dict]:
    map_rooms = region_payload.get("rooms") if isinstance(region_payload, dict) else None
    map_room_ids = set(map_rooms) if isinstance(map_rooms, dict) else set()
    eligible = []
    for case in source_results:
        if not isinstance(case, dict):
            continue
        prior = case.get("prior_fused")
        raw = prior.get("top5") if isinstance(prior, dict) else None
        candidates = list(dict.fromkeys(value for value in raw if isinstance(value, str))) if isinstance(raw, list) else []
        if len(candidates) == 5 and set(candidates) <= map_room_ids:
            eligible.append(case)
    return eligible


def _winning_hypothesis(result: dict) -> tuple[dict | None, dict | None]:
    predicted = result.get("predicted_room_id")
    for hypothesis in result.get("candidate_hypotheses", []):
        if isinstance(hypothesis, dict) and hypothesis.get("room_id") == predicted:
            selected = hypothesis.get("selected_view_0_allocentric_direction")
            for rotation in hypothesis.get("rotation_tests", []):
                if isinstance(rotation, dict) and rotation.get("view_0_allocentric_direction") == selected:
                    return hypothesis, rotation
            return hypothesis, None
    return None, None


def independent_support_directions(result: dict) -> list[str]:
    directions: set[str] = set()
    hypothesis, _ = _winning_hypothesis(result)
    if isinstance(hypothesis, dict):
        for check in hypothesis.get("directional_neighbor_checks", []):
            if not isinstance(check, dict):
                continue
            if check.get("status") == "specific_match" and check.get("evidence_specificity") in SPECIFICITIES:
                direction = check.get("allocentric_direction")
                if isinstance(direction, str):
                    directions.add(direction)
    for trace in result.get("map_alignment_trace", []):
        if not isinstance(trace, dict):
            continue
        if trace.get("evidence_type") not in {"neighbor", "map_layout", "transition"}:
            continue
        if trace.get("evidence_specificity") not in SPECIFICITIES:
            continue
        direction = trace.get("allocentric_direction")
        if isinstance(direction, str):
            directions.add(direction)
    return sorted(directions)


def gated_map_decision(result: dict, bayesian_room_id: str | None) -> dict:
    raw_room_id = result.get("predicted_room_id")
    comparison = result.get("top1_vs_top2")
    margin = comparison.get("margin") if isinstance(comparison, dict) else None
    _, rotation = _winning_hypothesis(result)
    layout_score = rotation.get("directional_layout") if isinstance(rotation, dict) else None
    contradiction_penalty = rotation.get("contradiction_penalty") if isinstance(rotation, dict) else None
    support_directions = independent_support_directions(result)
    checks = {
        "margin_at_least_2": isinstance(margin, (int, float)) and float(margin) >= 2.0,
        "layout_at_least_2": isinstance(layout_score, (int, float)) and float(layout_score) >= 2.0,
        "contradiction_at_most_1": (
            isinstance(contradiction_penalty, (int, float)) and float(contradiction_penalty) <= 1.0
        ),
        "two_independent_directions": len(support_directions) >= 2,
    }
    differs = isinstance(raw_room_id, str) and raw_room_id != bayesian_room_id
    eligible = all(checks.values())
    override_applied = differs and eligible
    final_room_id = raw_room_id if (not differs or override_applied) else bayesian_room_id
    return {
        "raw_map_room_id": raw_room_id,
        "bayesian_room_id": bayesian_room_id,
        "final_room_id": final_room_id,
        "override_attempted": differs,
        "override_applied": override_applied,
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "top1_top2_margin": margin,
        "directional_layout": layout_score,
        "contradiction_penalty": contradiction_penalty,
        "support_directions": support_directions,
    }


def _load_resume_records(path: Path) -> dict[str, dict]:
    if not path.is_file():
        return {}
    records: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        pano_id = record.get("pano_id")
        if isinstance(pano_id, str):
            records[pano_id] = record
    return records


def resumable_completed_panos(records: dict[str, dict]) -> set[str]:
    return {pano_id for pano_id, record in records.items() if record.get("status") == "completed"}


def _usage_values(usage: object) -> tuple[int | None, int | None]:
    if not isinstance(usage, dict):
        return None, None
    input_value = usage.get("input_tokens", usage.get("promptTokenCount", usage.get("prompt_tokens")))
    output_value = usage.get("output_tokens", usage.get("candidatesTokenCount", usage.get("completion_tokens")))
    return (
        int(input_value) if isinstance(input_value, (int, float)) else None,
        int(output_value) if isinstance(output_value, (int, float)) else None,
    )


def _case_record_base(case: dict, candidates: list[str]) -> dict:
    expected = str(case["expected_room_id"])
    bayes = _stage_top1(case, "prior_fused")
    spatial = _stage_top1(case, "spatial_aligned")
    return {
        "status": "error",
        "pano_id": str(case["pano_id"]),
        "expected_room_id": expected,
        "source_change_category": source_change_category(case),
        "candidate_room_ids": candidates,
        "candidate_contains_expected_room": expected in candidates,
        "bayesian_top1": bayes,
        "bayesian_correct": bayes == expected,
        "current_spatial_top1": spatial,
        "current_spatial_correct": spatial == expected,
        "source_manifest_path": case.get("manifest_path"),
    }


def _run_case(
    *,
    case: dict,
    candidates: list[str],
    room_graph: dict[str, dict],
    region_payload: dict,
    map_image_path: Path,
    output_dir: Path,
    experiment: MapGroundedSpatialAlignmentExperiment,
) -> dict:
    record = _case_record_base(case, candidates)
    pano_id = str(case["pano_id"])
    started = time.perf_counter()
    try:
        manifest_value = case.get("manifest_path")
        if not isinstance(manifest_value, str):
            raise ValueError("Source case is missing manifest_path.")
        manifest_path = resolve_project_path(Path(manifest_value))
        manifest, observation, views = load_manifest_observation(manifest_path=manifest_path, pano_id=pano_id)
        if manifest.get("heading_mode") != "museum":
            raise ValueError("Map-grounded alignment requires heading_mode=museum.")
        semantic_map_path = output_dir / "maps" / f"{sanitize_name(pano_id)}.png"
        map_diagnostics = render_semantic_map(
            base_map_path=map_image_path,
            region_payload=region_payload,
            room_graph=room_graph,
            candidate_room_ids=candidates,
            output_path=semantic_map_path,
        )
        wrapped = experiment.run_once(
            observation=observation,
            manifest=manifest,
            candidate_room_ids=candidates,
            semantic_map_path=semantic_map_path,
        )
        raw_response = wrapped.pop("alignment_response_payload", None)
        top1_source = wrapped.get("top1_source")
        audit_warnings = wrapped.get("audit_warnings")
        winning_audit_complete = wrapped.get("winning_hypothesis_audit_complete") is True
        ranking_complete = wrapped.get("ranking_complete") is True
        spatial_reasoning_complete = wrapped.get("spatial_reasoning_complete") is True
        result = wrapped.get("result")
        if not isinstance(result, dict):
            raise ValueError("Map-grounded response did not contain a result object.")
        raw_map_top1 = result.get("predicted_room_id")
        gate = gated_map_decision(result, record["bayesian_top1"])
        gated_top1 = gate.get("final_room_id")
        expected_direction, expected_offset = expected_orientation_from_manifest(manifest)
        predicted_direction = result.get("view_0_allocentric_direction")
        predicted_offset = result.get("geocentric_to_allocentric_offset_deg")
        usage = response_diagnostics(raw_response).get("usage")
        record.update(
            {
                "status": "completed",
                "views": views,
                "map": map_diagnostics,
                "raw_map_top1": raw_map_top1,
                "raw_map_correct": raw_map_top1 == record["expected_room_id"],
                "gated_map_top1": gated_top1,
                "gated_map_correct": gated_top1 == record["expected_room_id"],
                "gate": gate,
                "predicted_view_0_allocentric_direction": predicted_direction,
                "orientation_correct": predicted_direction == expected_direction if expected_direction else None,
                "offset_error_deg": angular_error_deg(predicted_offset, expected_offset),
                "usage": usage,
                "result": result,
                "decision_valid": True,
                "top1_valid": True,
                "top1_source": top1_source,
                "ranking_complete": ranking_complete,
                "spatial_reasoning_complete": spatial_reasoning_complete,
                "winning_hypothesis_audit_complete": winning_audit_complete,
                "audit_warnings": audit_warnings if isinstance(audit_warnings, list) else [],
                "candidate_hypothesis_coverage_count": len(result.get("candidate_hypotheses", [])),
                "candidate_hypothesis_full_coverage": len(result.get("candidate_hypotheses", [])) == len(candidates),
            }
        )
    except Exception as exc:
        record["error"] = {"type": type(exc).__name__, "message": str(exc)}
        if experiment.last_parsed_output is not None:
            record["invalid_result"] = experiment.last_parsed_output
        if experiment.last_validation_errors:
            record["validation_errors"] = list(experiment.last_validation_errors)
        if experiment.last_normalization_changes:
            record["normalization_changes"] = list(experiment.last_normalization_changes)
    record["latency_seconds"] = round(time.perf_counter() - started, 6)
    return record


def _is_quota_error(record: dict) -> bool:
    error = record.get("error")
    message = error.get("message") if isinstance(error, dict) else None
    return isinstance(message, str) and "HTTP 429" in message


def _stage_metrics(records: list[dict], prediction_key: str) -> dict:
    completed = [record for record in records if record.get("status") == "completed"]
    correct = [record for record in completed if record.get(prediction_key) == record.get("expected_room_id")]
    by_room: dict[str, list[dict]] = defaultdict(list)
    for record in completed:
        by_room[str(record.get("expected_room_id"))].append(record)
    return {
        "completed_count": len(completed),
        "correct_count": len(correct),
        "top1_accuracy": len(correct) / len(completed) if completed else None,
        "per_room": {
            room_id: {
                "correct": sum(item.get(prediction_key) == room_id for item in values),
                "total": len(values),
                "top1_accuracy": sum(item.get(prediction_key) == room_id for item in values) / len(values),
            }
            for room_id, values in sorted(by_room.items())
        },
    }


def _change_counts(records: list[dict], prediction_key: str) -> dict:
    counts: Counter[str] = Counter()
    for record in records:
        if record.get("status") != "completed":
            continue
        bayes_correct = record.get("bayesian_top1") == record.get("expected_room_id")
        proposed_correct = record.get(prediction_key) == record.get("expected_room_id")
        if bayes_correct and proposed_correct:
            counts["stable_correct"] += 1
        elif bayes_correct:
            counts["regression"] += 1
        elif proposed_correct:
            counts["correction"] += 1
        else:
            counts["unresolved"] += 1
    return {name: counts.get(name, 0) for name in ("correction", "regression", "stable_correct", "unresolved")}


def smoke_acceptance(selected_cases: list[dict], records: list[dict]) -> dict:
    selected_ids = {str(case["pano_id"]) for case in selected_cases}
    smoke_records = [record for record in records if record.get("pano_id") in selected_ids]
    completed = [record for record in smoke_records if record.get("status") == "completed"]
    decision_valid_rate = len(completed) / len(selected_cases) if selected_cases else 0.0
    recall = (
        sum(record.get("candidate_contains_expected_room") is True for record in smoke_records) / len(selected_cases)
        if selected_cases
        else 0.0
    )
    gated_accuracy = (
        sum(record.get("gated_map_correct") is True for record in completed) / len(completed) if completed else 0.0
    )
    raw_accuracy = (
        sum(record.get("raw_map_correct") is True for record in completed) / len(completed) if completed else 0.0
    )
    spatial_accuracy = (
        sum(record.get("current_spatial_correct") is True for record in completed) / len(completed) if completed else 0.0
    )
    spatial_reasoning_complete_rate = (
        sum(record.get("spatial_reasoning_complete") is True for record in completed) / len(completed)
        if completed
        else 0.0
    )
    raw_stable_regressions = sum(
        record.get("source_change_category") == "stable_correct" and record.get("raw_map_correct") is not True
        for record in completed
    )
    overrides = [record for record in completed if record.get("gate", {}).get("override_applied") is True]
    every_override_has_two_directions = all(
        len(record.get("gate", {}).get("support_directions", [])) >= 2 for record in overrides
    )
    checks = {
        "decision_valid_at_least_95_percent": decision_valid_rate >= 0.95,
        "candidate_recall_at_5_is_100_percent": recall == 1.0,
        "raw_map_accuracy_not_below_current_spatial": raw_accuracy >= spatial_accuracy,
        "no_raw_map_stable_correct_regression": raw_stable_regressions == 0,
    }
    return {
        "evaluated": len(smoke_records) == len(selected_cases),
        "passed": len(smoke_records) == len(selected_cases) and all(checks.values()),
        "checks": checks,
        "decision_valid_rate": decision_valid_rate,
        "schema_valid_rate": decision_valid_rate,
        "candidate_recall_at_5": recall,
        "raw_map_top1_accuracy": raw_accuracy,
        "gated_top1_accuracy": gated_accuracy,
        "current_spatial_top1_accuracy": spatial_accuracy,
        "spatial_reasoning_complete_rate": spatial_reasoning_complete_rate,
        "raw_map_stable_correct_regression_count": raw_stable_regressions,
        "stable_correct_regression_count": raw_stable_regressions,
        "override_count": len(overrides),
        "every_override_has_two_directions": every_override_has_two_directions,
    }


def _case_rows(records: list[dict]) -> list[dict]:
    rows = []
    for record in records:
        input_tokens, output_tokens = _usage_values(record.get("usage"))
        gate = record.get("gate") if isinstance(record.get("gate"), dict) else {}
        error = record.get("error") if isinstance(record.get("error"), dict) else {}
        rows.append(
            {
                "pano_id": record.get("pano_id"),
                "expected_room_id": record.get("expected_room_id"),
                "source_change_category": record.get("source_change_category"),
                "candidate_room_ids": " | ".join(record.get("candidate_room_ids", [])),
                "bayesian_top1": record.get("bayesian_top1"),
                "bayesian_correct": record.get("bayesian_correct"),
                "current_spatial_top1": record.get("current_spatial_top1"),
                "current_spatial_correct": record.get("current_spatial_correct"),
                "raw_map_top1": record.get("raw_map_top1"),
                "raw_map_correct": record.get("raw_map_correct"),
                "top1_source": record.get("top1_source"),
                "spatial_reasoning_complete": record.get("spatial_reasoning_complete"),
                "gated_map_top1": record.get("gated_map_top1"),
                "gated_map_correct": record.get("gated_map_correct"),
                "override_applied": gate.get("override_applied"),
                "gate_failed_checks": " | ".join(gate.get("failed_checks", [])),
                "margin": gate.get("top1_top2_margin"),
                "directional_layout": gate.get("directional_layout"),
                "contradiction_penalty": gate.get("contradiction_penalty"),
                "support_directions": " | ".join(gate.get("support_directions", [])),
                "latency_seconds": record.get("latency_seconds"),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "status": record.get("status"),
                "error": error.get("message"),
            }
        )
    return rows


def _write_case_reports(output_dir: Path, records: list[dict], summary: dict) -> None:
    rows = _case_rows(records)
    csv_path = output_dir / "case_report.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["pano_id"])
        writer.writeheader()
        writer.writerows(rows)
    stages = summary.get("stage_metrics", {})
    lines = [
        "# Map-grounded spatial alignment",
        "",
        f"- Selection: `{summary.get('sampling_mode')}`",
        f"- Selected: {summary.get('selected_count')}",
        f"- Completed: {summary.get('completed_count')}",
        "",
        "| Stage | Correct/Total | Top-1 |",
        "|---|---:|---:|",
    ]
    for name in ("bayesian", "current_spatial", "raw_map", "gated_map"):
        metrics = stages.get(name, {})
        accuracy = metrics.get("top1_accuracy")
        rendered = f"{accuracy:.1%}" if isinstance(accuracy, (int, float)) else "—"
        lines.append(f"| {name} | {metrics.get('correct_count', 0)}/{metrics.get('completed_count', 0)} | {rendered} |")
    lines.extend(["", "## Cases", "", "| Pano | Room | Source category | Bayesian | Current spatial | Raw map | Gated map |", "|---|---|---|---|---|---|---|"])
    for row in rows:
        lines.append(
            f"| {row['pano_id']} | {row['expected_room_id']} | {row['source_change_category']} | "
            f"{row['bayesian_top1']} | {row['current_spatial_top1']} | {row['raw_map_top1']} | {row['gated_map_top1']} |"
        )
    (output_dir / "case_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_summary(
    *,
    args: argparse.Namespace,
    sampling_mode: str,
    selected_cases: list[dict],
    records: list[dict],
    smoke_result: dict | None,
) -> dict:
    selected_ids = {str(case["pano_id"]) for case in selected_cases}
    selected_records = [record for record in records if record.get("pano_id") in selected_ids]
    completed = [record for record in selected_records if record.get("status") == "completed"]
    return {
        "schema": "sa_nav.map_grounded_bayesian_top5_evaluation",
        "schema_version": 2,
        "sampling_mode": sampling_mode,
        "selected_count": len(selected_cases),
        "processed_count": len(selected_records),
        "completed_count": len(completed),
        "pending_count": len(selected_cases) - len(completed),
        "config": {
            "source_summary": str(resolve_project_path(Path(args.source_summary))),
            "render_output_dir": args.render_output_dir,
            "candidate_source": "prior_fused.top5",
            "candidate_scores_sent_to_vlm": False,
            "candidate_rank_sent_to_vlm": False,
            "model": args.llm_model,
            "api_kind": args.llm_api_kind,
            "checkpoint_every": args.checkpoint_every,
            "resume": args.resume,
            "auto_source_all_on_pass": args.auto_source_all_on_pass,
            "output_mode": "compact_spatial_reasoning",
            "llm_max_output_tokens": args.llm_max_output_tokens,
            "logical_vlm_call_limit_per_case": 1,
        },
        "selection_counts": dict(Counter(source_change_category(case) for case in selected_cases)),
        "stage_metrics": {
            "bayesian": _stage_metrics(selected_records, "bayesian_top1"),
            "current_spatial": _stage_metrics(selected_records, "current_spatial_top1"),
            "raw_map": _stage_metrics(selected_records, "raw_map_top1"),
            "gated_map": _stage_metrics(selected_records, "gated_map_top1"),
        },
        "compact_spatial_reasoning": {
            "complete_count": sum(record.get("spatial_reasoning_complete") is True for record in completed),
            "complete_rate": (
                sum(record.get("spatial_reasoning_complete") is True for record in completed) / len(completed)
                if completed
                else None
            ),
        },
        "raw_map_changes_from_bayesian": _change_counts(selected_records, "raw_map_top1"),
        "gated_map_changes_from_bayesian": _change_counts(selected_records, "gated_map_top1"),
        "smoke_acceptance": smoke_result,
        "paths": {
            "results": str(resolve_project_path(Path(args.output_dir)) / "results.jsonl"),
            "case_csv": str(resolve_project_path(Path(args.output_dir)) / "case_report.csv"),
            "case_markdown": str(resolve_project_path(Path(args.output_dir)) / "case_report.md"),
        },
    }


def _checkpoint(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    records_by_pano: dict[str, dict],
    selected_cases: list[dict],
    sampling_mode: str,
    smoke_result: dict | None,
) -> dict:
    records = _sorted_cases(list(records_by_pano.values()))
    _write_jsonl(output_dir / "results.jsonl", records)
    summary = build_summary(
        args=args,
        sampling_mode=sampling_mode,
        selected_cases=selected_cases,
        records=records,
        smoke_result=smoke_result,
    )
    _write_json(output_dir / "summary.json", summary)
    _write_case_reports(output_dir, records, summary)
    return summary


def _run_phase(
    *,
    args: argparse.Namespace,
    sampling_mode: str,
    selected_cases: list[dict],
    records_by_pano: dict[str, dict],
    room_graph: dict[str, dict],
    region_payload: dict,
    map_image_path: Path,
    output_dir: Path,
    experiment: MapGroundedSpatialAlignmentExperiment,
    smoke_result: dict | None,
) -> dict:
    completed_panos = resumable_completed_panos(records_by_pano) if args.resume else set()
    processed_since_checkpoint = 0
    for index, case in enumerate(selected_cases, start=1):
        pano_id = str(case["pano_id"])
        if pano_id in completed_panos:
            continue
        candidates = bayesian_top5(case)
        quota_retry_count = 0
        while True:
            record = _run_case(
                case=case,
                candidates=candidates,
                room_graph=room_graph,
                region_payload=region_payload,
                map_image_path=map_image_path,
                output_dir=output_dir,
                experiment=experiment,
            )
            if not _is_quota_error(record):
                break
            records_by_pano[pano_id] = record
            _write_json(output_dir / "cases" / f"{sanitize_name(pano_id)}.json", record)
            _checkpoint(
                args=args,
                output_dir=output_dir,
                records_by_pano=records_by_pano,
                selected_cases=selected_cases,
                sampling_mode=sampling_mode,
                smoke_result=smoke_result,
            )
            if quota_retry_count >= args.max_quota_retries or args.quota_retry_delay <= 0:
                raise RuntimeError(
                    f"Gemini quota remained exhausted for {pano_id} after {quota_retry_count} delayed retries."
                )
            quota_retry_count += 1
            print(
                f"[{sampling_mode} {index}/{len(selected_cases)}] {pano_id}: HTTP 429; "
                f"quota retry {quota_retry_count}/{args.max_quota_retries} in {args.quota_retry_delay:.0f}s",
                flush=True,
            )
            time.sleep(args.quota_retry_delay)
        records_by_pano[pano_id] = record
        _write_json(output_dir / "cases" / f"{sanitize_name(pano_id)}.json", record)
        processed_since_checkpoint += 1
        print(
            f"[{sampling_mode} {index}/{len(selected_cases)}] {pano_id}: "
            f"{record.get('status')} raw={record.get('raw_map_top1')} gated={record.get('gated_map_top1')}",
            flush=True,
        )
        if processed_since_checkpoint >= args.checkpoint_every:
            _checkpoint(
                args=args,
                output_dir=output_dir,
                records_by_pano=records_by_pano,
                selected_cases=selected_cases,
                sampling_mode=sampling_mode,
                smoke_result=smoke_result,
            )
            processed_since_checkpoint = 0
        if args.delay > 0:
            time.sleep(args.delay)
    return _checkpoint(
        args=args,
        output_dir=output_dir,
        records_by_pano=records_by_pano,
        selected_cases=selected_cases,
        sampling_mode=sampling_mode,
        smoke_result=smoke_result,
    )


def evaluate(args: argparse.Namespace) -> dict:
    if args.checkpoint_every < 1:
        raise ValueError("--checkpoint-every must be at least 1.")
    if args.quota_retry_delay < 0 or args.max_quota_retries < 0:
        raise ValueError("Quota retry settings must be non-negative.")
    source_payload = load_json(resolve_project_path(Path(args.source_summary)))
    source_results = source_payload.get("results")
    if not isinstance(source_results, list):
        raise ValueError("Source summary must contain a results array.")
    region_payload = load_json(resolve_project_path(Path(args.map_regions)))
    all_source_cases = select_source_cases(source_results, "source-all")
    source_catalog_inventory = source_inventory(all_source_cases, region_payload)
    eligible_results = map_eligible_source_cases(source_results, region_payload)
    selected_cases = select_source_cases(eligible_results, args.sampling_mode)
    inventory = source_inventory(selected_cases, region_payload)
    if args.dry_run:
        return {
            "status": "dry_run",
            "sampling_mode": args.sampling_mode,
            "selected_count": len(selected_cases),
            "selection_counts": dict(Counter(source_change_category(case) for case in selected_cases)),
            "source_catalog_inventory": source_catalog_inventory,
            "inventory": inventory,
            "pano_ids": [case["pano_id"] for case in selected_cases],
        }

    if inventory["ready_count"] != inventory["selected_count"]:
        raise ValueError(
            "Selected cases are not runnable with exactly five candidates on the supplied map; "
            f"ready={inventory['ready_count']}/{inventory['selected_count']}. Run with --dry-run for details."
        )

    artifacts_dir = resolve_project_path(Path(args.artifacts_dir))
    room_graph = load_json(preferred_room_graph_path(artifacts_dir))
    validate_map_regions(room_graph, region_payload)
    map_image_path = resolve_project_path(Path(args.map_image))
    output_dir = resolve_project_path(Path(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    records_by_pano = _load_resume_records(output_dir / "results.jsonl") if args.resume else {}
    experiment = MapGroundedSpatialAlignmentExperiment(
        room_graph=room_graph,
        model=args.llm_model,
        api_key=args.llm_api_key,
        api_base=args.llm_api_base,
        api_kind=args.llm_api_kind,
        request_timeout=args.llm_timeout,
        num_ctx=args.llm_num_ctx,
        max_http_retries=args.max_http_retries,
        validation_retries=args.validation_retries,
        max_output_tokens=args.llm_max_output_tokens,
        allow_incomplete_winning_hypothesis=True,
        compact_decision_output=True,
    )
    summary = _run_phase(
        args=args,
        sampling_mode=args.sampling_mode,
        selected_cases=selected_cases,
        records_by_pano=records_by_pano,
        room_graph=room_graph,
        region_payload=region_payload,
        map_image_path=map_image_path,
        output_dir=output_dir,
        experiment=experiment,
        smoke_result=None,
    )
    if args.sampling_mode == "change-balanced-smoke":
        records = list(records_by_pano.values())
        smoke_result = smoke_acceptance(selected_cases, records)
        summary = _checkpoint(
            args=args,
            output_dir=output_dir,
            records_by_pano=records_by_pano,
            selected_cases=selected_cases,
            sampling_mode=args.sampling_mode,
            smoke_result=smoke_result,
        )
        _write_json(output_dir / "smoke_summary.json", summary)
        if args.auto_source_all_on_pass and smoke_result.get("passed") is True:
            all_cases = select_source_cases(eligible_results, "source-all")
            summary = _run_phase(
                args=args,
                sampling_mode="source-all",
                selected_cases=all_cases,
                records_by_pano=records_by_pano,
                room_graph=room_graph,
                region_payload=region_payload,
                map_image_path=map_image_path,
                output_dir=output_dir,
                experiment=experiment,
                smoke_result=smoke_result,
            )
    return summary


def main() -> int:
    args = build_parser().parse_args()
    try:
        payload = evaluate(args)
    except Exception as exc:
        print(render_json({"status": "error", "error": {"type": type(exc).__name__, "message": str(exc)}}))
        return 1
    print(render_json(payload))
    if payload.get("status") == "dry_run":
        return 0
    return 0 if payload.get("pending_count") == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
