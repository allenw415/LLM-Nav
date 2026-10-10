from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sa_nav import load_dotenv, resolve_model_environment, resolve_task_num_ctx  # noqa: E402
from sa_nav.cli._common import load_json, render_json, resolve_project_path  # noqa: E402
from sa_nav.cli.run_map_grounded_heading import angular_error_deg, stable_majority  # noqa: E402
from sa_nav.cli.run_spatial_alignment import request_diagnostics, response_diagnostics  # noqa: E402
from sa_nav.common.env import DEFAULT_GEMINI_API_BASE  # noqa: E402
from sa_nav.common.room_profiles import preferred_room_graph_path  # noqa: E402
from sa_nav.experiments.map_grounded_heading import (  # noqa: E402
    DIRECTION_TO_HEADING,
    MAP_MODES,
    MapGroundedHeadingExperiment,
    describe_plain_heading_map,
    render_passage_heading_map,
    validate_heading_map_regions,
)
from sa_nav.perception.renderer import sanitize_name  # noqa: E402


load_dotenv(PROJECT_ROOT / ".env")
MODEL_ENV = resolve_model_environment(
    default_model="gemma-4-31b-it",
    default_api_base=DEFAULT_GEMINI_API_BASE,
    default_api_kind="responses",
)
DEFAULT_LLM_NUM_CTX = resolve_task_num_ctx(
    "localization",
    fallback_num_ctx=MODEL_ENV.num_ctx,
    default_num_ctx=16384,
)
EXPECTED_LEVEL0_DIRECTED_CASE_COUNT = 76
FEASIBILITY_THRESHOLDS = {
    "schema_valid_rate": 0.95,
    "stable_case_rate": 0.90,
    "majority_within_45_accuracy": 0.80,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate map-only VLM passage headings on all directed Level 0 adjacent-room edges."
    )
    parser.add_argument("--artifacts-dir", default="dataset/sites/british_museum/normalized")
    parser.add_argument(
        "--map-image",
        default="dataset/sites/british_museum/maps/level_0_allocentric.png",
    )
    parser.add_argument(
        "--map-regions",
        default="dataset/sites/british_museum/maps/level_0_room_regions.json",
    )
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument(
        "--map-mode",
        choices=MAP_MODES,
        default="highlighted",
        help="Use generated room highlights or reuse the original unmodified map for every case.",
    )
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--current-room-id", action="append", default=[])
    parser.add_argument("--llm-model", default=MODEL_ENV.model_name)
    parser.add_argument("--llm-api-key", default=MODEL_ENV.api_key)
    parser.add_argument("--llm-api-kind", default=MODEL_ENV.api_kind)
    parser.add_argument("--llm-api-base", default=MODEL_ENV.api_base)
    parser.add_argument("--llm-timeout", type=float, default=MODEL_ENV.request_timeout or 180.0)
    parser.add_argument("--llm-num-ctx", type=int, default=DEFAULT_LLM_NUM_CTX)
    parser.add_argument("--delay", type=float, default=0.0)
    parser.add_argument("--low-confidence-threshold", type=float, default=0.6)
    parser.add_argument(
        "--output-dir",
        default="outputs/map_grounded_heading/all_level0_directed",
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def build_heading_cases(
    room_graph: dict[str, dict],
    region_payload: dict,
    *,
    current_room_ids: set[str] | None = None,
) -> list[dict]:
    validate_heading_map_regions(region_payload)
    region_room_ids = set(region_payload["rooms"])
    unknown_selected = sorted((current_room_ids or set()) - region_room_ids)
    if unknown_selected:
        raise ValueError("Selected current rooms are missing from map regions: " + ", ".join(unknown_selected))

    cases: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for current_room_id in sorted(region_room_ids):
        if current_room_ids and current_room_id not in current_room_ids:
            continue
        node = room_graph.get(current_room_id)
        if not isinstance(node, dict):
            continue
        for neighbor in node.get("neighbors", []):
            if not isinstance(neighbor, dict):
                continue
            target_room_id = neighbor.get("target_room_id")
            expected_heading = neighbor.get("allocentric_heading_deg")
            expected_direction = neighbor.get("allocentric_direction")
            if (
                not isinstance(target_room_id, str)
                or target_room_id not in region_room_ids
                or not isinstance(expected_heading, (int, float))
                or not isinstance(expected_direction, str)
            ):
                continue
            edge = (current_room_id, target_room_id)
            if edge in seen:
                raise ValueError(f"Duplicate directed room edge: {current_room_id} -> {target_room_id}")
            seen.add(edge)
            normalized_heading = float(expected_heading) % 360.0
            cases.append(
                {
                    "case_id": f"{current_room_id} -> {target_room_id}",
                    "current_room_id": current_room_id,
                    "target_room_id": target_room_id,
                    "expected_allocentric_direction": expected_direction,
                    "expected_heading_deg": normalized_heading,
                }
            )
    cases.sort(key=lambda record: (record["current_room_id"], record["target_room_id"]))
    return cases


def aggregate_cases(records: list[dict], cases: list[dict], *, repeat: int) -> list[dict]:
    by_case: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_case[str(record.get("case_id"))].append(record)
    summaries: list[dict] = []
    for case in cases:
        case_records = by_case.get(str(case["case_id"]), [])
        completed = [record for record in case_records if record.get("status") == "completed"]
        directions = [str(record["predicted_direction"]) for record in completed]
        pairs = list(combinations(directions, 2))
        majority_direction = stable_majority(directions)
        majority_heading = DIRECTION_TO_HEADING.get(majority_direction) if majority_direction else None
        majority_error = angular_error_deg(majority_heading, float(case["expected_heading_deg"]))
        confidences = [
            float(record["confidence"])
            for record in completed
            if isinstance(record.get("confidence"), (int, float))
        ]
        summaries.append(
            {
                **case,
                "requested_run_count": repeat,
                "completed_run_count": len(completed),
                "predicted_directions": directions,
                "direction_counts": dict(Counter(directions)),
                "unanimous_prediction": (
                    len(completed) == repeat and bool(directions) and len(set(directions)) == 1
                ),
                "pairwise_prediction_agreement": (
                    sum(left == right for left, right in pairs) / len(pairs) if pairs else None
                ),
                "stable_majority": majority_direction is not None,
                "majority_direction": majority_direction,
                "majority_heading_deg": majority_heading,
                "majority_angular_error_deg": majority_error,
                "majority_exact_correct": majority_error == 0.0 if majority_error is not None else None,
                "majority_within_45_correct": majority_error <= 45.0 if majority_error is not None else None,
                "mean_confidence": sum(confidences) / len(confidences) if confidences else None,
            }
        )
    return summaries


def evaluation_metrics(records: list[dict], case_summaries: list[dict]) -> dict:
    attempted = list(records)
    completed = [record for record in records if record.get("status") == "completed"]
    errors = [
        float(record["angular_error_deg"])
        for record in completed
        if isinstance(record.get("angular_error_deg"), (int, float))
    ]
    latencies = [
        float(record["latency_seconds"])
        for record in completed
        if isinstance(record.get("latency_seconds"), (int, float))
    ]
    pairwise = [
        float(case["pairwise_prediction_agreement"])
        for case in case_summaries
        if isinstance(case.get("pairwise_prediction_agreement"), (int, float))
    ]
    stable = [case for case in case_summaries if case.get("stable_majority") is True]
    majority_errors = [
        float(case["majority_angular_error_deg"])
        for case in stable
        if isinstance(case.get("majority_angular_error_deg"), (int, float))
    ]

    by_expected_direction: dict[str, dict] = {}
    for direction in ("north", "east", "south", "west"):
        subset = [case for case in case_summaries if case.get("expected_allocentric_direction") == direction]
        subset_stable = [case for case in subset if case.get("stable_majority") is True]
        by_expected_direction[direction] = {
            "case_count": len(subset),
            "stable_case_rate": len(subset_stable) / len(subset) if subset else None,
            "majority_within_45_accuracy": (
                sum(case.get("majority_within_45_correct") is True for case in subset_stable)
                / len(subset_stable)
                if subset_stable
                else None
            ),
        }

    return {
        "case_count": len(case_summaries),
        "attempted_run_count": len(attempted),
        "completed_run_count": len(completed),
        "schema_valid_rate": len(completed) / len(attempted) if attempted else None,
        "per_run_exact_accuracy": sum(error == 0.0 for error in errors) / len(errors) if errors else None,
        "per_run_within_45_accuracy": (
            sum(error <= 45.0 for error in errors) / len(errors) if errors else None
        ),
        "mean_angular_error_deg": sum(errors) / len(errors) if errors else None,
        "stable_case_count": len(stable),
        "stable_case_rate": len(stable) / len(case_summaries) if case_summaries else None,
        "unanimous_prediction_rate": (
            sum(case.get("unanimous_prediction") is True for case in case_summaries) / len(case_summaries)
            if case_summaries
            else None
        ),
        "pairwise_prediction_agreement": sum(pairwise) / len(pairwise) if pairwise else None,
        "majority_exact_accuracy": (
            sum(error == 0.0 for error in majority_errors) / len(majority_errors)
            if majority_errors
            else None
        ),
        "majority_within_45_accuracy": (
            sum(error <= 45.0 for error in majority_errors) / len(majority_errors)
            if majority_errors
            else None
        ),
        "all_case_majority_within_45_rate": (
            sum(case.get("majority_within_45_correct") is True for case in case_summaries)
            / len(case_summaries)
            if case_summaries
            else None
        ),
        "mean_majority_angular_error_deg": (
            sum(majority_errors) / len(majority_errors) if majority_errors else None
        ),
        "mean_latency_seconds": sum(latencies) / len(latencies) if latencies else None,
        "by_expected_direction": by_expected_direction,
    }


def feasibility_acceptance(metrics: dict) -> dict:
    checks = {
        "schema_valid_rate_at_least_95_percent": (
            isinstance(metrics.get("schema_valid_rate"), (int, float))
            and float(metrics["schema_valid_rate"]) >= FEASIBILITY_THRESHOLDS["schema_valid_rate"]
        ),
        "stable_case_rate_at_least_90_percent": (
            isinstance(metrics.get("stable_case_rate"), (int, float))
            and float(metrics["stable_case_rate"]) >= FEASIBILITY_THRESHOLDS["stable_case_rate"]
        ),
        "majority_within_45_accuracy_at_least_80_percent": (
            isinstance(metrics.get("majority_within_45_accuracy"), (int, float))
            and float(metrics["majority_within_45_accuracy"])
            >= FEASIBILITY_THRESHOLDS["majority_within_45_accuracy"]
        ),
    }
    evaluated = all(isinstance(metrics.get(key), (int, float)) for key in FEASIBILITY_THRESHOLDS)
    return {
        "evaluated": evaluated,
        "passed": all(checks.values()) if evaluated else None,
        "thresholds": dict(FEASIBILITY_THRESHOLDS),
        "checks": checks,
    }


def review_cases(case_summaries: list[dict], *, low_confidence_threshold: float) -> list[dict]:
    records: list[dict] = []
    for case in case_summaries:
        reasons = []
        if case.get("stable_majority") is not True:
            reasons.append("unstable_no_two_run_majority")
        if case.get("majority_within_45_correct") is False:
            reasons.append("majority_angular_error_over_45_deg")
        confidence = case.get("mean_confidence")
        if isinstance(confidence, (int, float)) and float(confidence) < low_confidence_threshold:
            reasons.append("low_mean_confidence")
        if int(case.get("completed_run_count", 0)) < int(case.get("requested_run_count", 0)):
            reasons.append("incomplete_runs")
        if reasons:
            records.append(
                {
                    **case,
                    "review_reasons": reasons,
                    "manual_category": None,
                    "allowed_manual_categories": [
                        "vlm_map_reading_error",
                        "map_passage_unclear",
                        "room_region_annotation_issue",
                        "room_graph_ground_truth_too_coarse",
                    ],
                }
            )
    return records


def _format_rate(value: object) -> str:
    return f"{float(value) * 100:.1f}%" if isinstance(value, (int, float)) else "—"


def _format_number(value: object, *, digits: int = 1) -> str:
    return f"{float(value):.{digits}f}" if isinstance(value, (int, float)) else "—"


def build_meeting_summary(payload: dict) -> str:
    metrics = payload["metrics"]
    acceptance = payload["acceptance"]
    reviews = payload["review_cases"]
    lines = [
        "# Map-only VLM 通道 Heading 實驗",
        "",
        "## 實驗設定",
        "",
        f"- 測試 {payload['case_count']} 條 Level 0 有向相鄰展廳轉移，每條重複 {payload['config']['repeat']} 次。",
        f"- 模型：`{payload['config']['model']}`；總呼叫數：{metrics['attempted_run_count']}。",
        "- VLM 僅收到 north-up 地圖、目前展廳 ID 與目標展廳 ID；room-graph heading 只用於事後評分。",
        "",
        "## 結果",
        "",
        "| 指標 | 結果 |",
        "|---|---:|",
        f"| Schema-valid rate | {_format_rate(metrics['schema_valid_rate'])} |",
        f"| Stable-case rate | {_format_rate(metrics['stable_case_rate'])} |",
        f"| Unanimous prediction rate | {_format_rate(metrics['unanimous_prediction_rate'])} |",
        f"| Pairwise agreement | {_format_rate(metrics['pairwise_prediction_agreement'])} |",
        f"| Per-run exact accuracy | {_format_rate(metrics['per_run_exact_accuracy'])} |",
        f"| Per-run ±45° accuracy | {_format_rate(metrics['per_run_within_45_accuracy'])} |",
        f"| Majority exact accuracy | {_format_rate(metrics['majority_exact_accuracy'])} |",
        f"| Majority ±45° accuracy | {_format_rate(metrics['majority_within_45_accuracy'])} |",
        f"| Mean angular error | {_format_number(metrics['mean_angular_error_deg'])}° |",
        "",
        "## Feasibility 判定",
        "",
        f"- 結果：{'通過' if acceptance.get('passed') is True else '未通過' if acceptance.get('passed') is False else '資料不足'}。",
        f"- Gate：schema-valid ≥95%、stable cases ≥90%、stable-majority 的 ±45° accuracy ≥80%。",
        "",
        "## 需要人工檢查的案例",
        "",
    ]
    if not reviews:
        lines.append("沒有符合 review 條件的案例。")
    else:
        lines.extend(
            [
                "| Current → Target | Expected | Predictions | Majority | Error | Review reason |",
                "|---|---:|---|---|---:|---|",
            ]
        )
        for case in reviews:
            expected = f"{case['expected_allocentric_direction']} / {case['expected_heading_deg']:.0f}°"
            predictions = ", ".join(case["predicted_directions"]) or "—"
            majority = case.get("majority_direction") or "—"
            error = _format_number(case.get("majority_angular_error_deg"))
            reasons = ", ".join(case["review_reasons"])
            lines.append(
                f"| {case['current_room_id']} → {case['target_room_id']} | {expected} | {predictions} | {majority} | {error}° | {reasons} |"
            )
    lines.extend(
        [
            "",
            "## 解讀限制",
            "",
            "這個實驗只能判斷 VLM 能否從地圖推斷 room-level 通道位於目前展廳哪一側。",
            "同一組 current/target room 的輸入不會隨 agent 在房內的位置改變，因此即使通過，也不能證明已解決位置相依的第一步 heading 問題。",
            "",
        ]
    )
    return "\n".join(lines)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_json(payload), encoding="utf-8")


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def evaluate(args: argparse.Namespace) -> dict:
    if args.repeat < 1:
        raise ValueError("--repeat must be at least 1.")
    if args.max_cases is not None and args.max_cases < 1:
        raise ValueError("--max-cases must be at least 1 when provided.")
    if not 0.0 <= args.low_confidence_threshold <= 1.0:
        raise ValueError("--low-confidence-threshold must be between 0 and 1.")

    artifacts_dir = resolve_project_path(args.artifacts_dir)
    room_graph_path = preferred_room_graph_path(artifacts_dir)
    map_image_path = resolve_project_path(args.map_image)
    map_regions_path = resolve_project_path(args.map_regions)
    output_dir = resolve_project_path(args.output_dir)
    room_graph = load_json(room_graph_path)
    region_payload = load_json(map_regions_path)
    all_cases = build_heading_cases(room_graph, region_payload)
    if len(all_cases) != EXPECTED_LEVEL0_DIRECTED_CASE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_LEVEL0_DIRECTED_CASE_COUNT} Level 0 directed map-heading cases; found {len(all_cases)}."
        )
    selected_rooms = set(args.current_room_id) if args.current_room_id else None
    cases = build_heading_cases(room_graph, region_payload, current_room_ids=selected_rooms)
    if args.max_cases is not None:
        cases = cases[: args.max_cases]

    experiment = MapGroundedHeadingExperiment(
        model=args.llm_model,
        api_key=args.llm_api_key,
        api_base=args.llm_api_base,
        api_kind=args.llm_api_kind,
        request_timeout=args.llm_timeout,
        num_ctx=args.llm_num_ctx,
        max_http_retries=1,
    )
    if not experiment.model_client.is_configured():
        raise RuntimeError("Map-grounded heading evaluation requires configured VLM access.")

    records: list[dict] = []
    runs_path = output_dir / "runs.jsonl"
    for case_index, case in enumerate(cases, start=1):
        if args.map_mode == "highlighted":
            case_slug = (
                f"{sanitize_name(case['current_room_id'])}"
                f"__to__{sanitize_name(case['target_room_id'])}"
            )
            semantic_map_path = output_dir / "maps" / f"{case_index:03d}_{case_slug}.png"
            map_diagnostics = render_passage_heading_map(
                base_map_path=map_image_path,
                region_payload=region_payload,
                current_room_id=str(case["current_room_id"]),
                target_room_id=str(case["target_room_id"]),
                output_path=semantic_map_path,
            )
        else:
            semantic_map_path = map_image_path
            map_diagnostics = describe_plain_heading_map(
                base_map_path=map_image_path,
                current_room_id=str(case["current_room_id"]),
                target_room_id=str(case["target_room_id"]),
            )
        if not args.quiet:
            print(
                f"[{case_index}/{len(cases)}] {case['case_id']} expected={case['expected_heading_deg']:.0f}°",
                flush=True,
            )
        for repeat_index in range(1, args.repeat + 1):
            started = time.perf_counter()
            record: dict = {
                "schema": "sa_nav.map_grounded_heading_run",
                "schema_version": 1,
                "status": "error",
                **case,
                "case_index": case_index,
                "repeat_index": repeat_index,
                "map_mode": args.map_mode,
                "semantic_map_path": str(semantic_map_path),
                "map": map_diagnostics,
                "model": {
                    "name": args.llm_model,
                    "api_kind": args.llm_api_kind,
                    "temperature": 0.0,
                },
                "input_policy": {
                    "spatial_inputs": ["map_image", "current_room_id", "target_room_id"],
                    "room_highlights_rendered": map_diagnostics["room_highlights_rendered"],
                    "compass_sidebar_rendered": map_diagnostics["compass_sidebar_rendered"],
                    "room_graph_heading_sent": False,
                    "topology_text_sent": False,
                    "panorama_sent": False,
                    "pano_id_sent": False,
                    "candidate_actions_sent": False,
                    "ground_truth_sent": False,
                },
            }
            try:
                wrapped = experiment.run_once(
                    current_room_id=str(case["current_room_id"]),
                    target_room_id=str(case["target_room_id"]),
                    semantic_map_path=semantic_map_path,
                    map_mode=args.map_mode,
                )
                response_payload = wrapped.pop("response_payload", None)
                result = wrapped["result"]
                predicted_direction = str(result["passage_direction"])
                predicted_heading = float(result["heading_deg"])
                error = angular_error_deg(predicted_heading, float(case["expected_heading_deg"]))
                record.update(
                    {
                        "status": "completed",
                        "request": request_diagnostics(experiment.last_request_body),
                        "response": response_diagnostics(response_payload),
                        "result": result,
                        "predicted_direction": predicted_direction,
                        "predicted_heading_deg": predicted_heading,
                        "confidence": float(result["confidence"]),
                        "angular_error_deg": error,
                        "exact_correct": error == 0.0,
                        "within_45_correct": error <= 45.0,
                    }
                )
            except Exception as exc:
                record.update(
                    {
                        "error": {"type": type(exc).__name__, "message": str(exc)},
                        "request": request_diagnostics(experiment.last_request_body),
                        "response": response_diagnostics(experiment.last_response_payload),
                    }
                )
            record["latency_seconds"] = round(time.perf_counter() - started, 6)
            records.append(record)
            _write_jsonl(runs_path, records)
            if args.delay > 0:
                time.sleep(args.delay)

    case_summaries = aggregate_cases(records, cases, repeat=args.repeat)
    metrics = evaluation_metrics(records, case_summaries)
    acceptance = feasibility_acceptance(metrics)
    reviews = review_cases(
        case_summaries,
        low_confidence_threshold=args.low_confidence_threshold,
    )
    payload = {
        "schema": "sa_nav.map_grounded_heading_evaluation",
        "schema_version": 1,
        "config": {
            "artifacts_dir": str(artifacts_dir),
            "room_graph_path": str(room_graph_path),
            "map_image_path": str(map_image_path),
            "map_regions_path": str(map_regions_path),
            "map_mode": args.map_mode,
            "repeat": args.repeat,
            "model": args.llm_model,
            "api_kind": args.llm_api_kind,
            "temperature": 0.0,
            "current_room_ids": sorted(selected_rooms) if selected_rooms else None,
            "max_cases": args.max_cases,
            "low_confidence_threshold": args.low_confidence_threshold,
            "request_spatial_inputs": ["map_image", "current_room_id", "target_room_id"],
        },
        "catalog_case_count": len(all_cases),
        "case_count": len(cases),
        "expected_run_count": len(cases) * args.repeat,
        "metrics": metrics,
        "acceptance": acceptance,
        "case_summaries": case_summaries,
        "review_case_count": len(reviews),
        "review_cases": reviews,
        "paths": {
            "runs": str(runs_path),
            "case_summaries": str(output_dir / "case_summaries.jsonl"),
            "review_cases": str(output_dir / "review_cases.jsonl"),
            "summary": str(output_dir / "summary.json"),
            "meeting_summary": str(output_dir / "meeting_summary.md"),
            "maps": str(output_dir / "maps") if args.map_mode == "highlighted" else None,
            "input_map": str(map_image_path),
            "output_dir": str(output_dir),
        },
        "interpretation": {
            "supported_claim": "The VLM can infer a room-level passage side from a north-up map.",
            "unsupported_claim": "The VLM solves position-dependent first-action heading inside a room.",
            "reason": "The input is identical for the same current/target room pair and contains no agent position or current view.",
        },
    }
    _write_jsonl(output_dir / "case_summaries.jsonl", case_summaries)
    _write_jsonl(output_dir / "review_cases.jsonl", reviews)
    _write_json(output_dir / "summary.json", payload)
    (output_dir / "meeting_summary.md").write_text(build_meeting_summary(payload), encoding="utf-8")
    return payload


def main() -> int:
    args = build_parser().parse_args()
    try:
        payload = evaluate(args)
    except Exception as exc:
        print(render_json({"status": "error", "error": {"type": type(exc).__name__, "message": str(exc)}}))
        return 1
    if args.json:
        print(render_json(payload))
    else:
        print(
            render_json(
                {
                    "case_count": payload["case_count"],
                    "expected_run_count": payload["expected_run_count"],
                    "metrics": payload["metrics"],
                    "acceptance": payload["acceptance"],
                    "paths": payload["paths"],
                }
            )
        )
    return 0 if payload["metrics"]["completed_run_count"] == payload["expected_run_count"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
