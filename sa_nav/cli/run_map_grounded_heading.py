from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from itertools import combinations
from pathlib import Path

from ._common import PROJECT_ROOT, ensure_project_root_on_path, load_json, render_json, resolve_project_path

ensure_project_root_on_path()

from sa_nav import load_dotenv, resolve_model_environment, resolve_task_num_ctx  # noqa: E402
from sa_nav.cli.run_spatial_alignment import request_diagnostics, response_diagnostics  # noqa: E402
from sa_nav.common.env import DEFAULT_GEMINI_API_BASE  # noqa: E402
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one standalone map-only VLM passage-heading case without changing navigation."
    )
    parser.add_argument("--current-room-id", required=True)
    parser.add_argument("--target-room-id", required=True)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument(
        "--map-mode",
        choices=MAP_MODES,
        default="highlighted",
        help="Use generated room highlights or the original unmodified map.",
    )
    parser.add_argument(
        "--map-image",
        default="dataset/sites/british_museum/maps/level_0_allocentric.png",
    )
    parser.add_argument(
        "--map-regions",
        default="dataset/sites/british_museum/maps/level_0_room_regions.json",
    )
    parser.add_argument("--expected-heading-deg", type=float)
    parser.add_argument("--llm-model", default=MODEL_ENV.model_name)
    parser.add_argument("--llm-api-key", default=MODEL_ENV.api_key)
    parser.add_argument("--llm-api-kind", default=MODEL_ENV.api_kind)
    parser.add_argument("--llm-api-base", default=MODEL_ENV.api_base)
    parser.add_argument("--llm-timeout", type=float, default=MODEL_ENV.request_timeout or 180.0)
    parser.add_argument("--llm-num-ctx", type=int, default=DEFAULT_LLM_NUM_CTX)
    parser.add_argument("--output-dir")
    parser.add_argument("--json", action="store_true")
    return parser


def default_output_dir(current_room_id: str, target_room_id: str) -> Path:
    case_slug = f"{sanitize_name(current_room_id)}__to__{sanitize_name(target_room_id)}"
    return resolve_project_path(Path("outputs/map_grounded_heading/single") / case_slug)


def angular_error_deg(left: float | None, right: float | None) -> float | None:
    if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
        return None
    return abs(((float(left) - float(right) + 180.0) % 360.0) - 180.0)


def stable_majority(directions: list[str], *, required_count: int = 2) -> str | None:
    counts = Counter(directions)
    eligible = [direction for direction, count in counts.items() if count >= required_count]
    if len(eligible) != 1:
        return None
    return eligible[0]


def summarize_runs(runs: list[dict], *, expected_heading_deg: float | None) -> dict:
    completed = [run for run in runs if run.get("status") == "completed"]
    directions = [str(run["result"]["passage_direction"]) for run in completed]
    pairs = list(combinations(directions, 2))
    majority_direction = stable_majority(directions)
    majority_heading = DIRECTION_TO_HEADING.get(majority_direction) if majority_direction else None
    majority_error = angular_error_deg(majority_heading, expected_heading_deg)
    errors = [
        float(run["evaluation"]["angular_error_deg"])
        for run in completed
        if isinstance(run.get("evaluation", {}).get("angular_error_deg"), (int, float))
    ]
    latencies = [float(run["latency_seconds"]) for run in completed]
    confidences = [float(run["result"]["confidence"]) for run in completed]
    return {
        "requested_run_count": len(runs),
        "completed_run_count": len(completed),
        "schema_valid_rate": len(completed) / len(runs) if runs else None,
        "directions": directions,
        "unanimous_prediction": (
            len(completed) == len(runs) and bool(directions) and len(set(directions)) == 1
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
        "per_run_exact_accuracy": (
            sum(error == 0.0 for error in errors) / len(errors) if errors else None
        ),
        "per_run_within_45_accuracy": (
            sum(error <= 45.0 for error in errors) / len(errors) if errors else None
        ),
        "mean_angular_error_deg": sum(errors) / len(errors) if errors else None,
        "mean_confidence": sum(confidences) / len(confidences) if confidences else None,
        "mean_latency_seconds": sum(latencies) / len(latencies) if latencies else None,
    }


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_json(payload), encoding="utf-8")


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def run_experiment(args: argparse.Namespace) -> dict:
    if args.repeat < 1:
        raise ValueError("--repeat must be at least 1.")
    current_room_id = str(args.current_room_id).strip()
    target_room_id = str(args.target_room_id).strip()
    if not current_room_id or not target_room_id:
        raise ValueError("Room IDs must not be empty.")
    if current_room_id == target_room_id:
        raise ValueError("Current and target rooms must be different.")
    output_dir = (
        resolve_project_path(args.output_dir)
        if args.output_dir
        else default_output_dir(current_room_id, target_room_id)
    )
    map_image_path = resolve_project_path(args.map_image)
    map_regions_path = resolve_project_path(args.map_regions)
    region_payload = load_json(map_regions_path)
    validate_heading_map_regions(region_payload)
    if args.map_mode == "highlighted":
        semantic_map_path = output_dir / "semantic_map.png"
        map_diagnostics = render_passage_heading_map(
            base_map_path=map_image_path,
            region_payload=region_payload,
            current_room_id=current_room_id,
            target_room_id=target_room_id,
            output_path=semantic_map_path,
        )
    else:
        semantic_map_path = map_image_path
        map_diagnostics = describe_plain_heading_map(
            base_map_path=map_image_path,
            current_room_id=current_room_id,
            target_room_id=target_room_id,
        )
    experiment = MapGroundedHeadingExperiment(
        model=args.llm_model,
        api_key=args.llm_api_key,
        api_base=args.llm_api_base,
        api_kind=args.llm_api_kind,
        request_timeout=args.llm_timeout,
        num_ctx=args.llm_num_ctx,
        max_http_retries=1,
    )
    runs: list[dict] = []
    for repeat_index in range(1, args.repeat + 1):
        started = time.perf_counter()
        record: dict = {
            "schema": "sa_nav.map_grounded_heading_run",
            "schema_version": 1,
            "status": "error",
            "current_room_id": current_room_id,
            "target_room_id": target_room_id,
            "map_mode": args.map_mode,
            "repeat_index": repeat_index,
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
            },
        }
        try:
            wrapped = experiment.run_once(
                current_room_id=current_room_id,
                target_room_id=target_room_id,
                semantic_map_path=semantic_map_path,
                map_mode=args.map_mode,
            )
            response_payload = wrapped.pop("response_payload", None)
            result = wrapped["result"]
            predicted_heading = float(result["heading_deg"])
            error = angular_error_deg(predicted_heading, args.expected_heading_deg)
            record.update(
                {
                    "status": "completed",
                    "request": request_diagnostics(experiment.last_request_body),
                    "response": response_diagnostics(response_payload),
                    "result": result,
                    "evaluation": {
                        "expected_heading_deg": args.expected_heading_deg,
                        "predicted_heading_deg": predicted_heading,
                        "angular_error_deg": error,
                        "exact_correct": error == 0.0 if error is not None else None,
                        "within_45_correct": error <= 45.0 if error is not None else None,
                    },
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
        runs.append(record)

    metrics = summarize_runs(runs, expected_heading_deg=args.expected_heading_deg)
    summary = {
        "schema": "sa_nav.map_grounded_heading_summary",
        "schema_version": 1,
        "current_room_id": current_room_id,
        "target_room_id": target_room_id,
        "config": {
            "repeat": args.repeat,
            "map_mode": args.map_mode,
            "model": args.llm_model,
            "api_kind": args.llm_api_kind,
            "temperature": 0.0,
            "expected_heading_deg": args.expected_heading_deg,
        },
        "paths": {
            "map_image": str(map_image_path),
            "map_regions": str(map_regions_path),
            "semantic_map": str(semantic_map_path),
            "runs": str(output_dir / "runs.jsonl"),
            "output_dir": str(output_dir),
        },
        "map": map_diagnostics,
        "metrics": metrics,
    }
    _write_jsonl(output_dir / "runs.jsonl", runs)
    _write_json(output_dir / "summary.json", summary)
    return summary


def main() -> int:
    args = build_parser().parse_args()
    try:
        summary = run_experiment(args)
    except Exception as exc:
        print(render_json({"status": "error", "error": {"type": type(exc).__name__, "message": str(exc)}}))
        return 1
    if args.json:
        print(render_json(summary))
    else:
        metrics = summary["metrics"]
        print(
            "\n".join(
                [
                    "Map-only passage-heading experiment completed",
                    f"Output: {summary['paths']['output_dir']}",
                    f"Directions: {', '.join(metrics['directions']) or '(none)'}",
                    f"Stable majority: {metrics['majority_direction']}",
                    f"Schema-valid rate: {metrics['schema_valid_rate']}",
                ]
            )
        )
    return 0 if summary["metrics"]["completed_run_count"] == summary["metrics"]["requested_run_count"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
