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
from sa_nav.cli.run_spatial_alignment import (  # noqa: E402
    angular_error_deg,
    expected_orientation_from_manifest,
    expected_room_from_grounding,
    load_manifest_observation,
    request_diagnostics,
    resolve_input_paths,
    response_diagnostics,
    select_candidate_rooms,
)
from sa_nav.common.room_profiles import preferred_room_graph_path  # noqa: E402
from sa_nav.experiments.map_grounded_alignment import (  # noqa: E402
    MapGroundedSpatialAlignmentExperiment,
    render_semantic_map,
    validate_map_regions,
)
from sa_nav.perception.renderer import sanitize_name  # noqa: E402


load_dotenv(PROJECT_ROOT / ".env")
MODEL_ENV = resolve_model_environment(
    default_model="gpt-5-mini",
    default_api_base="https://api.openai.com/v1",
    default_api_kind="responses",
)
DEFAULT_LLM_NUM_CTX = resolve_task_num_ctx(
    "localization",
    fallback_num_ctx=MODEL_ENV.num_ctx,
    default_num_ctx=16384,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run standalone map-grounded, nine-image spatial alignment without changing navigation."
    )
    parser.add_argument("--pano-id", required=True)
    parser.add_argument("--render-output-dir", default="renders/room_grounding_fov90")
    parser.add_argument("--manifest-path")
    parser.add_argument("--scores-json-path")
    parser.add_argument("--artifacts-dir", default="dataset/sites/british_museum/normalized")
    parser.add_argument("--candidate-top-k", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument(
        "--map-image",
        default="dataset/sites/british_museum/maps/level_0_allocentric.png",
    )
    parser.add_argument(
        "--map-regions",
        default="dataset/sites/british_museum/maps/level_0_room_regions.json",
    )
    parser.add_argument("--expected-room-id")
    parser.add_argument("--llm-model", default=MODEL_ENV.model_name)
    parser.add_argument("--llm-api-key", default=MODEL_ENV.api_key)
    parser.add_argument("--llm-api-kind", default=MODEL_ENV.api_kind)
    parser.add_argument("--llm-api-base", default=MODEL_ENV.api_base)
    parser.add_argument("--llm-timeout", type=float, default=MODEL_ENV.request_timeout or 180.0)
    parser.add_argument("--llm-num-ctx", type=int, default=DEFAULT_LLM_NUM_CTX)
    parser.add_argument("--llm-max-output-tokens", type=int, default=1024)
    parser.add_argument("--output-dir")
    parser.add_argument("--json", action="store_true")
    return parser


def default_output_dir(pano_id: str) -> Path:
    return resolve_project_path(Path("outputs/spatial_alignment_map_grounded") / sanitize_name(pano_id))


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_json(payload), encoding="utf-8")


def _majority(values: list[str]) -> str | None:
    if not values:
        return None
    counts = Counter(values)
    return sorted(counts, key=lambda value: (-counts[value], value))[0]


def summarize_runs(
    runs: list[dict],
    *,
    expected_room_id: str | None,
    candidate_room_ids: list[str],
) -> dict:
    completed = [run for run in runs if run.get("status") == "completed"]
    predictions = [str(run["evaluation"]["predicted_room_id"]) for run in completed]
    directions = [
        str(direction)
        for run in completed
        if isinstance((direction := run["evaluation"].get("predicted_view_0_allocentric_direction")), str)
    ]
    pairs = list(combinations(predictions, 2))
    pairwise_agreement = (
        sum(1 for left, right in pairs if left == right) / len(pairs)
        if pairs
        else (1.0 if len(predictions) == 1 else None)
    )
    majority_prediction = _majority(predictions)
    room_correct_values = [run["evaluation"].get("room_correct") for run in completed]
    orientation_correct_values = [run["evaluation"].get("orientation_correct") for run in completed]
    latency_values = [float(run["latency_seconds"]) for run in completed]
    return {
        "requested_run_count": len(runs),
        "completed_run_count": len(completed),
        "schema_valid_rate": len(completed) / len(runs) if runs else None,
        "candidate_recall_at_5": expected_room_id in candidate_room_ids if expected_room_id else None,
        "predictions": predictions,
        "unanimous_prediction": len(predictions) == len(runs) and len(set(predictions)) == 1 if runs else None,
        "pairwise_prediction_agreement": pairwise_agreement,
        "majority_prediction": majority_prediction,
        "majority_vote_correct": majority_prediction == expected_room_id if expected_room_id and majority_prediction else None,
        "per_run_room_accuracy": (
            sum(1 for value in room_correct_values if value is True) / len(room_correct_values)
            if room_correct_values
            else None
        ),
        "directions": directions,
        "unanimous_orientation": len(directions) == len(runs) and len(set(directions)) == 1 if runs else None,
        "per_run_orientation_accuracy": (
            sum(1 for value in orientation_correct_values if value is True) / len(orientation_correct_values)
            if any(value is not None for value in orientation_correct_values)
            else None
        ),
        "mean_latency_seconds": sum(latency_values) / len(latency_values) if latency_values else None,
    }


def run_experiment(args: argparse.Namespace) -> dict:
    if args.candidate_top_k != 5:
        raise ValueError("Map-grounded v1 requires --candidate-top-k 5.")
    if args.repeat < 1:
        raise ValueError("--repeat must be at least 1.")

    output_dir = resolve_project_path(args.output_dir) if args.output_dir else default_output_dir(args.pano_id)
    manifest_path, scores_path = resolve_input_paths(
        pano_id=args.pano_id,
        render_output_dir=args.render_output_dir,
        manifest_path=args.manifest_path,
        scores_json_path=args.scores_json_path,
    )
    artifacts_dir = resolve_project_path(args.artifacts_dir)
    room_graph_path = preferred_room_graph_path(artifacts_dir)
    grounding_path = artifacts_dir / "pano_room_grounding.json"
    map_image_path = resolve_project_path(args.map_image)
    map_regions_path = resolve_project_path(args.map_regions)
    semantic_map_path = output_dir / "semantic_map.png"

    room_graph = load_json(room_graph_path)
    grounding_payload = load_json(grounding_path)
    region_payload = load_json(map_regions_path)
    validate_map_regions(room_graph, region_payload)
    manifest, observation, views = load_manifest_observation(manifest_path=manifest_path, pano_id=args.pano_id)
    if manifest.get("heading_mode") != "museum":
        raise ValueError("Map-grounded alignment requires a manifest with heading_mode=museum.")
    selected_candidates = select_candidate_rooms(
        scores_path=scores_path,
        room_graph=room_graph,
        top_k=args.candidate_top_k,
    )
    candidate_room_ids = sorted(str(record["room_id"]) for record in selected_candidates)
    map_diagnostics = render_semantic_map(
        base_map_path=map_image_path,
        region_payload=region_payload,
        room_graph=room_graph,
        candidate_room_ids=candidate_room_ids,
        output_path=semantic_map_path,
    )
    expected_room_id = args.expected_room_id or expected_room_from_grounding(grounding_payload, args.pano_id)
    expected_direction, expected_offset = expected_orientation_from_manifest(manifest)

    experiment = MapGroundedSpatialAlignmentExperiment(
        room_graph=room_graph,
        model=args.llm_model,
        api_key=args.llm_api_key,
        api_base=args.llm_api_base,
        api_kind=args.llm_api_kind,
        request_timeout=args.llm_timeout,
        num_ctx=args.llm_num_ctx,
        max_http_retries=1,
        validation_retries=0,
        max_output_tokens=args.llm_max_output_tokens,
        compact_decision_output=True,
    )
    runs: list[dict] = []
    for repeat_index in range(1, args.repeat + 1):
        started_at = time.perf_counter()
        run_payload: dict = {
            "schema": "sa_nav.map_grounded_spatial_alignment",
            "schema_version": 1,
            "status": "error",
            "mode": "map_grounded_direct_images_single_call",
            "pano_id": args.pano_id,
            "repeat_index": repeat_index,
            "model": {"name": args.llm_model, "api_kind": args.llm_api_kind, "temperature": 0.0},
            "request_policy": {
                "logical_vlm_call_limit": 1,
                "http_retry_limit": 1,
                "reasoning_effort": "medium",
                "perception_fallback_enabled": False,
                "output_mode": "compact_spatial_reasoning",
                "max_output_tokens": args.llm_max_output_tokens,
            },
            "views": views,
            "map": map_diagnostics,
            "candidate_room_ids_sent_to_vlm": sorted(candidate_room_ids),
            "candidate_scores_sent_to_vlm": False,
            "ground_truth_sent_to_vlm": False,
            "manifest_cardinal_labels_sent_to_vlm": False,
        }
        try:
            result = experiment.run_once(
                observation=observation,
                manifest=manifest,
                candidate_room_ids=candidate_room_ids,
                semantic_map_path=semantic_map_path,
            )
            raw_response = result.pop("alignment_response_payload", None)
            parsed = result.get("result", {})
            run_payload["request"] = request_diagnostics(experiment.last_request_body)
            run_payload["request"]["base64_images_saved"] = False
            run_payload["response"] = response_diagnostics(raw_response)
            run_payload["raw_response_payload"] = raw_response
            run_payload["candidate_cards"] = result.get("candidate_cards", [])
            run_payload["result"] = parsed
            predicted_room_id = parsed.get("predicted_room_id")
            predicted_direction = parsed.get("view_0_allocentric_direction")
            predicted_offset = parsed.get("geocentric_to_allocentric_offset_deg")
            run_payload["evaluation"] = {
                "expected_room_id": expected_room_id,
                "candidate_contains_expected_room": expected_room_id in candidate_room_ids if expected_room_id else None,
                "predicted_room_id": predicted_room_id,
                "room_correct": predicted_room_id == expected_room_id if expected_room_id else None,
                "expected_view_0_allocentric_direction": expected_direction,
                "predicted_view_0_allocentric_direction": predicted_direction,
                "orientation_correct": predicted_direction == expected_direction if expected_direction else None,
                "expected_geocentric_to_allocentric_offset_deg": expected_offset,
                "predicted_geocentric_to_allocentric_offset_deg": predicted_offset,
                "offset_error_deg": angular_error_deg(predicted_offset, expected_offset),
            }
            run_payload["status"] = "completed"
        except Exception as exc:
            run_payload["error"] = {"type": type(exc).__name__, "message": str(exc)}
            run_payload["request"] = request_diagnostics(experiment.last_request_body)
            run_payload["request"]["base64_images_saved"] = False
            run_payload["response"] = response_diagnostics(experiment.last_response_payload)
            run_payload["raw_response_payload"] = experiment.last_response_payload
        run_payload["latency_seconds"] = round(time.perf_counter() - started_at, 6)
        _write_json(output_dir / f"run_{repeat_index:02d}.json", run_payload)
        runs.append(run_payload)

    summary = {
        "schema": "sa_nav.map_grounded_spatial_alignment_summary",
        "schema_version": 1,
        "pano_id": args.pano_id,
        "mode": "map_grounded_direct_images_single_call",
        "paths": {
            "manifest": str(manifest_path),
            "scores": str(scores_path),
            "room_graph": str(room_graph_path),
            "grounding": str(grounding_path),
            "map_image": str(map_image_path),
            "map_regions": str(map_regions_path),
            "semantic_map": str(semantic_map_path),
            "output_dir": str(output_dir),
        },
        "config": {
            "candidate_top_k": args.candidate_top_k,
            "repeat": args.repeat,
            "model": args.llm_model,
            "api_kind": args.llm_api_kind,
            "temperature": 0.0,
            "reasoning_effort": "medium",
            "output_mode": "compact_spatial_reasoning",
            "max_output_tokens": args.llm_max_output_tokens,
        },
        "candidate_selection": {
            "source": "visual_localization.room_scores",
            "records": selected_candidates,
            "scores_sent_to_alignment_vlm": False,
            "candidate_room_ids_sent_to_vlm": sorted(candidate_room_ids),
        },
        "expected_room_id": expected_room_id,
        "map": map_diagnostics,
        "metrics": summarize_runs(
            runs,
            expected_room_id=expected_room_id,
            candidate_room_ids=candidate_room_ids,
        ),
        "run_files": [str(output_dir / f"run_{index:02d}.json") for index in range(1, args.repeat + 1)],
    }
    _write_json(output_dir / "summary.json", summary)
    return summary


def main() -> int:
    args = build_parser().parse_args()
    try:
        summary = run_experiment(args)
    except Exception as exc:
        payload = {"status": "error", "error": {"type": type(exc).__name__, "message": str(exc)}}
        print(render_json(payload))
        return 1
    if args.json:
        print(render_json(summary))
    else:
        metrics = summary["metrics"]
        print(
            "\n".join(
                [
                    "Map-grounded spatial alignment completed",
                    f"Output: {summary['paths']['output_dir']}",
                    f"Predictions: {', '.join(metrics['predictions']) or '(none)'}",
                    f"Expected room: {summary.get('expected_room_id')}",
                    f"Unanimous: {metrics.get('unanimous_prediction')}",
                    f"Per-run accuracy: {metrics.get('per_run_room_accuracy')}",
                ]
            )
        )
    return 0 if summary["metrics"]["completed_run_count"] == summary["metrics"]["requested_run_count"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
