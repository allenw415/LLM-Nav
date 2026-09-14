from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import tempfile
import time

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from st_nav import NavigationPipelineConfig, build_navigation_pipeline, load_dotenv, resolve_model_environment
from st_nav.cli._common import load_normalized_artifacts
from st_nav.common.model_client import MODEL_TELEMETRY_COUNTER_KEYS
from st_nav.evaluation import (
    build_cache_inventory,
    compute_navigation_spl_metrics,
    compute_step_budget,
    filter_floor_artifacts,
    pano_path_distance_m,
    summarize_batch_results,
)
from st_nav.evaluation.navigation_batch import load_jsonl


DEFAULT_TEST_SET = (
    "outputs/navigation_paths/floor0_semantic_targets_topology_constrained/"
    "room_count_2to5_ratio_1p0_1p8_per10/test_set.jsonl"
)
DEFAULT_OUTPUT_DIR = (
    "outputs/navigation_evaluation/floor0_semantic_targets_topology_constrained/"
    "room_count_2to5_ratio_1p0_1p8_per10/llm_nav"
)
DEFAULT_RENDER_ROOT = "renders/room_grounding_fov90"
DEFAULT_ARTIFACTS = "dataset/sites/british_museum/normalized"

load_dotenv(PROJECT_ROOT / ".env")
MODEL_ENV = resolve_model_environment(
    default_model="gpt-5-mini",
    default_api_base="https://api.openai.com/v1",
    default_api_kind="responses",
)


def resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate.resolve() if candidate.is_absolute() else (PROJECT_ROOT / candidate).resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the floor-0 LLM-Nav benchmark sequentially with a shared render cache."
    )
    parser.add_argument("--test-set", default=DEFAULT_TEST_SET)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--artifacts-dir", default=DEFAULT_ARTIFACTS)
    parser.add_argument("--render-output-dir", default=DEFAULT_RENDER_ROOT)
    parser.add_argument(
        "--render-heading-mode",
        choices=["museum", "cardinal", "cardinal4"],
        default="museum",
        help="Render eight museum views or four absolute cardinal views.",
    )
    parser.add_argument("--floor", default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--test-id", action="append", default=[])
    parser.add_argument(
        "--inventory-only",
        action="store_true",
        help="Print dataset/cache coverage and exit without calling any model or image API.",
    )
    parser.add_argument(
        "--render-missing",
        action="store_true",
        help="Allow lazy Street View image requests for actually visited cache misses.",
    )
    parser.add_argument("--render-api-key", default=os.environ.get("GMAPS_API_KEY"))
    parser.add_argument("--llm-model", default=MODEL_ENV.model_name)
    parser.add_argument("--llm-api-key", default=MODEL_ENV.api_key)
    parser.add_argument("--llm-api-kind", default=MODEL_ENV.api_kind)
    parser.add_argument("--llm-api-base", default=MODEL_ENV.api_base)
    parser.add_argument("--llm-timeout", type=float, default=MODEL_ENV.request_timeout or 300.0)
    parser.add_argument("--localization-interval", type=int, default=3)
    parser.add_argument(
        "--localization-effective-stage",
        choices=["visual", "bayesian", "spatial"],
        default="spatial",
        help=(
            "Choose which localization stage updates navigation room state. "
            "Step 0 still runs spatial alignment when initial-spatial heading alignment is enabled."
        ),
    )
    parser.add_argument("--start-heading", type=float, default=0.0)
    parser.add_argument(
        "--heading-alignment-mode",
        choices=["fixed", "initial-spatial"],
        default="fixed",
    )
    parser.add_argument("--fixed-heading-offset-deg", type=float, default=30.0)
    parser.add_argument(
        "--force-initial-visual-refresh",
        action="store_true",
        help="Bypass cached VLM detection output for step-0 localization only.",
    )
    parser.add_argument(
        "--direction-policy",
        choices=["geometry", "geometry-edge-visit"],
        default="geometry",
        help="Use the unchanged geometry baseline or opt into edge-visit cycle avoidance.",
    )
    parser.add_argument("--no-detection-cache", action="store_true")
    return parser


def _atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        temporary_path = Path(stream.name)
    os.replace(temporary_path, path)


def _dataset_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_tests(tests: list[dict], *, floor: str, pano_graph: dict[str, dict]) -> None:
    test_ids: set[str] = set()
    for index, test in enumerate(tests, start=1):
        test_id = test.get("test_id")
        if not isinstance(test_id, str) or not test_id:
            raise ValueError(f"Dataset record {index} has no test_id.")
        if test_id in test_ids:
            raise ValueError(f"Duplicate test_id: {test_id}")
        test_ids.add(test_id)
        if str(test.get("floor")) != str(floor):
            raise ValueError(f"{test_id} is on floor {test.get('floor')}, expected {floor}.")
        start_pano_id = test.get("start_pano_id")
        if start_pano_id not in pano_graph:
            raise ValueError(f"{test_id} start pano is absent from the floor graph: {start_pano_id}")


def _grounding_mappings(payload: dict) -> dict:
    mappings = payload.get("mappings", payload)
    return mappings if isinstance(mappings, dict) else {}


def _pipeline_model_clients(pipeline) -> dict[str, object]:
    clients: dict[str, object] = {}

    parser = getattr(getattr(pipeline, "source_resolution_workflow", None), "instruction_parser", None)
    parser_client = getattr(parser, "model_client", None)
    if hasattr(parser_client, "telemetry_snapshot"):
        clients["query_parser"] = parser_client

    runner = getattr(pipeline, "episode_runner", None)
    perception_provider = getattr(runner, "perception_provider", None)
    perception_pipeline = getattr(perception_provider, "pipeline", None)
    detector = getattr(perception_pipeline, "detector", None)
    detector_client = getattr(detector, "model_client", None)
    if hasattr(detector_client, "telemetry_snapshot"):
        clients["detection"] = detector_client

    spatial_engine = getattr(runner, "spatial_engine", None)
    state_estimator = getattr(spatial_engine, "state_estimator", None)
    localizer = getattr(state_estimator, "localizer", None)
    spatial_refiner = getattr(localizer, "spatial_refiner", None)
    spatial_client = getattr(spatial_refiner, "model_client", None)
    if hasattr(spatial_client, "telemetry_snapshot"):
        clients["spatial_alignment"] = spatial_client

    policy_client = getattr(getattr(runner, "policy", None), "model_client", None)
    if hasattr(policy_client, "telemetry_snapshot"):
        clients["action_policy"] = policy_client
    return clients


def _model_telemetry_snapshot(clients: dict[str, object]) -> dict[str, dict[str, int]]:
    return {
        stage: client.telemetry_snapshot()
        for stage, client in clients.items()
    }


def _model_telemetry_delta(
    clients: dict[str, object],
    before: dict[str, dict[str, int]],
) -> dict:
    stages: dict[str, dict[str, int]] = {}
    totals = {key: 0 for key in MODEL_TELEMETRY_COUNTER_KEYS}
    for stage, client in clients.items():
        after = client.telemetry_snapshot()
        stage_before = before.get(stage, {})
        delta = {
            key: int(after.get(key, 0)) - int(stage_before.get(key, 0))
            for key in MODEL_TELEMETRY_COUNTER_KEYS
        }
        stages[stage] = delta
        for key, value in delta.items():
            totals[key] += value
    return {
        "schema": "model_call_telemetry",
        "schema_version": 1,
        "stages": stages,
        "total": totals,
    }


class BatchProgress:
    def __init__(self) -> None:
        self.cache_hit_events = 0
        self.cache_miss_events = 0
        self.cache_hit_pano_ids: set[str] = set()
        self.cache_miss_pano_ids: set[str] = set()
        self.new_render_pano_ids: set[str] = set()
        self.new_image_requests = 0
        self.stop_events: list[str] = []
        self.trace_movements: list[dict] = []
        self.localization_events: list[dict] = []
        self.module_timings: dict[str, dict] = {}
        self.end_to_end_wall_ms: float | None = None
        self.episode_runner_wall_ms: float | None = None

    def __call__(self, event: dict) -> None:
        event_name = event.get("event")
        pano_id = event.get("pano_id")
        if event_name == "module_timing":
            module = event.get("module")
            if isinstance(module, str) and module:
                record = self.module_timings.setdefault(
                    module,
                    {
                        "total_wall_ms": 0.0,
                        "vlm_response_ms": 0.0,
                        "program_module_ms": 0.0,
                        "call_count": 0,
                        "vlm_timing_missing_count": 0,
                    },
                )
                for key in ("total_wall_ms", "program_module_ms"):
                    value = event.get(key)
                    if isinstance(value, (int, float)):
                        record[key] += float(value)
                vlm_value = event.get("vlm_response_ms")
                if isinstance(vlm_value, (int, float)):
                    record["vlm_response_ms"] += float(vlm_value)
                else:
                    record["vlm_timing_missing_count"] += 1
                count = event.get("call_count")
                record["call_count"] += int(count) if isinstance(count, int) else 1
        elif event_name == "pipeline_done":
            value = event.get("end_to_end_wall_ms")
            if isinstance(value, (int, float)):
                self.end_to_end_wall_ms = float(value)
        elif event_name == "episode_done":
            value = event.get("episode_runner_wall_ms")
            if isinstance(value, (int, float)):
                self.episode_runner_wall_ms = float(value)
        elif event_name == "manifest_cache_hit":
            self.cache_hit_events += 1
            if isinstance(pano_id, str):
                self.cache_hit_pano_ids.add(pano_id)
        elif event_name == "manifest_cache_miss":
            self.cache_miss_events += 1
            if isinstance(pano_id, str):
                self.cache_miss_pano_ids.add(pano_id)
        elif event_name == "render_done":
            if isinstance(pano_id, str):
                self.new_render_pano_ids.add(pano_id)
        elif event_name == "render_capture_start":
            self.new_image_requests += 1
        elif isinstance(event_name, str) and event_name.startswith("stop_"):
            self.stop_events.append(event_name)
        elif event_name == "localization_done":
            localization_event = event.get("localization_event")
            if isinstance(localization_event, dict):
                self.localization_events.append(json.loads(json.dumps(localization_event)))
        elif event_name == "trace_recorded":
            trace = event.get("trace")
            action = trace.get("policy_output", {}).get("action") if isinstance(trace, dict) else None
            if isinstance(action, dict):
                self.trace_movements.append(
                    {
                        "from_pano_id": trace.get("pano_id"),
                        "to_pano_id": action.get("target_pano_id"),
                    }
                )

    def timing_summary(self, *, fallback_end_to_end_ms: float) -> dict:
        end_to_end_ms = (
            self.end_to_end_wall_ms
            if self.end_to_end_wall_ms is not None
            else float(fallback_end_to_end_ms)
        )
        exclusive_total_ms = sum(
            float(record.get("total_wall_ms", 0.0))
            for record in self.module_timings.values()
        )
        total_vlm_ms = sum(
            float(record.get("vlm_response_ms", 0.0))
            for record in self.module_timings.values()
        )
        total_program_ms = sum(
            float(record.get("program_module_ms", 0.0))
            for record in self.module_timings.values()
        )
        return {
            "schema": "navigation_episode_timing",
            "schema_version": 1,
            "end_to_end_wall_ms": end_to_end_ms,
            "episode_runner_wall_ms": self.episode_runner_wall_ms,
            "total_vlm_response_ms": total_vlm_ms,
            "total_program_module_ms": total_program_ms,
            "exclusive_module_total_ms": exclusive_total_ms,
            "unattributed_overhead_ms": max(0.0, end_to_end_ms - exclusive_total_ms),
            "timing_accounting_error_ms": end_to_end_ms - exclusive_total_ms,
            "modules": json.loads(json.dumps(self.module_timings)),
        }

    def render_summary(self) -> dict:
        return {
            "cache_hit_events": self.cache_hit_events,
            "cache_miss_events": self.cache_miss_events,
            "cache_hit_pano_ids": sorted(self.cache_hit_pano_ids),
            "cache_miss_pano_ids": sorted(self.cache_miss_pano_ids),
            "new_render_pano_count": len(self.new_render_pano_ids),
            "new_render_pano_ids": sorted(self.new_render_pano_ids),
            "new_image_requests": self.new_image_requests,
        }


def _movement_records(result) -> list[dict]:
    movements = []
    for trace in result.traces:
        action = trace.policy_output.action
        if action is None:
            continue
        movements.append(
            {
                "step_index": trace.step_index,
                "from_pano_id": trace.pano_id,
                "to_pano_id": action.target_pano_id,
                "estimated_room_id": trace.room_id,
                "subgoal_room_id": trace.subgoal_room_id,
                "geocentric_heading_deg": action.absolute_heading,
                "allocentric_heading_deg": action.metadata.get(
                    "allocentric_heading_deg"
                ),
                "heading_offset_used_deg": action.metadata.get(
                    "heading_offset_used_deg"
                ),
            }
        )
    return movements


def _stop_reason(*, success: bool, movement_steps: int, step_budget: int, events: BatchProgress) -> str:
    if success:
        return "goal_reached"
    if events.stop_events:
        return events.stop_events[-1]
    if movement_steps >= step_budget:
        return "step_budget_exhausted"
    return "navigation_incomplete"


def _offset_error_deg(predicted: object, expected: float = 30.0) -> float | None:
    if not isinstance(predicted, (int, float)):
        return None
    difference = abs((float(predicted) - float(expected)) % 360.0)
    return min(difference, 360.0 - difference)


def _case_result(
    *,
    test: dict,
    result,
    elapsed_seconds: float,
    step_budget: int,
    mappings: dict,
    pano_graph: dict[str, dict],
    events: BatchProgress,
    model_calls: dict,
    ground_truth_offset_deg: float = 30.0,
) -> dict:
    movements = _movement_records(result)
    pano_path = [test["start_pano_id"]] + [record["to_pano_id"] for record in movements]
    final_pano_id = result.final_state.current_pano_id
    final_ground_truth_room_id = mappings.get(final_pano_id)
    if final_ground_truth_room_id == "null":
        final_ground_truth_room_id = None
    acceptable = test.get("acceptable_target_room_ids")
    if not isinstance(acceptable, list) or not acceptable:
        acceptable = [test.get("target_room_id")]
    acceptable = [room_id for room_id in acceptable if isinstance(room_id, str) and room_id]
    success = final_ground_truth_room_id in set(acceptable)
    movement_steps = len(movements)
    shortest_steps = int(test["pano_step_count"])
    spl_metrics = compute_navigation_spl_metrics(
        success=success,
        shortest_path_distance_m=float(test["path_distance_m"]),
        actual_path_distance_m=pano_path_distance_m(pano_path, pano_graph),
        shortest_path_pano_steps=shortest_steps,
        movement_steps=movement_steps,
    )
    estimated_offset = result.final_state.geocentric_to_allocentric_offset_deg
    offset_error = _offset_error_deg(
        estimated_offset,
        expected=ground_truth_offset_deg,
    )
    return {
        "test_id": test["test_id"],
        "path_id": test.get("path_id"),
        "query_id": test.get("query_id"),
        "query": test["query"],
        "floor": str(test.get("floor")),
        "difficulty": test.get("difficulty"),
        "start_pano_id": test["start_pano_id"],
        "start_room_id": test.get("start_room_id"),
        "dataset_target_room_id": test.get("target_room_id"),
        "acceptable_target_room_ids": acceptable,
        "parsed_goal_room_ids": list(result.task.goal_room_ids),
        "parsed_waypoint_room_ids": list(result.task.waypoint_room_ids),
        "final_pano_id": final_pano_id,
        "final_ground_truth_room_id": final_ground_truth_room_id,
        "final_estimated_room_id": result.final_state.current_room_id,
        "success": success,
        **spl_metrics,
        "shortest_path_pano_steps": shortest_steps,
        "movement_steps": movement_steps,
        "step_budget": step_budget,
        "baseline_max_local_steps": 60,
        "llm_nav_local_step_limit": None,
        "stop_reason": _stop_reason(
            success=success,
            movement_steps=movement_steps,
            step_budget=step_budget,
            events=events,
        ),
        "elapsed_seconds": elapsed_seconds,
        "episode_timing": events.timing_summary(
            fallback_end_to_end_ms=elapsed_seconds * 1000.0
        ),
        "initial_heading_alignment": {
            "mode": result.final_state.heading_alignment_mode,
            "status": result.final_state.heading_alignment_status,
            "source_step_index": result.final_state.heading_alignment_source_step,
            "estimated_offset_deg": estimated_offset,
            "ground_truth_offset_deg": ground_truth_offset_deg,
            "circular_offset_error_deg": offset_error,
            "offset_correct": (
                abs(offset_error) <= 1e-6 if offset_error is not None else None
            ),
            "view_0_geocentric_heading_deg": (
                result.final_state.heading_alignment_view_0_geocentric_heading_deg
            ),
            "view_0_allocentric_direction": (
                result.final_state.heading_alignment_view_0_allocentric_direction
            ),
            "confidence": result.final_state.heading_alignment_confidence,
            "failure_reason": result.final_state.heading_alignment_failure_reason,
        },
        "pano_path": pano_path,
        "movements": movements,
        "render": events.render_summary(),
        "model_calls": model_calls,
        "localization_event_schema_version": 2,
        "localization_event_count": len(events.localization_events),
        "localization_events": events.localization_events,
        "error": None,
    }


def _error_result(
    *,
    test: dict,
    error: Exception,
    elapsed_seconds: float,
    step_budget: int,
    pano_graph: dict[str, dict],
    events: BatchProgress,
    model_calls: dict,
) -> dict:
    pano_path = [test["start_pano_id"]] + [
        movement["to_pano_id"] for movement in events.trace_movements
    ]
    movement_steps = len(events.trace_movements)
    shortest_steps = int(test["pano_step_count"])
    spl_metrics = compute_navigation_spl_metrics(
        success=False,
        shortest_path_distance_m=float(test["path_distance_m"]),
        actual_path_distance_m=pano_path_distance_m(pano_path, pano_graph),
        shortest_path_pano_steps=shortest_steps,
        movement_steps=movement_steps,
    )
    return {
        "test_id": test["test_id"],
        "path_id": test.get("path_id"),
        "query_id": test.get("query_id"),
        "query": test["query"],
        "floor": str(test.get("floor")),
        "difficulty": test.get("difficulty"),
        "start_pano_id": test["start_pano_id"],
        "dataset_target_room_id": test.get("target_room_id"),
        "acceptable_target_room_ids": test.get("acceptable_target_room_ids", []),
        "success": False,
        **spl_metrics,
        "shortest_path_pano_steps": shortest_steps,
        "movement_steps": movement_steps,
        "step_budget": step_budget,
        "baseline_max_local_steps": 60,
        "llm_nav_local_step_limit": None,
        "stop_reason": "error",
        "elapsed_seconds": elapsed_seconds,
        "episode_timing": events.timing_summary(
            fallback_end_to_end_ms=elapsed_seconds * 1000.0
        ),
        "pano_path": pano_path,
        "movements": events.trace_movements,
        "render": events.render_summary(),
        "model_calls": model_calls,
        "localization_event_schema_version": 2,
        "localization_event_count": len(events.localization_events),
        "localization_events": events.localization_events,
        "error": {"type": type(error).__name__, "message": str(error)},
    }


def _read_existing_results(path: Path) -> list[dict]:
    return load_jsonl(path) if path.is_file() else []


def _write_summary(path: Path, *, results: list[dict], expected_case_count: int) -> None:
    payload = summarize_batch_results(results)
    payload["expected_case_count"] = expected_case_count
    payload["completed_case_count"] = len(results)
    payload["pending_case_count"] = max(expected_case_count - len(results), 0)
    _atomic_write_json(path, payload)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.localization_interval < 1:
        parser.error("--localization-interval must be at least 1.")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1.")

    test_set_path = resolve(args.test_set)
    output_dir = resolve(args.output_dir)
    render_root = resolve(args.render_output_dir)
    artifacts_dir = resolve(args.artifacts_dir)
    tests = load_jsonl(test_set_path)
    if args.test_id:
        requested = set(args.test_id)
        tests = [test for test in tests if test.get("test_id") in requested]
        missing = requested - {test["test_id"] for test in tests}
        if missing:
            parser.error(f"Unknown --test-id values: {sorted(missing)}")

    artifacts = load_normalized_artifacts(
        artifacts_dir,
        room_graph=True,
        pano_graph=True,
        pano_room_grounding=True,
    )
    full_pano_graph = artifacts.pano_graph or {}
    floor_panos, floor_rooms, floor_grounding = filter_floor_artifacts(
        pano_graph=full_pano_graph,
        room_graph=artifacts.room_graph or {},
        pano_room_grounding=artifacts.pano_room_grounding or {},
        floor=args.floor,
    )
    _validate_tests(tests, floor=args.floor, pano_graph=floor_panos)

    reference_panos = {
        pano_id
        for test in tests
        for pano_id in test.get("shortest_path_panos", [])
        if isinstance(pano_id, str)
    }
    start_panos = {test["start_pano_id"] for test in tests}
    inventory = build_cache_inventory(
        pano_graph=full_pano_graph,
        render_root=render_root,
        floor=args.floor,
        reference_pano_ids=reference_panos,
        start_pano_ids=start_panos,
        heading_mode=args.render_heading_mode,
    )
    inventory["test_case_count"] = len(tests)
    print(json.dumps({"cache_inventory": inventory}, ensure_ascii=False, indent=2))
    if args.inventory_only:
        return 0

    if not args.llm_api_key:
        parser.error("Batch evaluation requires --llm-api-key or an active ST_NAV profile.")
    if args.render_missing and not args.render_api_key:
        parser.error("--render-missing requires --render-api-key or GMAPS_API_KEY.")

    direction_policy = getattr(args, "direction_policy", "geometry")
    config = {
        "schema": "llm_nav_floor0_batch_config",
        "schema_version": 6,
        "test_set": str(test_set_path),
        "dataset_sha256": _dataset_digest(test_set_path),
        "expected_case_count": len(tests),
        "floor": str(args.floor),
        "seed": args.seed,
        "direction_policy": direction_policy,
        "localization_interval": args.localization_interval,
        "localization_effective_stage": args.localization_effective_stage,
        "start_heading": args.start_heading,
        "heading_alignment": {
            "mode": args.heading_alignment_mode,
            "fixed_heading_offset_deg": args.fixed_heading_offset_deg,
            "force_initial_visual_refresh": bool(
                args.force_initial_visual_refresh
            ),
            "offset_write_policy": "step_0_once_then_locked",
            "initial_failure_policy": "stop_without_fixed_offset_fallback",
        },
        "step_budget_formula": "min(300, max(50, ceil(3 * pano_step_count)))",
        "evaluation_metrics": {
            "spl": (
                "success * shortest_path_distance_m / "
                "max(actual_path_distance_m, shortest_path_distance_m)"
            ),
            "distance_unit": "metres",
            "step_spl_legacy_retained": True,
        },
        "baseline_max_local_steps": 60,
        "llm_nav_local_step_limit": None,
        "render": {
            "root": str(render_root),
            "render_missing": bool(args.render_missing),
            "heading_mode": args.render_heading_mode,
            "pitch": 0.0,
            "fov": 90,
            "width": 512,
            "height": 512,
            "candidate_theme_fov": None,
        },
        "llm": {
            "model": args.llm_model,
            "api_kind": args.llm_api_kind,
            "api_base": args.llm_api_base,
            "timeout": args.llm_timeout,
        },
        "model_call_telemetry": {
            "enabled": True,
            "logical_calls_include_internal_http_retries": False,
            "image_calls_are_requests_containing_input_image_blocks": True,
        },
        "localization_events": {
            "schema_version": 2,
            "unit": "localization_call",
            "tracking_steps_excluded": True,
            "stages": ["visual", "bayesian", "spatial"],
            "effective_stage": args.localization_effective_stage,
            "step_0_offset_estimation_stage": (
                "spatial" if args.heading_alignment_mode == "initial-spatial" else None
            ),
            "spatial_fallback": "bayesian_when_alignment_not_applied",
            "visual_cache_enabled": not args.no_detection_cache,
            "cache_validation": "schema_and_candidate_room_ids_without_model_prompt_fingerprint",
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "config.json"
    if config_path.is_file():
        existing_config = json.loads(config_path.read_text(encoding="utf-8"))
        if existing_config != config:
            raise RuntimeError(
                f"Existing config differs from this run: {config_path}. Use a new --output-dir."
            )
    else:
        _atomic_write_json(config_path, config)

    results_path = output_dir / "results.jsonl"
    summary_path = output_dir / "summary.json"
    existing_results = _read_existing_results(results_path)
    completed_ids = {record.get("test_id") for record in existing_results}
    pending_tests = [test for test in tests if test["test_id"] not in completed_ids]
    if args.limit is not None:
        pending_tests = pending_tests[: args.limit]

    random.seed(args.seed)
    pipeline = build_navigation_pipeline(
        room_graph=floor_rooms,
        pano_graph=floor_panos,
        grounding_payload={},
        pano_room_grounding=floor_grounding,
        config=NavigationPipelineConfig(
            llm_model=args.llm_model,
            llm_api_key=args.llm_api_key,
            llm_api_base=args.llm_api_base,
            llm_api_kind=args.llm_api_kind,
            llm_timeout=args.llm_timeout,
            localization_interval=args.localization_interval,
            localization_effective_stage=args.localization_effective_stage,
            direction_policy=direction_policy,
            heading_alignment_mode=args.heading_alignment_mode,
            fixed_heading_offset_deg=args.fixed_heading_offset_deg,
            force_initial_visual_refresh=bool(args.force_initial_visual_refresh),
            use_detection_cache=not args.no_detection_cache,
            renderer_seed=args.seed,
            render_metadata_preflight=True,
        ),
    )
    mappings = _grounding_mappings(floor_grounding)
    model_clients = _pipeline_model_clients(pipeline)

    all_results = list(existing_results)
    with results_path.open("a", encoding="utf-8") as result_stream:
        for position, test in enumerate(pending_tests, start=1):
            print(
                f"[start {position}/{len(pending_tests)}] {test['test_id']}",
                flush=True,
            )
            step_budget = compute_step_budget(test["pano_step_count"])
            events = BatchProgress()
            model_calls_before = _model_telemetry_snapshot(model_clients)
            started = time.perf_counter()
            try:
                result = pipeline.run(
                    test["query"],
                    start_pano_id=test["start_pano_id"],
                    manifest_paths={},
                    start_heading=args.start_heading,
                    step_budget=step_budget,
                    render_api_key=args.render_api_key,
                    render_missing=args.render_missing,
                    render_output_dir=str(render_root),
                    render_heading_mode=args.render_heading_mode,
                    render_pitch=0.0,
                    render_fov=90,
                    render_width=512,
                    render_height=512,
                    candidate_theme_fov=None,
                    candidate_theme_output_dir=None,
                    render_graph_path=artifacts_dir / "pano_graph.json",
                    progress_callback=events,
                )
                record = _case_result(
                    test=test,
                    result=result,
                    elapsed_seconds=time.perf_counter() - started,
                    step_budget=step_budget,
                    mappings=mappings,
                    pano_graph=floor_panos,
                    events=events,
                    model_calls=_model_telemetry_delta(model_clients, model_calls_before),
                )
            except Exception as error:
                record = _error_result(
                    test=test,
                    error=error,
                    elapsed_seconds=time.perf_counter() - started,
                    step_budget=step_budget,
                    pano_graph=floor_panos,
                    events=events,
                    model_calls=_model_telemetry_delta(model_clients, model_calls_before),
                )
            result_stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            result_stream.flush()
            os.fsync(result_stream.fileno())
            all_results.append(record)
            _write_summary(
                summary_path,
                results=all_results,
                expected_case_count=len(tests),
            )
            print(
                f"[{position}/{len(pending_tests)}] {record['test_id']} "
                f"success={record['success']} steps={record['movement_steps']}/{step_budget} "
                f"new_images={record['render']['new_image_requests']}",
                flush=True,
            )

    _write_summary(summary_path, results=all_results, expected_case_count=len(tests))
    print(json.dumps(summarize_batch_results(all_results), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
