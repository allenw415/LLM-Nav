from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from ._common import (
    PROJECT_ROOT,
    ensure_project_root_on_path,
    load_json,
    load_normalized_artifacts,
    render_json,
    resolve_project_path,
    write_text_if_requested,
)

ensure_project_root_on_path()

from st_nav import NavigationPipelineConfig, build_navigation_pipeline, load_dotenv, resolve_model_environment

load_dotenv(PROJECT_ROOT / ".env")
MODEL_ENV = resolve_model_environment(
    default_model="gpt-5-mini",
    default_api_base="https://api.openai.com/v1",
    default_api_kind="responses",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the end-to-end navigation loop.")
    parser.add_argument(
        "--instruction",
        help="Navigation query containing only the desired goal and optional ordered waypoints.",
    )
    parser.add_argument(
        "--start-pano-id",
        help="Required start panorama; its grounding determines the source room.",
    )
    parser.add_argument(
        "--target-room-id",
        help="Explicit target used instead of --instruction.",
    )
    parser.add_argument(
        "--waypoint-room-ids",
        nargs="*",
        default=None,
        help="Optional ordered room waypoints for explicit start/target navigation.",
    )
    parser.add_argument("--artifacts-dir", default="dataset/sites/british_museum/normalized")
    parser.add_argument("--alignment-candidate-ratio-threshold", type=float, default=0.5)
    parser.add_argument("--alignment-candidate-max", type=int, default=5)
    parser.add_argument(
        "--u-turn-threshold-deg",
        type=float,
        default=150.0,
        help=(
            "Treat candidates at or above this incoming-heading turn error as U-turns. "
            "Geometry policy excludes them when another direction is available, but allows them at dead ends."
        ),
    )
    parser.add_argument(
        "--history-return-threshold-deg",
        type=float,
        default=30.0,
        help=(
            "Avoid candidates within this angle of the reverse of the recent two-heading displacement. "
            "The filter uses heading history only and is relaxed when no alternative exists."
        ),
    )
    parser.add_argument(
        "--direction-policy",
        choices=["geometry", "geometry-edge-visit", "llm", "visual-theme"],
        default="geometry",
        help=(
            "Select plain geometry, opt-in geometry with edge-visit cycle avoidance, "
            "LLM reasoning, or visual-theme view selection."
        ),
    )
    parser.add_argument("--direction-cache-dir")
    parser.add_argument("--direction-max-output-tokens", type=int)
    parser.add_argument("--direction-schema-validation-retries", type=int, default=1)
    parser.add_argument("--manifest-map-json")
    parser.add_argument(
        "--no-detection-cache",
        action="store_true",
        help="Ignore sibling *_detections.json files and rerun VLM perception/localization.",
    )
    parser.add_argument("--step-budget", type=int, default=15)
    parser.add_argument(
        "--localization-interval",
        type=int,
        default=3,
        help="Run full visual localization initially and after this many movement actions.",
    )
    parser.add_argument(
        "--start-heading",
        type=float,
        default=0.0,
        help="Initial rendering reference. Return avoidance begins after graph-action heading history is available.",
    )
    parser.add_argument("--llm-model", default=MODEL_ENV.model_name)
    parser.add_argument("--llm-api-key", default=MODEL_ENV.api_key)
    parser.add_argument("--llm-api-kind", default=MODEL_ENV.api_kind)
    parser.add_argument("--llm-api-base", default=MODEL_ENV.api_base)
    parser.add_argument("--llm-timeout", type=float, default=MODEL_ENV.request_timeout or 30.0)
    parser.add_argument("--render-api-key", default=os.environ.get("GMAPS_API_KEY"))
    parser.add_argument(
        "--render-missing",
        action="store_true",
        help="Allow paid image requests only after the shared render cache misses.",
    )
    parser.add_argument("--render-output-dir", default="renders/room_grounding_fov90")
    parser.add_argument(
        "--render-heading-mode",
        choices=["museum", "cardinal", "cardinal4", "graph"],
        default="museum",
    )
    parser.add_argument("--render-pitch", type=float, default=0.0)
    parser.add_argument("--render-fov", type=int, default=90)
    parser.add_argument(
        "--candidate-theme-fov",
        type=int,
        default=None,
        help="Optional wider FOV used only for candidate theme/spatial-context reasoning. Defaults to the main render FOV.",
    )
    parser.add_argument("--render-width", type=int, default=512)
    parser.add_argument("--render-height", type=int, default=512)
    parser.add_argument(
        "--candidate-theme-output-dir",
        default=None,
        help="Optional output dir for candidate-theme renders. Defaults to --render-output-dir when omitted.",
    )
    parser.add_argument("--output-path")
    parser.add_argument("--log-path")
    return parser


def serialize_candidate(candidate) -> dict:
    return {
        "target_pano_id": candidate.target_pano_id,
        "target_room_id": candidate.target_room_id,
        "absolute_heading": candidate.absolute_heading,
        "relative_heading": candidate.relative_heading,
        "relative_label": candidate.relative_label,
        "route_step_index": candidate.route_step_index,
        "score": candidate.score,
        "reason": candidate.reason,
        "metadata": candidate.metadata,
    }


def serialize_trace(trace) -> dict:
    observation_metadata = trace.observation.metadata
    return {
        "step_index": trace.step_index,
        "pano_id": trace.pano_id,
        "room_id": trace.room_id,
        "grounded_room_id": observation_metadata.get("grounded_room_id"),
        "route": list(trace.route),
        "subgoal_room_id": trace.subgoal_room_id,
        "completed_waypoint_room_ids": list(
            trace.completed_waypoint_room_ids
        ),
        "remaining_waypoint_room_ids": list(
            trace.remaining_waypoint_room_ids
        ),
        "current_room_context": dict(trace.current_room_context),
        "visible_passages": list(trace.visible_passages),
        "view_contexts": list(trace.view_contexts),
        "candidates": [serialize_candidate(candidate) for candidate in trace.candidates],
        "observation": {
            "pano_id": trace.observation.pano_id,
            "heading_estimate": trace.observation.heading_estimate,
            "localization_performed": observation_metadata.get("localization_performed"),
            "localization_interval": observation_metadata.get("localization_interval"),
            "movement_steps_since_last_localization": observation_metadata.get(
                "movement_steps_since_last_localization"
            ),
            "localization_skipped_reason": observation_metadata.get("localization_skipped_reason"),
            "localized_room_id": observation_metadata.get("localized_room_id"),
            "localized_room_id_raw": observation_metadata.get("localized_room_id_raw"),
            "localization_stabilized": observation_metadata.get("localization_stabilized"),
            "pending_room_id": observation_metadata.get("pending_room_id"),
            "pending_room_confirmation_count": observation_metadata.get("pending_room_confirmation_count"),
            "room_switch_confirmations_required": observation_metadata.get(
                "room_switch_confirmations_required"
            ),
            "grounded_room_id": observation_metadata.get("grounded_room_id"),
            "transition_support": observation_metadata.get("transition_support")
            or observation_metadata.get("transition_room_support"),
            "evidence_distribution": observation_metadata.get("evidence_distribution"),
            "base_predicted_room_id": observation_metadata.get("base_predicted_room_id"),
            "base_room_belief": observation_metadata.get("base_room_belief"),
            "alignment_candidate_room_ids": observation_metadata.get("alignment_candidate_room_ids"),
            "alignment_top_k": observation_metadata.get("alignment_top_k"),
            "alignment_predicted_room_id": observation_metadata.get("alignment_predicted_room_id"),
            "alignment_applied": observation_metadata.get("alignment_applied"),
            "alignment_skipped_reason": observation_metadata.get("alignment_skipped_reason"),
            "observation_likelihood": observation_metadata.get("observation_likelihood")
            or observation_metadata.get("observation_distribution"),
            "room_belief": observation_metadata.get("room_belief"),
            "visual_localization": observation_metadata.get("visual_localization"),
            "spatial_alignment": observation_metadata.get("spatial_alignment"),
            "inside_entities": observation_metadata.get("inside_entities"),
            "outside_entities": observation_metadata.get("outside_entities"),
            "entities": [
                {
                    "name": entity.name,
                    "kind": entity.kind,
                    "confidence": entity.confidence,
                    "source_view": entity.source_view,
                    "source_views": entity.metadata.get("source_views"),
                    "location_scope": entity.location_scope,
                }
                for entity in trace.observation.entities
            ],
        },
        "policy_output": {
            "rationale": trace.policy_output.rationale,
            "action": serialize_candidate(trace.policy_output.action) if trace.policy_output.action else None,
        },
        "policy_debug": {
            "request": trace.policy_request,
            "response": trace.policy_response,
        },
    }


def build_pano_viewer_episode(
    navigation_payload: dict,
    *,
    step_budget: int | None = None,
) -> dict:
    source = navigation_payload.get("source")
    source = source if isinstance(source, dict) else {}
    task = navigation_payload.get("task")
    task = task if isinstance(task, dict) else {}
    final_state = navigation_payload.get("final_state")
    final_state = final_state if isinstance(final_state, dict) else {}
    traces = [
        trace
        for trace in navigation_payload.get("traces", [])
        if isinstance(trace, dict)
    ]

    start_pano_id = source.get("source_pano_id")
    if not isinstance(start_pano_id, str) or not start_pano_id:
        start_pano_id = (
            traces[0].get("pano_id")
            if traces
            else final_state.get("current_pano_id")
        )
    if not isinstance(start_pano_id, str) or not start_pano_id:
        raise ValueError(
            "Navigation output has no start panorama for pano viewer playback."
        )

    waypoint_room_ids = [
        room_id
        for room_id in task.get("waypoint_room_ids", [])
        if isinstance(room_id, str) and room_id
    ]
    goal_room_ids = [
        room_id
        for room_id in task.get("goal_room_ids", [])
        if isinstance(room_id, str) and room_id
    ]
    target_room_id = goal_room_ids[-1] if goal_room_ids else None
    waypoint_room_ids = [
        room_id for room_id in waypoint_room_ids if room_id != target_room_id
    ]
    pano_path = [start_pano_id]
    rounds = []
    completed_waypoints: list[str] = []
    next_waypoint_index = 0

    def trace_room_id(trace: dict) -> str | None:
        grounded_room_id = trace.get("grounded_room_id")
        if isinstance(grounded_room_id, str) and grounded_room_id:
            return grounded_room_id
        room_id = trace.get("room_id")
        return room_id if isinstance(room_id, str) and room_id else None

    def apply_trace_waypoint_progress(trace: dict) -> bool:
        nonlocal completed_waypoints, next_waypoint_index
        raw_completed = trace.get("completed_waypoint_room_ids")
        raw_remaining = trace.get("remaining_waypoint_room_ids")
        if not isinstance(raw_completed, list) or not isinstance(raw_remaining, list):
            return False
        completed_waypoints = [
            room_id
            for room_id in raw_completed
            if isinstance(room_id, str) and room_id
        ]
        next_waypoint_index = len(completed_waypoints)
        return True

    def advance_waypoint(room_id: str | None) -> None:
        nonlocal next_waypoint_index
        if (
            next_waypoint_index < len(waypoint_room_ids)
            and room_id == waypoint_room_ids[next_waypoint_index]
        ):
            completed_waypoints.append(waypoint_room_ids[next_waypoint_index])
            next_waypoint_index += 1

    for trace_position, trace in enumerate(traces):
        has_recorded_progress = apply_trace_waypoint_progress(trace)
        current_room_id = trace_room_id(trace)
        if not has_recorded_progress:
            advance_waypoint(current_room_id)

        policy_output = trace.get("policy_output")
        policy_output = policy_output if isinstance(policy_output, dict) else {}
        action = policy_output.get("action")
        if not isinstance(action, dict):
            break

        current_pano_id = trace.get("pano_id")
        next_pano_id = action.get("target_pano_id")
        if not isinstance(current_pano_id, str) or not isinstance(next_pano_id, str):
            raise ValueError(
                f"Navigation trace {trace_position} has an incomplete panorama action."
            )
        if current_pano_id != pano_path[-1]:
            raise ValueError(
                f"Navigation trace {trace_position} starts at {current_pano_id}, "
                f"but the trajectory is currently at {pano_path[-1]}."
            )

        next_trace = (
            traces[trace_position + 1]
            if trace_position + 1 < len(traces)
            else None
        )
        next_room_id = None
        if (
            isinstance(next_trace, dict)
            and next_trace.get("pano_id") == next_pano_id
        ):
            next_room_id = trace_room_id(next_trace)
        action_metadata = action.get("metadata")
        action_metadata = action_metadata if isinstance(action_metadata, dict) else {}
        if not next_room_id:
            grounded_target_room_id = action_metadata.get(
                "grounded_target_room_id"
            )
            if (
                isinstance(grounded_target_room_id, str)
                and grounded_target_room_id
            ):
                next_room_id = grounded_target_room_id
        if not next_room_id:
            target_action_room_id = action.get("target_room_id")
            if isinstance(target_action_room_id, str) and target_action_room_id:
                next_room_id = target_action_room_id

        movement = {
            "local_step_index": trace.get("step_index", trace_position),
            "current_pano_id": current_pano_id,
            "next_pano_id": next_pano_id,
            "current_room_id": current_room_id,
            "next_room_id": next_room_id,
            "selected_action_heading": action.get("absolute_heading"),
            "subgoal_room_id": trace.get("subgoal_room_id"),
            "active_target_room_id": target_room_id,
            "score": action.get("score"),
            "reason": action.get("reason"),
        }
        if not has_recorded_progress:
            advance_waypoint(next_room_id)
        round_payload = {
            "round_index": trace_position,
            "start_pano_id": current_pano_id,
            "active_target_room_id": target_room_id,
            "subgoal_room_id": trace.get("subgoal_room_id"),
            "completed_waypoints": list(completed_waypoints),
            "remaining_waypoints": list(
                waypoint_room_ids[next_waypoint_index:]
            ),
            "movement_steps": [movement],
        }
        if (
            current_room_id
            and next_room_id
            and current_room_id != next_room_id
        ):
            round_payload["room_boundary"] = {
                "from_room_id": current_room_id,
                "to_room_id": next_room_id,
                "at_pano_id": next_pano_id,
            }

        rounds.append(round_payload)
        pano_path.append(next_pano_id)

    final_trace = traces[-1] if traces else {}
    apply_trace_waypoint_progress(final_trace)
    final_observed_room_id = trace_room_id(final_trace)
    waypoints_complete = completed_waypoints == waypoint_room_ids
    success = bool(
        target_room_id
        and final_observed_room_id == target_room_id
        and waypoints_complete
    )
    if success:
        reason = "goal_reached"
    elif traces and not isinstance(
        traces[-1].get("policy_output", {}).get("action"),
        dict,
    ):
        reason = "stopped_without_action"
    elif step_budget is not None and len(traces) >= step_budget:
        reason = "step_budget_exhausted"
    else:
        reason = "navigation_incomplete"

    return {
        "schema": "st_nav_navigation_episode",
        "schema_version": 1,
        "start_pano_id": start_pano_id,
        "final_pano_id": pano_path[-1],
        "target_room_id": target_room_id,
        "pano_path": pano_path,
        "waypoint_room_ids": waypoint_room_ids,
        "completed_waypoints": completed_waypoints,
        "remaining_waypoints": waypoint_room_ids[next_waypoint_index:],
        "ordered_targets": waypoint_room_ids + goal_room_ids,
        "success": success,
        "reason": reason,
        "rounds": rounds,
    }

def print_trace_summary(result, *, stream=None) -> None:
    stream = stream or sys.stderr
    lines = ["Traversed panoramas:"]
    if not result.traces:
        lines.append(
            f"- start: {result.source.source_pano.pano_id} -> {result.task.source_room_id or 'unknown room'}"
        )
    else:
        first_trace = result.traces[0]
        start_room_label = (
            first_trace.observation.metadata.get("grounded_room_id")
            or result.task.source_room_id
            or first_trace.room_id
            or "unknown room"
        )
        lines.append(f"- start: {result.source.source_pano.pano_id} -> {start_room_label}")
        for trace in result.traces:
            room_label = trace.observation.metadata.get("grounded_room_id") or trace.room_id or "unknown room"
            lines.append(f"- step {trace.step_index}: {trace.pano_id} -> {room_label}")
        final_room_label = getattr(result.final_state, "grounded_room_id", None) or result.final_state.current_room_id or "unknown room"
        trace_room_label = result.traces[-1].observation.metadata.get("grounded_room_id") or result.traces[-1].room_id or "unknown room"
        if result.final_state.current_pano_id != result.traces[-1].pano_id or final_room_label != trace_room_label:
            lines.append(f"- final: {result.final_state.current_pano_id} -> {final_room_label}")
    print("\n".join(lines), file=stream)


def print_progress(event: dict, *, stream=None) -> None:
    stream = stream or sys.stderr
    event_name = event.get("event")
    if event_name == "pipeline_start":
        print("[progress] pipeline start", file=stream)
        return
    if event_name == "source_resolution_start":
        print("[progress] parsing instruction and resolving source pano", file=stream)
        return
    if event_name == "source_resolution_done":
        print(
            "[progress] source resolved:"
            f" room={event.get('source_room_id')} pano={event.get('source_pano_id')}"
            f" goals={event.get('goal_room_ids')}",
            file=stream,
        )
        return
    if event_name == "episode_start":
        print(
            "[progress] episode start:"
            f" pano={event.get('start_pano_id')} room={event.get('start_room_id')}"
            f" step_budget={event.get('step_budget')}",
            file=stream,
        )
        return
    if event_name == "step_start":
        print(
            f"[progress] step {event.get('step_index')} start:"
            f" pano={event.get('current_pano_id')}",
            file=stream,
        )
        return
    if event_name == "manifest_cache_miss":
        print(
            f"[progress] manifest cache miss: pano={event.get('pano_id')}"
            f" reason={event.get('reason')} manifest={event.get('manifest_path')}",
            file=stream,
        )
        return
    if event_name == "manifest_cache_hit":
        print(
            f"[progress] manifest cache hit: pano={event.get('pano_id')}"
            f" source={event.get('source')} manifest={event.get('manifest_path')}",
            file=stream,
        )
        return
    if event_name == "render_cached":
        print(
            f"[progress] render cached: pano={event.get('pano_id')} manifest={event.get('manifest_path')}",
            file=stream,
        )
        return
    if event_name == "render_capture_start":
        print(
            f"[progress] rendering capture {event.get('capture_index')}/{event.get('capture_count')}:"
            f" pano={event.get('pano_id')} label={event.get('label')} heading={event.get('heading')}",
            file=stream,
        )
        return
    if event_name == "render_done":
        print(
            f"[progress] render done: pano={event.get('pano_id')} manifest={event.get('manifest_path')}",
            file=stream,
        )
        return
    if event_name == "perception_start":
        print(
            f"[progress] perception start: step={event.get('step_index')} pano={event.get('pano_id')}",
            file=stream,
        )
        return
    if event_name == "perception_done":
        print(
            f"[progress] perception done: step={event.get('step_index')}"
            f" pano={event.get('pano_id')} views={event.get('view_count')} entities={event.get('entity_count')}",
            file=stream,
        )
        return
    if event_name in {"localization_done", "localization_skipped"}:
        status = "done" if event_name == "localization_done" else "skipped"
        pending_room_id = event.get("pending_room_id")
        pending_text = ""
        if pending_room_id:
            pending_text = (
                f" pending_room={pending_room_id}"
                f" confirmation={event.get('pending_room_confirmation_count')}"
                f"/{event.get('room_switch_confirmations_required')}"
            )
        print(
            f"[progress] localization {status}: step={event.get('step_index')}"
            f" pano={event.get('current_pano_id')}"
            f" moves_since_last={event.get('movement_steps_since_last_localization')}"
            f" interval={event.get('localization_interval')}"
            f" localized_room={event.get('current_room_id')}"
            f" raw_localized_room={event.get('raw_localized_room_id')}"
            f" grounded_room={event.get('grounded_room_id')}"
            f"{pending_text}",
            file=stream,
        )
        return
    if event_name == "waypoint_completed":
        print(
            f"[progress] waypoint completed: step={event.get('step_index')}"
            f" room={event.get('room_id')}"
            f" remaining={event.get('remaining_waypoint_room_ids')}",
            file=stream,
        )
        return
    if event_name == "route_done":
        print(
            f"[progress] route ready: step={event.get('step_index')}"
            f" subgoal={event.get('subgoal_room_id')} candidates={event.get('candidate_count')}"
            f" route={event.get('route')}",
            file=stream,
        )
        return
    if event_name == "reasoning_done":
        print(
            f"[progress] reasoning done: step={event.get('step_index')}"
            f" chosen_pano={event.get('chosen_pano_id')}"
            f" chosen_room={event.get('chosen_room_id')}"
            f" chosen_grounded_room={event.get('chosen_grounded_room_id')}",
            file=stream,
        )
        return
    if event_name == "action_applied":
        print(
            f"[progress] action applied: step={event.get('step_index')}"
            f" next_pano={event.get('next_pano_id')}",
            file=stream,
        )
        return
    if event_name == "goal_reached":
        print(
            f"[progress] goal reached: step={event.get('step_index')}"
            f" pano={event.get('current_pano_id')}"
            f" localized_room={event.get('current_room_id')}"
            f" grounded_room={event.get('grounded_room_id')}",
            file=stream,
        )
        return
    if event_name == "stop_no_action":
        print(
            f"[progress] stop: no action at step={event.get('step_index')}"
            f" pano={event.get('current_pano_id')}"
            f" localized_room={event.get('current_room_id')}"
            f" grounded_room={event.get('grounded_room_id')}",
            file=stream,
        )
        return
    if event_name == "stop_no_manifest":
        print(
            f"[progress] stop: missing manifest at step={event.get('step_index')}"
            f" pano={event.get('pano_id')}",
            file=stream,
        )
        return
    if event_name == "episode_done":
        print(
            f"[progress] episode done: final_pano={event.get('final_pano_id')}"
            f" final_room={event.get('final_room_id')}"
            f" grounded_room={event.get('final_grounded_room_id')}"
            f" traces={event.get('trace_count')}",
            file=stream,
        )
        return
    if event_name == "pipeline_done":
        print(
            f"[progress] pipeline done: final_pano={event.get('final_pano_id')}"
            f" final_room={event.get('final_room_id')}",
            file=stream,
        )
        return


def make_progress_callback(*, log_stream=None):
    trace_stream = None
    if log_stream is not None:
        trace_path = Path(log_stream.name + ".trace.jsonl")
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_stream = trace_path.open("w", encoding="utf-8")

    def _callback(event):
        if event.get("event") != "trace_recorded":
            print_progress(event, stream=sys.stderr)
            sys.stderr.flush()
        if log_stream is not None:
            if event.get("event") != "trace_recorded":
                print_progress(event, stream=log_stream)
            log_stream.flush()
        if trace_stream is not None and event.get("event") == "trace_recorded":
            print(render_json(event.get("trace")), file=trace_stream)
            trace_stream.flush()

    _callback.trace_stream = trace_stream
    return _callback


def validate_navigation_args(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> None:
    has_instruction = bool(args.instruction and args.instruction.strip())
    has_start = bool(args.start_pano_id and args.start_pano_id.strip())
    has_target = bool(args.target_room_id and args.target_room_id.strip())
    if not has_start:
        parser.error("--start-pano-id is required for all navigation modes.")
    if has_instruction and has_target:
        parser.error("--instruction cannot be combined with --target-room-id.")
    if has_instruction and args.waypoint_room_ids:
        parser.error("--instruction cannot be combined with --waypoint-room-ids.")
    if not has_instruction and not has_target:
        parser.error(
            "provide --instruction with --start-pano-id, or provide "
            "--start-pano-id with --target-room-id."
        )
    if args.waypoint_room_ids and not has_target:
        parser.error("--waypoint-room-ids requires --target-room-id.")
    if args.localization_interval < 1:
        parser.error("--localization-interval must be at least 1.")
    if args.direction_max_output_tokens is not None and args.direction_max_output_tokens < 1:
        parser.error("--direction-max-output-tokens must be positive.")
    if args.direction_schema_validation_retries < 0:
        parser.error("--direction-schema-validation-retries must be non-negative.")
    if args.render_missing and not args.render_api_key:
        parser.error("--render-missing requires --render-api-key or GMAPS_API_KEY.")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    validate_navigation_args(parser, args)

    artifacts = load_normalized_artifacts(
        args.artifacts_dir,
        room_graph=True,
        pano_graph=True,
        pano_room_grounding=True,
    )
    room_graph = artifacts.room_graph or {}
    pano_graph = artifacts.pano_graph or {}
    pipeline = build_navigation_pipeline(
        room_graph=room_graph,
        pano_graph=pano_graph,
        grounding_payload={},
        pano_room_grounding=artifacts.pano_room_grounding or {},
        config=NavigationPipelineConfig(
            llm_model=args.llm_model,
            llm_api_key=args.llm_api_key,
            llm_api_base=args.llm_api_base,
            llm_api_kind=args.llm_api_kind,
            llm_timeout=args.llm_timeout,
            alignment_candidate_ratio_threshold=args.alignment_candidate_ratio_threshold,
            alignment_candidate_max=args.alignment_candidate_max,
            u_turn_threshold_deg=args.u_turn_threshold_deg,
            history_return_threshold_deg=args.history_return_threshold_deg,
            localization_interval=args.localization_interval,
            localization_effective_stage=(
                "visual" if args.direction_policy == "visual-theme" else "spatial"
            ),
            direction_policy=args.direction_policy,
            heading_alignment_mode=(
                "none" if args.direction_policy == "visual-theme" else "fixed"
            ),
            direction_cache_dir=(
                str(resolve_project_path(args.direction_cache_dir))
                if args.direction_cache_dir
                else None
            ),
            direction_max_output_tokens=args.direction_max_output_tokens,
            direction_schema_validation_retries=args.direction_schema_validation_retries,
            use_detection_cache=not args.no_detection_cache,
        ),
    )

    manifest_paths = {}
    if args.manifest_map_json:
        manifest_paths = load_json(resolve_project_path(args.manifest_map_json))
    render_output_dir = None
    if args.render_output_dir:
        render_output_dir = str(resolve_project_path(args.render_output_dir))
    candidate_theme_output_dir = None
    if args.candidate_theme_output_dir:
        candidate_theme_output_dir = str(resolve_project_path(args.candidate_theme_output_dir))

    log_path = resolve_project_path(args.log_path) if args.log_path else None
    log_stream = None
    try:
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_stream = log_path.open("w", encoding="utf-8")
        progress_callback = make_progress_callback(log_stream=log_stream)
        result = pipeline.run(
            args.instruction,
            progress_callback=progress_callback,
            start_pano_id=args.start_pano_id,
            target_room_id=args.target_room_id,
            waypoint_room_ids=args.waypoint_room_ids,
            manifest_paths=manifest_paths,
            start_heading=args.start_heading,
            step_budget=args.step_budget,
            render_api_key=args.render_api_key,
            render_missing=args.render_missing,
            render_output_dir=render_output_dir,
            render_heading_mode=args.render_heading_mode,
            render_pitch=args.render_pitch,
            render_fov=args.render_fov,
            render_width=args.render_width,
            render_height=args.render_height,
            candidate_theme_fov=args.candidate_theme_fov,
            candidate_theme_output_dir=candidate_theme_output_dir,
            render_graph_path=artifacts.artifacts_dir / "pano_graph.json",
        )
        if log_stream:
            print_trace_summary(result, stream=log_stream)
        else:
            print_trace_summary(result)

        navigation_payload = {
            "instruction": result.instruction,
            "task": {
                "task_type": result.task.task_type,
                "source_room_id": result.task.source_room_id,
                "waypoint_room_ids": list(result.task.waypoint_room_ids),
                "goal_room_ids": list(result.task.goal_room_ids),
            },
            "source": {
                "source_room_id": result.source.source_pano.source_room_id,
                "source_pano_id": result.source.source_pano.pano_id,
            },
            "final_state": {
                "current_pano_id": result.final_state.current_pano_id,
                "current_room_id": result.final_state.current_room_id,
                "pending_room_id": result.final_state.pending_room_id,
                "pending_room_confirmation_count": result.final_state.pending_room_confirmation_count,
                "grounded_room_id": getattr(result.final_state, "grounded_room_id", None),
                "current_heading": result.final_state.current_heading,
                "incoming_travel_heading": result.final_state.incoming_travel_heading,
                "recent_travel_headings": list(result.final_state.recent_travel_headings),
                "visited_panos": sorted(result.final_state.visited_panos),
                "visited_rooms": sorted(result.final_state.visited_rooms),
                "room_belief": result.final_state.room_belief,
            },
            "trace_count": len(result.traces),
            "traces": [serialize_trace(trace) for trace in result.traces],
        }
        viewer_episode = build_pano_viewer_episode(navigation_payload, step_budget=args.step_budget)
        payload = {**viewer_episode, **navigation_payload}
        output_text = render_json(payload)
        write_text_if_requested(output_text, args.output_path)
        trace_stream = getattr(progress_callback, "trace_stream", None)
        if trace_stream is not None:
            print(output_text, file=trace_stream)
            trace_stream.flush()
        if log_stream:
            print(f"Wrote navigation log to {log_path}")
        else:
            print(output_text)
    finally:
        trace_stream = locals().get("progress_callback", None)
        trace_stream = getattr(trace_stream, "trace_stream", None)
        if trace_stream is not None:
            trace_stream.close()
        if log_stream is not None:
            log_stream.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
