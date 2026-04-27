from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from _common import (
    PROJECT_ROOT,
    ensure_project_root_on_path,
    load_json,
    load_normalized_artifacts,
    render_json,
    resolve_project_path,
    write_text_if_requested,
)

ensure_project_root_on_path()

from st_nav import (
    EpisodeRunner,
    GroundingIndex,
    LLMActionPolicy,
    LLMInstructionParser,
    LLMRoomLocalizer,
    LLMSpatialAlignmentLocalizer,
    ManifestPerceptionProvider,
    NavigationPipeline,
    PanoramaRenderer,
    SourcePanoResolver,
    SourceResolutionWorkflow,
    SpatialEngine,
    load_dotenv,
    resolve_model_environment,
)

load_dotenv(PROJECT_ROOT / ".env")
MODEL_ENV = resolve_model_environment(
    default_model="gpt-5-mini",
    default_api_base="https://api.openai.com/v1",
    default_api_kind="responses",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the end-to-end navigation loop.")
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--artifacts-dir", default="dataset/sites/british_museum/normalized")
    parser.add_argument(
        "--localizer",
        choices=["heuristic", "llm", "spatial-alignment-a", "spatial-alignment-b"],
        default="spatial-alignment-a",
    )
    parser.add_argument("--manifest-map-json")
    parser.add_argument("--step-budget", type=int, default=10)
    parser.add_argument(
        "--start-heading",
        type=float,
        default=0.0,
        help="Rendering reference only. Reasoning uses spatial alignment instead of assuming this is the true agent heading.",
    )
    parser.add_argument("--llm-model", default=MODEL_ENV.model_name)
    parser.add_argument("--llm-api-key", default=MODEL_ENV.api_key)
    parser.add_argument("--llm-api-kind", default=MODEL_ENV.api_kind)
    parser.add_argument("--llm-api-base", default=MODEL_ENV.api_base)
    parser.add_argument("--llm-timeout", type=float, default=MODEL_ENV.request_timeout or 30.0)
    parser.add_argument("--render-api-key", default=os.environ.get("GMAPS_API_KEY"))
    parser.add_argument("--render-output-dir", default="renders/navigation_episode")
    parser.add_argument("--render-heading-mode", choices=["museum", "cardinal", "graph"], default="museum")
    parser.add_argument("--render-pitch", type=float, default=0.0)
    parser.add_argument("--render-fov", type=int, default=45)
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


def build_localizer(args, *, room_graph: dict[str, dict], grounding_index: GroundingIndex):
    if args.localizer == "heuristic":
        return None
    if args.localizer == "llm":
        return LLMRoomLocalizer(
            room_graph=room_graph,
            grounding_index=grounding_index,
            model=args.llm_model,
            api_key=args.llm_api_key,
            api_base=args.llm_api_base,
            api_kind=args.llm_api_kind,
            request_timeout=args.llm_timeout,
        )
    return LLMSpatialAlignmentLocalizer(
        room_graph=room_graph,
        grounding_index=grounding_index,
        alignment_mode="text_from_images" if args.localizer == "spatial-alignment-a" else "direct_images",
        model=args.llm_model,
        api_key=args.llm_api_key,
        api_base=args.llm_api_base,
        api_kind=args.llm_api_kind,
        request_timeout=args.llm_timeout,
    )


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
    return {
        "step_index": trace.step_index,
        "pano_id": trace.pano_id,
        "room_id": trace.room_id,
        "grounded_room_id": trace.observation.metadata.get("grounded_room_id"),
        "route": list(trace.route),
        "subgoal_room_id": trace.subgoal_room_id,
        "current_room_context": dict(trace.current_room_context),
        "visible_passages": list(trace.visible_passages),
        "view_contexts": list(trace.view_contexts),
        "candidates": [serialize_candidate(candidate) for candidate in trace.candidates],
        "observation": {
            "pano_id": trace.observation.pano_id,
            "heading_estimate": trace.observation.heading_estimate,
            "localized_room_id": trace.observation.metadata.get("localized_room_id"),
            "grounded_room_id": trace.observation.metadata.get("grounded_room_id"),
            "spatial_alignment": trace.observation.metadata.get("spatial_alignment"),
            "entities": [
                {
                    "name": entity.name,
                    "kind": entity.kind,
                    "confidence": entity.confidence,
                    "source_view": entity.source_view,
                    "source_views": entity.metadata.get("source_views"),
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
            f" pano={event.get('current_pano_id')}"
            f" localized_room={event.get('current_room_id')}"
            f" grounded_room={event.get('grounded_room_id')}",
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
    if event_name == "localization_done":
        print(
            f"[progress] localization done: step={event.get('step_index')}"
            f" pano={event.get('current_pano_id')}"
            f" localized_room={event.get('current_room_id')}"
            f" grounded_room={event.get('grounded_room_id')}",
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
    def _callback(event):
        if event.get("event") != "trace_recorded":
            print_progress(event, stream=sys.stderr)
            sys.stderr.flush()
        if log_stream is not None:
            if event.get("event") == "trace_recorded":
                print("[trace]", file=log_stream)
                print(render_json(event.get("trace")), file=log_stream)
            else:
                print_progress(event, stream=log_stream)
            log_stream.flush()

    return _callback


def main() -> int:
    args = build_parser().parse_args()
    artifacts = load_normalized_artifacts(
        args.artifacts_dir,
        room_graph=True,
        pano_graph=True,
        grounding=True,
        pano_room_grounding=True,
    )
    room_graph = artifacts.room_graph or {}
    pano_graph = artifacts.pano_graph or {}
    grounding_index = GroundingIndex(
        artifacts.grounding or {},
        pano_to_room=artifacts.pano_room_grounding or {},
    )

    parser = LLMInstructionParser(
        room_graph=room_graph,
        model=args.llm_model,
        api_key=args.llm_api_key,
        api_base=args.llm_api_base,
        api_kind=args.llm_api_kind,
        request_timeout=args.llm_timeout,
    )
    source_resolution = SourceResolutionWorkflow(
        instruction_parser=parser,
        source_pano_resolver=SourcePanoResolver(grounding_index),
    )
    spatial_engine = SpatialEngine(
        room_graph=room_graph,
        pano_graph=pano_graph,
        grounding_index=grounding_index,
        localizer=build_localizer(args, room_graph=room_graph, grounding_index=grounding_index),
    )
    runner = EpisodeRunner(
        perception_provider=ManifestPerceptionProvider(pano_graph=pano_graph),
        spatial_engine=spatial_engine,
        policy=LLMActionPolicy(
            model=args.llm_model,
            api_key=args.llm_api_key,
            api_base=args.llm_api_base,
            api_kind=args.llm_api_kind,
            request_timeout=args.llm_timeout,
        ),
        renderer=PanoramaRenderer(pano_graph),
    )
    pipeline = NavigationPipeline(
        source_resolution_workflow=source_resolution,
        episode_runner=runner,
    )

    manifest_paths = {}
    if args.manifest_map_json:
        manifest_paths = load_json(resolve_project_path(args.manifest_map_json))
    if not manifest_paths and not args.render_api_key:
        raise RuntimeError("Provide --manifest-map-json for cached inputs or configure --render-api-key / GMAPS_API_KEY.")

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
            manifest_paths=manifest_paths,
            start_heading=args.start_heading,
            step_budget=args.step_budget,
            render_api_key=args.render_api_key,
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

        payload = {
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
                "grounded_room_id": getattr(result.final_state, "grounded_room_id", None),
                "current_heading": result.final_state.current_heading,
                "visited_panos": sorted(result.final_state.visited_panos),
                "visited_rooms": sorted(result.final_state.visited_rooms),
                "room_belief": result.final_state.room_belief,
            },
            "trace_count": len(result.traces),
            "traces": [serialize_trace(trace) for trace in result.traces],
        }
        output_text = render_json(payload)
        write_text_if_requested(output_text, args.output_path)
        if log_stream:
            print(output_text, file=log_stream)
            log_stream.flush()
            print(f"Wrote navigation log to {log_path}")
        else:
            print(output_text)
    finally:
        if log_stream is not None:
            log_stream.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
