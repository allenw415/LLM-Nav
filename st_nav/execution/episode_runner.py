from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
from pathlib import Path
import time

from ..common.types import BeliefState, CandidateAction, JsonDict, Observation, PolicyOutput, ReasoningInput, RenderedView, TaskSpec
from ..perception.renderer import (
    expected_capture_headings,
    expected_capture_labels,
    manifest_cache_miss_reason,
    normalize_heading,
    sanitize_name,
)


@dataclass
class EpisodeTrace:
    step_index: int
    pano_id: str
    room_id: str | None
    route: list[str]
    observation: Observation
    policy_output: PolicyOutput
    subgoal_room_id: str | None = None
    completed_waypoint_room_ids: list[str] = field(default_factory=list)
    remaining_waypoint_room_ids: list[str] = field(default_factory=list)
    candidates: list[CandidateAction] = field(default_factory=list)
    current_room_context: JsonDict = field(default_factory=dict)
    visible_passages: list[JsonDict] = field(default_factory=list)
    view_contexts: list[JsonDict] = field(default_factory=list)
    policy_request: JsonDict | None = None
    policy_response: JsonDict | None = None


class EpisodeRunner:
    def __init__(
        self,
        *,
        perception_provider,
        spatial_engine,
        policy,
        renderer=None,
        localization_interval: int = 1,
        force_initial_visual_refresh: bool = False,
        stop_on_initial_alignment_failure: bool = True,
    ):
        if localization_interval < 1:
            raise ValueError("localization_interval must be at least 1.")
        self.perception_provider = perception_provider
        self.spatial_engine = spatial_engine
        self.policy = policy
        self.renderer = renderer
        self.localization_interval = int(localization_interval)
        self.force_initial_visual_refresh = bool(force_initial_visual_refresh)
        self.stop_on_initial_alignment_failure = bool(
            stop_on_initial_alignment_failure
        )

    def run(
        self,
        *,
        task: TaskSpec,
        start_pano_id: str,
        manifest_paths: dict[str, str | Path] | None = None,
        start_room_id: str | None = None,
        start_heading: float = 330.0,
        step_budget: int = 15,
        render_api_key: str | None = None,
        render_missing: bool = False,
        render_output_dir: str | Path | None = None,
        render_heading_mode: str = "museum",
        render_pitch: float = 0.0,
        render_fov: int = 90,
        render_width: int = 512,
        render_height: int = 512,
        candidate_theme_fov: int | None = None,
        candidate_theme_output_dir: str | Path | None = None,
        render_graph_path: str | Path | None = None,
        progress_callback=None,
    ) -> tuple[BeliefState, list[EpisodeTrace]]:
        episode_started = time.perf_counter()
        manifest_paths = dict(manifest_paths or {})
        reset_episode = getattr(self.policy, "reset_episode", None)
        if callable(reset_episode):
            reset_episode()
        self._emit_progress(
            progress_callback,
            {
                "event": "episode_start",
                "start_pano_id": start_pano_id,
                "start_room_id": start_room_id,
                "step_budget": step_budget,
            },
        )
        state = self.spatial_engine.initialize(
            start_pano_id=start_pano_id,
            start_room_id=start_room_id,
            start_heading=start_heading,
        )
        traces: list[EpisodeTrace] = []
        target_room_ids = set(task.goal_room_ids)
        ordered_waypoints: list[str] = []
        for room_id in task.waypoint_room_ids:
            if (
                isinstance(room_id, str)
                and room_id
                and room_id not in target_room_ids
                and room_id not in ordered_waypoints
            ):
                ordered_waypoints.append(room_id)
        completed_waypoints: list[str] = []
        next_waypoint_index = 0
        has_completed_localization = False
        movement_steps_since_localization = 0

        for step_index in range(step_budget):
            localization_requested = (
                not has_completed_localization
                or movement_steps_since_localization >= self.localization_interval
                or state.pending_room_id is not None
            )
            self._emit_progress(
                progress_callback,
                {
                    "event": "step_start",
                    "step_index": step_index,
                    "current_pano_id": state.current_pano_id,
                },
            )
            module_started = time.perf_counter()
            manifest_path = self._ensure_manifest(
                pano_id=state.current_pano_id,
                manifest_paths=manifest_paths,
                render_api_key=render_api_key,
                render_missing=render_missing,
                render_output_dir=render_output_dir,
                render_heading_mode=render_heading_mode,
                render_pitch=render_pitch,
                render_fov=render_fov,
                render_width=render_width,
                render_height=render_height,
                render_graph_path=render_graph_path,
                progress_callback=progress_callback,
            )
            self._emit_module_timing(
                progress_callback,
                module="render_manifest",
                total_wall_ms=(time.perf_counter() - module_started) * 1000.0,
            )
            if manifest_path is None:
                self._emit_progress(
                    progress_callback,
                    {
                        "event": "stop_no_manifest",
                        "step_index": step_index,
                        "pano_id": state.current_pano_id,
                    },
                )
                break

            self._emit_progress(
                progress_callback,
                {
                    "event": "perception_start",
                    "step_index": step_index,
                    "mode": "localization" if localization_requested else "tracking",
                    "pano_id": state.current_pano_id,
                    "manifest_path": str(manifest_path),
                },
            )
            force_initial_alignment = (
                step_index == 0
                and getattr(self.spatial_engine, "heading_alignment_mode", "fixed")
                == "initial-spatial"
            )
            module_started = time.perf_counter()
            observation = self._run_perception(
                manifest_path=manifest_path,
                current_heading=state.current_heading,
                run_localization=localization_requested,
                force_visual_refresh=(
                    force_initial_alignment and self.force_initial_visual_refresh
                ),
            )
            perception_total_ms = (time.perf_counter() - module_started) * 1000.0
            observation.metadata["force_spatial_alignment"] = force_initial_alignment
            self._emit_progress(
                progress_callback,
                {
                    "event": "perception_done",
                    "step_index": step_index,
                    "mode": "localization" if localization_requested else "tracking",
                    "pano_id": observation.pano_id,
                    "entity_count": len(observation.entities),
                    "view_count": len(observation.views),
                },
            )
            module_started = time.perf_counter()
            state, localization_performed = self._update_spatial_state(
                state, observation, run_localization=localization_requested
            )
            spatial_update_total_ms = (time.perf_counter() - module_started) * 1000.0
            alignment_ready = True
            if force_initial_alignment:
                alignment_ready = self.spatial_engine.initialize_heading_alignment(
                    state,
                    observation,
                    step_index=step_index,
                )
            self._emit_localization_module_timings(
                progress_callback,
                observation=observation,
                perception_total_ms=perception_total_ms,
                spatial_update_total_ms=spatial_update_total_ms,
                localization_performed=localization_performed,
            )
            observation.metadata["localization_interval"] = self.localization_interval
            observation.metadata["movement_steps_since_last_localization"] = (
                movement_steps_since_localization
            )
            if localization_performed:
                has_completed_localization = True
                movement_steps_since_localization = 0
            self._emit_progress(
                progress_callback,
                {
                    "event": (
                        "localization_done" if localization_performed else "localization_skipped"
                    ),
                    "step_index": step_index,
                    "localization_performed": localization_performed,
                    "localization_interval": self.localization_interval,
                    "movement_steps_since_last_localization": observation.metadata.get(
                        "movement_steps_since_last_localization"
                    ),
                    "current_pano_id": state.current_pano_id,
                    "current_room_id": state.current_room_id,
                    "grounded_room_id": state.grounded_room_id,
                    "room_belief": dict(state.room_belief),
                    "raw_localized_room_id": observation.metadata.get("localized_room_id_raw"),
                    "pending_room_id": state.pending_room_id,
                    "pending_room_confirmation_count": state.pending_room_confirmation_count,
                    "room_switch_confirmations_required": observation.metadata.get(
                        "room_switch_confirmations_required"
                    ),
                    "localization_event": (
                        self._serialize_localization_event(step_index, state, observation)
                        if localization_performed
                        else None
                    ),
                },
            )
            if (
                force_initial_alignment
                and not alignment_ready
                and self.stop_on_initial_alignment_failure
            ):
                self._emit_progress(
                    progress_callback,
                    {
                        "event": "stop_initial_alignment_failed",
                        "step_index": step_index,
                        "pano_id": state.current_pano_id,
                        "reason": state.heading_alignment_failure_reason,
                    },
                )
                break
            grounded_room_id = self.spatial_engine.grounding_index.room_for_pano(
                state.current_pano_id
            )
            if (
                next_waypoint_index < len(ordered_waypoints)
                and grounded_room_id == ordered_waypoints[next_waypoint_index]
            ):
                completed_room_id = ordered_waypoints[next_waypoint_index]
                completed_waypoints.append(completed_room_id)
                next_waypoint_index += 1
                self._emit_progress(
                    progress_callback,
                    {
                        "event": "waypoint_completed",
                        "step_index": step_index,
                        "room_id": completed_room_id,
                        "completed_waypoint_room_ids": list(completed_waypoints),
                        "remaining_waypoint_room_ids": list(
                            ordered_waypoints[next_waypoint_index:]
                        ),
                    },
                )

            remaining_waypoints = ordered_waypoints[next_waypoint_index:]
            active_task = replace(
                task,
                waypoint_room_ids=list(remaining_waypoints),
            )
            module_started = time.perf_counter()
            route = self.spatial_engine.route_to_goal(active_task, state)
            subgoal_room_id = self.spatial_engine.next_subgoal_room_id(state, route)
            current_room_context = self.spatial_engine.build_current_room_context(state, route)
            visible_passages = self.spatial_engine.extract_visible_passages(state, observation)
            self._emit_module_timing(
                progress_callback,
                module="route_planning",
                total_wall_ms=(time.perf_counter() - module_started) * 1000.0,
            )
            if (
                not remaining_waypoints
                and self.spatial_engine.goal_reached(task, state)
            ):
                policy_output = PolicyOutput(
                    action=None,
                    rationale="Goal already reached after localization; no further action required.",
                )
                traces.append(
                    EpisodeTrace(
                        step_index=step_index,
                        pano_id=state.current_pano_id,
                        room_id=state.current_room_id,
                        route=route,
                        subgoal_room_id=subgoal_room_id,
                        completed_waypoint_room_ids=list(completed_waypoints),
                        remaining_waypoint_room_ids=list(remaining_waypoints),
                        candidates=[],
                        current_room_context=dict(current_room_context),
                        visible_passages=list(visible_passages),
                        view_contexts=[],
                        observation=observation,
                        policy_output=policy_output,
                        policy_request=None,
                        policy_response=None,
                    )
                )
                self._emit_progress(
                    progress_callback,
                    {
                        "event": "trace_recorded",
                        "step_index": step_index,
                        "trace": self._serialize_trace_payload(traces[-1]),
                    },
                )
                self._emit_progress(
                    progress_callback,
                    {
                        "event": "goal_reached",
                        "step_index": step_index,
                        "current_pano_id": state.current_pano_id,
                        "current_room_id": state.current_room_id,
                        "grounded_room_id": state.grounded_room_id,
                    },
                )
                break

            module_started = time.perf_counter()
            reasoning_observation = self._build_reasoning_observation(
                state=state,
                base_observation=observation,
                manifest_paths=manifest_paths,
                render_api_key=render_api_key,
                render_missing=render_missing,
                render_output_dir=render_output_dir,
                render_heading_mode=render_heading_mode,
                render_pitch=render_pitch,
                render_fov=render_fov,
                candidate_theme_fov=candidate_theme_fov,
                candidate_theme_output_dir=candidate_theme_output_dir,
                render_width=render_width,
                render_height=render_height,
                render_graph_path=render_graph_path,
                progress_callback=progress_callback,
            )
            view_contexts = self.spatial_engine.describe_view_contexts(reasoning_observation)
            candidates = self.spatial_engine.generate_candidates(
                state,
                route,
                observation=observation,
                context_observation=reasoning_observation,
            )
            self._emit_module_timing(
                progress_callback,
                module="candidate_generation",
                total_wall_ms=(time.perf_counter() - module_started) * 1000.0,
            )
            self._emit_progress(
                progress_callback,
                {
                    "event": "route_done",
                    "step_index": step_index,
                    "route": list(route),
                    "subgoal_room_id": subgoal_room_id,
                    "candidate_count": len(candidates),
                },
            )
            module_started = time.perf_counter()
            policy_output = self._run_reasoning(
                task=active_task,
                route=route,
                candidates=candidates,
                current_pano_id=state.current_pano_id,
                current_room_id=state.current_room_id,
                current_room_context=current_room_context,
                visible_passages=visible_passages,
                view_contexts=view_contexts,
                spatial_alignment=observation.metadata.get("spatial_alignment"),
                subgoal_room_id=subgoal_room_id,
            )
            policy_total_ms = (time.perf_counter() - module_started) * 1000.0
            policy_client = getattr(self.policy, "model_client", None)
            policy_vlm_ms = getattr(policy_client, "last_call_wall_ms", 0.0)
            self._emit_module_timing(
                progress_callback,
                module="action_policy",
                total_wall_ms=policy_total_ms,
                vlm_response_ms=(
                    policy_vlm_ms
                    if isinstance(policy_vlm_ms, (int, float))
                    else 0.0
                ),
            )
            traces.append(
                EpisodeTrace(
                    step_index=step_index,
                    pano_id=state.current_pano_id,
                    room_id=state.current_room_id,
                    route=route,
                    subgoal_room_id=subgoal_room_id,
                    completed_waypoint_room_ids=list(completed_waypoints),
                    remaining_waypoint_room_ids=list(remaining_waypoints),
                    candidates=list(candidates),
                    current_room_context=dict(current_room_context),
                    visible_passages=list(visible_passages),
                    view_contexts=list(view_contexts),
                    observation=observation,
                    policy_output=policy_output,
                    policy_request=self._clone_json(getattr(self.policy, "last_request_body", None)),
                    policy_response=self._clone_json(getattr(self.policy, "last_response_payload", None)),
                )
            )
            self._emit_progress(
                progress_callback,
                {
                    "event": "trace_recorded",
                    "step_index": step_index,
                    "trace": self._serialize_trace_payload(traces[-1]),
                },
            )
            self._emit_progress(
                progress_callback,
                {
                    "event": "reasoning_done",
                    "step_index": step_index,
                    "chosen_pano_id": policy_output.action.target_pano_id if policy_output.action else None,
                    "chosen_room_id": policy_output.action.target_room_id if policy_output.action else None,
                    "chosen_grounded_room_id": (
                        policy_output.action.metadata.get("grounded_target_room_id")
                        if policy_output.action
                        else None
                    ),
                    "rationale": policy_output.rationale,
                },
            )

            if policy_output.action is None:
                self._emit_progress(
                    progress_callback,
                    {
                        "event": "stop_no_action",
                        "step_index": step_index,
                        "current_pano_id": state.current_pano_id,
                        "current_room_id": state.current_room_id,
                        "grounded_room_id": state.grounded_room_id,
                    },
                )
                break

            module_started = time.perf_counter()
            source_pano_id = state.current_pano_id
            record_transition = getattr(self.policy, "record_transition", None)
            if callable(record_transition):
                record_transition(
                    source_pano_id,
                    policy_output.action.target_pano_id,
                )
            state.previous_pano_id = source_pano_id
            state.current_pano_id = policy_output.action.target_pano_id
            state.current_heading = policy_output.action.absolute_heading
            travel_heading = policy_output.action.absolute_heading
            state.incoming_travel_heading = travel_heading
            state.recent_travel_headings.append(travel_heading)
            state.recent_travel_headings = state.recent_travel_headings[-2:]
            movement_steps_since_localization += 1
            self._emit_module_timing(
                progress_callback,
                module="movement_execution",
                total_wall_ms=(time.perf_counter() - module_started) * 1000.0,
            )
            self._emit_progress(
                progress_callback,
                {
                    "event": "action_applied",
                    "step_index": step_index,
                    "next_pano_id": state.current_pano_id,
                    "next_heading": state.current_heading,
                },
            )

        self._emit_progress(
            progress_callback,
            {
                "event": "episode_done",
                "final_pano_id": state.current_pano_id,
                "final_room_id": state.current_room_id,
                "final_grounded_room_id": state.grounded_room_id,
                "trace_count": len(traces),
                "episode_runner_wall_ms": (
                    time.perf_counter() - episode_started
                )
                * 1000.0,
            },
        )
        return state, traces

    def _run_perception(
        self,
        *,
        manifest_path: str | Path,
        current_heading: float,
        run_localization: bool,
        force_visual_refresh: bool = False,
    ) -> Observation:
        if not run_localization and hasattr(self.perception_provider, "observe_tracking"):
            return self.perception_provider.observe_tracking(
                manifest_path,
                current_heading=current_heading,
            )
        if hasattr(self.perception_provider, "observe_from_manifest"):
            kwargs = {"current_heading": current_heading}
            if force_visual_refresh:
                kwargs["force_refresh"] = True
            return self.perception_provider.observe_from_manifest(manifest_path, **kwargs)
        kwargs = {"current_heading": current_heading}
        if force_visual_refresh:
            kwargs["force_refresh"] = True
        return self.perception_provider.observe(manifest_path, **kwargs)

    def _update_spatial_state(
        self,
        state: BeliefState,
        observation: Observation,
        *,
        run_localization: bool,
    ) -> tuple[BeliefState, bool]:
        if not run_localization and hasattr(self.spatial_engine, "update_tracking"):
            return (
                self.spatial_engine.update_tracking(state, observation),
                False,
            )
        return self.spatial_engine.update(state, observation), True

    def _run_reasoning(
        self,
        *,
        task: TaskSpec,
        route: list[str],
        candidates,
        current_pano_id: str,
        current_room_id: str | None,
        current_room_context,
        visible_passages,
        view_contexts,
        spatial_alignment,
        subgoal_room_id: str | None,
    ) -> PolicyOutput:
        reasoning_input = ReasoningInput(
            task=task,
            route=route,
            candidates=candidates,
            current_pano_id=current_pano_id,
            current_room_id=current_room_id,
            subgoal_room_id=subgoal_room_id,
            current_room_context=dict(current_room_context or {}),
            visible_passages=list(visible_passages or []),
            spatial_alignment=dict(spatial_alignment or {}),
            view_contexts=list(view_contexts or []),
        )
        if hasattr(self.policy, "choose_next_action"):
            return self.policy.choose_next_action(reasoning_input)
        return self.policy.choose_action(task=task, route=route, candidates=candidates)

    def _build_reasoning_observation(
        self,
        *,
        state: BeliefState,
        base_observation: Observation,
        manifest_paths: dict[str, str | Path],
        render_api_key: str | None,
        render_missing: bool,
        render_output_dir: str | Path | None,
        render_heading_mode: str,
        render_pitch: float,
        render_fov: int,
        candidate_theme_fov: int | None,
        candidate_theme_output_dir: str | Path | None,
        render_width: int,
        render_height: int,
        render_graph_path: str | Path | None,
        progress_callback,
    ) -> Observation:
        if candidate_theme_fov is None or candidate_theme_fov == render_fov:
            return base_observation
        if not render_missing or self.renderer is None or render_api_key is None:
            return base_observation
        theme_output_dir = candidate_theme_output_dir or render_output_dir
        if theme_output_dir is None:
            return base_observation

        candidate_captures = self._candidate_theme_captures(state.current_pano_id)
        if not candidate_captures:
            return base_observation
        merged_views = []
        manifest_by_label: dict[str, Path] = {}
        base_room_id = base_observation.metadata.get("grounded_room_id")
        for index, (label, heading) in enumerate(candidate_captures):
            candidate_output_dir = (
                Path(theme_output_dir)
                / sanitize_name(state.current_pano_id)
                / sanitize_name(label)
            )
            manifest = self.renderer.render(
                pano_id=state.current_pano_id,
                api_key=render_api_key,
                output_dir=candidate_output_dir,
                heading_mode="explicit",
                custom_captures=[(label, heading)],
                pitch=render_pitch,
                fov=candidate_theme_fov,
                width=render_width,
                height=render_height,
                graph_path=render_graph_path,
                progress_callback=progress_callback,
            )
            theme_manifest_path = manifest["manifest_path"]
            if theme_manifest_path is None:
                continue
            manifest_by_label[label] = Path(theme_manifest_path)
            single_manifest_observation = Observation(
                pano_id=state.current_pano_id,
                views=[],
                entities=[],
                heading_estimate=base_observation.heading_estimate,
                metadata={
                    "grounded_room_id": base_room_id,
                    "manifest_path": str(theme_manifest_path),
                },
            )
            raw_manifest = manifest.get("captures", [])
            if not isinstance(raw_manifest, list) or not raw_manifest:
                continue
            capture = raw_manifest[0]
            path = capture.get("path")
            capture_heading = capture.get("heading")
            if not isinstance(path, str) or not isinstance(capture_heading, (int, float)):
                continue
            single_manifest_observation.views.append(
                RenderedView(
                    label=label,
                    heading=float(capture_heading),
                    path=path,
                    url=capture.get("url"),
                )
            )
            merged_views.extend(single_manifest_observation.views)

        if not merged_views:
            return base_observation
        merged_metadata: JsonDict = {"grounded_room_id": base_room_id}
        batch_theme_context = self._batch_candidate_theme_context(
            state=state,
            base_observation=base_observation,
            merged_views=merged_views,
        )
        if isinstance(batch_theme_context, dict):
            merged_metadata["ego_spatial_context"] = batch_theme_context
            self._write_candidate_theme_sidecars(
                merged_views=merged_views,
                batch_theme_context=batch_theme_context,
                manifest_by_label=manifest_by_label,
            )
        return Observation(
            pano_id=base_observation.pano_id,
            views=merged_views,
            entities=[],
            heading_estimate=base_observation.heading_estimate,
            metadata=merged_metadata,
        )

    def _batch_candidate_theme_context(
        self,
        *,
        state: BeliefState,
        base_observation: Observation,
        merged_views: list,
    ) -> JsonDict | None:
        localizer = getattr(self.spatial_engine.state_estimator, "localizer", None)
        if localizer is None:
            return None
        required_methods = (
            "model_client",
            "_candidate_room_ids",
            "_ordered_views",
            "_build_context_extraction_request_body",
            "_create_response",
            "_parse_output_payload",
            "_format_ego_spatial_context",
        )
        if any(not hasattr(localizer, name) for name in required_methods):
            return None
        model_client = getattr(localizer, "model_client", None)
        if model_client is None or not model_client.is_configured():
            return None

        theme_observation = Observation(
            pano_id=base_observation.pano_id,
            views=list(merged_views),
            entities=[],
            heading_estimate=base_observation.heading_estimate,
            metadata={"floor": base_observation.metadata.get("floor")},
        )
        candidate_room_ids = localizer._candidate_room_ids(base_observation)
        ordered_views = localizer._ordered_views(theme_observation)
        if not candidate_room_ids or not ordered_views:
            return None
        extraction_request = localizer._build_context_extraction_request_body(
            ordered_views=ordered_views,
            candidate_room_ids=candidate_room_ids,
        )
        payload = localizer._create_response(extraction_request)
        parsed = localizer._parse_output_payload(payload)
        return localizer._format_ego_spatial_context(parsed, ordered_views)

    def _candidate_theme_captures(self, pano_id: str) -> list[tuple[str, float]]:
        pano_record = self.spatial_engine.pano_graph.get(pano_id, {})
        raw_neighbors = pano_record.get("neighbors", [])
        if not isinstance(raw_neighbors, list):
            return []
        captures: list[tuple[str, float]] = []
        seen_target_ids: set[str] = set()
        sorted_neighbors = sorted(
            (
                neighbor
                for neighbor in raw_neighbors
                if isinstance(neighbor, dict)
                and isinstance(neighbor.get("target_pano_id"), str)
                and isinstance(neighbor.get("geocentric_heading_deg"), (int, float))
            ),
            key=lambda neighbor: normalize_heading(float(neighbor["geocentric_heading_deg"])),
        )
        for index, neighbor in enumerate(sorted_neighbors):
            target_pano_id = str(neighbor["target_pano_id"])
            if target_pano_id in seen_target_ids:
                continue
            seen_target_ids.add(target_pano_id)
            captures.append(
                (
                    f"candidate_{index:02d}_{target_pano_id}",
                    normalize_heading(float(neighbor["geocentric_heading_deg"])),
                )
            )
        return captures

    def _write_candidate_theme_sidecars(
        self,
        *,
        merged_views: list[RenderedView],
        batch_theme_context: JsonDict,
        manifest_by_label: dict[str, Path],
    ) -> None:
        raw_views = batch_theme_context.get("views")
        if not isinstance(raw_views, list):
            return
        for index, record in enumerate(raw_views):
            if index >= len(merged_views):
                break
            if not isinstance(record, dict):
                continue
            view = merged_views[index]
            manifest_path = manifest_by_label.get(view.label)
            if manifest_path is None:
                continue
            payload = {
                "pano_id": view.label,
                "source_pano_id": manifest_path.parent.name,
                "candidate_label": view.label,
                "heading": float(view.heading),
                "view_id": record.get("view_id"),
                "themes": self._clone_json(record.get("themes", [])),
                "summary": batch_theme_context.get("summary", ""),
                "text": batch_theme_context.get("text", ""),
            }
            sidecar_path = manifest_path.with_name(
                manifest_path.stem + "_candidate_themes.json"
            )
            sidecar_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _ensure_manifest(
        self,
        *,
        pano_id: str,
        manifest_paths: dict[str, str | Path],
        render_api_key: str | None,
        render_output_dir: str | Path | None,
        render_heading_mode: str,
        render_pitch: float,
        render_fov: int,
        render_width: int,
        render_height: int,
        render_graph_path: str | Path | None,
        progress_callback,
        render_missing: bool = False,
    ) -> str | Path | None:
        manifest_path = manifest_paths.get(pano_id)
        manifest_source = "manifest_map"
        if manifest_path is None:
            manifest_path = self._discover_render_manifest(
                pano_id=pano_id,
                render_output_dir=render_output_dir,
            )
            if manifest_path is not None:
                manifest_source = "render_output_dir"
                manifest_paths[pano_id] = manifest_path

        cache_miss_reason = self._manifest_cache_miss_reason(
            manifest_path,
            expected_pano_id=pano_id,
            expected_heading_mode=render_heading_mode,
            expected_pitch=render_pitch,
            expected_fov=render_fov,
            expected_width=render_width,
            expected_height=render_height,
        )
        if cache_miss_reason is None:
            self._emit_progress(
                progress_callback,
                {
                    "event": "manifest_cache_hit",
                    "pano_id": pano_id,
                    "manifest_path": str(manifest_path),
                    "source": manifest_source,
                },
            )
            return manifest_path

        self._emit_progress(
            progress_callback,
            {
                "event": "manifest_cache_miss",
                "pano_id": pano_id,
                "manifest_path": str(manifest_path) if manifest_path is not None else None,
                "reason": cache_miss_reason,
            },
        )
        if (
            not render_missing
            or self.renderer is None
            or render_api_key is None
            or render_output_dir is None
        ):
            return None

        manifest = self.renderer.render(
            pano_id=pano_id,
            api_key=render_api_key,
            output_dir=render_output_dir,
            heading_mode=render_heading_mode,
            pitch=render_pitch,
            fov=render_fov,
            width=render_width,
            height=render_height,
            graph_path=render_graph_path,
            progress_callback=progress_callback,
        )
        manifest_path = manifest["manifest_path"]
        manifest_paths[pano_id] = manifest_path
        return manifest_path

    @staticmethod
    def _discover_render_manifest(
        *,
        pano_id: str,
        render_output_dir: str | Path | None,
    ) -> Path | None:
        if render_output_dir is None:
            return None
        pano_slug = sanitize_name(pano_id)
        candidate = Path(render_output_dir) / pano_slug / f"{pano_slug}_manifest.json"
        return candidate if candidate.is_file() else None

    @staticmethod
    def _manifest_cache_miss_reason(
        manifest_path: str | Path | None,
        *,
        expected_pano_id: str | None = None,
        expected_heading_mode: str | None = None,
        expected_pitch: float | None = None,
        expected_fov: int | None = None,
        expected_width: int | None = None,
        expected_height: int | None = None,
    ) -> str | None:
        return manifest_cache_miss_reason(
            manifest_path,
            expected_pano_id=expected_pano_id,
            expected_heading_mode=expected_heading_mode,
            expected_pitch=expected_pitch,
            expected_fov=expected_fov,
            expected_width=expected_width,
            expected_height=expected_height,
            expected_labels=(
                expected_capture_labels(expected_heading_mode)
                if expected_heading_mode is not None
                else None
            ),
            expected_headings=(
                expected_capture_headings(expected_heading_mode)
                if expected_heading_mode is not None
                else None
            ),
        )

    @staticmethod
    def _emit_progress(progress_callback, payload: dict) -> None:
        if progress_callback is None:
            return
        progress_callback(dict(payload))

    @staticmethod
    def _emit_module_timing(
        progress_callback,
        *,
        module: str,
        total_wall_ms: float,
        vlm_response_ms: float | None = 0.0,
        call_count: int = 1,
    ) -> None:
        if progress_callback is None:
            return
        numeric_vlm_ms = (
            float(vlm_response_ms)
            if isinstance(vlm_response_ms, (int, float))
            else None
        )
        program_module_ms = (
            max(0.0, float(total_wall_ms) - numeric_vlm_ms)
            if numeric_vlm_ms is not None
            else float(total_wall_ms)
        )
        progress_callback(
            {
                "event": "module_timing",
                "module": module,
                "total_wall_ms": float(total_wall_ms),
                "vlm_response_ms": numeric_vlm_ms,
                "program_module_ms": program_module_ms,
                "call_count": int(call_count),
            }
        )

    def _emit_localization_module_timings(
        self,
        progress_callback,
        *,
        observation: Observation,
        perception_total_ms: float,
        spatial_update_total_ms: float,
        localization_performed: bool,
    ) -> None:
        if not localization_performed:
            self._emit_module_timing(
                progress_callback,
                module="tracking_perception",
                total_wall_ms=perception_total_ms + spatial_update_total_ms,
            )
            return

        timing = observation.metadata.get("localization_timing")
        timing = timing if isinstance(timing, dict) else {}
        visual = timing.get("visual")
        visual = visual if isinstance(visual, dict) else {}
        visual_vlm_ms = visual.get("vlm_response_ms")
        self._emit_module_timing(
            progress_callback,
            module="visual_localization",
            total_wall_ms=perception_total_ms,
            vlm_response_ms=visual_vlm_ms,
            call_count=int(visual.get("logical_vlm_call_count", 0) or 0),
        )

        nested_total_ms = 0.0
        for stage_name, module_name in (
            ("bayesian", "bayesian_filtering"),
            ("spatial", "spatial_alignment"),
        ):
            stage = timing.get(stage_name)
            if not isinstance(stage, dict):
                continue
            stage_total = stage.get("stage_total_ms")
            if not isinstance(stage_total, (int, float)):
                continue
            nested_total_ms += float(stage_total)
            self._emit_module_timing(
                progress_callback,
                module=module_name,
                total_wall_ms=float(stage_total),
                vlm_response_ms=stage.get("vlm_response_ms"),
                call_count=int(stage.get("logical_vlm_call_count", 0) or 0),
            )
        overhead_ms = max(0.0, spatial_update_total_ms - nested_total_ms)
        if overhead_ms > 0.0:
            self._emit_module_timing(
                progress_callback,
                module="localization_state_overhead",
                total_wall_ms=overhead_ms,
            )

    @staticmethod
    def _clone_json(payload):
        if payload is None:
            return None
        if isinstance(payload, dict):
            return {key: EpisodeRunner._clone_json(value) for key, value in payload.items()}
        if isinstance(payload, list):
            return [EpisodeRunner._clone_json(value) for value in payload]
        return payload

    @staticmethod
    def _serialize_localization_event(step_index: int, state: BeliefState, observation: Observation) -> JsonDict:
        metadata = observation.metadata
        return {
            "schema": "localization_stage_event",
            "schema_version": 2,
            "step_index": step_index,
            "pano_id": observation.pano_id,
            "ground_truth_room_id": metadata.get("grounded_room_id") or state.grounded_room_id,
            "visual_cache_hit": metadata.get("visual_cache_hit"),
            "visual_cache_version": metadata.get("visual_cache_version"),
            "visual_cache_path": metadata.get("visual_cache_path"),
            "visual_room_scores": EpisodeRunner._clone_json(metadata.get("visual_room_scores")),
            "visual_room_belief": EpisodeRunner._clone_json(metadata.get("visual_room_belief")),
            "visual_predicted_room_id": metadata.get("visual_predicted_room_id"),
            "prior_room_belief": EpisodeRunner._clone_json(metadata.get("prior_room_belief")),
            "transition_support": EpisodeRunner._clone_json(metadata.get("transition_support")),
            "bayesian_room_belief": EpisodeRunner._clone_json(metadata.get("bayesian_room_belief")),
            "bayesian_predicted_room_id": metadata.get("bayesian_predicted_room_id"),
            "alignment_candidate_room_ids": EpisodeRunner._clone_json(
                metadata.get("alignment_candidate_room_ids")
            ),
            "alignment_applied": bool(metadata.get("alignment_applied", False)),
            "alignment_skipped_reason": metadata.get("alignment_skipped_reason"),
            "alignment_top_k": EpisodeRunner._clone_json(metadata.get("alignment_top_k")),
            "alignment_predicted_room_id": metadata.get("alignment_predicted_room_id"),
            "spatial_predicted_room_id": metadata.get("spatial_predicted_room_id"),
            "effective_localization_stage": metadata.get(
                "effective_localization_stage"
            ),
            "raw_localized_room_id": metadata.get("localized_room_id_raw"),
            "stabilized_room_id": metadata.get("stabilized_room_id") or state.current_room_id,
            "pending_room_id": metadata.get("pending_room_id"),
            "pending_room_confirmation_count": metadata.get("pending_room_confirmation_count"),
            "room_switch_confirmations_required": metadata.get("room_switch_confirmations_required"),
            "localization_timing": EpisodeRunner._clone_json(
                metadata.get("localization_timing")
            ),
            "initial_heading_alignment_status": metadata.get(
                "initial_heading_alignment_status"
            ),
            "initial_heading_alignment_failure_reason": metadata.get(
                "initial_heading_alignment_failure_reason"
            ),
            "heading_alignment_mode": state.heading_alignment_mode,
            "heading_alignment_status": state.heading_alignment_status,
            "heading_alignment_source_step": state.heading_alignment_source_step,
            "heading_alignment_confidence": state.heading_alignment_confidence,
            "view_0_geocentric_heading_deg": (
                state.heading_alignment_view_0_geocentric_heading_deg
            ),
            "view_0_allocentric_direction": (
                state.heading_alignment_view_0_allocentric_direction
            ),
            "estimated_offset_deg": state.geocentric_to_allocentric_offset_deg,
        }

    @staticmethod
    def _serialize_candidate_payload(candidate: CandidateAction) -> JsonDict:
        return {
            "target_pano_id": candidate.target_pano_id,
            "target_room_id": candidate.target_room_id,
            "absolute_heading": candidate.absolute_heading,
            "relative_heading": candidate.relative_heading,
            "relative_label": candidate.relative_label,
            "route_step_index": candidate.route_step_index,
            "score": candidate.score,
            "reason": candidate.reason,
            "metadata": EpisodeRunner._clone_json(candidate.metadata),
        }

    @staticmethod
    def _serialize_trace_payload(trace: EpisodeTrace) -> JsonDict:
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
            "current_room_context": EpisodeRunner._clone_json(trace.current_room_context),
            "visible_passages": EpisodeRunner._clone_json(trace.visible_passages),
            "view_contexts": EpisodeRunner._clone_json(trace.view_contexts),
            "candidates": [
                EpisodeRunner._serialize_candidate_payload(candidate) for candidate in trace.candidates
            ],
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
                "pending_room_confirmation_count": observation_metadata.get(
                    "pending_room_confirmation_count"
                ),
                "room_switch_confirmations_required": observation_metadata.get(
                    "room_switch_confirmations_required"
                ),
                "grounded_room_id": observation_metadata.get("grounded_room_id"),
                "transition_support": EpisodeRunner._clone_json(
                    observation_metadata.get("transition_support")
                    or observation_metadata.get("transition_room_support")
                ),
                "evidence_distribution": EpisodeRunner._clone_json(
                    observation_metadata.get("evidence_distribution")
                ),
                "visual_cache_hit": observation_metadata.get("visual_cache_hit"),
                "visual_cache_version": observation_metadata.get("visual_cache_version"),
                "visual_room_scores": EpisodeRunner._clone_json(observation_metadata.get("visual_room_scores")),
                "visual_room_belief": EpisodeRunner._clone_json(observation_metadata.get("visual_room_belief")),
                "visual_predicted_room_id": observation_metadata.get("visual_predicted_room_id"),
                "prior_room_belief": EpisodeRunner._clone_json(observation_metadata.get("prior_room_belief")),
                "bayesian_room_belief": EpisodeRunner._clone_json(observation_metadata.get("bayesian_room_belief")),
                "bayesian_predicted_room_id": observation_metadata.get("bayesian_predicted_room_id"),
                "spatial_predicted_room_id": observation_metadata.get("spatial_predicted_room_id"),
                "effective_localization_stage": observation_metadata.get(
                    "effective_localization_stage"
                ),
                "stabilized_room_id": observation_metadata.get("stabilized_room_id"),
                "base_predicted_room_id": observation_metadata.get("base_predicted_room_id"),
                "base_room_belief": EpisodeRunner._clone_json(observation_metadata.get("base_room_belief")),
                "alignment_candidate_room_ids": EpisodeRunner._clone_json(
                    observation_metadata.get("alignment_candidate_room_ids")
                ),
                "alignment_top_k": EpisodeRunner._clone_json(observation_metadata.get("alignment_top_k")),
                "alignment_predicted_room_id": observation_metadata.get("alignment_predicted_room_id"),
                "alignment_applied": observation_metadata.get("alignment_applied"),
                "alignment_skipped_reason": observation_metadata.get("alignment_skipped_reason"),
                "observation_likelihood": EpisodeRunner._clone_json(
                    observation_metadata.get("observation_likelihood")
                    or observation_metadata.get("observation_distribution")
                ),
                "room_belief": EpisodeRunner._clone_json(observation_metadata.get("room_belief")),
                "spatial_alignment": EpisodeRunner._clone_json(
                    observation_metadata.get("spatial_alignment")
                ),
                "entities": [
                    {
                        "name": entity.name,
                        "kind": entity.kind,
                        "confidence": entity.confidence,
                        "source_view": entity.source_view,
                        "source_views": EpisodeRunner._clone_json(entity.metadata.get("source_views")),
                    }
                    for entity in trace.observation.entities
                ],
            },
            "policy_output": {
                "rationale": trace.policy_output.rationale,
                "action": (
                    EpisodeRunner._serialize_candidate_payload(trace.policy_output.action)
                    if trace.policy_output.action is not None
                    else None
                ),
            },
            "policy_debug": {
                "request": EpisodeRunner._clone_json(trace.policy_request),
                "response": EpisodeRunner._clone_json(trace.policy_response),
            },
        }
