from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path

from ..common.types import BeliefState, CandidateAction, JsonDict, Observation, PolicyOutput, ReasoningInput, RenderedView, TaskSpec
from ..perception.renderer import normalize_heading, sanitize_name


@dataclass
class EpisodeTrace:
    step_index: int
    pano_id: str
    room_id: str | None
    route: list[str]
    observation: Observation
    policy_output: PolicyOutput
    subgoal_room_id: str | None = None
    candidates: list[CandidateAction] = field(default_factory=list)
    current_room_context: JsonDict = field(default_factory=dict)
    visible_passages: list[JsonDict] = field(default_factory=list)
    view_contexts: list[JsonDict] = field(default_factory=list)
    policy_request: JsonDict | None = None
    policy_response: JsonDict | None = None


class EpisodeRunner:
    def __init__(self, *, perception_provider, spatial_engine, policy, renderer=None):
        self.perception_provider = perception_provider
        self.spatial_engine = spatial_engine
        self.policy = policy
        self.renderer = renderer

    def run(
        self,
        *,
        task: TaskSpec,
        start_pano_id: str,
        manifest_paths: dict[str, str | Path] | None = None,
        start_room_id: str | None = None,
        start_heading: float = 330.0,
        step_budget: int = 10,
        render_api_key: str | None = None,
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
        manifest_paths = dict(manifest_paths or {})
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

        for step_index in range(step_budget):
            self._emit_progress(
                progress_callback,
                {
                    "event": "step_start",
                    "step_index": step_index,
                    "current_pano_id": state.current_pano_id,
                    "current_room_id": state.current_room_id,
                    "grounded_room_id": state.grounded_room_id,
                },
            )
            manifest_path = self._ensure_manifest(
                pano_id=state.current_pano_id,
                manifest_paths=manifest_paths,
                render_api_key=render_api_key,
                render_output_dir=render_output_dir,
                render_heading_mode=render_heading_mode,
                render_pitch=render_pitch,
                render_fov=render_fov,
                render_width=render_width,
                render_height=render_height,
                render_graph_path=render_graph_path,
                progress_callback=progress_callback,
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
                    "pano_id": state.current_pano_id,
                    "manifest_path": str(manifest_path),
                },
            )
            observation = self._run_perception(
                manifest_path=manifest_path,
                current_heading=state.current_heading,
            )
            self._emit_progress(
                progress_callback,
                {
                    "event": "perception_done",
                    "step_index": step_index,
                    "pano_id": observation.pano_id,
                    "entity_count": len(observation.entities),
                    "view_count": len(observation.views),
                },
            )
            state = self.spatial_engine.update(state, observation)
            reasoning_observation = self._build_reasoning_observation(
                state=state,
                base_observation=observation,
                manifest_paths=manifest_paths,
                render_api_key=render_api_key,
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
            self._emit_progress(
                progress_callback,
                {
                    "event": "localization_done",
                    "step_index": step_index,
                    "current_pano_id": state.current_pano_id,
                    "current_room_id": state.current_room_id,
                    "grounded_room_id": state.grounded_room_id,
                    "room_belief": dict(state.room_belief),
                },
            )
            route = self.spatial_engine.route_to_goal(task, state)
            subgoal_room_id = self.spatial_engine.next_subgoal_room_id(state, route)
            current_room_context = self.spatial_engine.build_current_room_context(state, route)
            visible_passages = self.spatial_engine.extract_visible_passages(state, observation)
            view_contexts = self.spatial_engine.describe_view_contexts(reasoning_observation)
            candidates = self.spatial_engine.generate_candidates(
                state,
                route,
                observation=observation,
                context_observation=reasoning_observation,
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
            policy_output = self._run_reasoning(
                task=task,
                route=route,
                candidates=candidates,
                current_room_id=state.current_room_id,
                current_room_context=current_room_context,
                visible_passages=visible_passages,
                view_contexts=view_contexts,
                spatial_alignment=observation.metadata.get("spatial_alignment"),
                subgoal_room_id=subgoal_room_id,
            )
            traces.append(
                EpisodeTrace(
                    step_index=step_index,
                    pano_id=state.current_pano_id,
                    room_id=state.current_room_id,
                    route=route,
                    subgoal_room_id=subgoal_room_id,
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

            if self.spatial_engine.goal_reached(task, state):
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

            state.current_pano_id = policy_output.action.target_pano_id
            state.current_heading = policy_output.action.absolute_heading
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
            },
        )
        return state, traces

    def _run_perception(self, *, manifest_path: str | Path, current_heading: float) -> Observation:
        if hasattr(self.perception_provider, "observe_from_manifest"):
            return self.perception_provider.observe_from_manifest(
                manifest_path,
                current_heading=current_heading,
            )
        return self.perception_provider.observe(
            manifest_path,
            current_heading=current_heading,
        )

    def _run_reasoning(
        self,
        *,
        task: TaskSpec,
        route: list[str],
        candidates,
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
        if self.renderer is None or render_api_key is None:
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
    ) -> str | Path | None:
        manifest_path = manifest_paths.get(pano_id)
        if manifest_path is not None:
            return manifest_path

        if self.renderer is None or render_api_key is None or render_output_dir is None:
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
    def _emit_progress(progress_callback, payload: dict) -> None:
        if progress_callback is None:
            return
        progress_callback(dict(payload))

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
        return {
            "step_index": trace.step_index,
            "pano_id": trace.pano_id,
            "room_id": trace.room_id,
            "grounded_room_id": trace.observation.metadata.get("grounded_room_id"),
            "route": list(trace.route),
            "subgoal_room_id": trace.subgoal_room_id,
            "current_room_context": EpisodeRunner._clone_json(trace.current_room_context),
            "visible_passages": EpisodeRunner._clone_json(trace.visible_passages),
            "view_contexts": EpisodeRunner._clone_json(trace.view_contexts),
            "candidates": [
                EpisodeRunner._serialize_candidate_payload(candidate) for candidate in trace.candidates
            ],
            "observation": {
                "pano_id": trace.observation.pano_id,
                "heading_estimate": trace.observation.heading_estimate,
                "localized_room_id": trace.observation.metadata.get("localized_room_id"),
                "grounded_room_id": trace.observation.metadata.get("grounded_room_id"),
                "spatial_alignment": EpisodeRunner._clone_json(
                    trace.observation.metadata.get("spatial_alignment")
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
