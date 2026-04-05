from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .models import BeliefState, Observation, PolicyOutput, ReasoningInput, TaskSpec


@dataclass
class EpisodeTrace:
    step_index: int
    pano_id: str
    room_id: str | None
    route: list[str]
    observation: Observation
    policy_output: PolicyOutput


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
        render_graph_path: str | Path | None = None,
    ) -> tuple[BeliefState, list[EpisodeTrace]]:
        manifest_paths = dict(manifest_paths or {})
        state = self.spatial_engine.initialize(
            start_pano_id=start_pano_id,
            start_room_id=start_room_id,
            start_heading=start_heading,
        )
        traces: list[EpisodeTrace] = []

        for step_index in range(step_budget):
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
            )
            if manifest_path is None:
                break

            observation = self._run_perception(
                manifest_path=manifest_path,
                current_heading=state.current_heading,
            )
            state = self.spatial_engine.update(state, observation)
            route = self.spatial_engine.route_to_goal(task, state)
            candidates = self.spatial_engine.generate_candidates(state, route)
            policy_output = self._run_reasoning(
                task=task,
                route=route,
                candidates=candidates,
                current_room_id=state.current_room_id,
            )
            traces.append(
                EpisodeTrace(
                    step_index=step_index,
                    pano_id=state.current_pano_id,
                    room_id=state.current_room_id,
                    route=route,
                    observation=observation,
                    policy_output=policy_output,
                )
            )

            if self.spatial_engine.goal_reached(task, state):
                break

            if policy_output.action is None:
                break

            state.current_pano_id = policy_output.action.target_pano_id
            state.current_heading = policy_output.action.absolute_heading

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
    ) -> PolicyOutput:
        reasoning_input = ReasoningInput(
            task=task,
            route=route,
            candidates=candidates,
            current_room_id=current_room_id,
        )
        if hasattr(self.policy, "choose_next_action"):
            return self.policy.choose_next_action(reasoning_input)
        return self.policy.choose_action(task=task, route=route, candidates=candidates)

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
        )
        manifest_path = manifest["manifest_path"]
        manifest_paths[pano_id] = manifest_path
        return manifest_path
