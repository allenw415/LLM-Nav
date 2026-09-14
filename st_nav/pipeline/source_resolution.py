from __future__ import annotations

from dataclasses import dataclass, replace

from ..common.types import SourcePanoResolution, TaskSpec


@dataclass
class SourceResolutionResult:
    instruction: str
    task: TaskSpec
    source_pano: SourcePanoResolution


class SourceResolutionWorkflow:
    """
    Resolve a caller-provided start panorama and combine it with parsed goals.
    """

    def __init__(self, *, instruction_parser, source_pano_resolver):
        self.instruction_parser = instruction_parser
        self.source_pano_resolver = source_pano_resolver

    def run(
        self,
        instruction: str,
        *,
        start_pano_id: str,
    ) -> SourceResolutionResult:
        task = self.instruction_parser.parse(instruction)
        self._validate_parsed_task(task)
        start_pano_id, start_room_id = self._resolve_start_pano(start_pano_id)
        task = replace(
            task,
            source_room_id=start_room_id,
            source_entity=None,
        )
        return SourceResolutionResult(
            instruction=instruction,
            task=task,
            source_pano=self._source_pano_result(
                start_pano_id=start_pano_id,
                start_room_id=start_room_id,
            ),
        )

    def run_explicit(
        self,
        *,
        start_pano_id: str,
        target_room_id: str,
        waypoint_room_ids: list[str] | None = None,
    ) -> SourceResolutionResult:
        start_pano_id, start_room_id = self._resolve_start_pano(start_pano_id)
        target_room_id = target_room_id.strip()
        if not target_room_id:
            raise ValueError("target_room_id must not be empty.")

        room_graph = self.instruction_parser.room_graph
        if target_room_id not in room_graph:
            raise ValueError(f"Unknown target room: {target_room_id}")

        waypoints: list[str] = []
        for value in waypoint_room_ids or []:
            room_id = str(value).strip()
            if (
                room_id
                and room_id != target_room_id
                and room_id not in waypoints
            ):
                waypoints.append(room_id)
        unknown_waypoints = [
            room_id for room_id in waypoints if room_id not in room_graph
        ]
        if unknown_waypoints:
            raise ValueError(
                f"Unknown waypoint rooms: {', '.join(unknown_waypoints)}"
            )

        raw_instruction = (
            f"Navigate from panorama {start_pano_id} in {start_room_id} "
            f"to {target_room_id}."
        )
        task = TaskSpec(
            task_type=(
                "gallery_instruction_following_navigation"
                if waypoints
                else "gallery_goal_navigation"
            ),
            raw_instruction=raw_instruction,
            source_room_id=start_room_id,
            source_entity=None,
            goal_room_ids=[target_room_id],
            waypoint_room_ids=waypoints,
        )
        return SourceResolutionResult(
            instruction=raw_instruction,
            task=task,
            source_pano=self._source_pano_result(
                start_pano_id=start_pano_id,
                start_room_id=start_room_id,
            ),
        )

    def _resolve_start_pano(self, start_pano_id: str) -> tuple[str, str]:
        start_pano_id = str(start_pano_id or "").strip()
        if not start_pano_id:
            raise ValueError("start_pano_id must not be empty.")

        grounding_index = self.source_pano_resolver.grounding_index
        start_room_id = grounding_index.room_for_pano(start_pano_id)
        if not start_room_id:
            raise ValueError(
                f"Start panorama is not grounded to a room: {start_pano_id}"
            )
        if start_room_id not in self.instruction_parser.room_graph:
            raise ValueError(
                f"Start panorama grounds to an unknown room: "
                f"{start_pano_id} -> {start_room_id}"
            )
        return start_pano_id, start_room_id

    def _validate_parsed_task(self, task: TaskSpec) -> None:
        room_graph = self.instruction_parser.room_graph
        if not task.goal_room_ids:
            raise ValueError("Navigation query did not resolve any goal room ids.")

        unknown_goals = [
            room_id for room_id in task.goal_room_ids if room_id not in room_graph
        ]
        if unknown_goals:
            raise ValueError(f"Unknown goal rooms: {', '.join(unknown_goals)}")

        unknown_waypoints = [
            room_id
            for room_id in task.waypoint_room_ids
            if room_id not in room_graph
        ]
        if unknown_waypoints:
            raise ValueError(
                f"Unknown waypoint rooms: {', '.join(unknown_waypoints)}"
            )

    @staticmethod
    def _source_pano_result(
        *,
        start_pano_id: str,
        start_room_id: str,
    ) -> SourcePanoResolution:
        return SourcePanoResolution(
            source_room_id=start_room_id,
            pano_id=start_pano_id,
            candidate_pano_ids=[start_pano_id],
            resolution_method="explicit_start_pano",
        )
