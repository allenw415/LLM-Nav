from __future__ import annotations

from ..common.types import BeliefState, CandidateAction, Observation, TaskSpec
from ..perception.renderer import normalize_heading
from .grounding import GroundingIndex
from .localization import RoomLocalizer
from .routing import RoutePlanner
from .state import StateEstimator


class SpatialEngine:
    def __init__(
        self,
        *,
        room_graph: dict[str, dict],
        pano_graph: dict[str, dict],
        grounding_index: GroundingIndex,
        localizer: RoomLocalizer | None = None,
    ):
        self.room_graph = room_graph
        self.pano_graph = pano_graph
        self.grounding_index = grounding_index
        self.route_planner = RoutePlanner(room_graph=room_graph)
        self.state_estimator = StateEstimator(
            room_graph=room_graph,
            pano_graph=pano_graph,
            grounding_index=grounding_index,
            localizer=localizer,
        )

    def initialize(
        self,
        *,
        start_pano_id: str,
        start_room_id: str | None = None,
        start_heading: float = 330.0,
    ) -> BeliefState:
        return self.state_estimator.initialize(
            start_pano_id=start_pano_id,
            start_room_id=start_room_id,
            start_heading=start_heading,
        )

    def update(self, state: BeliefState, observation: Observation) -> BeliefState:
        return self.state_estimator.update(state, observation)

    def route_to_goal(self, task: TaskSpec, state: BeliefState) -> list[str]:
        return self.route_planner.route_to_goal(task, state.current_room_id)

    def shortest_room_path(self, source_room_id: str, target_room_id: str) -> list[str]:
        return self.route_planner.shortest_room_path(source_room_id, target_room_id)

    def shortest_room_path_avoiding(
        self,
        source_room_id: str,
        target_room_id: str,
        forbidden_room_ids: set[str] | None,
    ) -> list[str]:
        return self.route_planner.shortest_room_path_avoiding(
            source_room_id,
            target_room_id,
            forbidden_room_ids,
        )

    def shortest_room_route(
        self,
        source_room_id: str,
        target_room_id: str,
        waypoint_room_ids: list[str] | None = None,
    ) -> list[str]:
        return self.route_planner.shortest_room_route(
            source_room_id,
            target_room_id,
            waypoint_room_ids,
        )

    def generate_candidates(self, state: BeliefState, route: list[str]) -> list[CandidateAction]:
        pano_record = self.pano_graph.get(state.current_pano_id, {})
        neighbors = pano_record.get("neighbors", [])
        desired_heading = self.route_planner.desired_heading_for_route(state.current_room_id, route)
        candidates: list[CandidateAction] = []
        for neighbor in neighbors:
            target_pano_id = str(neighbor["target_pano_id"])
            absolute_heading = float(neighbor["geocentric_heading_deg"])
            relative_heading = normalize_heading(absolute_heading - state.current_heading)
            target_room_id = None

            score = 0.0
            reason_parts = []
            if target_pano_id not in state.visited_panos:
                score += 1.0
                reason_parts.append("unvisited")
            if desired_heading is not None:
                diff = self._angular_distance(desired_heading, absolute_heading)
                score += max(0.0, 1.0 - diff / 180.0)
                reason_parts.append(f"heading_diff={diff:.1f}")
            if target_room_id and route and target_room_id in route:
                score += 0.5
                reason_parts.append("supports_route")

            candidates.append(
                CandidateAction(
                    target_pano_id=target_pano_id,
                    absolute_heading=absolute_heading,
                    relative_heading=relative_heading,
                    relative_label=self._relative_label(relative_heading),
                    target_room_id=target_room_id,
                    score=score,
                    reason=", ".join(reason_parts) if reason_parts else "fallback",
                )
            )

        if not candidates and state.junction_stack:
            target_pano_id = state.junction_stack[-1]
            candidates.append(
                CandidateAction(
                    target_pano_id=target_pano_id,
                    absolute_heading=state.current_heading,
                    relative_heading=0.0,
                    relative_label="front",
                    target_room_id=None,
                    score=0.1,
                    reason="backtrack_to_junction",
                )
            )

        candidates.sort(key=lambda item: (-item.score, item.target_pano_id))
        return candidates

    def goal_reached(self, task: TaskSpec, state: BeliefState) -> bool:
        return self.state_estimator.goal_reached(task, state)

    @staticmethod
    def _angular_distance(left: float, right: float) -> float:
        diff = abs(normalize_heading(left - right))
        return min(diff, 360.0 - diff)

    @staticmethod
    def _relative_label(relative_heading: float) -> str:
        angle = normalize_heading(relative_heading)
        if angle < 45 or angle >= 315:
            return "front"
        if angle < 135:
            return "right"
        if angle < 225:
            return "back"
        return "left"
