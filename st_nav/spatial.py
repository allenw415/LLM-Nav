from __future__ import annotations

from collections import deque

from .grounding import GroundingIndex
from .localization import RoomLocalizer
from .models import BeliefState, CandidateAction, Observation, TaskSpec
from .perception import normalize_heading


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
        self.localizer = localizer or RoomLocalizer(
            room_graph=room_graph,
            grounding_index=grounding_index,
        )

    def initialize(
        self,
        *,
        start_pano_id: str,
        start_room_id: str | None = None,
        start_heading: float = 330.0,
    ) -> BeliefState:
        room_belief = {start_room_id: 1.0} if start_room_id else {}
        visited_rooms = {start_room_id} if start_room_id else set()
        return BeliefState(
            current_pano_id=start_pano_id,
            current_room_id=start_room_id,
            current_heading=normalize_heading(start_heading),
            pano_belief={start_pano_id: 1.0},
            room_belief=room_belief,
            visited_panos={start_pano_id},
            visited_rooms=visited_rooms,
        )

    def update(self, state: BeliefState, observation: Observation) -> BeliefState:
        state.current_pano_id = observation.pano_id
        state.visited_panos.add(observation.pano_id)
        state.pano_belief = {observation.pano_id: 1.0}
        if observation.heading_estimate is not None:
            state.current_heading = normalize_heading(observation.heading_estimate)

        localized_room_id = observation.metadata.get("localized_room_id")
        localization_confidence = observation.metadata.get("localization_confidence", 1.0)
        if (
            isinstance(localized_room_id, str)
            and localized_room_id in self.room_graph
            and isinstance(localization_confidence, (int, float))
        ):
            state.current_room_id = localized_room_id
            room_belief = observation.metadata.get("room_belief")
            if isinstance(room_belief, dict):
                normalized_room_belief = {
                    room_id: float(probability)
                    for room_id, probability in room_belief.items()
                    if room_id in self.room_graph and isinstance(probability, (int, float))
                }
                state.room_belief = normalized_room_belief or {localized_room_id: float(localization_confidence)}
            else:
                state.room_belief = {localized_room_id: float(localization_confidence)}
            state.visited_rooms.add(localized_room_id)
        else:
            localization = self.localizer.localize(
                observation=observation,
                prior_room_belief=state.room_belief,
                fallback_room_id=state.current_room_id,
            )
            localized_room_id = localization.get("predicted_room_id")
            if isinstance(localized_room_id, str) and localized_room_id in self.room_graph:
                state.current_room_id = localized_room_id
                state.room_belief = dict(localization.get("room_belief", {}))
                state.visited_rooms.add(localized_room_id)
                observation.metadata["localized_room_id"] = localized_room_id
                observation.metadata["localization_confidence"] = float(localization.get("confidence", 0.0))
                observation.metadata["room_belief"] = dict(localization.get("room_belief", {}))
                observation.metadata["transition_room_support"] = dict(localization.get("transition_support", {}))
                observation.metadata["localization_evidence"] = list(localization.get("evidence", []))

        unexplored = self._unexplored_neighbor_count(state)
        if unexplored > 1 and (not state.junction_stack or state.junction_stack[-1] != state.current_pano_id):
            state.junction_stack.append(state.current_pano_id)

        return state

    def route_to_goal(self, task: TaskSpec, state: BeliefState) -> list[str]:
        if not state.current_room_id or not task.goal_room_ids:
            return []
        ordered_targets = list(task.waypoint_room_ids) + list(task.goal_room_ids)
        return self._compose_ordered_route(state.current_room_id, ordered_targets)

    def shortest_room_path(self, source_room_id: str, target_room_id: str) -> list[str]:
        if not source_room_id or not target_room_id:
            return []
        if source_room_id not in self.room_graph or target_room_id not in self.room_graph:
            return []
        return self._bfs_room_path(source_room_id, target_room_id)

    def shortest_room_path_avoiding(
        self,
        source_room_id: str,
        target_room_id: str,
        forbidden_room_ids: set[str] | None,
    ) -> list[str]:
        return self._bfs_room_path(source_room_id, target_room_id, forbidden_room_ids=forbidden_room_ids)

    def shortest_room_route(
        self,
        source_room_id: str,
        target_room_id: str,
        waypoint_room_ids: list[str] | None = None,
    ) -> list[str]:
        ordered_rooms = [source_room_id] + list(waypoint_room_ids or []) + [target_room_id]
        ordered_rooms = [room_id for room_id in ordered_rooms if room_id]
        if len(ordered_rooms) < 2:
            return ordered_rooms
        return self._compose_ordered_route(ordered_rooms[0], ordered_rooms[1:])

    def generate_candidates(self, state: BeliefState, route: list[str]) -> list[CandidateAction]:
        pano_record = self.pano_graph.get(state.current_pano_id, {})
        neighbors = pano_record.get("neighbors", [])
        desired_heading = self._desired_heading_for_route(state.current_room_id, route)
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
        if not task.goal_room_ids:
            return False
        return state.current_room_id == task.goal_room_ids[-1]

    def _compose_ordered_route(self, start_room_id: str, ordered_targets: list[str]) -> list[str]:
        route: list[str] = []
        cursor = start_room_id
        cleaned_targets = [room_id for room_id in ordered_targets if room_id]
        for target_room_id in cleaned_targets:
            if cursor == target_room_id:
                if not route:
                    route.append(cursor)
                continue
            segment = self.shortest_room_path(cursor, target_room_id)
            if not segment:
                return route
            if not route:
                route.extend(segment)
            else:
                route.extend(segment[1:])
            cursor = target_room_id
        return route

    def _bfs_room_path(
        self,
        start_room_id: str,
        goal_room_id: str,
        forbidden_room_ids: set[str] | None = None,
    ) -> list[str]:
        forbidden_room_ids = set(forbidden_room_ids or set())
        forbidden_room_ids.discard(start_room_id)
        forbidden_room_ids.discard(goal_room_id)
        queue: deque[str] = deque([start_room_id])
        parent = {start_room_id: None}
        while queue:
            room_id = queue.popleft()
            if room_id == goal_room_id:
                break
            for neighbor in self.room_graph.get(room_id, {}).get("neighbors", []):
                target_room_id = neighbor["target_room_id"]
                if target_room_id in forbidden_room_ids:
                    continue
                if target_room_id not in parent and target_room_id in self.room_graph:
                    parent[target_room_id] = room_id
                    queue.append(target_room_id)

        if goal_room_id not in parent:
            return []

        path: list[str] = []
        cursor: str | None = goal_room_id
        while cursor is not None:
            path.append(cursor)
            cursor = parent[cursor]
        path.reverse()
        return path

    def _desired_heading_for_route(self, current_room_id: str | None, route: list[str]) -> float | None:
        if not current_room_id or len(route) < 2:
            return None
        try:
            current_index = route.index(current_room_id)
        except ValueError:
            return None
        if current_index + 1 >= len(route):
            return None
        next_room_id = route[current_index + 1]
        for neighbor in self.room_graph.get(current_room_id, {}).get("neighbors", []):
            if neighbor["target_room_id"] == next_room_id:
                return neighbor.get("allocentric_heading_deg")
        return None

    def _unexplored_neighbor_count(self, state: BeliefState) -> int:
        pano_record = self.pano_graph.get(state.current_pano_id, {})
        return sum(
            1
            for neighbor in pano_record.get("neighbors", [])
            if neighbor["target_pano_id"] not in state.visited_panos
        )

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
