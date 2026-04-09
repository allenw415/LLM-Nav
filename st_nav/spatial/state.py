from __future__ import annotations

from ..common.types import BeliefState, Observation, TaskSpec
from ..perception.renderer import normalize_heading
from .grounding import GroundingIndex
from .localization import RoomLocalizer


class StateEstimator:
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

    @staticmethod
    def goal_reached(task: TaskSpec, state: BeliefState) -> bool:
        if not task.goal_room_ids:
            return False
        return state.current_room_id == task.goal_room_ids[-1]

    def _unexplored_neighbor_count(self, state: BeliefState) -> int:
        pano_record = self.pano_graph.get(state.current_pano_id, {})
        return sum(
            1
            for neighbor in pano_record.get("neighbors", [])
            if neighbor["target_pano_id"] not in state.visited_panos
        )
