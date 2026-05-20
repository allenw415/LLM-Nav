from __future__ import annotations

from ..common.types import BeliefState, CandidateAction, EntityDetection, JsonDict, Observation, TaskSpec
from ..perception.renderer import normalize_heading
from .grounding import GroundingIndex
from .localization import EvidenceScoreLocalizer
from .routing import RoutePlanner
from .state import StateEstimator


class SpatialEngine:
    def __init__(
        self,
        *,
        room_graph: dict[str, dict],
        pano_graph: dict[str, dict],
        grounding_index: GroundingIndex,
        localizer: EvidenceScoreLocalizer | None = None,
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

    def annotate_observation_context(self, state: BeliefState, observation: Observation) -> Observation:
        localized_room_id = observation.metadata.get("localized_room_id")
        localization_confidence = observation.metadata.get("localization_confidence")
        if (
            isinstance(localized_room_id, str)
            and localized_room_id in self.room_graph
            and isinstance(localization_confidence, (int, float))
        ):
            return observation
        localization = self.state_estimator.localizer.localize(
            observation=observation,
            prior_room_belief=state.room_belief,
            fallback_room_id=state.current_room_id,
        )
        localized_room_id = localization.get("predicted_room_id")
        if isinstance(localized_room_id, str) and localized_room_id in self.room_graph:
            observation.metadata["localized_room_id"] = localized_room_id
            observation.metadata["localization_confidence"] = float(localization.get("confidence", 0.0))
            observation.metadata["room_belief"] = dict(localization.get("room_belief", {}))
            observation.metadata["transition_support"] = dict(localization.get("transition_support", {}))
            observation.metadata["transition_room_support"] = dict(localization.get("transition_support", {}))
            observation.metadata["observation_likelihood"] = dict(localization.get("observation_likelihood", {}))
            observation.metadata["evidence_distribution"] = dict(localization.get("evidence_distribution", {}))
            observation.metadata["base_predicted_room_id"] = localization.get("base_predicted_room_id")
            observation.metadata["base_room_belief"] = dict(localization.get("base_room_belief", {}))
            observation.metadata["alignment_candidate_room_ids"] = list(localization.get("alignment_candidate_room_ids", []))
            observation.metadata["alignment_top_k"] = list(localization.get("alignment_top_k", []))
            observation.metadata["alignment_predicted_room_id"] = localization.get("alignment_predicted_room_id")
            observation.metadata["alignment_applied"] = bool(localization.get("alignment_applied", False))
            if localization.get("alignment_skipped_reason"):
                observation.metadata["alignment_skipped_reason"] = localization.get("alignment_skipped_reason")
            observation.metadata["localization_evidence"] = list(localization.get("evidence", []))
        visual_localization = localization.get("visual_localization")
        if isinstance(visual_localization, dict):
            observation.metadata["visual_localization"] = dict(visual_localization)
        spatial_alignment = localization.get("spatial_alignment")
        if isinstance(spatial_alignment, dict):
            observation.metadata["spatial_alignment"] = dict(spatial_alignment)
        ego_spatial_context = localization.get("ego_spatial_context")
        if isinstance(ego_spatial_context, dict):
            observation.metadata["ego_spatial_context"] = dict(ego_spatial_context)
        return observation

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

    def next_subgoal_room_id(self, state: BeliefState, route: list[str]) -> str | None:
        if not route:
            return None
        if not state.current_room_id:
            return route[0]
        try:
            current_index = route.index(state.current_room_id)
        except ValueError:
            return route[0]
        if current_index + 1 >= len(route):
            return None
        return route[current_index + 1]

    def build_current_room_context(self, state: BeliefState, route: list[str]) -> JsonDict:
        room_id = state.current_room_id
        node = self.room_graph.get(room_id or "", {})
        subgoal_room_id = self.next_subgoal_room_id(state, route)
        subgoal_node = self.room_graph.get(subgoal_room_id or "", {})
        remaining_route = self._remaining_route(state.current_room_id, route)
        neighbors = []
        for neighbor in node.get("neighbors", []):
            if not isinstance(neighbor, dict):
                continue
            target_room_id = neighbor.get("target_room_id")
            if not isinstance(target_room_id, str) or not target_room_id:
                continue
            target_node = self.room_graph.get(target_room_id, {})
            neighbors.append(
                {
                    "target_room_id": target_room_id,
                    "target_title": target_node.get("title"),
                    "allocentric_direction": neighbor.get("allocentric_direction"),
                    "allocentric_heading_deg": neighbor.get("allocentric_heading_deg"),
                    "transition_type": neighbor.get("transition_type"),
                    "is_subgoal": target_room_id == subgoal_room_id,
                    "is_on_remaining_route": target_room_id in remaining_route[1:],
                }
            )
        return {
            "room_id": room_id,
            "title": node.get("title"),
            "category": node.get("category"),
            "subgoal_room_id": subgoal_room_id,
            "subgoal_title": subgoal_node.get("title"),
            "subgoal_theme_labels": self._room_theme_labels(subgoal_node),
            "remaining_route": remaining_route,
            "neighbors": neighbors,
        }

    @staticmethod
    def _room_theme_labels(node: dict) -> list[str]:
        labels: list[str] = []
        seen: set[str] = set()
        for key in ("title", "category"):
            value = node.get(key)
            if isinstance(value, str):
                label = value.strip()
                if label and label not in seen:
                    seen.add(label)
                    labels.append(label)
        return labels

    def extract_visible_passages(self, state: BeliefState, observation: Observation) -> list[JsonDict]:
        current_room_neighbors = self.room_graph.get(state.current_room_id or "", {}).get("neighbors", [])
        direction_to_room_ids: dict[str, list[str]] = {}
        for neighbor in current_room_neighbors:
            if not isinstance(neighbor, dict):
                continue
            direction = neighbor.get("allocentric_direction")
            target_room_id = neighbor.get("target_room_id")
            if not isinstance(direction, str) or not isinstance(target_room_id, str):
                continue
            direction_to_room_ids.setdefault(direction, []).append(target_room_id)

        view_directions = self._infer_view_directions(observation)
        passages: list[JsonDict] = []
        for entity in observation.entities:
            if entity.kind != "passage":
                continue
            source_views = list(entity.metadata.get("source_views", [])) or [entity.source_view]
            source_views = [view_id for view_id in source_views if isinstance(view_id, str) and view_id]
            inferred_directions: list[str] = []
            matched_room_ids: list[str] = []
            for source_view in source_views:
                direction = view_directions.get(source_view)
                if not direction or direction in inferred_directions:
                    continue
                inferred_directions.append(direction)
                for room_id in direction_to_room_ids.get(direction, []):
                    if room_id not in matched_room_ids:
                        matched_room_ids.append(room_id)
            passages.append(
                {
                    "name": entity.name,
                    "confidence": float(entity.confidence),
                    "source_views": source_views,
                    "allocentric_directions": inferred_directions,
                    "matched_room_ids": matched_room_ids,
                }
            )
        return passages

    def describe_view_contexts(self, observation: Observation) -> list[JsonDict]:
        if not observation.views:
            return []
        spatial_alignment = observation.metadata.get("spatial_alignment")
        spatial_alignment = spatial_alignment if isinstance(spatial_alignment, dict) else {}
        view_directions = self._infer_view_directions(observation)
        ego_context_by_id: dict[str, JsonDict] = {}
        raw_ego_views = spatial_alignment.get("ego_context_views")
        if not isinstance(raw_ego_views, list):
            raw_ego_spatial_context = observation.metadata.get("ego_spatial_context")
            if isinstance(raw_ego_spatial_context, dict):
                raw_ego_views = raw_ego_spatial_context.get("views")
        if isinstance(raw_ego_views, list):
            for record in raw_ego_views:
                if not isinstance(record, dict):
                    continue
                view_id = record.get("view_id")
                if isinstance(view_id, str) and view_id:
                    ego_context_by_id[view_id] = record

        entities_by_label: dict[str, list[EntityDetection]] = {}
        for entity in observation.entities:
            source_views = entity.metadata.get("source_views")
            if isinstance(source_views, list):
                labels = [value for value in source_views if isinstance(value, str) and value]
            elif isinstance(entity.source_view, str) and entity.source_view:
                labels = [entity.source_view]
            else:
                labels = []
            for label in labels:
                entities_by_label.setdefault(label, []).append(entity)

        contexts: list[JsonDict] = []
        for index, view in enumerate(observation.views):
            view_id = f"view_{index}"
            label = view.label if isinstance(view.label, str) else view_id
            aligned_direction = view_directions.get(label)
            entities = entities_by_label.get(label, [])
            serialized_entities = [
                {
                    "name": entity.name,
                    "kind": entity.kind,
                    "confidence": float(entity.confidence),
                }
                for entity in sorted(entities, key=lambda item: (-float(item.confidence), item.name.lower()))[:6]
            ]
            passage_names = [
                entity.name
                for entity in sorted(entities, key=lambda item: (-float(item.confidence), item.name.lower()))
                if entity.kind == "passage"
            ][:3]
            ego_context = ego_context_by_id.get(view_id, {})
            raw_themes = ego_context.get("themes")
            themes = []
            if isinstance(raw_themes, list):
                for theme in raw_themes[:3]:
                    if not isinstance(theme, dict):
                        continue
                    label_text = theme.get("label")
                    confidence = theme.get("confidence")
                    if isinstance(label_text, str) and label_text:
                        themes.append(
                            {
                                "label": label_text,
                                "confidence": float(confidence) if isinstance(confidence, (int, float)) else 0.0,
                            }
                        )
            contexts.append(
                {
                    "view_id": view_id,
                    "label": label,
                    "heading": float(view.heading),
                    "allocentric_direction": aligned_direction,
                    "themes": themes,
                    "entities": serialized_entities,
                    "passages": passage_names,
                }
            )
        return contexts

    def generate_candidates(
        self,
        state: BeliefState,
        route: list[str],
        observation: Observation | None = None,
        context_observation: Observation | None = None,
    ) -> list[CandidateAction]:
        pano_record = self.pano_graph.get(state.current_pano_id, {})
        neighbors = pano_record.get("neighbors", [])
        context_source = context_observation if context_observation is not None else observation
        view_contexts = self.describe_view_contexts(context_source) if context_source is not None else []
        desired_heading = self.route_planner.desired_heading_for_route(state.current_room_id, route)
        subgoal_room_id = self.next_subgoal_room_id(state, route)
        inferred_current_heading = self._infer_current_heading_from_alignment(observation)
        target_relative_heading = None
        target_heading_metadata: JsonDict = {}
        if desired_heading is not None:
            target_relative_heading = desired_heading
            target_heading_metadata = {
                "target_heading_source": "allocentric_subgoal",
                "target_allocentric_direction": self._heading_to_allocentric_direction(desired_heading),
            }
        candidates: list[CandidateAction] = []
        for neighbor in neighbors:
            target_pano_id = str(neighbor["target_pano_id"])
            is_immediate_backtrack = (
                isinstance(state.previous_pano_id, str)
                and state.previous_pano_id
                and target_pano_id == state.previous_pano_id
            )
            absolute_heading = float(neighbor["geocentric_heading_deg"])
            allocentric_heading = self._candidate_allocentric_heading(absolute_heading)
            inferred_target_room_id, room_transition = self._match_room_transition(
                state.current_room_id,
                absolute_heading,
                allocentric_heading=allocentric_heading,
            )
            grounded_target_room_id = self.grounding_index.room_for_pano(target_pano_id)
            target_room_id = grounded_target_room_id or inferred_target_room_id
            relative_heading = None
            relative_label = None
            if allocentric_heading is not None:
                relative_heading = allocentric_heading
                relative_label = self._relative_label(relative_heading)

            route_step_index = None
            if target_room_id and target_room_id in route:
                route_step_index = route.index(target_room_id)

            score = 0.0
            reason_parts = []
            if target_pano_id not in state.visited_panos:
                score += 1.0
                reason_parts.append("unvisited")
            if target_room_id == subgoal_room_id:
                score += 3.0
                reason_parts.append("matches_subgoal")
            elif route_step_index is not None:
                score += 1.5
                reason_parts.append("on_route")
            if target_room_id and target_room_id not in state.visited_rooms:
                score += 0.25
                reason_parts.append("leads_to_unvisited_room")
            target_relative_diff = None
            if target_relative_heading is not None and relative_heading is not None:
                target_relative_diff = self._angular_distance(target_relative_heading, relative_heading)
                score += 2.0 * max(0.0, 1.0 - target_relative_diff / 180.0)
                reason_parts.append(f"target_relative_diff={target_relative_diff:.1f}")
            elif desired_heading is not None:
                heading_reference = allocentric_heading if allocentric_heading is not None else absolute_heading
                diff = self._angular_distance(desired_heading, heading_reference)
                score += max(0.0, 1.0 - diff / 180.0)
                reason_parts.append(f"heading_diff={diff:.1f}")

            metadata = {
                "inferred_current_heading": inferred_current_heading,
                "allocentric_heading_deg": allocentric_heading,
                "desired_allocentric_heading_deg": desired_heading,
                "target_relative_heading_deg": target_relative_heading,
                "target_relative_diff_deg": target_relative_diff,
                "grounded_target_room_id": grounded_target_room_id,
                "inferred_target_room_id": inferred_target_room_id,
                "is_immediate_backtrack": is_immediate_backtrack,
            }
            if view_contexts:
                metadata["spatial_context"] = self._candidate_spatial_context(
                    absolute_heading=absolute_heading,
                    allocentric_heading=allocentric_heading,
                    view_contexts=view_contexts,
                )
            metadata.update(target_heading_metadata)
            if room_transition:
                metadata.update(room_transition)
                heading_diff = room_transition.get("heading_diff_deg")
                if isinstance(heading_diff, (int, float)):
                    reason_parts.append(f"room_transition_diff={float(heading_diff):.1f}")

            candidates.append(
                CandidateAction(
                    target_pano_id=target_pano_id,
                    absolute_heading=absolute_heading,
                    relative_heading=relative_heading,
                    relative_label=relative_label,
                    target_room_id=target_room_id,
                    route_step_index=route_step_index,
                    score=score,
                    reason=", ".join(reason_parts) if reason_parts else "fallback",
                    metadata=metadata,
                )
            )

        non_backtracking_candidates = [
            candidate
            for candidate in candidates
            if not candidate.metadata.get("is_immediate_backtrack")
        ]
        if non_backtracking_candidates:
            candidates = non_backtracking_candidates

        if not candidates and state.junction_stack:
            target_pano_id = state.junction_stack[-1]
            candidates.append(
                CandidateAction(
                    target_pano_id=target_pano_id,
                    absolute_heading=state.current_heading,
                    relative_heading=0.0 if inferred_current_heading is not None else None,
                    relative_label="front" if inferred_current_heading is not None else None,
                    target_room_id=None,
                    route_step_index=None,
                    score=0.1,
                    reason="backtrack_to_junction",
                    metadata={"inferred_current_heading": inferred_current_heading},
                )
            )

        candidates.sort(
            key=lambda item: (
                item.metadata.get("target_relative_diff_deg")
                if isinstance(item.metadata.get("target_relative_diff_deg"), (int, float))
                else 10**6,
                -(item.target_room_id == subgoal_room_id),
                item.route_step_index if item.route_step_index is not None else 10**6,
                -item.score,
                item.target_pano_id,
            )
        )
        return candidates

    @staticmethod
    def _candidate_allocentric_heading(geocentric_heading: float) -> float:
        # British Museum pano headings use a stable +30deg offset from the room-map allocentric frame.
        return normalize_heading(float(geocentric_heading) + 30.0)

    def _candidate_spatial_context(
        self,
        *,
        absolute_heading: float,
        allocentric_heading: float | None,
        view_contexts: list[JsonDict],
    ) -> JsonDict:
        if not view_contexts:
            return {}
        candidate_specific_views = all(
            isinstance(record.get("label"), str) and str(record.get("label")).startswith("candidate_")
            for record in view_contexts
        )
        ranked_views = sorted(
            view_contexts,
            key=lambda record: (
                self._angular_distance(absolute_heading, float(record.get("heading", 0.0))),
                str(record.get("view_id", "")),
            ),
        )
        supporting_views = []
        max_supporting_views = 1 if candidate_specific_views else 2
        for record in ranked_views[:max_supporting_views]:
            view_heading = float(record.get("heading", 0.0))
            supporting_views.append(
                {
                    "view_id": record.get("view_id"),
                    "label": record.get("label"),
                    "heading": view_heading,
                    "angular_distance_deg": self._angular_distance(absolute_heading, view_heading),
                    "allocentric_direction": record.get("allocentric_direction"),
                    "themes": list(record.get("themes", [])),
                    "entities": list(record.get("entities", [])),
                    "passages": list(record.get("passages", [])),
                }
            )

        salient_entities: list[JsonDict] = []
        seen_entity_names: set[str] = set()
        for record in supporting_views:
            for entity in record.get("entities", []):
                if not isinstance(entity, dict):
                    continue
                name = entity.get("name")
                if not isinstance(name, str) or not name or name in seen_entity_names:
                    continue
                seen_entity_names.add(name)
                salient_entities.append(entity)
                if len(salient_entities) >= 6:
                    break
            if len(salient_entities) >= 6:
                break

        theme_hints: list[JsonDict] = []
        seen_theme_labels: set[str] = set()
        for record in supporting_views:
            for theme in record.get("themes", []):
                if not isinstance(theme, dict):
                    continue
                label = theme.get("label")
                if not isinstance(label, str) or not label or label in seen_theme_labels:
                    continue
                seen_theme_labels.add(label)
                theme_hints.append(theme)
                if len(theme_hints) >= 4:
                    break
            if len(theme_hints) >= 4:
                break

        return {
            "candidate_geocentric_heading_deg": float(absolute_heading),
            "candidate_allocentric_heading_deg": float(allocentric_heading) if allocentric_heading is not None else None,
            "supporting_views": supporting_views,
            "salient_entities": salient_entities,
            "theme_hints": theme_hints,
        }

    def goal_reached(self, task: TaskSpec, state: BeliefState) -> bool:
        return self.state_estimator.goal_reached(task, state)

    def _remaining_route(self, current_room_id: str | None, route: list[str]) -> list[str]:
        if not route:
            return []
        if not current_room_id:
            return list(route)
        try:
            current_index = route.index(current_room_id)
        except ValueError:
            return list(route)
        return list(route[current_index:])

    def _match_room_transition(
        self,
        current_room_id: str | None,
        absolute_heading: float,
        *,
        allocentric_heading: float | None = None,
    ) -> tuple[str | None, JsonDict]:
        if not current_room_id:
            return None, {}
        room_neighbors = [
            neighbor
            for neighbor in self.room_graph.get(current_room_id, {}).get("neighbors", [])
            if isinstance(neighbor, dict)
        ]

        if allocentric_heading is not None:
            inferred_direction = self._heading_to_allocentric_direction(allocentric_heading)
            directional_neighbors = [
                neighbor
                for neighbor in room_neighbors
                if neighbor.get("allocentric_direction") == inferred_direction
            ]
            if directional_neighbors:
                best_neighbor = min(
                    directional_neighbors,
                    key=lambda neighbor: self._angular_distance(
                        float(neighbor.get("allocentric_heading_deg", 0.0)),
                        allocentric_heading,
                    ),
                )
                best_diff = self._angular_distance(
                    float(best_neighbor.get("allocentric_heading_deg", 0.0)),
                    allocentric_heading,
                )
                return str(best_neighbor.get("target_room_id")), {
                    "allocentric_direction": best_neighbor.get("allocentric_direction"),
                    "allocentric_heading_deg": float(best_neighbor.get("allocentric_heading_deg")),
                    "heading_diff_deg": float(best_diff),
                    "transition_type": best_neighbor.get("transition_type"),
                    "matching_strategy": "spatial_alignment_direction",
                    "candidate_allocentric_heading_deg": float(allocentric_heading),
                }

        best_neighbor = None
        best_diff = None
        for neighbor in room_neighbors:
            if not isinstance(neighbor, dict):
                continue
            room_heading = neighbor.get("allocentric_heading_deg")
            if not isinstance(room_heading, (int, float)):
                continue
            reference_heading = allocentric_heading if allocentric_heading is not None else absolute_heading
            diff = self._angular_distance(float(room_heading), reference_heading)
            if best_diff is None or diff < best_diff:
                best_diff = diff
                best_neighbor = neighbor

        if best_neighbor is None or best_diff is None or best_diff > 60.0:
            return None, {}

        return str(best_neighbor.get("target_room_id")), {
            "allocentric_direction": best_neighbor.get("allocentric_direction"),
            "allocentric_heading_deg": float(best_neighbor.get("allocentric_heading_deg")),
            "heading_diff_deg": float(best_diff),
            "transition_type": best_neighbor.get("transition_type"),
            "matching_strategy": "allocentric_heading_fallback" if allocentric_heading is not None else "geocentric_fallback",
            "candidate_allocentric_heading_deg": float(allocentric_heading) if allocentric_heading is not None else None,
        }

    def _infer_current_heading_from_alignment(self, observation: Observation | None) -> float | None:
        if observation is None or not observation.views:
            return None
        spatial_alignment = observation.metadata.get("spatial_alignment")
        if not isinstance(spatial_alignment, dict):
            return None
        direction = spatial_alignment.get("view_0_allocentric_direction")
        allocentric_heading = self._allocentric_heading(direction)
        if allocentric_heading is None:
            return None
        view_0_heading = observation.views[0].heading
        return normalize_heading(float(view_0_heading) - allocentric_heading)

    def _infer_view_directions(self, observation: Observation) -> dict[str, str]:
        if not observation.views:
            return {}
        spatial_alignment = observation.metadata.get("spatial_alignment")
        if not isinstance(spatial_alignment, dict):
            return {}
        direction = spatial_alignment.get("view_0_allocentric_direction")
        start_heading = self._allocentric_heading(direction)
        if start_heading is None:
            return {}
        headings_by_view: dict[str, str] = {}
        for view in observation.views:
            if not isinstance(view.label, str) or not view.label:
                continue
            offset = normalize_heading(float(view.heading) - float(observation.views[0].heading))
            allocentric_heading = normalize_heading(start_heading + offset)
            inferred_direction = self._heading_to_allocentric_direction(allocentric_heading)
            if inferred_direction is not None:
                headings_by_view[view.label] = inferred_direction
        return headings_by_view

    def _target_relative_heading_from_sector_alignment(
        self,
        observation: Observation | None,
        *,
        subgoal_room_id: str | None,
        desired_heading: float | None,
        inferred_current_heading: float | None,
    ) -> tuple[float | None, JsonDict]:
        if observation is None or inferred_current_heading is None:
            return None, {}
        spatial_alignment = observation.metadata.get("spatial_alignment")
        if not isinstance(spatial_alignment, dict):
            return None, {}
        raw_sector_alignment = spatial_alignment.get("sector_alignment")
        if not isinstance(raw_sector_alignment, list):
            return None, {}

        desired_direction = self._heading_to_allocentric_direction(desired_heading) if desired_heading is not None else None
        ranked_matches: list[tuple[tuple[int, int, float, str], float, JsonDict]] = []
        for record in raw_sector_alignment:
            if not isinstance(record, dict):
                continue
            view_id = record.get("view_id")
            if not isinstance(view_id, str) or not view_id.startswith("view_"):
                continue
            try:
                view_index = int(view_id.split("_", 1)[1])
            except (IndexError, ValueError):
                continue
            if view_index < 0 or view_index >= len(observation.views):
                continue
            view = observation.views[view_index]
            relative_heading = normalize_heading(float(view.heading) - inferred_current_heading)
            allocentric_direction = record.get("allocentric_direction")
            matched_room_id = record.get("matched_room_id")
            room_match = isinstance(subgoal_room_id, str) and matched_room_id == subgoal_room_id
            direction_match = isinstance(desired_direction, str) and allocentric_direction == desired_direction
            desired_diff = (
                self._angular_distance(desired_heading, self._allocentric_heading(allocentric_direction))
                if desired_heading is not None and self._allocentric_heading(allocentric_direction) is not None
                else 360.0
            )
            ranked_matches.append(
                (
                    (
                        0 if room_match else 1,
                        0 if direction_match else 1,
                        desired_diff,
                        view_id,
                    ),
                    relative_heading,
                    {
                        "target_heading_source": "sector_alignment",
                        "target_view_id": view_id,
                        "target_allocentric_direction": allocentric_direction,
                        "target_matched_room_id": matched_room_id,
                    },
                )
            )

        if not ranked_matches:
            return None, {}
        ranked_matches.sort(key=lambda item: item[0])
        _, relative_heading, metadata = ranked_matches[0]
        return relative_heading, metadata

    @staticmethod
    def _allocentric_heading(direction: object) -> float | None:
        if direction == "north":
            return 0.0
        if direction == "east":
            return 90.0
        if direction == "south":
            return 180.0
        if direction == "west":
            return 270.0
        return None

    @staticmethod
    def _heading_to_allocentric_direction(heading: float) -> str | None:
        angle = normalize_heading(heading)
        if angle < 45 or angle >= 315:
            return "north"
        if angle < 135:
            return "east"
        if angle < 225:
            return "south"
        return "west"

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
