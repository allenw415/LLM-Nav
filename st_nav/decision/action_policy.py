from __future__ import annotations

import json
from typing import Callable

from ..common.env import resolve_model_environment
from ..common.model_client import DEFAULT_OPENAI_API_BASE, ModelResponseClient, parse_json_output, resolve_api_kind
from ..common.types import CandidateAction, PolicyOutput, ReasoningInput


class GreedyActionPolicy:
    """
    Heuristic fallback policy.
    """

    def __init__(
        self,
        *,
        u_turn_threshold_deg: float = 150.0,
        history_return_threshold_deg: float = 30.0,
    ) -> None:
        if not 0.0 < u_turn_threshold_deg <= 180.0:
            raise ValueError("u_turn_threshold_deg must be in the interval (0, 180].")
        if not 0.0 <= history_return_threshold_deg <= 180.0:
            raise ValueError("history_return_threshold_deg must be in the interval [0, 180].")
        self.u_turn_threshold_deg = float(u_turn_threshold_deg)
        self.history_return_threshold_deg = float(history_return_threshold_deg)

    @staticmethod
    def _target_direction_error(candidate: CandidateAction) -> float:
        target_diff = candidate.metadata.get("target_relative_diff_deg")
        if isinstance(target_diff, (int, float)):
            return float(target_diff)
        return 10**6

    @staticmethod
    def _turn_error(candidate: CandidateAction) -> float:
        turn_error = candidate.metadata.get("turn_error_deg")
        if isinstance(turn_error, (int, float)):
            return float(turn_error)
        return 10**6

    @staticmethod
    def _candidate_index(candidate: CandidateAction) -> int:
        candidate_index = candidate.metadata.get("candidate_index")
        if isinstance(candidate_index, int) and candidate_index >= 0:
            return candidate_index
        return 10**6

    def _is_u_turn(self, candidate: CandidateAction) -> bool:
        turn_error = candidate.metadata.get("turn_error_deg")
        return isinstance(turn_error, (int, float)) and float(turn_error) >= self.u_turn_threshold_deg

    def _avoid_u_turns_when_possible(
        self,
        candidates: list[CandidateAction],
    ) -> tuple[list[CandidateAction], list[CandidateAction]]:
        non_u_turn_candidates = [candidate for candidate in candidates if not self._is_u_turn(candidate)]
        if not non_u_turn_candidates:
            return candidates, []
        excluded_u_turn_candidates = [candidate for candidate in candidates if self._is_u_turn(candidate)]
        return non_u_turn_candidates, excluded_u_turn_candidates

    def _is_history_return(self, candidate: CandidateAction) -> bool:
        return_error = candidate.metadata.get("history_return_error_deg")
        return isinstance(return_error, (int, float)) and float(return_error) <= self.history_return_threshold_deg

    def _avoid_history_returns_when_possible(
        self,
        candidates: list[CandidateAction],
    ) -> tuple[list[CandidateAction], list[CandidateAction]]:
        non_return_candidates = [candidate for candidate in candidates if not self._is_history_return(candidate)]
        if not non_return_candidates:
            return candidates, []
        excluded_return_candidates = [candidate for candidate in candidates if self._is_history_return(candidate)]
        return non_return_candidates, excluded_return_candidates

    def choose_next_action(self, reasoning_input: ReasoningInput) -> PolicyOutput:
        if not reasoning_input.candidates:
            return PolicyOutput(action=None, rationale="No candidate actions available.")

        candidate_pool, excluded_u_turn_candidates = self._avoid_u_turns_when_possible(
            list(reasoning_input.candidates)
        )
        candidate_pool, excluded_history_return_candidates = self._avoid_history_returns_when_possible(
            candidate_pool
        )
        ranked_candidates = sorted(
            candidate_pool,
            key=lambda item: (
                self._target_direction_error(item),
                self._turn_error(item),
                self._candidate_index(item),
            ),
        )
        best_action = ranked_candidates[0]
        rationale_parts = [
            f"Selected candidate_index={self._candidate_index(best_action)}"
        ]
        if excluded_u_turn_candidates:
            excluded_ids = ", ".join(candidate.target_pano_id for candidate in excluded_u_turn_candidates)
            rationale_parts.append(
                f"avoided_u_turns=[{excluded_ids}] at threshold={self.u_turn_threshold_deg:.1f}"
            )
        if excluded_history_return_candidates:
            excluded_ids = ", ".join(candidate.target_pano_id for candidate in excluded_history_return_candidates)
            rationale_parts.append(
                f"avoided_history_returns=[{excluded_ids}] at threshold={self.history_return_threshold_deg:.1f}"
            )
        target_relative_diff = best_action.metadata.get("target_relative_diff_deg")
        if isinstance(target_relative_diff, (int, float)):
            rationale_parts.append(f"target_relative_diff={float(target_relative_diff):.1f}")
        turn_error = best_action.metadata.get("turn_error_deg")
        if isinstance(turn_error, (int, float)):
            rationale_parts.append(f"turn_error={float(turn_error):.1f}")
        rationale = ", ".join(rationale_parts)
        if best_action.reason:
            rationale = f"{rationale} ({best_action.reason})"
        return PolicyOutput(action=best_action, rationale=rationale)

    def choose_action(self, *, task, route, candidates) -> PolicyOutput:
        return self.choose_next_action(
            ReasoningInput(
                task=task,
                route=route,
                candidates=list(candidates),
            )
        )


class EdgeVisitActionPolicy(GreedyActionPolicy):
    """Geometry policy that favors less-used graph edges on repeated contexts."""

    def __init__(
        self,
        *,
        u_turn_threshold_deg: float = 150.0,
        history_return_threshold_deg: float = 30.0,
    ) -> None:
        super().__init__(
            u_turn_threshold_deg=u_turn_threshold_deg,
            history_return_threshold_deg=history_return_threshold_deg,
        )
        self.edge_visit_counts: dict[tuple[str, str], int] = {}
        self.pano_subgoal_visit_counts: dict[tuple[str, str | None], int] = {}

    @staticmethod
    def edge_key(pano_a: str, pano_b: str) -> tuple[str, str]:
        """Return one stable key for both directions of an undirected edge."""
        return tuple(sorted((pano_a, pano_b)))

    def reset_episode(self) -> None:
        self.edge_visit_counts.clear()
        self.pano_subgoal_visit_counts.clear()

    def record_transition(self, source_pano_id: str, target_pano_id: str) -> None:
        key = self.edge_key(source_pano_id, target_pano_id)
        self.edge_visit_counts[key] = self.edge_visit_counts.get(key, 0) + 1

    @staticmethod
    def _edge_visit_count(candidate: CandidateAction) -> int:
        count = candidate.metadata.get("edge_visit_count")
        if isinstance(count, int) and count >= 0:
            return count
        return 0

    def _annotate_candidates(
        self,
        *,
        current_pano_id: str,
        candidates: list[CandidateAction],
        context_visit_count: int,
        cycle_avoidance_active: bool,
    ) -> None:
        for candidate in candidates:
            key = self.edge_key(current_pano_id, candidate.target_pano_id)
            candidate.metadata["edge_key"] = list(key)
            candidate.metadata["edge_visit_count"] = self.edge_visit_counts.get(key, 0)
            candidate.metadata["pano_subgoal_visit_count"] = context_visit_count
            candidate.metadata["edge_cycle_avoidance_active"] = cycle_avoidance_active

    def choose_next_action(self, reasoning_input: ReasoningInput) -> PolicyOutput:
        if not reasoning_input.candidates:
            return PolicyOutput(action=None, rationale="No candidate actions available.")

        current_pano_id = reasoning_input.current_pano_id
        if not isinstance(current_pano_id, str) or not current_pano_id:
            return super().choose_next_action(reasoning_input)

        context_key = (current_pano_id, reasoning_input.subgoal_room_id)
        context_visit_count = self.pano_subgoal_visit_counts.get(context_key, 0) + 1
        self.pano_subgoal_visit_counts[context_key] = context_visit_count
        cycle_avoidance_active = context_visit_count > 1
        candidates = list(reasoning_input.candidates)
        self._annotate_candidates(
            current_pano_id=current_pano_id,
            candidates=candidates,
            context_visit_count=context_visit_count,
            cycle_avoidance_active=cycle_avoidance_active,
        )

        if not cycle_avoidance_active:
            output = super().choose_next_action(reasoning_input)
            return PolicyOutput(
                action=output.action,
                rationale=(
                    f"{output.rationale}, edge_cycle_avoidance_active=false, "
                    f"pano_subgoal_visit_count={context_visit_count}"
                ),
            )

        candidate_pool, excluded_u_turn_candidates = self._avoid_u_turns_when_possible(
            candidates
        )
        candidate_pool, excluded_history_return_candidates = (
            self._avoid_history_returns_when_possible(candidate_pool)
        )
        ranked_candidates = sorted(
            candidate_pool,
            key=lambda item: (
                self._edge_visit_count(item),
                self._target_direction_error(item),
                self._turn_error(item),
                self._candidate_index(item),
            ),
        )
        best_action = ranked_candidates[0]
        rationale_parts = [
            f"Selected candidate_index={self._candidate_index(best_action)}",
            "edge_cycle_avoidance_active=true",
            f"pano_subgoal_visit_count={context_visit_count}",
            f"edge_visit_count={self._edge_visit_count(best_action)}",
        ]
        if excluded_u_turn_candidates:
            excluded_ids = ", ".join(
                candidate.target_pano_id
                for candidate in excluded_u_turn_candidates
            )
            rationale_parts.append(
                f"avoided_u_turns=[{excluded_ids}] "
                f"at threshold={self.u_turn_threshold_deg:.1f}"
            )
        if excluded_history_return_candidates:
            excluded_ids = ", ".join(
                candidate.target_pano_id
                for candidate in excluded_history_return_candidates
            )
            rationale_parts.append(
                f"avoided_history_returns=[{excluded_ids}] "
                f"at threshold={self.history_return_threshold_deg:.1f}"
            )
        target_relative_diff = best_action.metadata.get("target_relative_diff_deg")
        if isinstance(target_relative_diff, (int, float)):
            rationale_parts.append(
                f"target_relative_diff={float(target_relative_diff):.1f}"
            )
        turn_error = best_action.metadata.get("turn_error_deg")
        if isinstance(turn_error, (int, float)):
            rationale_parts.append(f"turn_error={float(turn_error):.1f}")
        rationale = ", ".join(rationale_parts)
        if best_action.reason:
            rationale = f"{rationale} ({best_action.reason})"
        return PolicyOutput(action=best_action, rationale=rationale)


class LLMActionPolicy:
    """
    LLM-based navigation decision policy.

    The model receives room/subgoal context, egocentric-allocentric alignment,
    visible passages, and per-candidate spatial context, then chooses the best
    next pano without relying on angle minimization alone.
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
        api_kind: str | None = None,
        request_timeout: float | None = None,
        response_client: Callable[[dict], dict] | None = None,
        fallback_policy: GreedyActionPolicy | None = None,
    ) -> None:
        settings = resolve_model_environment(
            default_model="gpt-5-mini",
            default_api_base=DEFAULT_OPENAI_API_BASE,
            default_api_kind="responses",
        )
        self.model = model or settings.model_name or "gpt-5-mini"
        self.api_key = api_key or settings.api_key
        self.api_base = (api_base or settings.api_base or DEFAULT_OPENAI_API_BASE).rstrip("/")
        self.api_kind = resolve_api_kind(api_kind or settings.api_kind)
        self.request_timeout = float(request_timeout if request_timeout is not None else (settings.request_timeout or 30.0))
        self.response_client = response_client
        self.model_client = ModelResponseClient(
            provider=settings.provider,
            api_key=self.api_key,
            api_base=self.api_base,
            api_kind=self.api_kind,
            request_timeout=self.request_timeout,
            num_ctx=settings.num_ctx,
            temperature=settings.temperature,
            response_client=self.response_client,
        )
        self.fallback_policy = fallback_policy or GreedyActionPolicy()
        self.last_request_body: dict | None = None
        self.last_response_payload: dict | None = None

    def choose_next_action(self, reasoning_input: ReasoningInput) -> PolicyOutput:
        if not reasoning_input.candidates:
            return PolicyOutput(action=None, rationale="No candidate actions available.")
        if len(reasoning_input.candidates) == 1:
            action = reasoning_input.candidates[0]
            return PolicyOutput(
                action=action,
                rationale=f"Only one candidate action available: {action.target_pano_id}.",
            )
        if not self.model_client.is_configured():
            return self.fallback_policy.choose_next_action(reasoning_input)

        request_body = self._build_request_body(reasoning_input)
        self.last_request_body = self._clone_json(request_body)
        payload = self.model_client.create(request_body)
        self.last_response_payload = self._clone_json(payload)
        parsed = parse_json_output(payload)
        action = self._resolve_action(parsed, reasoning_input.candidates)
        if action is None:
            fallback = self.fallback_policy.choose_next_action(reasoning_input)
            return PolicyOutput(
                action=fallback.action,
                rationale=f"LLM policy returned an invalid candidate selection; fallback applied. {fallback.rationale}",
            )

        rationale = parsed.get("rationale")
        if not isinstance(rationale, str) or not rationale.strip():
            rationale = f"LLM selected {action.target_pano_id}."
        return PolicyOutput(action=action, rationale=rationale.strip())

    def choose_action(self, *, task, route, candidates) -> PolicyOutput:
        return self.choose_next_action(
            ReasoningInput(
                task=task,
                route=route,
                candidates=list(candidates),
            )
        )

    def _build_request_body(self, reasoning_input: ReasoningInput) -> dict:
        candidate_ids = [candidate.target_pano_id for candidate in reasoning_input.candidates]
        return {
            "model": self.model,
            "instructions": self._instructions(),
            "input": self._input_text(reasoning_input),
            "reasoning": {"effort": "low"},
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "navigation_decision",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "selected_target_pano_id": {"type": "string", "enum": candidate_ids},
                            "rationale": {"type": "string"},
                        },
                        "required": ["selected_target_pano_id", "rationale"],
                        "additionalProperties": False,
                    },
                }
            },
        }

    @staticmethod
    def _instructions() -> str:
        return " ".join(
            [
                "You are the navigation reasoning layer for museum indoor navigation.",
                "Choose exactly one next candidate action from the provided candidate list.",
                "Do not minimize angle difference alone.",
                "Use target_direction_error_deg as the geometric baseline for progress toward the subgoal.",
                "Use turn_error_deg and history_return_error_deg to recognize candidates that immediately reverse or return toward the recent path.",
                "Avoid a return candidate when another plausible candidate is available, but allow returning when it is the only exit.",
                "Use the allocentric target relation and each candidate's spatial context.",
                "Spatial context includes visible passages, landmarks, exhibit themes, and nearby semantic cues.",
                "A candidate with a worse angular difference can still be better if its semantic context is more consistent with the goal room or more likely to connect toward it.",
                "First judge whether each candidate is directionally plausible toward the target, rather than simply preferring the numerically closest angle.",
                "If multiple candidates are directionally plausible, use semantic evidence and passage context to choose among them.",
                "If no candidate is clearly directionally plausible, rely more on semantic evidence, visible passages, and local spatial context.",
                "Do not over-weight small angular differences when all candidates are poorly aligned with the target direction.",
                "The current_room_context may include subgoal_theme_labels; treat those labels as direct goal-theme supervision.",
                "A direct match between a candidate's local themes and subgoal_theme_labels is strong evidence in favor of that candidate.",
                "Evaluate all candidate actions jointly instead of privileging a prefiltered subset.",
                "Use the provided headings to judge directional plausibility yourself.",
                "Do not let any single heuristic field dominate if semantic evidence and passage connectivity suggest a better route.",
                "Reason about direction, semantics, and likely connectivity together.",
                "Return JSON only.",
            ]
        )

    def _input_text(self, reasoning_input: ReasoningInput) -> str:
        ordered_candidates = sorted(
            reasoning_input.candidates,
            key=lambda candidate: (float(candidate.absolute_heading), candidate.target_pano_id),
        )
        payload = {
            "instruction": reasoning_input.task.raw_instruction,
            "current_room_id": reasoning_input.current_room_id,
            "subgoal_room_id": reasoning_input.subgoal_room_id,
            "route": list(reasoning_input.route),
            "current_room_context": self._serialize_current_room_context(reasoning_input.current_room_context),
            "visible_passages": self._serialize_visible_passages(reasoning_input.visible_passages),
            "candidate_actions": [self._serialize_candidate(candidate) for candidate in ordered_candidates],
        }
        return "\n".join(
            [
                "Choose the best next action from these candidates.",
                "Use room-to-room allocentric relations plus candidate spatial context.",
                json.dumps(payload, ensure_ascii=False, indent=2),
            ]
        )

    def _alignment_payload(self, reasoning_input: ReasoningInput) -> dict:
        spatial_alignment = reasoning_input.spatial_alignment if isinstance(reasoning_input.spatial_alignment, dict) else {}
        return {
            "view_0_allocentric_direction": spatial_alignment.get("view_0_allocentric_direction"),
            "sector_alignment": spatial_alignment.get("sector_alignment"),
            "alignment_summary": spatial_alignment.get("alignment_summary"),
            "view_contexts": list(reasoning_input.view_contexts),
        }

    @staticmethod
    def _serialize_current_room_context(context: dict | None) -> dict:
        if not isinstance(context, dict):
            return {}
        return {
            "room_id": context.get("room_id"),
            "title": context.get("title"),
            "category": context.get("category"),
            "subgoal_room_id": context.get("subgoal_room_id"),
            "subgoal_title": context.get("subgoal_title"),
            "subgoal_theme_labels": list(context.get("subgoal_theme_labels", []))
            if isinstance(context.get("subgoal_theme_labels"), list)
            else [],
            "remaining_route": list(context.get("remaining_route", []))
            if isinstance(context.get("remaining_route"), list)
            else [],
        }

    @staticmethod
    def _serialize_visible_passages(passages: list[dict] | None) -> list[dict]:
        cleaned: list[dict] = []
        for passage in passages or []:
            if not isinstance(passage, dict):
                continue
            cleaned.append(
                {
                    "name": passage.get("name"),
                    "confidence": passage.get("confidence"),
                    "source_views": list(passage.get("source_views", []))
                    if isinstance(passage.get("source_views"), list)
                    else [],
                }
            )
        return cleaned

    @staticmethod
    def _serialize_candidate(candidate: CandidateAction) -> dict:
        metadata = dict(candidate.metadata or {})
        spatial_context = metadata.get("spatial_context")
        return {
            "target_pano_id": candidate.target_pano_id,
            "candidate_geocentric_heading_deg": candidate.absolute_heading,
            "route_step_index": candidate.route_step_index,
            "candidate_allocentric_heading_deg": metadata.get(
                "candidate_allocentric_heading_deg",
                metadata.get("allocentric_heading_deg"),
            ),
            "target_allocentric_heading_deg": metadata.get("desired_allocentric_heading_deg"),
            "target_heading_source": metadata.get("target_heading_source"),
            "target_allocentric_direction": metadata.get("target_allocentric_direction"),
            "incoming_geocentric_heading_deg": metadata.get("incoming_geocentric_heading_deg"),
            "turn_error_deg": metadata.get("turn_error_deg"),
            "history_return_heading_deg": metadata.get("history_return_heading_deg"),
            "history_return_error_deg": metadata.get("history_return_error_deg"),
            "target_direction_error_deg": metadata.get("target_relative_diff_deg"),
            "spatial_context": spatial_context if isinstance(spatial_context, dict) else {},
        }

    @staticmethod
    def _resolve_action(parsed: dict, candidates: list[CandidateAction]) -> CandidateAction | None:
        target_pano_id = parsed.get("selected_target_pano_id")
        if not isinstance(target_pano_id, str) or not target_pano_id:
            return None
        for candidate in candidates:
            if candidate.target_pano_id == target_pano_id:
                return candidate
        return None

    @staticmethod
    def _clone_json(payload: dict) -> dict:
        return json.loads(json.dumps(payload))
