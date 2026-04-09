from __future__ import annotations

from ..common.types import PolicyOutput, ReasoningInput


class GreedyActionPolicy:
    """
    Minimal action policy that selects the highest-scoring spatial candidate.

    The route planner and spatial layer provide candidate actions plus route context.
    The decision layer is responsible for choosing the final next action.
    """

    def choose_next_action(self, reasoning_input: ReasoningInput) -> PolicyOutput:
        if not reasoning_input.candidates:
            return PolicyOutput(action=None, rationale="No candidate actions available.")

        ranked_candidates = sorted(
            reasoning_input.candidates,
            key=lambda item: (-item.score, item.target_pano_id),
        )
        best_action = ranked_candidates[0]
        rationale = f"Selected {best_action.target_pano_id} with score {best_action.score:.2f}"
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
