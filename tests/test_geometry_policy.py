from __future__ import annotations

import unittest

from sa_nav.common.types import BeliefState, CandidateAction, ReasoningInput, TaskSpec
from sa_nav.decision.action_policy import EdgeVisitActionPolicy, GreedyActionPolicy
from sa_nav.execution.episode_runner import EpisodeRunner
from sa_nav.pipeline.factory import NavigationPipelineConfig, build_navigation_pipeline


class SimplifiedGeometryPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = GreedyActionPolicy()
        self.task = TaskSpec(
            task_type="test",
            raw_instruction="Navigate to Room B",
            source_room_id="Room A",
            goal_room_ids=["Room B"],
        )

    @staticmethod
    def _candidate(
        pano_id: str,
        *,
        target_diff: float,
        turn_error: float,
        candidate_index: int,
        target_room_id: str,
        score: float,
        route_step_index: int | None,
    ) -> CandidateAction:
        return CandidateAction(
            target_pano_id=pano_id,
            target_room_id=target_room_id,
            absolute_heading=0.0,
            relative_heading=0.0,
            relative_label="front",
            route_step_index=route_step_index,
            score=score,
            reason="test candidate",
            metadata={
                "target_relative_diff_deg": target_diff,
                "turn_error_deg": turn_error,
                "candidate_index": candidate_index,
            },
        )

    def _choose(self, candidates: list[CandidateAction]) -> CandidateAction:
        output = self.policy.choose_next_action(
            ReasoningInput(
                task=self.task,
                route=["Room A", "Room B"],
                candidates=candidates,
                current_room_id="Room A",
                subgoal_room_id="Room B",
            )
        )
        self.assertIsNotNone(output.action)
        return output.action

    def test_target_direction_error_has_first_priority(self) -> None:
        oracle_favored = self._candidate(
            "pano-a",
            target_diff=20.0,
            turn_error=0.0,
            candidate_index=0,
            target_room_id="Room B",
            score=999.0,
            route_step_index=1,
        )
        direction_favored = self._candidate(
            "pano-z",
            target_diff=10.0,
            turn_error=100.0,
            candidate_index=1,
            target_room_id="Room C",
            score=-999.0,
            route_step_index=None,
        )

        self.assertIs(self._choose([oracle_favored, direction_favored]), direction_favored)

    def test_turn_error_breaks_equal_direction_ties(self) -> None:
        larger_turn = self._candidate(
            "pano-a",
            target_diff=10.0,
            turn_error=40.0,
            candidate_index=0,
            target_room_id="Room B",
            score=999.0,
            route_step_index=1,
        )
        smaller_turn = self._candidate(
            "pano-z",
            target_diff=10.0,
            turn_error=20.0,
            candidate_index=1,
            target_room_id="Room C",
            score=-999.0,
            route_step_index=None,
        )

        self.assertIs(self._choose([larger_turn, smaller_turn]), smaller_turn)

    def test_candidate_index_is_the_final_stable_tie_break(self) -> None:
        lexical_and_score_favored = self._candidate(
            "pano-a",
            target_diff=10.0,
            turn_error=20.0,
            candidate_index=1,
            target_room_id="Room B",
            score=999.0,
            route_step_index=1,
        )
        index_favored = self._candidate(
            "pano-z",
            target_diff=10.0,
            turn_error=20.0,
            candidate_index=0,
            target_room_id="Room C",
            score=-999.0,
            route_step_index=None,
        )

        self.assertIs(self._choose([lexical_and_score_favored, index_favored]), index_favored)


class EdgeVisitActionPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = EdgeVisitActionPolicy()
        self.task = TaskSpec(
            task_type="test",
            raw_instruction="Navigate to Room B",
            source_room_id="Room A",
            goal_room_ids=["Room B"],
        )

    @staticmethod
    def _candidate(
        pano_id: str,
        *,
        target_diff: float,
        turn_error: float,
        candidate_index: int,
        history_return_error: float | None = None,
    ) -> CandidateAction:
        return CandidateAction(
            target_pano_id=pano_id,
            target_room_id=None,
            absolute_heading=0.0,
            relative_heading=0.0,
            relative_label="front",
            route_step_index=None,
            score=0.0,
            reason="test candidate",
            metadata={
                "target_relative_diff_deg": target_diff,
                "turn_error_deg": turn_error,
                "candidate_index": candidate_index,
                "history_return_error_deg": history_return_error,
            },
        )

    def _choose(
        self,
        candidates: list[CandidateAction],
        *,
        current_pano_id: str = "pano-current",
        subgoal_room_id: str = "Room B",
    ):
        return self.policy.choose_next_action(
            ReasoningInput(
                task=self.task,
                route=["Room A", "Room B"],
                candidates=candidates,
                current_pano_id=current_pano_id,
                current_room_id="Room A",
                subgoal_room_id=subgoal_room_id,
            )
        )

    def test_first_visit_uses_unchanged_geometry_ranking(self) -> None:
        geometry_best = self._candidate(
            "pano-a", target_diff=5.0, turn_error=10.0, candidate_index=0
        )
        other = self._candidate(
            "pano-b", target_diff=50.0, turn_error=10.0, candidate_index=1
        )

        output = self._choose([geometry_best, other])

        self.assertIs(output.action, geometry_best)
        self.assertFalse(geometry_best.metadata["edge_cycle_avoidance_active"])
        self.assertEqual(geometry_best.metadata["pano_subgoal_visit_count"], 1)

    def test_revisit_prefers_less_used_edge_before_geometry(self) -> None:
        geometry_best = self._candidate(
            "pano-a", target_diff=5.0, turn_error=10.0, candidate_index=0
        )
        unused = self._candidate(
            "pano-b", target_diff=50.0, turn_error=10.0, candidate_index=1
        )
        first = self._choose([geometry_best, unused])
        self.policy.record_transition("pano-current", first.action.target_pano_id)

        second = self._choose([geometry_best, unused])

        self.assertIs(second.action, unused)
        self.assertTrue(unused.metadata["edge_cycle_avoidance_active"])
        self.assertEqual(geometry_best.metadata["edge_visit_count"], 1)
        self.assertEqual(unused.metadata["edge_visit_count"], 0)
        self.assertIn("edge_visit_count=0", second.rationale)

    def test_reverse_transition_increments_the_same_undirected_edge(self) -> None:
        self.policy.record_transition("pano-a", "pano-b")
        self.policy.record_transition("pano-b", "pano-a")

        self.assertEqual(
            self.policy.edge_visit_counts[
                self.policy.edge_key("pano-a", "pano-b")
            ],
            2,
        )
        self.assertEqual(
            self.policy.edge_key("pano-a", "pano-b"),
            self.policy.edge_key("pano-b", "pano-a"),
        )

    def test_equal_edge_counts_fall_back_to_geometry_ranking(self) -> None:
        geometry_best = self._candidate(
            "pano-a", target_diff=5.0, turn_error=10.0, candidate_index=0
        )
        other = self._candidate(
            "pano-b", target_diff=50.0, turn_error=10.0, candidate_index=1
        )
        self._choose([geometry_best, other])
        self.policy.record_transition("pano-current", "pano-a")
        self.policy.record_transition("pano-current", "pano-b")

        output = self._choose([geometry_best, other])

        self.assertIs(output.action, geometry_best)
        self.assertEqual(geometry_best.metadata["edge_visit_count"], 1)
        self.assertEqual(other.metadata["edge_visit_count"], 1)

    def test_same_pano_with_new_subgoal_starts_with_geometry(self) -> None:
        geometry_best = self._candidate(
            "pano-a", target_diff=5.0, turn_error=10.0, candidate_index=0
        )
        unused = self._candidate(
            "pano-b", target_diff=50.0, turn_error=10.0, candidate_index=1
        )
        first = self._choose([geometry_best, unused], subgoal_room_id="Room B")
        self.policy.record_transition("pano-current", first.action.target_pano_id)

        new_subgoal = self._choose(
            [geometry_best, unused], subgoal_room_id="Room C"
        )

        self.assertIs(new_subgoal.action, geometry_best)
        self.assertFalse(geometry_best.metadata["edge_cycle_avoidance_active"])
        self.assertEqual(geometry_best.metadata["pano_subgoal_visit_count"], 1)

    def test_reset_episode_clears_both_count_tables(self) -> None:
        candidate = self._candidate(
            "pano-a", target_diff=5.0, turn_error=10.0, candidate_index=0
        )
        self._choose([candidate])
        self.policy.record_transition("pano-current", "pano-a")

        self.policy.reset_episode()

        self.assertEqual(self.policy.edge_visit_counts, {})
        self.assertEqual(self.policy.pano_subgoal_visit_counts, {})

    def test_revisit_preserves_u_turn_and_history_return_filters(self) -> None:
        safe = self._candidate(
            "pano-safe",
            target_diff=100.0,
            turn_error=10.0,
            candidate_index=0,
            history_return_error=90.0,
        )
        u_turn = self._candidate(
            "pano-u-turn",
            target_diff=0.0,
            turn_error=170.0,
            candidate_index=1,
            history_return_error=90.0,
        )
        history_return = self._candidate(
            "pano-history",
            target_diff=0.0,
            turn_error=10.0,
            candidate_index=2,
            history_return_error=0.0,
        )
        first = self._choose([safe, u_turn, history_return])
        self.policy.record_transition("pano-current", first.action.target_pano_id)

        second = self._choose([safe, u_turn, history_return])

        self.assertIs(second.action, safe)
        self.assertIn("avoided_u_turns=[pano-u-turn]", second.rationale)
        self.assertIn("avoided_history_returns=[pano-history]", second.rationale)

    def test_revisit_allows_u_turn_at_dead_end(self) -> None:
        dead_end = self._candidate(
            "pano-only", target_diff=0.0, turn_error=180.0, candidate_index=0
        )
        first = self._choose([dead_end])
        self.policy.record_transition("pano-current", first.action.target_pano_id)

        second = self._choose([dead_end])

        self.assertIs(second.action, dead_end)

    def test_episode_runner_resets_policy_before_every_episode(self) -> None:
        class InitializeOnlySpatialEngine:
            @staticmethod
            def initialize(*, start_pano_id, start_room_id, start_heading):
                return BeliefState(
                    current_pano_id=start_pano_id,
                    current_room_id=start_room_id,
                    current_heading=start_heading,
                )

        self.policy.record_transition("pano-a", "pano-b")
        self.policy.pano_subgoal_visit_counts[("pano-a", "Room B")] = 2
        runner = EpisodeRunner(
            perception_provider=None,
            spatial_engine=InitializeOnlySpatialEngine(),
            policy=self.policy,
        )

        runner.run(
            task=self.task,
            start_pano_id="pano-a",
            start_room_id="Room A",
            step_budget=0,
        )

        self.assertEqual(self.policy.edge_visit_counts, {})
        self.assertEqual(self.policy.pano_subgoal_visit_counts, {})

    def test_factory_keeps_geometry_default_and_adds_opt_in_policy(self) -> None:
        baseline = build_navigation_pipeline(room_graph={}, pano_graph={})
        edge_visit = build_navigation_pipeline(
            room_graph={},
            pano_graph={},
            config=NavigationPipelineConfig(
                direction_policy="geometry-edge-visit"
            ),
        )

        self.assertIs(type(baseline.episode_runner.policy), GreedyActionPolicy)
        self.assertIsInstance(edge_visit.episode_runner.policy, EdgeVisitActionPolicy)


if __name__ == "__main__":
    unittest.main()
