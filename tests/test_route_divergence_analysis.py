from __future__ import annotations

import unittest

from tools.evaluation.navigation.analyze_route_divergence import (
    actual_room_visits,
    build_step_diagnostics,
    classify_divergence,
    decision_window_indices,
    find_first_divergence,
    normalize_grounding,
    select_decision_row,
    shortest_room_next_hops,
)


def decision_row(
    *,
    movement_index: int,
    ground_truth: str | None,
    estimated: str | None,
    subgoal: str | None,
) -> dict:
    return {
        "movement_index": movement_index,
        "step_index": movement_index,
        "ground_truth_room_id": ground_truth,
        "estimated_room_id": estimated,
        "stored_subgoal_room_id": subgoal,
    }


class RouteDivergenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.room_graph = {
            "A": {
                "neighbors": [
                    {"target_room_id": "B", "allocentric_heading_deg": 90.0},
                    {"target_room_id": "C", "allocentric_heading_deg": 270.0},
                    {"target_room_id": "D", "allocentric_heading_deg": 180.0},
                ]
            },
            "B": {
                "neighbors": [
                    {"target_room_id": "G", "allocentric_heading_deg": 90.0}
                ]
            },
            "C": {
                "neighbors": [
                    {"target_room_id": "G", "allocentric_heading_deg": 270.0}
                ]
            },
            "D": {
                "neighbors": [
                    {"target_room_id": "A", "allocentric_heading_deg": 360.0}
                ]
            },
            "G": {"neighbors": []},
        }

    def test_all_equal_shortest_next_rooms_are_valid(self) -> None:
        self.assertEqual(
            shortest_room_next_hops(self.room_graph, "A", ["G"]), ["B", "C"]
        )
        divergence = find_first_divergence(
            visits=[
                {"room_id": "A"},
                {"room_id": "C"},
                {"room_id": "G"},
            ],
            room_graph=self.room_graph,
            goal_room_ids=["G"],
        )
        self.assertEqual(divergence["divergence_kind"], "goal_reached_not_recognized")
        self.assertEqual(divergence["divergence_room_id"], "G")

    def test_first_invalid_room_transition_is_found_before_terminal_loop(self) -> None:
        divergence = find_first_divergence(
            visits=[
                {"room_id": "A"},
                {"room_id": "D"},
                {"room_id": "A"},
                {"room_id": "D"},
            ],
            room_graph=self.room_graph,
            goal_room_ids=["G"],
        )
        self.assertEqual(divergence["divergence_kind"], "wrong_room_transition")
        self.assertEqual(divergence["divergence_room_id"], "A")
        self.assertEqual(divergence["valid_expected_next_room_ids"], ["B", "C"])
        self.assertEqual(divergence["actual_next_room_id"], "D")

    def test_stall_uses_final_known_room_as_divergence(self) -> None:
        divergence = find_first_divergence(
            visits=[{"room_id": "A"}, {"room_id": "B"}],
            room_graph=self.room_graph,
            goal_room_ids=["G"],
        )
        self.assertEqual(
            divergence["divergence_kind"], "stalled_before_expected_transition"
        )
        self.assertEqual(divergence["divergence_room_id"], "B")
        self.assertEqual(divergence["valid_expected_next_room_ids"], ["G"])
        self.assertIsNone(divergence["actual_next_room_id"])

    def test_classification_uses_last_decision_state_not_transient_earlier_error(self) -> None:
        divergence = {
            "divergence_kind": "wrong_room_transition",
            "divergence_room_id": "A",
            "valid_expected_next_room_ids": ["B", "C"],
            "actual_next_room_id": "D",
        }
        transient_wrong = decision_row(
            movement_index=0, ground_truth="A", estimated="D", subgoal="A"
        )
        recovered = decision_row(
            movement_index=1, ground_truth="A", estimated="A", subgoal="B"
        )
        cause, evidence = classify_divergence(
            divergence=divergence,
            decision_row=recovered,
            result={"final_estimated_room_id": "D"},
            goal_room_ids=["G"],
        )
        self.assertNotEqual(transient_wrong["estimated_room_id"], "A")
        self.assertEqual(cause, "direction/exit_selection")
        self.assertIn("subgoal_were_correct", evidence)

    def test_wrong_effective_room_at_decision_is_localization(self) -> None:
        cause, _ = classify_divergence(
            divergence={
                "divergence_kind": "wrong_room_transition",
                "divergence_room_id": "A",
                "valid_expected_next_room_ids": ["B", "C"],
                "actual_next_room_id": "D",
            },
            decision_row=decision_row(
                movement_index=2, ground_truth="A", estimated="D", subgoal="A"
            ),
            result={},
            goal_room_ids=["G"],
        )
        self.assertEqual(cause, "localization")

    def test_persistent_wrong_state_is_localization_even_if_last_step_recovers(self) -> None:
        window = [
            decision_row(movement_index=0, ground_truth="A", estimated="D", subgoal="A"),
            decision_row(movement_index=1, ground_truth="A", estimated="D", subgoal="A"),
            decision_row(movement_index=2, ground_truth="A", estimated="D", subgoal="A"),
            decision_row(movement_index=3, ground_truth="A", estimated="A", subgoal="B"),
        ]
        cause, evidence = classify_divergence(
            divergence={
                "divergence_kind": "wrong_room_transition",
                "divergence_room_id": "A",
                "valid_expected_next_room_ids": ["B", "C"],
                "actual_next_room_id": "D",
            },
            decision_row=window[-1],
            decision_window_rows=window,
            result={},
            goal_room_ids=["G"],
        )
        self.assertEqual(cause, "localization")
        self.assertIn("dominated", evidence)

    def test_correct_room_but_wrong_subgoal_is_planning(self) -> None:
        cause, _ = classify_divergence(
            divergence={
                "divergence_kind": "wrong_room_transition",
                "divergence_room_id": "A",
                "valid_expected_next_room_ids": ["B", "C"],
                "actual_next_room_id": "D",
            },
            decision_row=decision_row(
                movement_index=2, ground_truth="A", estimated="A", subgoal="D"
            ),
            result={},
            goal_room_ids=["G"],
        )
        self.assertEqual(cause, "planning/replanning")

    def test_goal_reached_without_recognition_is_localization(self) -> None:
        cause, _ = classify_divergence(
            divergence={
                "divergence_kind": "goal_reached_not_recognized",
                "divergence_room_id": "G",
                "valid_expected_next_room_ids": [],
                "actual_next_room_id": None,
            },
            decision_row=decision_row(
                movement_index=2, ground_truth="G", estimated="B", subgoal="G"
            ),
            result={"final_estimated_room_id": "B"},
            goal_room_ids=["G"],
        )
        self.assertEqual(cause, "localization")


class TransitAndStepEvidenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.grounding = {"pA": "A", "pB": "B", "pG": "G"}
        self.room_graph = {
            "A": {
                "neighbors": [
                    {"target_room_id": "B", "allocentric_heading_deg": 90.0}
                ]
            },
            "B": {
                "neighbors": [
                    {"target_room_id": "G", "allocentric_heading_deg": 90.0}
                ]
            },
            "G": {"neighbors": []},
        }
        self.pano_graph = {
            "pA": {
                "neighbors": [
                    {"target_pano_id": "pTransit", "geocentric_heading_deg": 60.0}
                ]
            },
            "pTransit": {
                "neighbors": [
                    {"target_pano_id": "pB", "geocentric_heading_deg": 60.0}
                ]
            },
            "pB": {
                "neighbors": [
                    {"target_pano_id": "pG", "geocentric_heading_deg": 60.0}
                ]
            },
            "pG": {"neighbors": []},
        }

    def test_transit_is_ignored_in_room_sequence_but_retained_in_decision_window(self) -> None:
        visits = actual_room_visits(
            ["pA", "pTransit", "pB", "pG"], self.grounding
        )
        self.assertEqual([visit["room_id"] for visit in visits], ["A", "B", "G"])
        divergence = {
            "visit_index": 0,
            "divergence_room_id": "A",
        }
        indices = decision_window_indices(
            visits=visits, divergence=divergence, movement_count=3
        )
        self.assertEqual(indices, [0, 1])
        rows = [
            decision_row(movement_index=0, ground_truth="A", estimated="A", subgoal="B"),
            decision_row(
                movement_index=1, ground_truth=None, estimated="A", subgoal="B"
            ),
            decision_row(movement_index=2, ground_truth="B", estimated="B", subgoal="G"),
        ]
        selected = select_decision_row(
            step_rows=rows, window_indices=indices, divergence_room_id="A"
        )
        self.assertEqual(selected["movement_index"], 0)

    def test_unknown_transit_step_is_not_a_localization_error(self) -> None:
        result = {
            "test_id": "T1",
            "success": False,
            "parsed_goal_room_ids": ["G"],
            "pano_path": ["pA", "pTransit", "pB", "pG"],
            "movements": [
                {
                    "step_index": 0,
                    "from_pano_id": "pA",
                    "to_pano_id": "pTransit",
                    "estimated_room_id": "A",
                    "subgoal_room_id": "B",
                },
                {
                    "step_index": 1,
                    "from_pano_id": "pTransit",
                    "to_pano_id": "pB",
                    "estimated_room_id": "A",
                    "subgoal_room_id": "B",
                },
                {
                    "step_index": 2,
                    "from_pano_id": "pB",
                    "to_pano_id": "pG",
                    "estimated_room_id": "B",
                    "subgoal_room_id": "G",
                },
            ],
            "localization_events": [
                {
                    "step_index": 1,
                    "stabilized_room_id": "A",
                    "ground_truth_room_id": None,
                }
            ],
        }
        rows = build_step_diagnostics(
            condition="without_trajectory",
            case={"test_id": "T1", "target_room_id": "G"},
            result=result,
            room_graph=self.room_graph,
            pano_graph=self.pano_graph,
            grounding=self.grounding,
        )
        self.assertIsNone(rows[1]["ground_truth_room_id"])
        self.assertIsNone(rows[1]["effective_state_correct"])
        self.assertIsNone(rows[1]["localization_call_correct"])
        self.assertTrue(rows[1]["localization_performed"])

    def test_string_null_grounding_is_removed(self) -> None:
        self.assertEqual(
            normalize_grounding(
                {"pA": "A", "pTransit": "null", "pUnknown": "UNKNOWN"}
            ),
            {"pA": "A"},
        )


if __name__ == "__main__":
    unittest.main()
