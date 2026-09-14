from __future__ import annotations

import unittest

from tools.evaluation.navigation.analyze_room_exit_accuracy import (
    build_first_room_attempts,
    summarize_room_attempts,
)


class RoomLevelExitAccuracyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.room_graph = {
            "A": {"neighbors": [{"target_room_id": "B"}, {"target_room_id": "C"}]},
            "B": {"neighbors": [{"target_room_id": "G"}]},
            "C": {"neighbors": [{"target_room_id": "G"}]},
            "D": {"neighbors": [{"target_room_id": "A"}]},
            "G": {"neighbors": []},
        }
        self.grounding = {
            "pA": "A",
            "pB": "B",
            "pC": "C",
            "pD": "D",
            "pG": "G",
        }

    def build(self, pano_path: list[str]) -> list[dict]:
        return build_first_room_attempts(
            condition="without_trajectory",
            case={"test_id": "T1", "target_room_id": "G"},
            result={
                "test_id": "T1",
                "path_id": "P1",
                "parsed_goal_room_ids": ["G"],
                "pano_path": pano_path,
                "success": False,
            },
            room_graph=self.room_graph,
            grounding=self.grounding,
            paired_outcome="both_fail",
        )

    def test_equal_shortest_next_rooms_are_both_correct(self) -> None:
        attempts = self.build(["pA", "pC", "pG"])
        self.assertEqual(attempts[0]["valid_expected_next_room_ids"], ["B", "C"])
        self.assertTrue(attempts[0]["correct"])

    def test_wrong_room_can_recover_and_first_room_is_counted_once(self) -> None:
        attempts = self.build(["pA", "pD", "pA", "pB", "pG"])
        by_room = {row["current_room_id"]: row for row in attempts}
        self.assertEqual(sum(row["current_room_id"] == "A" for row in attempts), 1)
        self.assertTrue(by_room["A"]["wrong"])
        self.assertTrue(by_room["A"]["wrong_room"])
        self.assertTrue(by_room["A"]["recovered_after_wrong"])
        self.assertEqual(by_room["A"]["recovered_expected_room_id"], "B")

    def test_no_exit_is_wrong(self) -> None:
        attempts = self.build(["pA"])
        self.assertEqual(len(attempts), 1)
        self.assertTrue(attempts[0]["wrong"])
        self.assertTrue(attempts[0]["no_exit"])

    def test_visits_after_physical_goal_are_not_counted(self) -> None:
        attempts = self.build(["pA", "pB", "pG", "pD"])
        self.assertEqual([row["current_room_id"] for row in attempts], ["A", "B"])

    def test_summary_combines_wrong_room_and_no_exit(self) -> None:
        wrong_room = self.build(["pA", "pD"])[0]
        no_exit = self.build(["pA"])[0]
        correct = self.build(["pA", "pB", "pG"])[0]
        rows = summarize_room_attempts([wrong_room, no_exit, correct])
        room_a = next(row for row in rows if row["room_id"] == "A")
        self.assertEqual(room_a["attempt_case_count"], 3)
        self.assertEqual(room_a["correct_case_count"], 1)
        self.assertEqual(room_a["wrong_case_count"], 2)
        self.assertEqual(room_a["wrong_room_case_count"], 1)
        self.assertEqual(room_a["no_exit_case_count"], 1)
        self.assertAlmostEqual(room_a["wrong_rate"], 2 / 3)


if __name__ == "__main__":
    unittest.main()
