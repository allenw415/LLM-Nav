from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


def load_eval_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "tools/evaluation/localization/eval_map_grounded_spatial_alignment.py"
    )
    spec = importlib.util.spec_from_file_location("map_grounded_eval_test", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load evaluator: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_case(index: int, category: str) -> dict:
    room = f"Room {index + 1}"
    wrong = f"Wrong {index + 1}"
    bayesian = room if category in {"regression", "stable_correct"} else wrong
    spatial = room if category in {"correction", "stable_correct"} else wrong
    return {
        "pano_id": f"pano-{index:03d}",
        "expected_room_id": room,
        "prior_fused": {"top1": bayesian, "top5": [room, "A", "B", "C", "D"]},
        "spatial_aligned": {"top1": spatial},
    }


def gate_result(*, margin: float = 2.0, layout: float = 2.0, penalty: float = 1.0) -> dict:
    return {
        "predicted_room_id": "Room 8",
        "top1_vs_top2": {"margin": margin},
        "candidate_hypotheses": [
            {
                "room_id": "Room 8",
                "selected_view_0_allocentric_direction": "north",
                "rotation_tests": [
                    {
                        "view_0_allocentric_direction": "north",
                        "directional_layout": layout,
                        "contradiction_penalty": penalty,
                    }
                ],
                "directional_neighbor_checks": [
                    {
                        "status": "specific_match",
                        "evidence_specificity": "specific",
                        "allocentric_direction": "east",
                    },
                    {
                        "status": "specific_match",
                        "evidence_specificity": "exact",
                        "allocentric_direction": "west",
                    },
                ],
            }
        ],
        "map_alignment_trace": [],
    }


class MapGroundedAlignmentEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_eval_module()

    def test_parser_defaults_to_one_call_compact_output(self) -> None:
        args = self.module.build_parser().parse_args([])
        self.assertEqual(args.validation_retries, 0)
        self.assertEqual(args.llm_max_output_tokens, 1024)

    def test_change_balanced_smoke_selection_is_reproducible_and_has_approved_mix(self) -> None:
        cases = []
        index = 0
        for category, count in (("regression", 4), ("correction", 10), ("unresolved", 6), ("stable_correct", 8)):
            for _ in range(count):
                cases.append(source_case(index, category))
                index += 1
        first = self.module.select_source_cases(list(reversed(cases)), "change-balanced-smoke")
        second = self.module.select_source_cases(cases, "change-balanced-smoke")
        self.assertEqual([case["pano_id"] for case in first], [case["pano_id"] for case in second])
        self.assertEqual(len(first), 20)
        counts = {}
        for case in first:
            category = self.module.source_change_category(case)
            counts[category] = counts.get(category, 0) + 1
        self.assertEqual(counts, self.module.SMOKE_TARGETS)

    def test_bayesian_candidates_are_unique_and_sorted_without_scores_or_rank(self) -> None:
        case = source_case(0, "stable_correct")
        case["prior_fused"]["top5"] = ["Room 9", "Room 2", "Room 8", "Room 7", "Room 10"]
        self.assertEqual(
            self.module.bayesian_top5(case),
            ["Room 10", "Room 2", "Room 7", "Room 8", "Room 9"],
        )
        case["prior_fused"]["top5"][-1] = "Room 9"
        with self.assertRaisesRegex(ValueError, "five unique Bayesian candidates"):
            self.module.bayesian_top5(case)

    def test_gated_override_applies_at_every_boundary_and_requires_two_directions(self) -> None:
        accepted = self.module.gated_map_decision(gate_result(), "Room 7")
        self.assertTrue(accepted["override_applied"])
        self.assertEqual(accepted["final_room_id"], "Room 8")
        for kwargs, failed_check in (
            ({"margin": 1.99}, "margin_at_least_2"),
            ({"layout": 1.99}, "layout_at_least_2"),
            ({"penalty": 1.01}, "contradiction_at_most_1"),
        ):
            rejected = self.module.gated_map_decision(gate_result(**kwargs), "Room 7")
            self.assertFalse(rejected["override_applied"])
            self.assertIn(failed_check, rejected["failed_checks"])
            self.assertEqual(rejected["final_room_id"], "Room 7")
        one_direction = gate_result()
        one_direction["candidate_hypotheses"][0]["directional_neighbor_checks"].pop()
        rejected = self.module.gated_map_decision(one_direction, "Room 7")
        self.assertFalse(rejected["override_applied"])
        self.assertIn("two_independent_directions", rejected["failed_checks"])

    def test_resume_only_skips_completed_records_and_retries_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "results.jsonl"
            path.write_text(
                "\n".join(
                    json.dumps(record)
                    for record in (
                        {"pano_id": "done", "status": "completed"},
                        {"pano_id": "retry", "status": "error"},
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            records = self.module._load_resume_records(path)
        self.assertEqual(self.module.resumable_completed_panos(records), {"done"})

    def test_quota_error_detection_only_matches_http_429(self) -> None:
        self.assertTrue(
            self.module._is_quota_error(
                {"error": {"type": "RuntimeError", "message": "Model request failed with HTTP 429"}}
            )
        )
        self.assertFalse(
            self.module._is_quota_error(
                {"error": {"type": "RuntimeError", "message": "Model request failed with HTTP 400"}}
            )
        )

    def test_inventory_rejects_short_candidate_lists_and_rooms_missing_from_map(self) -> None:
        ready = source_case(0, "stable_correct")
        ready["manifest_path"] = __file__
        short = source_case(1, "stable_correct")
        short["manifest_path"] = __file__
        short["prior_fused"]["top5"] = ["Room 2", "Room 33"]
        regions = {"rooms": {room_id: {} for room_id in ["Room 1", "A", "B", "C", "D"]}}
        inventory = self.module.source_inventory([ready, short], regions)
        self.assertEqual(inventory["ready_count"], 1)
        self.assertEqual(inventory["invalid_candidate_count_cases"][0]["candidate_count"], 2)
        self.assertEqual(inventory["candidates_missing_from_map_cases"][0]["room_ids"], ["Room 2", "Room 33"])

    def test_map_eligible_catalog_excludes_non_five_candidate_cases(self) -> None:
        ready = source_case(0, "stable_correct")
        short = source_case(1, "stable_correct")
        short["prior_fused"]["top5"] = ["Room 2", "Room 33"]
        regions = {"rooms": {room_id: {} for room_id in ["Room 1", "A", "B", "C", "D"]}}
        eligible = self.module.map_eligible_source_cases([short, ready], regions)
        self.assertEqual([case["pano_id"] for case in eligible], [ready["pano_id"]])

    def test_smoke_acceptance_allows_one_decision_error_but_blocks_raw_stable_regression(self) -> None:
        selected = []
        index = 0
        for category, count in (("regression", 4), ("correction", 8), ("unresolved", 4), ("stable_correct", 4)):
            for _ in range(count):
                selected.append(source_case(index, category))
                index += 1
        records = []
        for case in selected:
            category = self.module.source_change_category(case)
            records.append(
                {
                    "pano_id": case["pano_id"],
                    "status": "completed",
                    "candidate_contains_expected_room": True,
                    "source_change_category": category,
                    "current_spatial_correct": category in {"correction", "stable_correct"},
                    "raw_map_correct": True,
                    "gated_map_correct": True,
                    "gate": {"override_applied": False, "support_directions": []},
                }
            )
        records[0]["status"] = "error"
        accepted = self.module.smoke_acceptance(selected, records)
        self.assertTrue(accepted["evaluated"])
        self.assertTrue(accepted["passed"])
        stable = next(record for record in records if record["source_change_category"] == "stable_correct")
        stable["raw_map_correct"] = False
        rejected = self.module.smoke_acceptance(selected, records)
        self.assertFalse(rejected["passed"])
        self.assertFalse(rejected["checks"]["no_raw_map_stable_correct_regression"])


if __name__ == "__main__":
    unittest.main()
