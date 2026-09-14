from __future__ import annotations

import unittest

from tools.evaluation.navigation.compare_navigation_policies import (
    apply_error_replacements,
    build_paired_cases,
    build_summary,
    records_by_test_id,
    trajectory_metrics,
    validate_result_identity,
)


def dataset_record(test_id: str, room_count: int, ratio: str) -> dict:
    return {
        "test_id": test_id,
        "path_id": f"path-{test_id}",
        "visited_room_count": room_count,
        "ratio_stratum": ratio,
    }


def result_record(
    test_id: str,
    *,
    success: bool,
    pano_path: list[str] | None = None,
    error: dict | None = None,
    path_id: str | None = None,
) -> dict:
    return {
        "test_id": test_id,
        "path_id": path_id or f"path-{test_id}",
        "success": success,
        "spl": 0.5 if success else 0.0,
        "movement_steps": 4,
        "actual_path_distance_m": 20.0,
        "pano_path": pano_path or ["A", "B", "C"],
        "error": error,
    }


class NavigationPolicyComparisonTests(unittest.TestCase):
    def test_partial_report_pairs_only_candidate_ids_and_keeps_sparse_cell(self) -> None:
        dataset = {
            "TEST001": dataset_record("TEST001", 2, "1.0-1.2"),
            "TEST002": dataset_record("TEST002", 2, "1.8-2.0"),
            "TEST003": dataset_record("TEST003", 3, "1.0-1.2"),
        }
        baseline = {
            test_id: result_record(test_id, success=True) for test_id in dataset
        }
        candidate = {
            "TEST001": result_record("TEST001", success=False),
            "TEST002": result_record("TEST002", success=True),
        }

        pairs = build_paired_cases(dataset, baseline, candidate)
        summary = build_summary(dataset, pairs)

        self.assertEqual([row["test_id"] for row in pairs], ["TEST001", "TEST002"])
        self.assertEqual(summary["completed_candidate_case_count"], 2)
        self.assertEqual(summary["pending_candidate_case_count"], 1)
        sparse = summary["grid"]["cells"]["2"]["1.8-2.0"]
        self.assertEqual(sparse["case_count"], 1)
        self.assertEqual(sparse["candidate"]["success_count"], 1)

    def test_249_case_partial_grid_preserves_nine_case_sparse_denominator(self) -> None:
        dataset: dict[str, dict] = {}
        baseline: dict[str, dict] = {}
        candidate: dict[str, dict] = {}
        next_id = 1
        for room_count in (2, 3, 4, 5, 6):
            for ratio in ("1.0-1.2", "1.2-1.4", "1.4-1.6", "1.6-1.8", "1.8-2.0"):
                count = 9 if (room_count, ratio) == (2, "1.8-2.0") else 10
                for _ in range(count):
                    test_id = f"TEST{next_id:03d}"
                    dataset[test_id] = dataset_record(test_id, room_count, ratio)
                    baseline[test_id] = result_record(test_id, success=True)
                    candidate[test_id] = result_record(test_id, success=True)
                    next_id += 1

        pairs = build_paired_cases(dataset, baseline, candidate)
        summary = build_summary(dataset, pairs)

        self.assertEqual(len(pairs), 249)
        self.assertEqual(
            summary["grid"]["cells"]["2"]["1.8-2.0"]["case_count"], 9
        )

    def test_duplicate_unknown_and_path_mismatch_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate test_id"):
            records_by_test_id(
                [
                    result_record("TEST001", success=True),
                    result_record("TEST001", success=False),
                ],
                label="candidate",
            )
        dataset = {"TEST001": dataset_record("TEST001", 2, "1.0-1.2")}
        with self.assertRaisesRegex(ValueError, "unknown test IDs"):
            validate_result_identity(
                dataset,
                {"TEST999": result_record("TEST999", success=True)},
                label="candidate",
            )
        with self.assertRaisesRegex(ValueError, "path_id differs"):
            validate_result_identity(
                dataset,
                {
                    "TEST001": result_record(
                        "TEST001", success=True, path_id="wrong-path"
                    )
                },
                label="candidate",
            )

    def test_replacement_requires_original_error_and_completed_rerun(self) -> None:
        original_error = result_record(
            "TEST001",
            success=False,
            error={"type": "TimeoutError", "message": "timeout"},
        )
        completed_failure = result_record("TEST001", success=False)

        effective, audit = apply_error_replacements(
            {"TEST001": original_error},
            [("rerun/results.jsonl", [completed_failure])],
        )

        self.assertIs(effective["TEST001"], completed_failure)
        self.assertEqual(audit[0]["replacement_source"], "rerun/results.jsonl")
        self.assertFalse(audit[0]["replacement_success"])
        with self.assertRaisesRegex(ValueError, "only allowed for an original error"):
            apply_error_replacements(
                {"TEST001": completed_failure},
                [("rerun/results.jsonl", [completed_failure])],
            )
        with self.assertRaisesRegex(ValueError, "did not complete successfully"):
            apply_error_replacements(
                {"TEST001": original_error},
                [("rerun/results.jsonl", [original_error])],
            )

    def test_trajectory_metrics_count_repeated_panos_and_undirected_edges(self) -> None:
        metrics = trajectory_metrics(["A", "B", "A", "C", "A"])

        self.assertEqual(metrics["repeated_pano_visits"], 2)
        self.assertEqual(metrics["repeated_undirected_edge_traversals"], 2)
        self.assertTrue(metrics["has_repeated_undirected_edge"])

    def test_success_transition_and_candidate_minus_baseline_deltas(self) -> None:
        dataset = {"TEST001": dataset_record("TEST001", 2, "1.0-1.2")}
        baseline = {
            "TEST001": result_record(
                "TEST001", success=False, pano_path=["A", "B", "A"]
            )
        }
        candidate = {
            "TEST001": result_record(
                "TEST001", success=True, pano_path=["A", "B", "C"]
            )
        }

        pair = build_paired_cases(dataset, baseline, candidate)[0]

        self.assertEqual(pair["success_transition"], "baseline_fail_candidate_success")
        self.assertEqual(pair["delta_repeated_undirected_edge_traversals"], -1)


if __name__ == "__main__":
    unittest.main()
