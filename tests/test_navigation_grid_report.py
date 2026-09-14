from __future__ import annotations

import unittest

from tools.evaluation.navigation.report_navigation_grid import (
    apply_error_replacements,
    build_experiment_summary,
    derive_dataset_partitions,
    build_meeting_markdown,
)


def dataset_record(test_id: str, *, room_count: int, ratio: str, path_id: str) -> dict:
    return {
        "test_id": test_id,
        "path_id": path_id,
        "visited_room_count": room_count,
        "ratio_stratum": ratio,
    }


def result_record(
    test_id: str,
    *,
    path_id: str,
    success: bool,
    spl: float,
    error: dict | None = None,
) -> dict:
    return {
        "test_id": test_id,
        "path_id": path_id,
        "success": success,
        "spl": spl,
        "movement_steps": 10,
        "elapsed_seconds": 20.0,
        "render": {
            "cache_hit_events": 1,
            "cache_miss_events": 0,
            "new_render_pano_count": 0,
            "new_image_requests": 0,
        },
        "model_calls": {
            "stages": {
                "query_parser": {
                    "logical_call_count": 1,
                    "successful_logical_call_count": 1,
                    "text_logical_call_count": 1,
                    "http_attempt_count": 1,
                    "successful_http_attempt_count": 1,
                    "text_http_attempt_count": 1,
                }
            }
        },
        "error": error,
    }


class NavigationGridReportTests(unittest.TestCase):
    def test_summary_counts_errors_as_failures_and_keeps_sparse_cells_empty(self) -> None:
        dataset = {
            "TEST161": dataset_record(
                "TEST161", room_count=2, ratio="1.8-2.0", path_id="path-161"
            ),
            "TEST162": dataset_record(
                "TEST162", room_count=6, ratio="1.0-1.2", path_id="path-162"
            ),
        }
        results = [
            result_record("TEST161", path_id="path-161", success=True, spl=0.5),
            result_record(
                "TEST162",
                path_id="path-162",
                success=False,
                spl=0.0,
                error={"type": "RuntimeError", "message": "API failed"},
            ),
        ]

        summary = build_experiment_summary(dataset, results)

        self.assertEqual(summary["case_count"], 2)
        self.assertEqual(summary["success_count"], 1)
        self.assertEqual(summary["failure_count"], 1)
        comparison_markdown = build_meeting_markdown(
            title="test",
            summary=summary,
            comparison=[("Existing 1", summary), ("Extension 1", summary)],
        )
        self.assertIn("## Existing 1 vs Extension 1", comparison_markdown)

        self.assertEqual(summary["error_count"], 1)
        self.assertEqual(summary["error_test_ids"], ["TEST162"])
        self.assertEqual(summary["grid"]["cells"]["2"]["1.8-2.0"]["case_count"], 1)
        self.assertEqual(summary["grid"]["cells"]["2"]["1.0-1.2"]["case_count"], 0)
        self.assertEqual(summary["model_calls"]["coverage_case_count"], 2)
        markdown = build_meeting_markdown(title="test", summary=summary)
        self.assertIn("Errors are counted as failures", markdown)
        self.assertIn("—", markdown)

    def test_dataset_partitions_are_derived_from_extension_dataset(self) -> None:
        dataset = {
            "TEST001": dataset_record(
                "TEST001", room_count=2, ratio="1.0-1.2", path_id="path-1"
            ),
            "TEST002": dataset_record(
                "TEST002", room_count=2, ratio="1.2-1.4", path_id="path-2"
            ),
            "TEST003": dataset_record(
                "TEST003", room_count=3, ratio="1.4-1.6", path_id="path-3"
            ),
        }
        extension = {"TEST003": dataset["TEST003"]}

        legacy_ids, extension_ids, all_ids = derive_dataset_partitions(
            dataset,
            extension,
        )

        self.assertEqual(legacy_ids, ("TEST001", "TEST002"))
        self.assertEqual(extension_ids, ("TEST003",))
        self.assertEqual(all_ids, ("TEST001", "TEST002", "TEST003"))
        with self.assertRaisesRegex(ValueError, "unknown test IDs"):
            derive_dataset_partitions(
                dataset,
                {"TEST999": extension["TEST003"]},
            )

    def test_error_replacement_only_accepts_completed_rerun(self) -> None:
        original = result_record(
            "TEST161",
            path_id="path-161",
            success=False,
            spl=0.0,
            error={"type": "RuntimeError", "message": "API failed"},
        )
        completed = result_record(
            "TEST161",
            path_id="path-161",
            success=True,
            spl=0.75,
        )

        effective, audit = apply_error_replacements({"TEST161": original}, [completed])

        self.assertIs(effective["TEST161"], completed)
        self.assertTrue(audit[0]["accepted"])

        with self.assertRaisesRegex(ValueError, "only allowed for an original error"):
            apply_error_replacements({"TEST161": completed}, [completed])


if __name__ == "__main__":
    unittest.main()
