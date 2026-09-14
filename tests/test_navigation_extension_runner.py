from __future__ import annotations

from pathlib import Path
import unittest

from tools.evaluation.navigation.run_extension236 import (
    build_initial_status,
    parse_evaluator_progress_line,
    validate_complete_result_ids,
)


class NavigationExtensionRunnerTests(unittest.TestCase):
    def test_progress_lines_capture_current_and_completed_cases(self) -> None:
        self.assertEqual(
            parse_evaluator_progress_line("[start 1/236] TEST250"),
            {
                "event": "started",
                "position": 1,
                "total": 236,
                "test_id": "TEST250",
            },
        )
        self.assertEqual(
            parse_evaluator_progress_line(
                "[1/236] TEST250 success=True steps=5/50 new_images=0"
            ),
            {
                "event": "completed",
                "position": 1,
                "total": 236,
                "test_id": "TEST250",
                "success": True,
            },
        )
        self.assertIsNone(parse_evaluator_progress_line("cache inventory"))

    def test_complete_result_validation_detects_missing_and_duplicate_ids(self) -> None:
        expected = ("TEST250", "TEST251")
        validate_complete_result_ids(
            expected,
            [{"test_id": "TEST250"}, {"test_id": "TEST251"}],
            label="results",
        )

        with self.assertRaisesRegex(ValueError, "missing=.*TEST251"):
            validate_complete_result_ids(
                expected,
                [{"test_id": "TEST250"}, {"test_id": "TEST250"}],
                label="results",
            )

    def test_initial_status_preserves_start_time_and_increments_attempt(self) -> None:
        status = build_initial_status(
            test_set=Path("/tmp/extension.jsonl"),
            output_dir=Path("/tmp/output"),
            expected_case_count=236,
            existing_results=[
                {"test_id": "TEST250", "success": True},
                {"test_id": "TEST251", "success": False},
            ],
            previous_status={
                "started_at": "2026-08-02T00:00:00+00:00",
                "attempt_count": 1,
                "last_progress_at": "2026-08-02T00:30:00+00:00",
            },
        )

        self.assertEqual(status["started_at"], "2026-08-02T00:00:00+00:00")
        self.assertEqual(status["attempt_count"], 2)
        self.assertEqual(status["process_restart_count"], 1)
        self.assertEqual(status["expected_case_count"], 236)
        self.assertEqual(status["completed_case_count"], 2)
        self.assertEqual(status["last_completed_test_id"], "TEST251")
        self.assertFalse(status["last_completed_success"])
        self.assertEqual(
            status["last_progress_at"],
            "2026-08-02T00:30:00+00:00",
        )


if __name__ == "__main__":
    unittest.main()
