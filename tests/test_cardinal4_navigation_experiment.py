from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from tools.evaluation.navigation import run_cardinal4_initial_offset_incremental as campaign
from tools.evaluation.navigation import retry_cardinal4_api_errors as cardinal4_retry
from tools.evaluation.navigation.retry_cardinal4_smoke_error import replace_target_error
from tools.evaluation.navigation.report_view_count_comparison import build_report


class Cardinal4CampaignTests(unittest.TestCase):
    def test_evaluation_command_uses_four_view_profile_without_exposing_key(self) -> None:
        command = campaign.evaluation_command(
            test_ids=["TEST001"],
            output_dir=Path("/tmp/cardinal4-output"),
            direction_policy="geometry",
        )

        self.assertIn("--render-heading-mode", command)
        self.assertEqual(command[command.index("--render-heading-mode") + 1], "cardinal4")
        self.assertIn(str(campaign.RENDER_ROOT), command)
        self.assertNotIn("--llm-api-key", command)

    def test_key_audit_requires_distinct_keys_without_persisting_them(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            experiment_root = Path(tmpdir) / "experiment"
            environment = {
                "ST_NAV_ACTIVE_PROFILE": "gemini",
                "ST_NAV_PROFILE_GEMINI_API_KEY": "without-secret",
                "ST_NAV_WITH_TRAJECTORY_API_KEY": "with-secret",
                "GMAPS_API_KEY": "maps-secret",
            }
            with mock.patch.object(campaign, "EXPERIMENT_ROOT", experiment_root), mock.patch.dict(
                os.environ,
                environment,
                clear=True,
            ):
                keys = campaign.validate_api_keys()

            audit_text = (experiment_root / "api_key_audit.json").read_text(encoding="utf-8")
            self.assertEqual(set(keys), {"without_trajectory", "with_trajectory"})
            self.assertNotIn("without-secret", audit_text)
            self.assertNotIn("with-secret", audit_text)
            self.assertNotIn("maps-secret", audit_text)
            self.assertTrue(json.loads(audit_text)["keys_distinct"])

    def test_key_audit_rejects_reused_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.object(campaign, "EXPERIMENT_ROOT", Path(tmpdir)), mock.patch.dict(
                os.environ,
                {
                    "ST_NAV_ACTIVE_PROFILE": "gemini",
                    "ST_NAV_PROFILE_GEMINI_API_KEY": "same-secret",
                    "ST_NAV_WITH_TRAJECTORY_API_KEY": "same-secret",
                    "GMAPS_API_KEY": "maps-secret",
                },
                clear=True,
            ):
                with self.assertRaisesRegex(RuntimeError, "distinct API keys"):
                    campaign.validate_api_keys()

    def test_smoke_retry_replaces_only_the_requested_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source.jsonl"
            retry = root / "retry.jsonl"
            successful = {"test_id": "TEST003", "error": None, "success": True}
            failed = {
                "test_id": "TEST004",
                "error": {"type": "TimeoutError", "message": "timed out"},
                "stop_reason": "error",
                "success": False,
            }
            replacement = {
                "test_id": "TEST004",
                "error": None,
                "stop_reason": "goal_reached",
                "success": True,
            }
            source.write_text(
                "".join(json.dumps(record) + "\n" for record in (successful, failed)),
                encoding="utf-8",
            )
            retry.write_text(json.dumps(replacement) + "\n", encoding="utf-8")

            audit = replace_target_error(
                source_path=source,
                retry_path=retry,
                target_test_id="TEST004",
            )

            merged = campaign.load_jsonl(source)
            self.assertEqual(merged[0], successful)
            self.assertEqual(merged[1], replacement)
            self.assertEqual(audit["replaced_count"], 1)
            self.assertEqual(audit["preserved_successful_record_count"], 1)

    def test_full_retry_command_keeps_cardinal4_profile_and_extends_timeout(self) -> None:
        command = cardinal4_retry.retry_command(
            test_ids=["TEST055"],
            output_dir=Path("/tmp/cardinal4-retry"),
            direction_policy="geometry",
            llm_timeout=600.0,
        )

        self.assertEqual(command[command.index("--render-heading-mode") + 1], "cardinal4")
        self.assertEqual(command[command.index("--llm-timeout") + 1], "600")
        self.assertIn(str(campaign.RENDER_ROOT), command)

    def test_full_retry_merge_preserves_successes_and_failed_retries(self) -> None:
        originals = [
            {"test_id": "TEST001", "error": None, "success": True},
            {
                "test_id": "TEST002",
                "error": {"type": "TimeoutError", "message": "timed out"},
                "success": False,
            },
            {
                "test_id": "TEST003",
                "error": {"type": "TimeoutError", "message": "timed out"},
                "success": False,
            },
        ]
        successful_retry = {"test_id": "TEST002", "error": None, "success": False}
        failed_retry = {
            "test_id": "TEST003",
            "error": {"type": "TimeoutError", "message": "timed out again"},
            "success": False,
        }

        merged, audit = cardinal4_retry.merge_retry_records(
            originals=originals,
            requested_ids=["TEST002", "TEST003"],
            retry_by_id={"TEST002": successful_retry, "TEST003": failed_retry},
        )

        self.assertIs(merged[0], originals[0])
        self.assertEqual(merged[1], successful_retry)
        self.assertEqual(merged[2], originals[2])
        self.assertEqual(audit["replaced_ids"], ["TEST002"])
        self.assertEqual(audit["retry_error_ids"], ["TEST003"])
        self.assertEqual(audit["remaining_api_error_ids"], ["TEST003"])


class ViewCountComparisonTests(unittest.TestCase):
    @staticmethod
    def _record(test_id: str, *, success: bool, offset: float) -> dict:
        return {
            "test_id": test_id,
            "error": None,
            "stop_reason": "goal_reached" if success else "navigation_incomplete",
            "success": success,
            "spl": 0.8 if success else 0.0,
            "actual_over_shortest_ratio": 1.25,
            "initial_heading_alignment": {
                "estimated_offset_deg": offset,
                "ground_truth_offset_deg": 30.0,
            },
            "localization_events": [
                {
                    "step_index": 0,
                    "ground_truth_room_id": "Room 1",
                    "visual_predicted_room_id": "Room 1",
                    "bayesian_predicted_room_id": "Room 1",
                    "spatial_predicted_room_id": "Room 1",
                }
            ],
        }

    @staticmethod
    def _write_condition(root: Path, condition: str, records: list[dict]) -> None:
        destination = root / condition
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "results.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )

    def test_report_uses_common_api_valid_pairs_and_preserves_offset_gt(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            eight_root = root / "eight"
            four_root = root / "four"
            for condition in campaign.CONDITIONS:
                name = condition["name"]
                self._write_condition(
                    eight_root,
                    name,
                    [self._record("TEST001", success=True, offset=30.0)],
                )
                self._write_condition(
                    four_root,
                    name,
                    [self._record("TEST001", success=False, offset=0.0)],
                )

            summary = build_report(
                eight_root=eight_root,
                four_root=four_root,
                output_dir=root / "report",
            )

            without = summary["conditions"]["without_trajectory"]
            self.assertEqual(without["paired_api_valid_count"], 1)
            self.assertEqual(
                without["boolean_metrics"]["success"]["four_minus_eight_rate"],
                -1.0,
            )
            self.assertEqual(
                without["continuous_metrics"]["offset_error_deg"][
                    "mean_paired_difference_four_minus_eight"
                ],
                30.0,
            )
            self.assertTrue((root / "report" / "summary.json").is_file())
            self.assertTrue((root / "report" / "per_case.csv").is_file())


if __name__ == "__main__":
    unittest.main()
