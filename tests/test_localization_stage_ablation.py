from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools.evaluation.navigation import report_localization_stage_ablation as report
from tools.evaluation.navigation import run_localization_stage_ablation as campaign


class LocalizationStageAblationTests(unittest.TestCase):
    def test_campaign_command_holds_nonlocalization_settings_constant(self) -> None:
        visual = campaign.evaluation_command(
            output_dir=Path("/tmp/visual"),
            effective_stage="visual",
        )
        bayesian = campaign.evaluation_command(
            output_dir=Path("/tmp/bayesian"),
            effective_stage="bayesian",
        )
        self.assertIn("museum", visual)
        self.assertIn("geometry-edge-visit", visual)
        self.assertIn("initial-spatial", visual)
        self.assertIn("--force-initial-visual-refresh", visual)
        self.assertEqual(visual[visual.index("--localization-interval") + 1], "3")
        self.assertEqual(visual[visual.index("--start-heading") + 1], "0")
        self.assertEqual(
            visual[visual.index("--localization-effective-stage") + 1],
            "visual",
        )
        self.assertEqual(
            bayesian[bayesian.index("--localization-effective-stage") + 1],
            "bayesian",
        )

    def test_report_uses_fixed_and_valid_denominators(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            records = {
                "visual": [
                    {"test_id": "TEST001", "error": None, "success": True, "spl": 0.5},
                    {"test_id": "TEST002", "error": "api", "success": False, "spl": 0.0},
                ],
                "bayesian": [
                    {"test_id": "TEST001", "error": None, "success": False, "spl": 0.0},
                    {"test_id": "TEST002", "error": None, "success": True, "spl": 0.8},
                ],
                "spatial": [
                    {"test_id": "TEST001", "error": None, "success": True, "spl": 0.6},
                    {"test_id": "TEST002", "error": None, "success": True, "spl": 0.7},
                ],
            }
            paths = {}
            for condition, condition_records in records.items():
                condition_root = root / condition
                condition_root.mkdir()
                paths[condition] = condition_root
                (condition_root / "results.jsonl").write_text(
                    "".join(json.dumps(record) + "\n" for record in condition_records),
                    encoding="utf-8",
                )
            payload = report.build_report(
                visual=paths["visual"],
                bayesian=paths["bayesian"],
                spatial=paths["spatial"],
                output_dir=root / "comparison",
                expected_case_count=2,
            )
            visual = payload["conditions"]["visual"]
            self.assertEqual(visual["error_count"], 1)
            self.assertEqual(visual["fixed_denominator"]["success_rate"], 0.5)
            self.assertEqual(visual["fixed_denominator"]["average_spl"], 0.25)
            self.assertEqual(visual["valid_only"]["success_rate"], 1.0)
            self.assertEqual(payload["paired_common_valid"]["common_valid_count"], 1)


if __name__ == "__main__":
    unittest.main()
