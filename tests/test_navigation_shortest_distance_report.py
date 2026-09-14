from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    PROJECT_ROOT
    / "tools/evaluation/navigation/report_navigation_by_shortest_distance.py"
)
SPEC = importlib.util.spec_from_file_location("distance_report", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
distance_report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(distance_report)


class NavigationShortestDistanceReportTest(unittest.TestCase):
    def test_bins_are_lower_inclusive_and_upper_exclusive(self) -> None:
        bins = distance_report.build_distance_bins(39.0, width_m=20.0)

        self.assertEqual(
            [item["label"] for item in bins], ["0-20 m", "20-40 m"]
        )
        self.assertEqual(distance_report.distance_bin_label(0.0, bins), "0-20 m")
        self.assertEqual(distance_report.distance_bin_label(19.999, bins), "0-20 m")
        self.assertEqual(distance_report.distance_bin_label(20.0, bins), "20-40 m")

    def test_cell_summary_uses_canonical_spl_and_keeps_failures(self) -> None:
        records = [
            {
                "success": True,
                "error": None,
                "spl": 0.5,
                "step_spl_legacy": 0.25,
                "shortest_path_distance_m": 10.0,
                "actual_path_distance_m": 20.0,
                "movement_steps": 4,
                "elapsed_seconds": 2.0,
            },
            {
                "success": False,
                "error": "api_error",
                "spl": 0.0,
                "step_spl_legacy": 0.0,
                "shortest_path_distance_m": 10.0,
                "actual_path_distance_m": 30.0,
                "movement_steps": 5,
                "elapsed_seconds": 4.0,
            },
        ]

        summary = distance_report.cell_summary(records)

        self.assertEqual(summary["case_count"], 2)
        self.assertEqual(summary["success_count"], 1)
        self.assertEqual(summary["error_count"], 1)
        self.assertEqual(summary["success_rate"], 0.5)
        self.assertEqual(summary["average_spl"], 0.25)
        self.assertEqual(summary["average_step_spl_legacy"], 0.125)

    def test_pano_path_distance_uses_haversine_segments(self) -> None:
        graph = {
            "a": {"lat": 0.0, "lng": 0.0},
            "b": {"lat": 0.0, "lng": 0.001},
            "c": {"lat": 0.0, "lng": 0.002},
        }

        distance = distance_report.pano_path_distance_m(["a", "b", "c"], graph)

        self.assertAlmostEqual(distance, 222.389853, places=5)


if __name__ == "__main__":
    unittest.main()
