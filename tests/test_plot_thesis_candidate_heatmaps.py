from __future__ import annotations

import csv
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tools.data.plot_thesis_candidate_heatmaps import (
    COUNT_HEATMAP_FILENAME,
    COUNT_MATRIX_FILENAME,
    PERCENTAGE_HEATMAP_FILENAME,
    PERCENTAGE_MATRIX_FILENAME,
    SUMMARY_FILENAME,
    CandidateMetric,
    build_heatmap_data,
    generate_heatmaps,
)


class PlotThesisCandidateHeatmapsTests(unittest.TestCase):
    def test_builds_counts_and_column_normalized_percentages(self) -> None:
        metrics = [
            CandidateMetric("TASK00001", 2, 1.0),
            CandidateMetric("TASK00002", 2, 1.1),
            CandidateMetric("TASK00003", 2, 1.5),
            CandidateMetric("TASK00004", 3, 1.1),
            CandidateMetric("TASK00005", 3, 2.0),
        ]

        data = build_heatmap_data(metrics, bin_width=0.2)

        self.assertEqual(data.room_counts, (2, 3))
        self.assertEqual(int(data.counts.sum()), 5)
        np.testing.assert_array_equal(data.counts.sum(axis=0), [3, 2])
        np.testing.assert_allclose(data.percentages.sum(axis=0), [100.0, 100.0])
        self.assertEqual(data.detour_ratio_labels[-1], ">2.0")

    def test_generates_both_heatmaps_and_auditable_matrices(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_path = root / "candidate_tasks.psv"
            output_dir = root / "output"
            with input_path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=("task_id", "visited_room_count", "detour_ratio"),
                    delimiter="|",
                    lineterminator="\n",
                )
                writer.writeheader()
                writer.writerows(
                    [
                        {
                            "task_id": "TASK00001",
                            "visited_room_count": 2,
                            "detour_ratio": 1.0,
                        },
                        {
                            "task_id": "TASK00002",
                            "visited_room_count": 2,
                            "detour_ratio": 1.4,
                        },
                        {
                            "task_id": "TASK00003",
                            "visited_room_count": 3,
                            "detour_ratio": 1.8,
                        },
                    ]
                )

            result = generate_heatmaps(
                input_path=input_path,
                output_dir=output_dir,
                expected_task_count=3,
                bin_width=0.2,
            )

            self.assertEqual(result["validation"]["cell_count_total"], 3)
            self.assertEqual(result["validation"]["status"], "valid")
            self.assertEqual(
                {path.name for path in output_dir.iterdir()},
                {
                    PERCENTAGE_HEATMAP_FILENAME,
                    COUNT_HEATMAP_FILENAME,
                    PERCENTAGE_MATRIX_FILENAME,
                    COUNT_MATRIX_FILENAME,
                    SUMMARY_FILENAME,
                },
            )
            with (output_dir / COUNT_MATRIX_FILENAME).open(
                "r", encoding="utf-8", newline=""
            ) as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[-1]["detour_ratio_bin"], "TOTAL")
            self.assertEqual(rows[-1]["row_total"], "3")

    def test_rejects_an_unexpected_task_count(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_path = root / "candidate_tasks.psv"
            input_path.write_text(
                "task_id|visited_room_count|detour_ratio\nTASK00001|2|1.0\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "Expected 2 candidate tasks"):
                generate_heatmaps(
                    input_path=input_path,
                    output_dir=root / "output",
                    expected_task_count=2,
                )


if __name__ == "__main__":
    unittest.main()
