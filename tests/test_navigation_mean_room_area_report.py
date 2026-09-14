from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from tools.evaluation.navigation.report_navigation_by_mean_room_area import (
    PROJECT_ROOT,
    actual_ground_truth_room_sequence,
    actual_path_gallery_area,
    assign_quartiles,
    assign_quantile_groups,
    assign_fixed_width_groups,
    assign_primary_fixed_interval_groups,
    build_binned_summary,
    build_filtered_actual_fixed_interval_summary,
    build_fixed_width_summary,
    build_primary_fixed_interval_summary,
    build_summary,
    build_trajectory_distribution_report,
    fixed_histogram_edges,
    fit_logistic_regression,
    histogram_counts,
    logistic_prediction,
    load_area_catalog,
    load_json,
    load_pano_room_grounding,
    render_histogram_plot,
    render_route_level_scatter_plot,
    render_scatter_plot,
    route_gallery_area,
    validate_paired_inputs,
    wilson_interval,
)


ROOM_AREA_PATH = (
    PROJECT_ROOT
    / "dataset/sites/british_museum/analysis/floor0_estimated_gallery_areas.json"
)


class MeanRoomAreaReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.room_areas, cls.excluded_room_ids = load_area_catalog(
            load_json(ROOM_AREA_PATH)
        )

    def test_repo_area_catalog_contains_28_galleries_and_circulation_exclusions(self) -> None:
        self.assertEqual(len(self.room_areas), 28)
        self.assertIn("East stairs", self.excluded_room_ids)
        self.assertIn("West stairs", self.excluded_room_ids)
        self.assertNotIn("East stairs", self.room_areas)

    def test_room9_to_room8_mean_matches_expected_value(self) -> None:
        area = route_gallery_area(
            ["Room 9", "Room 8"], self.room_areas, self.excluded_room_ids
        )

        self.assertEqual(area["gallery_room_ids"], ["Room 9", "Room 8"])
        self.assertAlmostEqual(area["estimated_mean_gallery_area_m2"], 217.67732460984565)

    def test_circulation_is_excluded_and_repeated_gallery_is_counted_once(self) -> None:
        area = route_gallery_area(
            ["Room 27", "East stairs", "Room 1", "Room 1"],
            self.room_areas,
            self.excluded_room_ids,
        )

        self.assertEqual(area["gallery_room_ids"], ["Room 27", "Room 1"])
        self.assertEqual(area["excluded_route_room_ids"], ["East stairs"])
        self.assertEqual(area["gallery_count_for_mean"], 2)
        self.assertAlmostEqual(
            area["estimated_total_unique_gallery_area_m2"],
            self.room_areas["Room 27"]["effective_area_m2"]
            + self.room_areas["Room 1"]["effective_area_m2"],
        )

    def test_unknown_room_is_not_silently_excluded(self) -> None:
        with self.assertRaisesRegex(ValueError, "without an area or exclusion"):
            route_gallery_area(
                ["Room 9", "Unknown room"], self.room_areas, self.excluded_room_ids
            )

    def test_actual_path_uses_ground_truth_rooms_and_counts_each_gallery_once(self) -> None:
        grounding = load_pano_room_grounding(
            {
                "mappings": {
                    "p1": "Room 9",
                    "p2": "null",
                    "p3": "East stairs",
                    "p4": "Room 8",
                    "p5": "Room 9",
                }
            }
        )
        sequence, unlabeled_count = actual_ground_truth_room_sequence(
            ["p1", "p2", "p3", "p4", "p5"], grounding
        )
        area = actual_path_gallery_area(
            {"pano_path": ["p1", "p2", "p3", "p4", "p5"]},
            grounding,
            self.room_areas,
            self.excluded_room_ids,
        )

        self.assertEqual(
            sequence, ["Room 9", "East stairs", "Room 8", "Room 9"]
        )
        self.assertEqual(unlabeled_count, 1)
        self.assertEqual(area["gallery_room_ids"], ["Room 9", "Room 8"])
        self.assertEqual(area["excluded_route_room_ids"], ["East stairs"])
        self.assertEqual(area["gallery_count_for_mean"], 2)
        self.assertAlmostEqual(
            area["estimated_mean_gallery_area_m2"], 217.67732460984565
        )

    def test_actual_path_rejects_unknown_pano_ids(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown pano IDs"):
            actual_ground_truth_room_sequence(["missing"], {"known": "Room 9"})

    def test_paired_inputs_require_exact_test_ids_and_path_ids(self) -> None:
        dataset = {"TEST001": {"path_id": "PATH001"}}
        valid = {
            "TEST001": {
                "path_id": "PATH001",
                "success": True,
            }
        }
        validate_paired_inputs(dataset, valid, valid)

        with self.assertRaisesRegex(ValueError, "test IDs differ"):
            validate_paired_inputs(dataset, {}, valid)
        with self.assertRaisesRegex(ValueError, "path_id differs"):
            validate_paired_inputs(
                dataset,
                {"TEST001": {"path_id": "WRONG", "success": True}},
                valid,
            )

    def test_quartiles_cover_all_routes_and_reproduce_overall_success(self) -> None:
        rows = [
            {
                "test_id": f"TEST{index:03d}",
                "estimated_mean_gallery_area_m2": float(index),
                "without_trajectory_success": index in {1, 2, 4, 7},
                "with_trajectory_success": index in {1, 2, 3, 5, 6, 8},
            }
            for index in range(1, 9)
        ]

        boundaries = assign_quartiles(rows)
        summary = build_summary(rows, boundaries)

        self.assertEqual(sum(group["case_count"] for group in summary["quartiles"]), 8)
        self.assertEqual(
            {row["area_quartile"] for row in rows}, {"Q1", "Q2", "Q3", "Q4"}
        )
        self.assertEqual(summary["overall"]["without_trajectory"]["success_count"], 4)
        self.assertEqual(summary["overall"]["with_trajectory"]["success_count"], 6)

    def test_six_eight_and_ten_groups_cover_every_route(self) -> None:
        for group_count in (6, 8, 10):
            rows = [
                {
                    "test_id": f"TEST{index:03d}",
                    "estimated_mean_gallery_area_m2": float(index),
                    "without_trajectory_success": index % 3 != 0,
                    "with_trajectory_success": index % 5 != 0,
                }
                for index in range(1, 121)
            ]
            field = f"area_group_{group_count}"
            boundaries, labels = assign_quantile_groups(
                rows, group_count, field=field
            )
            summary = build_binned_summary(
                rows, boundaries, labels, field=field
            )

            self.assertEqual(summary["group_count"], group_count)
            self.assertEqual(sum(group["case_count"] for group in summary["groups"]), 120)
            self.assertEqual(len({row[field] for row in rows}), group_count)
            self.assertEqual(
                sum(group["without_trajectory"]["success_count"] for group in summary["groups"]),
                summary["overall"]["without_trajectory"]["success_count"],
            )

    def test_fixed_100m2_groups_use_equal_width_and_retain_empty_bins(self) -> None:
        rows = [
            {
                "estimated_mean_gallery_area_m2": area,
                "without_trajectory_success": success,
                "with_trajectory_success": True,
            }
            for area, success in ((50.0, True), (150.0, False), (350.0, True))
        ]
        specifications = assign_fixed_width_groups(
            rows, 100.0, field="fixed_group"
        )
        summary = build_fixed_width_summary(
            rows,
            specifications,
            field="fixed_group",
            width_m2=100.0,
        )

        self.assertEqual(
            [group["group_label"] for group in summary["groups"]],
            ["0-100", "100-200", "200-300", "300-400"],
        )
        self.assertEqual(
            [group["case_count"] for group in summary["groups"]], [1, 1, 0, 1]
        )
        self.assertIsNone(summary["groups"][2]["without_trajectory"])
        self.assertEqual(
            summary["overall"]["without_trajectory"]["success_count"], 2
        )

    def test_fixed_200m2_filter_excludes_only_values_over_800(self) -> None:
        rows = [
            {
                "estimated_mean_gallery_area_m2": area,
                "without_trajectory_success": True,
                "with_trajectory_success": True,
            }
            for area in (100.0, 300.0, 500.0, 700.0, 800.0, 801.0)
        ]
        filtered = [
            row
            for row in rows
            if float(row["estimated_mean_gallery_area_m2"]) <= 800.0
        ]
        specifications = assign_fixed_width_groups(
            filtered, 200.0, field="fixed_group", maximum_m2=800.0
        )
        summary = build_fixed_width_summary(
            filtered,
            specifications,
            field="fixed_group",
            width_m2=200.0,
        )

        self.assertEqual(len(filtered), 5)
        self.assertEqual(
            [group["group_label"] for group in summary["groups"]],
            ["0-200", "200-400", "400-600", "600-800"],
        )
        self.assertEqual(summary["groups"][-1]["case_count"], 2)

    def test_fixed_200m2_open_final_group_includes_values_over_800(self) -> None:
        rows = [
            {
                "estimated_mean_gallery_area_m2": area,
                "without_trajectory_success": True,
                "with_trajectory_success": True,
            }
            for area in (100.0, 300.0, 500.0, 600.0, 700.0, 868.8)
        ]
        specifications = assign_fixed_width_groups(
            rows,
            200.0,
            field="fixed_group",
            open_ended_from_m2=600.0,
        )
        summary = build_fixed_width_summary(
            rows,
            specifications,
            field="fixed_group",
            width_m2=200.0,
        )

        self.assertEqual(
            [group["group_label"] for group in summary["groups"]],
            ["0-200", "200-400", "400-600", ">600"],
        )
        self.assertEqual(summary["groups"][-2]["case_count"], 2)
        self.assertEqual(summary["groups"][-1]["case_count"], 2)
        self.assertAlmostEqual(
            summary["groups"][-1]["mean_of_route_mean_gallery_area_m2"],
            (700.0 + 868.8) / 2,
        )
        self.assertIsNone(summary["groups"][-1]["midpoint_m2"])

    def test_primary_fixed_intervals_retain_above_800_in_final_bin(self) -> None:
        rows = [
            {
                "test_id": f"TEST{index:03d}",
                "reference_estimated_mean_gallery_area_m2": area,
                "without_trajectory_success": index != 5,
                "with_trajectory_success": True,
            }
            for index, area in enumerate(
                (100.0, 200.0, 400.0, 600.0, 900.0), start=1
            )
        ]
        specifications = assign_primary_fixed_interval_groups(
            rows,
            area_key="reference_estimated_mean_gallery_area_m2",
            field="reference_fixed_200m2_group",
        )
        summary = build_primary_fixed_interval_summary(
            rows, path_kind="reference_shortest_path"
        )

        self.assertEqual(
            [specification["group_label"] for specification in specifications],
            ["0-200", "200-400", "400-600", "600-800"],
        )
        self.assertEqual(
            [group["without_trajectory"]["case_count"] for group in summary["groups"]],
            [1, 1, 1, 2],
        )
        self.assertEqual(
            summary["groups"][-1]["without_trajectory"][
                "above_800_m2_case_count"
            ],
            1,
        )
        self.assertEqual(
            sum(
                group["without_trajectory"]["success_count"]
                for group in summary["groups"]
            ),
            4,
        )

    def test_actual_path_bins_are_method_specific_but_cover_all_cases(self) -> None:
        rows = [
            {
                "test_id": "TEST001",
                "without_actual_estimated_mean_gallery_area_m2": 100.0,
                "with_actual_estimated_mean_gallery_area_m2": 300.0,
                "without_trajectory_success": False,
                "with_trajectory_success": True,
            },
            {
                "test_id": "TEST002",
                "without_actual_estimated_mean_gallery_area_m2": 300.0,
                "with_actual_estimated_mean_gallery_area_m2": 100.0,
                "without_trajectory_success": True,
                "with_trajectory_success": True,
            },
            {
                "test_id": "TEST003",
                "without_actual_estimated_mean_gallery_area_m2": 500.0,
                "with_actual_estimated_mean_gallery_area_m2": 700.0,
                "without_trajectory_success": True,
                "with_trajectory_success": False,
            },
            {
                "test_id": "TEST004",
                "without_actual_estimated_mean_gallery_area_m2": 900.0,
                "with_actual_estimated_mean_gallery_area_m2": 500.0,
                "without_trajectory_success": False,
                "with_trajectory_success": True,
            },
        ]
        summary = build_primary_fixed_interval_summary(
            rows, path_kind="actual_agent_path"
        )

        self.assertFalse(
            summary["grouping"]["shared_route_membership_between_methods"]
        )
        self.assertEqual(
            sum(group["without_trajectory"]["case_count"] for group in summary["groups"]),
            4,
        )
        self.assertEqual(
            sum(group["with_trajectory"]["case_count"] for group in summary["groups"]),
            4,
        )
        self.assertEqual(
            summary["groups"][-1]["without_trajectory"][
                "above_800_m2_case_count"
            ],
            1,
        )

    def test_filtered_actual_path_summary_excludes_method_specific_values_over_800(
        self,
    ) -> None:
        rows = [
            {
                "test_id": "TEST001",
                "without_actual_estimated_mean_gallery_area_m2": 100.0,
                "with_actual_estimated_mean_gallery_area_m2": 900.0,
                "without_trajectory_success": True,
                "with_trajectory_success": False,
            },
            {
                "test_id": "TEST002",
                "without_actual_estimated_mean_gallery_area_m2": 900.0,
                "with_actual_estimated_mean_gallery_area_m2": 700.0,
                "without_trajectory_success": False,
                "with_trajectory_success": True,
            },
            {
                "test_id": "TEST003",
                "without_actual_estimated_mean_gallery_area_m2": 700.0,
                "with_actual_estimated_mean_gallery_area_m2": 500.0,
                "without_trajectory_success": False,
                "with_trajectory_success": True,
            },
        ]

        summary = build_filtered_actual_fixed_interval_summary(rows)

        self.assertEqual(
            summary["retained_case_count"],
            {"without_trajectory": 2, "with_trajectory": 2},
        )
        self.assertEqual(
            summary["excluded_test_ids"],
            {"without_trajectory": ["TEST002"], "with_trajectory": ["TEST001"]},
        )
        self.assertEqual(
            summary["groups"][-1]["without_trajectory"]["case_count"], 1
        )
        self.assertEqual(
            summary["groups"][-1]["with_trajectory"]["case_count"], 1
        )
        self.assertEqual(
            summary["groups"][-1]["without_trajectory"][
                "above_800_m2_case_count"
            ],
            0,
        )

    def test_wilson_interval_is_bounded_and_contains_observed_rate(self) -> None:
        low, high = wilson_interval(7, 10)

        self.assertGreaterEqual(low, 0.0)
        self.assertLessEqual(high, 1.0)
        self.assertLess(low, 0.7)
        self.assertGreater(high, 0.7)

    def test_histogram_bins_cover_boundaries_once(self) -> None:
        values = (0.0, 20.0, 39.9, 40.0, 60.0)
        edges = fixed_histogram_edges(values, 20.0)

        self.assertEqual(edges, [0.0, 20.0, 40.0, 60.0])
        self.assertEqual(histogram_counts(values, edges), [1, 2, 2])

    def test_trajectory_distribution_report_sums_25_navigable_galleries(self) -> None:
        room_ids = list(self.room_areas)[:25]
        rows = []
        for index in range(2):
            rows.append(
                {
                    "reference_gallery_room_ids": room_ids,
                    "shortest_path_distance_m": 20.0 + index * 20.0,
                    "reference_estimated_total_unique_gallery_area_m2": 1000.0,
                    "without_actual_estimated_total_unique_gallery_area_m2": 1200.0,
                    "with_actual_estimated_total_unique_gallery_area_m2": 1400.0,
                }
            )

        report = build_trajectory_distribution_report(rows, self.room_areas)

        self.assertEqual(report["navigable_galleries"]["gallery_count"], 25)
        self.assertAlmostEqual(
            report["navigable_galleries"]["estimated_total_area_m2"],
            sum(
                self.room_areas[room_id]["effective_area_m2"]
                for room_id in room_ids
            ),
        )
        self.assertEqual(
            sum(report["shortest_path_distance_m"]["histogram"]["counts"]),
            2,
        )

    def test_logistic_regression_detects_declining_success_probability(self) -> None:
        rows = [
            {
                "estimated_mean_gallery_area_m2": float(index * 50),
                "success": index in {1, 2, 3, 5, 7},
            }
            for index in range(1, 11)
        ]

        model = fit_logistic_regression(rows, "success")
        low_area_probability, _, _ = logistic_prediction(model, 100.0)
        high_area_probability, _, _ = logistic_prediction(model, 450.0)

        self.assertLess(model["odds_ratio_per_100_m2"], 1.0)
        self.assertLess(
            model["odds_ratio_per_100_m2_95_ci_low"],
            model["odds_ratio_per_100_m2"],
        )
        self.assertGreater(
            model["odds_ratio_per_100_m2_95_ci_high"],
            model["odds_ratio_per_100_m2"],
        )
        self.assertGreater(low_area_probability, high_area_probability)

    def test_scatter_png_contains_expected_metadata_and_nontrivial_pixels(self) -> None:
        rows = [
            {
                "test_id": f"TEST{index:03d}",
                "estimated_mean_gallery_area_m2": float(index * 100),
                "without_trajectory_success": index % 2 == 0,
                "with_trajectory_success": index not in {4, 8},
            }
            for index in range(1, 9)
        ]
        boundaries = assign_quartiles(rows)
        summary = build_summary(rows, boundaries)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scatter.png"
            render_scatter_plot(summary, path)
            with Image.open(path) as image:
                self.assertEqual(image.size, (1600, 1000))
                self.assertIn("Without trajectory", image.info["Description"])
                self.assertIn("With trajectory", image.info["Description"])
                colors = image.convert("RGB").getcolors(maxcolors=1_000_000)
                self.assertIsNotNone(colors)
                self.assertGreater(len(colors or []), 10)
            minimal_path = Path(directory) / "scatter-minimal.png"
            render_scatter_plot(summary, minimal_path, minimal=True)
            with Image.open(minimal_path) as image:
                self.assertEqual(image.size, (1600, 1000))
                self.assertIn("4 fixed-interval group points", image.info["Description"])

            interval_path = Path(directory) / "scatter-intervals.png"
            render_scatter_plot(
                summary,
                interval_path,
                groups=summary["quartiles"],
                minimal=True,
                x_domain=(0.0, 800.0),
                x_axis_label="Mean gallery area interval (m²)",
                x_tick_labels=(
                    (100.0, "0–200"),
                    (300.0, "200–400"),
                    (500.0, "400–600"),
                    (700.0, "600–800"),
                ),
            )
            with Image.open(interval_path) as image:
                self.assertIn(
                    "X-axis interval labels: 0–200, 200–400, 400–600, 600–800.",
                    image.info["Description"],
                )
                self.assertIn(
                    "Error bars show 95% Wilson confidence intervals.",
                    image.info["Description"],
                )

    def test_route_level_scatter_has_two_panels_and_binary_outcome_description(self) -> None:
        rows = [
            {
                "test_id": f"TEST{index:03d}",
                "estimated_mean_gallery_area_m2": float(index * 100),
                "without_trajectory_success": index % 2 == 0,
                "with_trajectory_success": index not in {4, 8},
            }
            for index in range(1, 9)
        ]
        boundaries = assign_quartiles(rows)
        summary = build_summary(rows, boundaries)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "route-scatter.png"
            render_route_level_scatter_plot(rows, summary, path)
            with Image.open(path) as image:
                self.assertEqual(image.size, (1800, 1000))
                self.assertIn("Two-panel", image.info["Description"])
                self.assertIn("binary 0/100%", image.info["Description"])

    def test_histogram_png_records_bins_and_series(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "histogram.png"
            render_histogram_plot(
                (
                    ("Without trajectory", (100.0, 300.0, 700.0), "#0072B2"),
                    ("With trajectory", (200.0, 400.0, 800.0), "#D55E00"),
                ),
                (0.0, 500.0, 1000.0),
                path,
                x_axis_label="Sum of unique gallery areas (m²)",
            )
            with Image.open(path) as image:
                self.assertEqual(image.size, (1600, 1000))
                self.assertIn("2 equal-width bins", image.info["Description"])
                self.assertIn("Without trajectory", image.info["Description"])
                self.assertIn("With trajectory", image.info["Description"])


if __name__ == "__main__":
    unittest.main()
