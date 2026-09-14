from __future__ import annotations

from copy import deepcopy
import unittest

from tools.data.sample_navigation_grid import (
    build_partial_grid,
    excluded_room_hits,
    ratio_grid_stratum_for_value,
    sampling_summary_notes,
)


def candidate(
    path_id: str,
    *,
    room_count: int,
    ratio: float,
    start_room_id: str,
    target_group_id: str,
) -> dict:
    target_room_id = f"target-{target_group_id}"
    middle_rooms = [f"middle-{path_id}-{index}" for index in range(max(room_count - 2, 0))]
    shortest_rooms = [start_room_id, *middle_rooms, target_room_id]
    return {
        "path_id": path_id,
        "floor": "0",
        "difficulty": "easy" if room_count <= 3 else "medium",
        "visited_room_count": room_count,
        "room_transition_count": max(room_count - 1, 0),
        "detour_ratio": ratio,
        "path_distance_m": ratio * 10.0,
        "straight_line_distance_m": 10.0,
        "pano_step_count": room_count + 2,
        "target_group_id": target_group_id,
        "target_group_theme": f"Theme {target_group_id}",
        "reference_end_room_id": target_room_id,
        "target_room_id": target_room_id,
        "end_room_id": target_room_id,
        "acceptable_target_room_ids": [target_room_id],
        "start_room_id": start_room_id,
        "start_pano_id": f"start-{path_id}",
        "end_pano_id": f"end-{path_id}",
        "shortest_path_rooms": shortest_rooms,
        "shortest_path_panos": [f"pano-{path_id}-{index}" for index in range(room_count)],
        "shortest_path_pano_rooms": list(shortest_rooms),
        "null_pano_count": 0,
        "path_constraint_mode": "room_topology_bfs_constrained_dijkstra",
        "topology_validation_status": "valid",
    }


def query_mapping(*target_group_ids: str) -> dict[str, dict]:
    return {
        target_group_id: {
            "query_id": f"Q-{target_group_id}",
            "target_group_id": target_group_id,
            "acceptable_target_room_ids": [f"target-{target_group_id}"],
            "query": f"Take me to Theme {target_group_id} gallery.",
        }
        for target_group_id in target_group_ids
    }


class NavigationGridSamplerTests(unittest.TestCase):
    def test_ratio_grid_boundaries_are_lower_inclusive_upper_exclusive(self) -> None:
        expected = {
            1.0: "1.0-1.2",
            1.199999: "1.0-1.2",
            1.2: "1.2-1.4",
            1.4: "1.4-1.6",
            1.6: "1.6-1.8",
            1.8: "1.8-2.0",
            1.999999: "1.8-2.0",
        }
        for ratio, label in expected.items():
            self.assertEqual(
                ratio_grid_stratum_for_value(ratio, lower_bound=1.0, upper_bound=2.0)["label"],
                label,
            )
        self.assertIsNone(ratio_grid_stratum_for_value(0.999999, upper_bound=2.0))
        self.assertIsNone(ratio_grid_stratum_for_value(2.0, upper_bound=2.0))

    def test_sparse_cell_requires_explicit_opt_in(self) -> None:
        records = [
            candidate(
                f"P{index}",
                room_count=2,
                ratio=1.9,
                start_room_id=f"Room {index}",
                target_group_id="target",
            )
            for index in range(9)
        ]
        queries = query_mapping("target")

        with self.assertRaisesRegex(ValueError, "insufficient cells"):
            build_partial_grid(
                records,
                [],
                (),
                queries,
                room_counts=(2,),
                lower_bound=1.8,
                upper_bound=2.0,
                per_cell=10,
            )

        combined, extension, diagnostics, _, rerun = build_partial_grid(
            records,
            [],
            (),
            queries,
            room_counts=(2,),
            lower_bound=1.8,
            upper_bound=2.0,
            per_cell=10,
            allow_sparse_cells=True,
        )

        self.assertEqual(len(combined), 9)
        self.assertEqual(len(extension), 9)
        self.assertEqual(diagnostics["sample_shortfall_count"], 1)
        self.assertEqual(diagnostics["sparse_grid_cell_count"], 1)
        self.assertEqual(rerun["test_ids"], [f"TEST{index:03d}" for index in range(1, 10)])

    def test_partial_preserve_keeps_records_and_fills_only_missing_cells(self) -> None:
        records = []
        for ratio_label, ratio in (("low", 1.1), ("high", 1.3)):
            for index in range(4):
                records.append(
                    candidate(
                        f"{ratio_label}-{index}",
                        room_count=2,
                        ratio=ratio,
                        start_room_id=f"Room {ratio_label}-{index}",
                        target_group_id=f"target-{index % 2}",
                    )
                )
        queries = query_mapping("target-0", "target-1")
        full, _, _, _, _ = build_partial_grid(
            records,
            [],
            (),
            queries,
            room_counts=(2,),
            lower_bound=1.0,
            upper_bound=1.4,
            per_cell=2,
            seed=42,
        )
        previous = deepcopy([record for record in full if record["ratio_stratum"] == "1.0-1.2"])
        self.assertEqual([record["test_id"] for record in previous], ["TEST001", "TEST002"])

        combined, extension, diagnostics, manifest, rerun = build_partial_grid(
            records,
            previous,
            (),
            queries,
            room_counts=(2,),
            lower_bound=1.0,
            upper_bound=1.4,
            per_cell=2,
            seed=42,
        )

        self.assertEqual(combined[:2], previous)
        self.assertEqual([record["test_id"] for record in extension], ["TEST003", "TEST004"])
        self.assertTrue(all(record["ratio_stratum"] == "1.2-1.4" for record in extension))
        self.assertEqual(diagnostics["preserved_count"], 2)
        self.assertEqual(diagnostics["extension_count"], 2)
        self.assertEqual(manifest["preserved_test_ids"], ["TEST001", "TEST002"])
        self.assertEqual(rerun["test_ids"], ["TEST003", "TEST004"])

    def test_partial_preserve_refreshes_sampling_metadata_when_cap_grows(self) -> None:
        records = [
            candidate(
                f"P{index}",
                room_count=2,
                ratio=1.1,
                start_room_id=f"Room {index}",
                target_group_id="target",
            )
            for index in range(3)
        ]
        queries = query_mapping("target")
        previous = build_partial_grid(
            records,
            [],
            (),
            queries,
            room_counts=(2,),
            lower_bound=1.0,
            upper_bound=1.2,
            per_cell=2,
        )[0]
        self.assertTrue(all(not record["sampling_cell_sparse"] for record in previous))

        combined, extension, diagnostics, _, rerun = build_partial_grid(
            records,
            previous,
            (),
            queries,
            room_counts=(2,),
            lower_bound=1.0,
            upper_bound=1.2,
            per_cell=4,
            allow_sparse_cells=True,
        )

        self.assertEqual(
            [record["test_id"] for record in combined],
            ["TEST001", "TEST002", "TEST003"],
        )
        self.assertEqual([record["test_id"] for record in extension], ["TEST003"])
        self.assertTrue(
            all(record["sampling_cell_candidate_count"] == 3 for record in combined)
        )
        self.assertTrue(
            all(record["sampling_cell_selected_count"] == 3 for record in combined)
        )
        self.assertTrue(all(record["sampling_cell_sparse"] for record in combined))
        self.assertEqual(diagnostics["sample_shortfall_count"], 1)
        self.assertEqual(rerun["test_ids"], ["TEST003"])

    def test_sampling_summary_notes_report_all_sparse_cell_denominators(self) -> None:
        selection_note, reporting_scope = sampling_summary_notes(
            {
                "grid_cell_count": 3,
                "grid_cells": [
                    {
                        "visited_room_count": 2,
                        "ratio_stratum": "1.0-1.2",
                        "selected_count": 20,
                    },
                    {
                        "visited_room_count": 2,
                        "ratio_stratum": "1.6-1.8",
                        "selected_count": 16,
                    },
                    {
                        "visited_room_count": 2,
                        "ratio_stratum": "1.8-2.0",
                        "selected_count": 9,
                    },
                ],
            },
            per_cell=20,
        )

        self.assertIn("2 naturally undersupplied cells", selection_note)
        self.assertIn("rooms_2/1.6-1.8=16/20", reporting_scope)
        self.assertIn("rooms_2/1.8-2.0=9/20", reporting_scope)
        self.assertIn(
            "other 1 requested cells contain 20 cases each",
            reporting_scope,
        )

    def test_seeded_selection_is_reproducible_and_avoids_duplicate_paths(self) -> None:
        records = [
            candidate(
                f"P{index:02d}",
                room_count=2,
                ratio=1.1,
                start_room_id=f"Room {index % 3}",
                target_group_id=f"target-{index % 4}",
            )
            for index in range(20)
        ]
        queries = query_mapping("target-0", "target-1", "target-2", "target-3")

        first = build_partial_grid(
            records,
            [],
            (),
            queries,
            room_counts=(2,),
            lower_bound=1.0,
            upper_bound=1.2,
            per_cell=10,
            seed=42,
        )[0]
        second = build_partial_grid(
            records,
            [],
            (),
            queries,
            room_counts=(2,),
            lower_bound=1.0,
            upper_bound=1.2,
            per_cell=10,
            seed=42,
        )[0]

        self.assertEqual(first, second)
        self.assertEqual(len({record["path_id"] for record in first}), 10)

    def test_excluded_rooms_are_checked_across_route_fields(self) -> None:
        record = candidate(
            "excluded",
            room_count=2,
            ratio=1.1,
            start_room_id="Room 1",
            target_group_id="target",
        )
        record["acceptable_target_room_ids"].append("Room 18a")
        self.assertEqual(excluded_room_hits(record, ("Room 18a", "Room 18b")), ("Room 18a",))


if __name__ == "__main__":
    unittest.main()
