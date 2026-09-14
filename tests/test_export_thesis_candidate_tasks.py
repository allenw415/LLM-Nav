from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from tools.data.export_thesis_candidate_tasks import (
    BUNDLE_FILENAME,
    CANDIDATE_FILENAME,
    CROSSWALK_FILENAME,
    ExportExpectations,
    SUMMARY_FILENAME,
    TEST_TASK_IDS_FILENAME,
    build_thesis_export,
    export_thesis_candidate_tasks,
    validate_expectations,
    validate_export_directory,
)


TARGET_GROUPS = {
    "group-a": ["Room A"],
    "group-b": ["Room B1", "Room B2"],
    "group-c": ["Room C"],
}


def route(
    number: int,
    *,
    start_pano_id: str,
    start_room_id: str,
    target_group_id: str,
    end_room_id: str,
) -> dict:
    end_pano_id = f"end-{number}"
    return {
        "path_id": f"NP{number:06d}",
        "start_pano_id": start_pano_id,
        "end_pano_id": end_pano_id,
        "start_room_id": start_room_id,
        "end_room_id": end_room_id,
        "target_group_id": target_group_id,
        "acceptable_target_room_ids": TARGET_GROUPS[target_group_id],
        "shortest_path_rooms": [start_room_id, end_room_id],
        "shortest_path_panos": [start_pano_id, end_pano_id],
        "visited_room_count": 2,
        "path_distance_m": 12.0,
        "straight_line_distance_m": 10.0,
        "detour_ratio": 1.2,
        "path_constraint_mode": "room_topology_bfs_constrained_dijkstra",
        "topology_validation_status": "valid",
    }


def source_records() -> list[dict]:
    return [
        route(
            1,
            start_pano_id="pano-a",
            start_room_id="Room A",
            target_group_id="group-b",
            end_room_id="Room B1",
        ),
        route(
            2,
            start_pano_id="pano-a",
            start_room_id="Room A",
            target_group_id="group-c",
            end_room_id="Room C",
        ),
        route(
            3,
            start_pano_id="pano-b",
            start_room_id="Room B2",
            target_group_id="group-a",
            end_room_id="Room A",
        ),
        route(
            4,
            start_pano_id="pano-b",
            start_room_id="Room B2",
            target_group_id="group-c",
            end_room_id="Room C",
        ),
        route(
            5,
            start_pano_id="pano-x",
            start_room_id="Room X",
            target_group_id="group-a",
            end_room_id="Room A",
        ),
        route(
            6,
            start_pano_id="pano-x",
            start_room_id="Room X",
            target_group_id="group-b",
            end_room_id="Room B1",
        ),
        route(
            7,
            start_pano_id="pano-x",
            start_room_id="Room X",
            target_group_id="group-c",
            end_room_id="Room C",
        ),
    ]


EXPECTATIONS = ExportExpectations(
    source_task_count=7,
    candidate_task_count=4,
    eligible_start_panorama_count=2,
    semantic_target_group_count=3,
    targets_per_start=2,
    test_task_count=2,
    excluded_start_panoramas_by_room=(("Room X", 1),),
)


class ExportThesisCandidateTasksTests(unittest.TestCase):
    def test_build_filters_non_target_starts_and_assigns_contiguous_ids(self) -> None:
        export = build_thesis_export(
            source_records(),
            [
                {"test_id": "TEST002", "path_id": "NP000004"},
                {"test_id": "TEST001", "path_id": "NP000001"},
            ],
        )
        validate_expectations(export, EXPECTATIONS)

        self.assertEqual(
            [row["task_id"] for row in export.candidate_rows],
            ["TASK00001", "TASK00002", "TASK00003", "TASK00004"],
        )
        self.assertEqual(
            {row["room_start"] for row in export.candidate_rows},
            {"Room A", "Room B2"},
        )
        self.assertEqual(
            export.test_task_id_rows,
            [
                {"test_id": "TEST001", "task_id": "TASK00001"},
                {"test_id": "TEST002", "task_id": "TASK00004"},
            ],
        )
        self.assertEqual(
            export.diagnostics["excluded_start_panoramas_by_room"],
            {"Room X": 1},
        )
        self.assertEqual(export.diagnostics["removed_task_count"], 3)

    def test_missing_test_path_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "absent from the thesis corpus"):
            build_thesis_export(
                source_records(),
                [{"test_id": "TEST001", "path_id": "NP000005"}],
            )

    def test_writes_and_independently_validates_psv_and_zip_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_path = root / "source.jsonl"
            test_path = root / "tests.jsonl"
            output_dir = root / "output"
            source_path.write_text(
                "".join(json.dumps(record) + "\n" for record in source_records()),
                encoding="utf-8",
            )
            test_path.write_text(
                "".join(
                    json.dumps(record) + "\n"
                    for record in (
                        {"test_id": "TEST001", "path_id": "NP000001"},
                        {"test_id": "TEST002", "path_id": "NP000004"},
                    )
                ),
                encoding="utf-8",
            )

            result = export_thesis_candidate_tasks(
                input_path=source_path,
                test_set_path=test_path,
                output_dir=output_dir,
                expectations=EXPECTATIONS,
            )
            verified = validate_export_directory(output_dir, EXPECTATIONS)

            self.assertEqual(result["status"], "valid")
            self.assertEqual(verified["candidate_task_count"], 4)
            self.assertEqual(
                {path.name for path in output_dir.iterdir()},
                {
                    CANDIDATE_FILENAME,
                    TEST_TASK_IDS_FILENAME,
                    CROSSWALK_FILENAME,
                    SUMMARY_FILENAME,
                    BUNDLE_FILENAME,
                },
            )
            candidate_lines = (output_dir / CANDIDATE_FILENAME).read_text(
                encoding="utf-8"
            ).splitlines()
            self.assertEqual(len(candidate_lines), 5)
            self.assertTrue(candidate_lines[0].startswith("task_id|panorama_id_start|"))
            self.assertTrue(all(line.count("|") == 10 for line in candidate_lines))


if __name__ == "__main__":
    unittest.main()
