from __future__ import annotations

import json
from pathlib import Path
import unittest

from tools.evaluation.navigation.rerun_room4_room9_affected import (
    DEFAULT_DATASET,
    DEFAULT_SOURCE_RESULTS,
    DEFAULT_VISUAL_ROOM_GRAPH,
    merge_authorized_replacements,
    select_affected_cases,
    validate_corrected_room_edge,
    validate_expected_selection,
)


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


class Room4Room9RerunTests(unittest.TestCase):
    def test_real_selection_is_the_expected_26_33_34_union(self) -> None:
        selection = select_affected_cases(
            _load_jsonl(DEFAULT_DATASET),
            _load_jsonl(DEFAULT_SOURCE_RESULTS),
        )

        validate_expected_selection(selection)
        self.assertEqual(selection["reference_only_test_ids"], ["TEST294"])
        self.assertEqual(
            selection["runtime_only_test_ids"],
            [
                "TEST074",
                "TEST112",
                "TEST150",
                "TEST193",
                "TEST217",
                "TEST244",
                "TEST460",
                "TEST483",
            ],
        )

    def test_corrected_visual_room_graph_has_expected_bidirectional_headings(self) -> None:
        room_graph = json.loads(DEFAULT_VISUAL_ROOM_GRAPH.read_text(encoding="utf-8"))
        validate_corrected_room_edge(room_graph)

    def test_merge_replaces_only_authorized_records_and_preserves_order(self) -> None:
        dataset = [
            {"test_id": "TEST001", "path_id": "P1"},
            {"test_id": "TEST002", "path_id": "P2"},
        ]
        source = [
            {"test_id": "TEST001", "path_id": "P1", "success": False, "spl": 0.0},
            {"test_id": "TEST002", "path_id": "P2", "success": True, "spl": 0.5},
        ]
        replacement = [
            {
                "test_id": "TEST001",
                "path_id": "P1",
                "success": True,
                "spl": 0.75,
                "error": None,
            }
        ]

        merged, audit = merge_authorized_replacements(
            dataset,
            source,
            replacement,
            authorized_test_ids=["TEST001"],
        )

        self.assertEqual([record["test_id"] for record in merged], ["TEST001", "TEST002"])
        self.assertEqual(merged[0], replacement[0])
        self.assertEqual(merged[1], source[1])
        self.assertEqual(len(audit), 1)

    def test_merge_rejects_missing_extra_path_mismatch_and_error(self) -> None:
        dataset = [{"test_id": "TEST001", "path_id": "P1"}]
        source = [{"test_id": "TEST001", "path_id": "P1", "success": False}]

        with self.assertRaisesRegex(ValueError, "authorized set"):
            merge_authorized_replacements(
                dataset,
                source,
                [],
                authorized_test_ids=["TEST001"],
            )
        with self.assertRaisesRegex(ValueError, "path_id mismatch"):
            merge_authorized_replacements(
                dataset,
                source,
                [{"test_id": "TEST001", "path_id": "wrong", "error": None}],
                authorized_test_ids=["TEST001"],
            )
        with self.assertRaisesRegex(ValueError, "still contains an error"):
            merge_authorized_replacements(
                dataset,
                source,
                [{"test_id": "TEST001", "path_id": "P1", "error": {"type": "x"}}],
                authorized_test_ids=["TEST001"],
            )


if __name__ == "__main__":
    unittest.main()
