from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from st_nav_data.pano_room_grounding import rebuild_pano_room_grounding_from_batches


class PanoRoomGroundingRebuildTests(unittest.TestCase):
    def test_rebuild_from_batches_prefers_accepted_manual_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            batch_dir = Path(tmpdir)
            (batch_dir / "floor0_batch_0000_002.json").write_text(
                json.dumps(
                    {
                        "results": [
                            {"pano_id": "pano-a", "predicted_room_id": "Room 4"},
                            {"pano_id": "pano-b", "predicted_room_id": "Room 7"},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            (batch_dir / "floor0_batch_0000_002.manual.json").write_text(
                json.dumps(
                    {
                        "results": [
                            {"pano_id": "pano-b", "manual_status": "accepted", "manual_room_id": "Room 8"},
                            {"pano_id": "pano-c", "manual_status": "accepted", "manual_room_id": "Room 9"},
                        ]
                    }
                ),
                encoding="utf-8",
            )

            payload = rebuild_pano_room_grounding_from_batches(batch_dir)

        self.assertEqual(payload["summary"]["batch_file_count"], 1)
        self.assertEqual(payload["summary"]["manual_batch_file_count"], 1)
        self.assertEqual(payload["summary"]["grounding_result_count"], 2)
        self.assertEqual(payload["summary"]["manual_record_count"], 2)
        self.assertEqual(payload["mappings"], {"pano-a": "Room 4", "pano-b": "Room 8"})
        self.assertEqual(payload["sources"], {"pano-a": "gemini", "pano-b": "manual:accepted"})

    def test_rebuild_from_batches_lets_extra_manual_files_override_batch_manual_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            batch_dir = Path(tmpdir) / "batches"
            batch_dir.mkdir()
            root_manual = Path(tmpdir) / "room_grounding.manual.json"
            (batch_dir / "floor0_batch_0000_001.json").write_text(
                json.dumps({"results": [{"pano_id": "pano-a", "predicted_room_id": "Room 4"}]}),
                encoding="utf-8",
            )
            (batch_dir / "floor0_batch_0000_001.manual.json").write_text(
                json.dumps({"results": [{"pano_id": "pano-a", "manual_status": "accepted", "manual_room_id": "Room 8"}]}),
                encoding="utf-8",
            )
            root_manual.write_text(
                json.dumps({"results": [{"pano_id": "pano-a", "manual_status": "accepted", "manual_room_id": "Room 9"}]}),
                encoding="utf-8",
            )

            payload = rebuild_pano_room_grounding_from_batches(batch_dir, manual_paths=[root_manual])

        self.assertEqual(payload["summary"]["extra_manual_file_count"], 1)
        self.assertEqual(payload["mappings"], {"pano-a": "Room 9"})
        self.assertEqual(payload["sources"], {"pano-a": "manual:accepted"})

    def test_rebuild_from_batches_includes_extra_manual_only_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            batch_dir = Path(tmpdir) / "batches"
            batch_dir.mkdir()
            root_manual = Path(tmpdir) / "room_grounding.manual.json"
            (batch_dir / "floor0_batch_0000_001.json").write_text(
                json.dumps({"results": [{"pano_id": "pano-a", "predicted_room_id": "Room 4"}]}),
                encoding="utf-8",
            )
            root_manual.write_text(
                json.dumps(
                    {
                        "results": [
                            {"pano_id": "pano-b", "manual_status": "accepted", "manual_room_id": "Room 18"},
                            {"pano_id": "pano-c", "manual_status": "accepted", "manual_room_id": None},
                            {"pano_id": "pano-d", "manual_status": "pending", "manual_room_id": "Room 18"},
                        ]
                    }
                ),
                encoding="utf-8",
            )

            payload = rebuild_pano_room_grounding_from_batches(batch_dir, manual_paths=[root_manual])

        self.assertEqual(payload["summary"]["extra_manual_only_record_count"], 3)
        self.assertEqual(
            payload["mappings"],
            {"pano-a": "Room 4", "pano-b": "Room 18", "pano-c": "null"},
        )
        self.assertEqual(payload["sources"]["pano-b"], "manual:accepted")
        self.assertEqual(payload["sources"]["pano-c"], "manual:accepted")


if __name__ == "__main__":
    unittest.main()
