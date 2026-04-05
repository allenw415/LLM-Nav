from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from st_nav.room_grounder import (
    build_compact_pano_room_mapping,
    GeminiRoomGrounder,
    build_manual_annotation_records,
    build_room_candidates,
    collect_manual_seed_panos,
    collect_seed_panos_for_rooms,
    expand_seed_panos_by_region_growing,
    expand_seed_panos_by_hops,
    invert_room_grounding,
    merge_seed_panos_by_room,
    merge_records_by_pano_id,
    select_grounding_captures,
)


class RoomGroundingTests(unittest.TestCase):
    def test_invert_room_grounding_groups_rooms_by_pano(self) -> None:
        grounding = {
            "Room 18": {"pano_ids": ["pano-shared", "pano-18"]},
            "Room 18a": {"pano_ids": ["pano-shared"]},
            "Room 23": {"pano_ids": ["pano-23"]},
        }

        pano_to_rooms = invert_room_grounding(grounding)

        self.assertEqual(pano_to_rooms["pano-shared"], ["Room 18", "Room 18a"])
        self.assertEqual(pano_to_rooms["pano-23"], ["Room 23"])

    def test_build_room_candidates_can_filter_to_same_floor(self) -> None:
        room_graph = {
            "Room 8": {
                "display_name": "Room 8",
                "floor": "0",
                "title": "Assyria: Nimrud",
                "category": "Middle East",
                "aliases": ["Room 8"],
            },
            "Room 61": {
                "display_name": "Room 61",
                "floor": "3",
                "title": "Clockmaker's Museum",
                "category": "Britain",
                "aliases": ["Room 61"],
            },
        }
        grounding = {
            "Room 8": {
                "aliases": ["Assyria: Nimrud", "Room 8"],
                "anchor_entities": ["Lamassu", "Relief panels"],
            }
        }

        candidates = build_room_candidates(room_graph, grounding, floor="0", same_floor_only=True)

        self.assertEqual([candidate["room_id"] for candidate in candidates], ["Room 8"])
        self.assertEqual(candidates[0]["aliases"], ["Room 8", "Assyria: Nimrud"])
        self.assertEqual(candidates[0]["anchor_entities"], ["Lamassu", "Relief panels"])

    def test_grounder_uses_cached_output_when_available(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            image_path = tmp_path / "view.png"
            image_path.write_bytes(b"fake-image")
            manifest_path = tmp_path / "sample_manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "pano_id": "pano-8",
                        "floor": "0",
                        "heading_mode": "museum",
                        "captures": [{"label": "north", "heading": 330.0, "path": str(image_path)}],
                    }
                ),
                encoding="utf-8",
            )
            cached_output_path = tmp_path / "sample_manifest_room_grounding.json"
            cached_output_path.write_text(
                json.dumps(
                    {
                        "pano_id": "pano-8",
                        "floor": "0",
                        "manifest_path": str(manifest_path),
                        "candidate_room_ids": ["Room 8"],
                        "predicted_room_id": "Room 8",
                        "confidence": 0.95,
                        "evidence": ["large relief panels"],
                        "alternative_room_ids": [],
                        "summary": "Looks like Room 8.",
                    }
                ),
                encoding="utf-8",
            )

            def fail_if_called(_: dict) -> dict:
                raise AssertionError("response_client should not be called when cached output exists")

            grounder = GeminiRoomGrounder(response_client=fail_if_called, use_grounding_files=True)
            result = grounder.ground(
                manifest_path,
                room_graph={"Room 8": {"display_name": "Room 8", "floor": "0"}},
                room_grounding={"Room 8": {"pano_ids": ["pano-8"]}},
            )

            self.assertEqual(result["predicted_room_id"], "Room 8")
            self.assertEqual(result["confidence"], 0.95)

    def test_grounder_normalizes_prediction_against_candidate_room_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            image_path = tmp_path / "view.png"
            image_path.write_bytes(b"fake-image")
            manifest_path = tmp_path / "sample_manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "pano_id": "pano-8",
                        "floor": "0",
                        "heading_mode": "museum",
                        "captures": [{"label": "north", "heading": 330.0, "path": str(image_path)}],
                    }
                ),
                encoding="utf-8",
            )

            def fake_response(_: dict) -> dict:
                payload = {
                    "predicted_room_id": "Room 999",
                    "confidence": 0.62,
                    "evidence": ["stone reliefs", "assyrian guardian figure"],
                    "alternative_room_ids": ["Room 23", "Room 8", "Room 23"],
                    "summary": "Visual features look closer to the Assyria gallery.",
                }
                return {"candidates": [{"content": {"parts": [{"text": json.dumps(payload)}]}}]}

            room_graph = {
                "Room 8": {
                    "display_name": "Room 8",
                    "floor": "0",
                    "title": "Assyria: Nimrud",
                    "category": "Middle East",
                    "aliases": ["Room 8"],
                },
                "Room 23": {
                    "display_name": "Room 23",
                    "floor": "0",
                    "title": "Greek and Roman sculpture",
                    "category": "Ancient Greece and Rome",
                    "aliases": ["Room 23"],
                },
            }
            grounding = {
                "Room 8": {"aliases": ["Assyria: Nimrud"], "anchor_entities": ["Lamassu"]},
                "Room 23": {"aliases": ["Greek and Roman sculpture"], "anchor_entities": ["Marble statues"]},
            }

            grounder = GeminiRoomGrounder(response_client=fake_response, use_grounding_files=False)
            result = grounder.ground(manifest_path, room_graph=room_graph, room_grounding=grounding)

            self.assertIsNone(result["predicted_room_id"])
            self.assertEqual(result["alternative_room_ids"], ["Room 23", "Room 8"])
            self.assertEqual(result["candidate_room_ids"], ["Room 23", "Room 8"])
            self.assertEqual(result["evidence"], ["stone reliefs", "assyrian guardian figure"])

    def test_select_grounding_captures_prefers_four_cardinal_views(self) -> None:
        captures = [
            {"label": "north", "path": "n.png"},
            {"label": "north_to_east", "path": "ne.png"},
            {"label": "east", "path": "e.png"},
            {"label": "east_to_south", "path": "es.png"},
            {"label": "south", "path": "s.png"},
            {"label": "south_to_west", "path": "sw.png"},
            {"label": "west", "path": "w.png"},
            {"label": "west_to_north", "path": "wn.png"},
        ]

        selected = select_grounding_captures(captures, max_captures=4)

        self.assertEqual([capture["label"] for capture in selected], ["north", "east", "south", "west"])

    def test_collect_seed_panos_reports_missing_rooms(self) -> None:
        seed_panos_by_room, missing_rooms = collect_seed_panos_for_rooms(
            {
                "Room 7": {"pano_ids": ["pano-7"]},
                "Room 18": {"pano_ids": []},
            },
            ["Room 7", "Room 18", "Room 23"],
        )

        self.assertEqual(seed_panos_by_room, {"Room 7": ["pano-7"]})
        self.assertEqual(missing_rooms, ["Room 18", "Room 23"])

    def test_expand_seed_panos_by_hops_tracks_nearest_seed_rooms(self) -> None:
        pano_graph = {
            "pano-7": {"pano_id": "pano-7", "floor": "0", "neighbors": [{"target_pano_id": "mid"}]},
            "pano-8": {"pano_id": "pano-8", "floor": "0", "neighbors": [{"target_pano_id": "mid"}]},
            "mid": {
                "pano_id": "mid",
                "floor": "0",
                "neighbors": [{"target_pano_id": "pano-7"}, {"target_pano_id": "pano-8"}, {"target_pano_id": "far"}],
            },
            "far": {"pano_id": "far", "floor": "0", "neighbors": [{"target_pano_id": "mid"}]},
            "other-floor": {"pano_id": "other-floor", "floor": "1", "neighbors": []},
        }

        expanded = expand_seed_panos_by_hops(
            pano_graph,
            {"Room 7": ["pano-7"], "Room 8": ["pano-8"]},
            max_hops=1,
            floor="0",
        )

        self.assertEqual(expanded["pano-7"]["seed_distance_hops"], 0)
        self.assertEqual(expanded["pano-8"]["seed_distance_hops"], 0)
        self.assertEqual(expanded["mid"]["seed_distance_hops"], 1)
        self.assertEqual(expanded["mid"]["nearest_seed_room_ids"], ["Room 7", "Room 8"])
        self.assertNotIn("far", expanded)
        self.assertNotIn("other-floor", expanded)

    def test_region_growing_only_expands_matching_high_confidence_frontiers(self) -> None:
        pano_graph = {
            "seed-7": {"pano_id": "seed-7", "floor": "0", "neighbors": [{"target_pano_id": "mid"}]},
            "seed-8": {"pano_id": "seed-8", "floor": "0", "neighbors": [{"target_pano_id": "mid"}]},
            "mid": {"pano_id": "mid", "floor": "0", "neighbors": [{"target_pano_id": "far"}]},
            "far": {"pano_id": "far", "floor": "0", "neighbors": []},
        }
        classifications = {
            "seed-7": {"predicted_room_id": "Room 7", "confidence": 0.95},
            "seed-8": {"predicted_room_id": "Room 8", "confidence": 0.95},
            "mid": {"predicted_room_id": "Room 7", "confidence": 0.85},
            "far": {"predicted_room_id": "Room 7", "confidence": 0.9},
        }

        expanded = expand_seed_panos_by_region_growing(
            pano_graph,
            {"Room 7": ["seed-7"], "Room 8": ["seed-8"]},
            classify_pano=lambda pano_id: classifications[pano_id],
            max_depth=2,
            floor="0",
            min_confidence=0.8,
        )

        self.assertEqual(expanded["mid"]["frontier_room_ids"], ["Room 7", "Room 8"])
        self.assertEqual(expanded["mid"]["expansion_room_ids"], ["Room 7"])
        self.assertIn("far", expanded)
        self.assertEqual(expanded["far"]["frontier_room_ids"], ["Room 7"])

    def test_region_growing_stops_on_low_confidence_match(self) -> None:
        pano_graph = {
            "seed-7": {"pano_id": "seed-7", "floor": "0", "neighbors": [{"target_pano_id": "mid"}]},
            "mid": {"pano_id": "mid", "floor": "0", "neighbors": [{"target_pano_id": "far"}]},
            "far": {"pano_id": "far", "floor": "0", "neighbors": []},
        }
        classifications = {
            "seed-7": {"predicted_room_id": "Room 7", "confidence": 0.95},
            "mid": {"predicted_room_id": "Room 7", "confidence": 0.6},
        }

        expanded = expand_seed_panos_by_region_growing(
            pano_graph,
            {"Room 7": ["seed-7"]},
            classify_pano=lambda pano_id: classifications[pano_id],
            max_depth=2,
            floor="0",
            min_confidence=0.8,
        )

        self.assertIn("mid", expanded)
        self.assertEqual(expanded["mid"]["expansion_room_ids"], [])
        self.assertNotIn("far", expanded)

    def test_merge_records_by_pano_id_preserves_existing_and_updates_latest(self) -> None:
        merged = merge_records_by_pano_id(
            [
                {"pano_id": "pano-1", "predicted_room_id": "Room 7", "notes": "old"},
                {"pano_id": "pano-2", "predicted_room_id": "Room 8"},
            ],
            [
                {"pano_id": "pano-1", "predicted_room_id": "Room 8", "confidence": 0.9},
                {"pano_id": "pano-3", "predicted_room_id": "Room 9"},
            ],
        )

        self.assertEqual([record["pano_id"] for record in merged], ["pano-1", "pano-2", "pano-3"])
        self.assertEqual(merged[0]["predicted_room_id"], "Room 8")
        self.assertEqual(merged[0]["notes"], "old")
        self.assertEqual(merged[0]["confidence"], 0.9)

    def test_build_manual_annotation_records_preserves_manual_fields(self) -> None:
        records = build_manual_annotation_records(
            [
                {
                    "pano_id": "pano-1",
                    "manifest_path": "/tmp/pano-1.json",
                    "predicted_room_id": "Room 4",
                    "confidence": 0.95,
                    "alternative_room_ids": [],
                    "summary": "Egyptian sculpture",
                    "region_depth": 0,
                    "frontier_room_ids": ["Room 4"],
                    "expansion_room_ids": ["Room 4"],
                }
            ],
            existing_manual_records=[
                {
                    "pano_id": "pano-1",
                    "manual_status": "accepted",
                    "manual_room_id": "Room 4",
                    "notes": "checked",
                }
            ],
            min_confidence=0.75,
        )

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["manual_status"], "accepted")
        self.assertEqual(records[0]["manual_room_id"], "Room 4")
        self.assertEqual(records[0]["notes"], "checked")
        self.assertFalse(records[0]["needs_review"])

    def test_collect_manual_seed_panos_uses_only_accepted_records(self) -> None:
        seed_panos_by_room = collect_manual_seed_panos(
            [
                {"pano_id": "pano-4a", "manual_status": "accepted", "manual_room_id": "Room 4"},
                {"pano_id": "pano-4b", "manual_status": "pending", "manual_room_id": "Room 4"},
                {"pano_id": "pano-8a", "manual_status": "accepted", "manual_room_id": "Room 8"},
                {"pano_id": "pano-x", "manual_status": "accepted", "manual_room_id": ""},
            ],
            room_ids=["Room 4", "Room 8"],
        )

        self.assertEqual(seed_panos_by_room, {"Room 4": ["pano-4a"], "Room 8": ["pano-8a"]})

    def test_merge_seed_panos_by_room_deduplicates_across_sources(self) -> None:
        merged = merge_seed_panos_by_room(
            {"Room 4": ["pano-1"], "Room 8": ["pano-8a"]},
            {"Room 4": ["pano-2", "pano-1"]},
        )

        self.assertEqual(merged, {"Room 4": ["pano-1", "pano-2"], "Room 8": ["pano-8a"]})

    def test_build_compact_pano_room_mapping_prefers_manual_labels(self) -> None:
        compact = build_compact_pano_room_mapping(
            [
                {"pano_id": "pano-1", "predicted_room_id": "Room 4"},
                {"pano_id": "pano-2", "predicted_room_id": "Room 7"},
            ],
            manual_records=[
                {"pano_id": "pano-2", "manual_status": "accepted", "manual_room_id": "Room 8"},
                {"pano_id": "pano-3", "manual_status": "pending", "manual_room_id": "Room 9"},
            ],
        )

        self.assertEqual(compact["mappings"], {"pano-1": "Room 4", "pano-2": "Room 8"})
        self.assertEqual(compact["sources"], {"pano-1": "gemini", "pano-2": "manual:accepted"})


if __name__ == "__main__":
    unittest.main()
