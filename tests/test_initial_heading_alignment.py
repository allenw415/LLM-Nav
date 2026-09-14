from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from st_nav.common.types import Observation, RenderedView
from st_nav.spatial.engine import SpatialEngine
from st_nav.spatial.grounding import GroundingIndex
from st_nav.spatial.localization import SpatialAlignmentRefiner


class InitialHeadingAlignmentTest(unittest.TestCase):
    def setUp(self) -> None:
        self.room_graph = {
            "Room 1": {
                "floor": "0",
                "neighbors": [
                    {
                        "target_room_id": "Room 2",
                        "allocentric_heading_deg": 90.0,
                    }
                ],
            },
            "Room 2": {
                "floor": "0",
                "neighbors": [
                    {
                        "target_room_id": "Room 1",
                        "allocentric_heading_deg": 270.0,
                    }
                ],
            },
        }
        self.pano_graph = {
            "pano-1": {
                "neighbors": [
                    {
                        "target_pano_id": "pano-2",
                        "geocentric_heading_deg": 60.0,
                    }
                ]
            },
            "pano-2": {
                "neighbors": [
                    {
                        "target_pano_id": "pano-1",
                        "geocentric_heading_deg": 240.0,
                    }
                ]
            },
        }
        self.grounding = GroundingIndex(
            {},
            pano_to_room={"pano-1": "Room 1", "pano-2": "Room 2"},
        )

    def _engine(self) -> SpatialEngine:
        return SpatialEngine(
            room_graph=self.room_graph,
            pano_graph=self.pano_graph,
            grounding_index=self.grounding,
            heading_alignment_mode="initial-spatial",
        )

    def test_episode_offset_is_initialized_once_and_used_by_candidates(self) -> None:
        engine = self._engine()
        state = engine.initialize(
            start_pano_id="pano-1",
            start_room_id="Room 1",
            start_heading=0.0,
        )
        self.assertIsNone(state.geocentric_to_allocentric_offset_deg)
        self.assertEqual(state.heading_alignment_status, "pending")

        first = Observation(
            pano_id="pano-1",
            metadata={
                "spatial_alignment": {
                    "view_0_geocentric_heading_deg": 330.0,
                    "view_0_allocentric_direction": "north",
                    "geocentric_to_allocentric_offset_deg": 30.0,
                    "alignment_confidence": 0.9,
                }
            },
        )
        self.assertTrue(
            engine.initialize_heading_alignment(state, first, step_index=0)
        )
        self.assertEqual(state.geocentric_to_allocentric_offset_deg, 30.0)
        self.assertEqual(state.heading_alignment_source_step, 0)

        later = Observation(
            pano_id="pano-1",
            metadata={
                "spatial_alignment": {
                    "geocentric_to_allocentric_offset_deg": 120.0,
                }
            },
        )
        self.assertTrue(
            engine.initialize_heading_alignment(state, later, step_index=3)
        )
        self.assertEqual(state.geocentric_to_allocentric_offset_deg, 30.0)
        self.assertEqual(state.heading_alignment_source_step, 0)

        candidates = engine.generate_candidates(
            state,
            ["Room 1", "Room 2"],
        )
        self.assertEqual(candidates[0].relative_heading, 90.0)
        self.assertEqual(candidates[0].metadata["heading_offset_used_deg"], 30.0)

    def test_missing_initial_offset_does_not_fall_back_to_fixed_offset(self) -> None:
        engine = self._engine()
        state = engine.initialize(
            start_pano_id="pano-1",
            start_room_id="Room 1",
            start_heading=0.0,
        )
        observation = Observation(
            pano_id="pano-1",
            metadata={"alignment_skipped_reason": "invalid_alignment"},
        )

        self.assertFalse(
            engine.initialize_heading_alignment(state, observation, step_index=0)
        )
        self.assertIsNone(state.geocentric_to_allocentric_offset_deg)
        self.assertEqual(state.heading_alignment_status, "failed")
        self.assertEqual(
            state.heading_alignment_failure_reason,
            "invalid_alignment",
        )

    def test_two_stage_refiner_derives_offset_and_records_vlm_timing(self) -> None:
        responses = [
            {
                "output_text": json.dumps(
                    {
                        "views": [
                            {
                                "view_id": "view_0",
                                "themes": [],
                                "summary": "north-facing gallery view",
                            }
                        ],
                        "summary": "context extracted",
                    }
                )
            },
            {
                "output_text": json.dumps(
                    {
                        "predicted_room_id": "Room 1",
                        "confidence": 0.9,
                        "view_0_allocentric_direction": "north",
                        "evidence": ["map and view agree"],
                        "room_ranking": [
                            {"room_id": "Room 1", "score": 0.9},
                            {"room_id": "Room 2", "score": 0.1},
                        ],
                        "summary": "aligned",
                    }
                )
            },
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "view.png"
            image_path.write_bytes(b"fake-image")
            refiner = SpatialAlignmentRefiner(
                room_graph=self.room_graph,
                grounding_index=self.grounding,
                response_client=lambda _: responses.pop(0),
            )
            result = refiner.refine(
                observation=Observation(
                    pano_id="pano-1",
                    views=[
                        RenderedView(
                            label="north",
                            heading=330.0,
                            path=str(image_path),
                        )
                    ],
                    metadata={"floor": "0"},
                ),
                candidate_room_ids=["Room 1", "Room 2"],
            )

        alignment = result["spatial_alignment"]
        self.assertEqual(alignment["geocentric_to_allocentric_offset_deg"], 30.0)
        self.assertEqual(alignment["view_0_geocentric_heading_deg"], 330.0)
        self.assertEqual(result["logical_vlm_call_count"], 2)
        self.assertEqual(result["spatial_timing"]["logical_vlm_call_count"], 2)
        self.assertGreaterEqual(result["spatial_timing"]["vlm_response_ms"], 0.0)

    def test_two_stage_refiner_uses_four_cardinal_images_and_can_predict_zero_offset(self) -> None:
        responses = [
            {
                "output_text": json.dumps(
                    {
                        "views": [
                            {
                                "view_id": f"view_{index}",
                                "themes": [],
                                "summary": f"sector {index}",
                            }
                            for index in range(4)
                        ],
                        "summary": "four-view context extracted",
                    }
                )
            },
            {
                "output_text": json.dumps(
                    {
                        "predicted_room_id": "Room 1",
                        "confidence": 0.8,
                        "view_0_allocentric_direction": "north",
                        "evidence": ["four-view map match"],
                        "room_ranking": [
                            {"room_id": "Room 1", "score": 0.8},
                            {"room_id": "Room 2", "score": 0.2},
                        ],
                        "summary": "aligned to the closest cardinal hypothesis",
                    }
                )
            },
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            image_paths = []
            for index in range(4):
                image_path = Path(tmpdir) / f"view_{index}.png"
                image_path.write_bytes(b"fake-image")
                image_paths.append(image_path)
            refiner = SpatialAlignmentRefiner(
                room_graph=self.room_graph,
                grounding_index=self.grounding,
                response_client=lambda _: responses.pop(0),
            )
            result = refiner.refine(
                observation=Observation(
                    pano_id="pano-1",
                    views=[
                        RenderedView(
                            label=f"view_{index}",
                            heading=float(index * 90),
                            path=str(image_path),
                        )
                        for index, image_path in enumerate(image_paths)
                    ],
                    metadata={"floor": "0", "heading_mode": "cardinal4"},
                ),
                candidate_room_ids=["Room 1", "Room 2"],
            )

        extraction_content = refiner.last_extraction_request_body["input"][0]["content"]
        self.assertEqual(
            sum(item.get("type") == "input_image" for item in extraction_content),
            4,
        )
        self.assertIn(
            "These four sectors are spaced exactly 90 degrees apart",
            refiner.last_alignment_request_body["input"],
        )
        alignment = result["spatial_alignment"]
        self.assertEqual(alignment["view_0_geocentric_heading_deg"], 0.0)
        self.assertEqual(alignment["view_0_allocentric_direction"], "north")
        self.assertEqual(alignment["geocentric_to_allocentric_offset_deg"], 0.0)


if __name__ == "__main__":
    unittest.main()
