from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from sa_nav import GroundingIndex, Observation, RenderedView, SpatialAlignmentRefiner
from sa_nav.cli.run_spatial_alignment import (
    load_manifest_observation,
    request_diagnostics,
    response_diagnostics,
    select_candidate_rooms,
)


HEADINGS = [330.0, 345.25, 60.0, 141.91, 150.0, 216.30, 240.0, 302.69]


def test_room_graph() -> dict[str, dict]:
    return {
        "Room 8": {
            "title": "Assyria: Nimrud",
            "category": "Middle East",
            "visual_profile": {
                "short_description": "Nimrud palace relief gallery.",
                "visual_cues": ["Nimrud relief panels", "lamassu"],
            },
            "neighbors": [
                {
                    "target_room_id": "Room 9",
                    "allocentric_direction": "north",
                    "allocentric_heading_deg": 0.0,
                },
                {
                    "target_room_id": "Room 4",
                    "allocentric_direction": "east",
                    "allocentric_heading_deg": 90.0,
                },
                {
                    "target_room_id": "Room 23",
                    "allocentric_direction": "west",
                    "allocentric_heading_deg": 270.0,
                },
            ],
        },
        "Room 9": {
            "title": "Assyria: Nineveh",
            "category": "Middle East",
            "visual_profile": {
                "short_description": "Nineveh palace relief gallery.",
                "visual_cues": ["Sennacherib relief panels"],
            },
            "neighbors": [
                {
                    "target_room_id": "Room 8",
                    "allocentric_direction": "south",
                    "allocentric_heading_deg": 180.0,
                }
            ],
        },
        "Room 4": {
            "title": "Egyptian sculpture",
            "category": "Ancient Egypt",
            "visual_profile": {
                "short_description": "Ancient Egyptian monumental sculpture gallery.",
                "visual_cues": ["large Egyptian royal statues", "hieroglyphic monuments"],
            },
            "neighbors": [
                {
                    "target_room_id": "Room 8",
                    "allocentric_direction": "west",
                    "allocentric_heading_deg": 270.0,
                }
            ],
        },
        "Room 23": {
            "title": "Greek and Roman sculpture",
            "category": "Ancient Greece and Rome",
            "visual_profile": {
                "short_description": "Greek and Roman marble sculpture gallery.",
                "visual_cues": ["marble statues"],
            },
            "neighbors": [
                {
                    "target_room_id": "Room 8",
                    "allocentric_direction": "east",
                    "allocentric_heading_deg": 90.0,
                }
            ],
        },
    }


def valid_model_output() -> dict:
    view_observations = []
    for index in range(8):
        view_observations.append(
            {
                "view_id": f"view_{index}",
                "themes": ["Assyrian reliefs"],
                "location_scope": "current_room",
                "room_hint": "Room 8",
                "summary": "Assyrian relief gallery.",
            }
        )
    view_observations[2] = {
        "view_id": "view_2",
        "themes": ["Egyptian sculpture"],
        "location_scope": "adjacent_room",
        "room_hint": "Room 4",
        "summary": "The eastern opening shows Egyptian monumental sculpture.",
    }
    return {
        "predicted_room_id": "Room 8",
        "confidence": 0.91,
        "view_0_allocentric_direction": "north",
        "evidence": [
            "view_0 matches the northern Room 9 opening.",
            "view_6 matches Room 23 west of Room 8.",
        ],
        "room_ranking": [
            {"room_id": "Room 8", "score": 0.91},
            {"room_id": "Room 9", "score": 0.22},
        ],
        "candidate_hypotheses": [
            {
                "room_id": "Room 8",
                "score": 0.91,
                "current_room_support": ["view_0 shows Nimrud reliefs"],
                "neighbor_support": ["view_6 matches Room 23 west of Room 8"],
                "contradictions": [],
                "directional_neighbor_checks": [
                    {
                        "allocentric_direction": "north",
                        "expected_neighbor_room_id": "Room 9",
                        "inspected_view_ids": ["view_0"],
                        "status": "specific_match",
                        "observed_room_hint": "Room 9",
                        "rationale": "view_0 shows the northern Nineveh gallery.",
                    },
                    {
                        "allocentric_direction": "east",
                        "expected_neighbor_room_id": "Room 4",
                        "inspected_view_ids": ["view_2"],
                        "status": "specific_match",
                        "observed_room_hint": "Room 4",
                        "rationale": "view_2 shows eastern Egyptian monumental sculpture.",
                    },
                    {
                        "allocentric_direction": "west",
                        "expected_neighbor_room_id": "Room 23",
                        "inspected_view_ids": ["view_6"],
                        "status": "specific_match",
                        "observed_room_hint": "Room 23",
                        "rationale": "view_6 shows western Greek and Roman sculpture.",
                    },
                ],
            },
            {
                "room_id": "Room 9",
                "score": 0.22,
                "current_room_support": ["view_2 shows Assyrian reliefs"],
                "neighbor_support": [],
                "contradictions": ["view_6 has no matching west neighbor"],
                "directional_neighbor_checks": [
                    {
                        "allocentric_direction": "south",
                        "expected_neighbor_room_id": "Room 8",
                        "inspected_view_ids": ["view_4"],
                        "status": "uncertain",
                        "observed_room_hint": None,
                        "rationale": "view_4 is insufficient to identify Room 8.",
                    },
                ],
            },
        ],
        "summary": "Room 8 and a north-facing view_0 best explain the panorama.",
        "sector_alignment": [
            {
                "view_id": "view_0",
                "allocentric_direction": "north",
                "matched_room_id": "Room 9",
                "matched_theme": "Assyria: Nineveh",
                "rationale": "The northern opening shows the Nineveh gallery.",
            },
            {
                "view_id": "view_6",
                "allocentric_direction": "west",
                "matched_room_id": "Room 23",
                "matched_theme": "Greek and Roman sculpture",
                "rationale": "The western opening shows marble sculpture.",
            },
        ],
        "view_observations": view_observations,
    }


class SingleCallSpatialAlignmentTests(unittest.TestCase):
    def build_observation(self, directory: Path, *, view_count: int = 8) -> Observation:
        views = []
        for index, heading in enumerate(HEADINGS[:view_count]):
            image_path = directory / f"secret_manifest_label_{index}.png"
            image_path.write_bytes(b"fake-image")
            views.append(
                RenderedView(
                    label=f"north_to_east_secret_{index}",
                    heading=heading,
                    path=str(image_path),
                )
            )
        return Observation(
            pano_id="pano-test",
            views=views,
            metadata={
                "ground_truth_room_id": "GroundTruthOnly",
                "visual_localization_score": 9.99,
            },
        )

    def test_refine_single_call_sends_eight_images_once_without_label_or_score_leakage(self) -> None:
        requests = []

        def response_client(request_body: dict) -> dict:
            requests.append(request_body)
            return {
                "output_text": json.dumps(valid_model_output()),
                "usage": {"input_tokens": 123, "output_tokens": 45},
            }

        with tempfile.TemporaryDirectory() as tmpdir:
            observation = self.build_observation(Path(tmpdir))
            refiner = SpatialAlignmentRefiner(
                room_graph=test_room_graph(),
                grounding_index=GroundingIndex({}),
                response_client=response_client,
                max_http_retries=1,
            )
            result = refiner.refine_single_call(
                observation=observation,
                candidate_room_ids=["Room 8", "Room 9"],
            )

        self.assertEqual(len(requests), 1)
        content = requests[0]["input"][0]["content"]
        image_blocks = [block for block in content if block.get("type") == "input_image"]
        text = "\n".join(
            str(block.get("text", ""))
            for block in content
            if block.get("type") == "input_text"
        )
        self.assertEqual(len(image_blocks), 8)
        diagnostic_request = request_diagnostics(requests[0])
        self.assertEqual(diagnostic_request["image_count"], 8)
        self.assertNotIn("data:image", json.dumps(diagnostic_request))
        self.assertIn("view_0: 330.000000 degrees", text)
        self.assertIn("Room 23", text)
        self.assertIn("is west of Room 8", text)
        self.assertIn("Room 4 is east of Room 8", text)
        self.assertIn("large Egyptian royal statues", text)
        self.assertIn("neighbor short description", text)
        self.assertIn("marble statues", text)
        self.assertIn("exact room title", requests[0]["instructions"])
        self.assertIn(
            "do not choose a current room first",
            requests[0]["instructions"].lower(),
        )
        self.assertIn("one best room_hint", requests[0]["instructions"])
        self.assertIn(
            "non-neighbor",
            text,
        )
        self.assertIn(
            "every listed neighbor direction",
            requests[0]["instructions"],
        )
        self.assertIn("directional_neighbor_checks", text)
        self.assertIn("candidate_hypotheses", text)
        self.assertNotIn("north_to_east_secret", text)
        self.assertNotIn("GroundTruthOnly", text)
        self.assertNotIn("9.99", text)
        self.assertNotIn("secret_manifest_label", text)
        self.assertIsNone(refiner.last_extraction_request_body)
        self.assertEqual(result["logical_vlm_call_count"], 1)
        self.assertEqual(result["alignment_predicted_room_id"], "Room 8")
        self.assertEqual(
            result["spatial_alignment"]["geocentric_to_allocentric_offset_deg"],
            30.0,
        )
        self.assertTrue(result["spatial_alignment"]["sector_alignment_consistent"])
        self.assertEqual(len(result["candidate_hypotheses"]), 2)

        schema = requests[0]["text"]["format"]["schema"]
        matched_room_enum = schema["properties"]["sector_alignment"]["items"]["properties"][
            "matched_room_id"
        ]["enum"]
        self.assertIn("Room 23", matched_room_enum)
        self.assertEqual(schema["properties"]["view_observations"]["minItems"], 8)
        self.assertEqual(
            schema["properties"]["candidate_hypotheses"]["minItems"],
            2,
        )
        hypothesis_properties = schema["properties"]["candidate_hypotheses"]["items"][
            "properties"
        ]
        self.assertIn("directional_neighbor_checks", hypothesis_properties)

    def test_gemini_relaxed_schema_keeps_contract_shape_without_complex_constraints(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "room": {"type": "string", "enum": ["Room 8", "Room 9"]},
                "views": {
                    "type": "array",
                    "minItems": 8,
                    "maxItems": 8,
                    "items": {
                        "type": "object",
                        "properties": {"view_id": {"type": "string"}},
                        "required": ["view_id"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["room", "views"],
            "additionalProperties": False,
        }
        relaxed = SpatialAlignmentRefiner._relax_json_schema(schema)
        serialized = json.dumps(relaxed)
        self.assertIn('"properties"', serialized)
        self.assertIn('"required"', serialized)
        for removed_key in ("enum", "minItems", "maxItems", "additionalProperties"):
            self.assertNotIn(f'"{removed_key}"', serialized)

    def test_response_diagnostics_reads_gemini_usage_metadata(self) -> None:
        diagnostics = response_diagnostics(
            {
                "output_text": "{}",
                "usageMetadata": {
                    "promptTokenCount": 123,
                    "candidatesTokenCount": 45,
                },
            }
        )
        self.assertEqual(diagnostics["output_text"], "{}")
        self.assertEqual(diagnostics["usage"]["promptTokenCount"], 123)

    def test_refine_single_call_rejects_non_eight_view_input_without_calling_vlm(self) -> None:
        requests = []
        with tempfile.TemporaryDirectory() as tmpdir:
            observation = self.build_observation(Path(tmpdir), view_count=7)
            refiner = SpatialAlignmentRefiner(
                room_graph=test_room_graph(),
                grounding_index=GroundingIndex({}),
                response_client=lambda request: requests.append(request) or {},
                max_http_retries=1,
            )
            with self.assertRaisesRegex(ValueError, "exactly 8"):
                refiner.refine_single_call(
                    observation=observation,
                    candidate_room_ids=["Room 8", "Room 9"],
                )
        self.assertEqual(requests, [])

    def test_refine_single_call_rejects_unknown_room_before_calling_vlm(self) -> None:
        requests = []
        with tempfile.TemporaryDirectory() as tmpdir:
            observation = self.build_observation(Path(tmpdir))
            refiner = SpatialAlignmentRefiner(
                room_graph=test_room_graph(),
                grounding_index=GroundingIndex({}),
                response_client=lambda request: requests.append(request) or {},
            )
            with self.assertRaisesRegex(ValueError, "Unknown candidate rooms"):
                refiner.refine_single_call(
                    observation=observation,
                    candidate_room_ids=["Room 8", "Room 404"],
                )
        self.assertEqual(requests, [])

    def test_refine_single_call_rejects_incomplete_structured_output(self) -> None:
        malformed = valid_model_output()
        malformed["view_observations"] = malformed["view_observations"][:-1]
        with tempfile.TemporaryDirectory() as tmpdir:
            observation = self.build_observation(Path(tmpdir))
            refiner = SpatialAlignmentRefiner(
                room_graph=test_room_graph(),
                grounding_index=GroundingIndex({}),
                response_client=lambda _: {"output_text": json.dumps(malformed)},
                max_http_retries=1,
            )
            with self.assertRaisesRegex(ValueError, "every panorama view"):
                refiner.refine_single_call(
                    observation=observation,
                    candidate_room_ids=["Room 8", "Room 9"],
                )

    def test_refine_single_call_rejects_incomplete_candidate_hypotheses(self) -> None:
        malformed = valid_model_output()
        malformed["candidate_hypotheses"] = malformed["candidate_hypotheses"][:-1]
        with tempfile.TemporaryDirectory() as tmpdir:
            observation = self.build_observation(Path(tmpdir))
            refiner = SpatialAlignmentRefiner(
                room_graph=test_room_graph(),
                grounding_index=GroundingIndex({}),
                response_client=lambda _: {"output_text": json.dumps(malformed)},
                max_http_retries=1,
            )
            with self.assertRaisesRegex(ValueError, "candidate_hypotheses"):
                refiner.refine_single_call(
                    observation=observation,
                    candidate_room_ids=["Room 8", "Room 9"],
                )

    def test_refine_single_call_requires_every_directional_neighbor_check(self) -> None:
        malformed = valid_model_output()
        malformed["candidate_hypotheses"][0]["directional_neighbor_checks"] = [
            malformed["candidate_hypotheses"][0]["directional_neighbor_checks"][0]
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            observation = self.build_observation(Path(tmpdir))
            refiner = SpatialAlignmentRefiner(
                room_graph=test_room_graph(),
                grounding_index=GroundingIndex({}),
                response_client=lambda _: {"output_text": json.dumps(malformed)},
                max_http_retries=1,
            )
            with self.assertRaisesRegex(ValueError, "every listed neighbor of Room 8"):
                refiner.refine_single_call(
                    observation=observation,
                    candidate_room_ids=["Room 8", "Room 9"],
                )

    def test_candidate_selection_is_deterministic_and_requires_cache(self) -> None:
        scores = {
            "visual_localization": {
                "room_scores": [
                    {"room_id": "Room 9", "score": 8.0, "reason": "nine"},
                    {"room_id": "Room 23", "score": 7.0, "reason": "sculpture"},
                    {"room_id": "Room 8", "score": 8.0, "reason": "nimrud"},
                ]
            }
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            scores_path = Path(tmpdir) / "scores.json"
            scores_path.write_text(json.dumps(scores), encoding="utf-8")
            selected = select_candidate_rooms(
                scores_path=scores_path,
                room_graph=test_room_graph(),
                top_k=3,
            )
            self.assertEqual(
                [record["room_id"] for record in selected],
                ["Room 8", "Room 9", "Room 23"],
            )
            with self.assertRaisesRegex(FileNotFoundError, "will not run perception"):
                select_candidate_rooms(
                    scores_path=Path(tmpdir) / "missing.json",
                    room_graph=test_room_graph(),
                    top_k=3,
                )
            with self.assertRaisesRegex(ValueError, "at least 2"):
                select_candidate_rooms(
                    scores_path=scores_path,
                    room_graph=test_room_graph(),
                    top_k=1,
                )

    def test_manifest_loader_rejects_capture_count_and_missing_images(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            manifest_path = directory / "pano-test_manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "pano_id": "pano-test",
                        "captures": [
                            {"label": "north", "heading": heading, "path": "missing.png"}
                            for heading in HEADINGS[:7]
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "exactly 8"):
                load_manifest_observation(
                    manifest_path=manifest_path,
                    pano_id="pano-test",
                )

            manifest_path.write_text(
                json.dumps(
                    {
                        "pano_id": "pano-test",
                        "captures": [
                            {"label": "north", "heading": heading, "path": "missing.png"}
                            for heading in HEADINGS
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(FileNotFoundError, "Panorama image not found"):
                load_manifest_observation(
                    manifest_path=manifest_path,
                    pano_id="pano-test",
                )


if __name__ == "__main__":
    unittest.main()
