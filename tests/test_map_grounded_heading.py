from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from st_nav.experiments.map_grounded_heading import (
    DIRECTION_TO_HEADING,
    MapGroundedHeadingExperiment,
    build_heading_schema,
    describe_plain_heading_map,
    render_passage_heading_map,
    validate_heading_output,
)


def mini_regions() -> dict:
    return {
        "coordinate_system": "normalized_0_1",
        "north_up": True,
        "rooms": {
            "Room A": {
                "polygons": [[[0.1, 0.2], [0.45, 0.2], [0.45, 0.8], [0.1, 0.8]]],
                "label_anchor": [0.275, 0.5],
            },
            "Room B": {
                "polygons": [[[0.55, 0.2], [0.9, 0.2], [0.9, 0.8], [0.55, 0.8]]],
                "label_anchor": [0.725, 0.5],
            },
        },
    }


def valid_output() -> dict:
    return {
        "current_room_id": "Room A",
        "target_room_id": "Room B",
        "passage_direction": "E",
        "heading_deg": 90.0,
        "confidence": 0.85,
        "ambiguity_level": "low",
        "passage_description": "The connecting opening is on the east side of Room A.",
        "summary": "Travel east from Room A toward Room B.",
    }


class MapGroundedHeadingTests(unittest.TestCase):
    def test_renderer_highlights_only_current_and_target_rooms(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            base_map = directory / "map.png"
            output = directory / "semantic_map.png"
            Image.new("RGBA", (400, 300), (235, 235, 235, 255)).save(base_map)
            diagnostics = render_passage_heading_map(
                base_map_path=base_map,
                region_payload=mini_regions(),
                current_room_id="Room A",
                target_room_id="Room B",
                output_path=output,
            )
            with Image.open(output) as image:
                size = image.size
        self.assertEqual(size, (930, 300))
        self.assertEqual(diagnostics["spatial_inputs"], ["map_image", "current_room_id", "target_room_id"])
        self.assertFalse(diagnostics["graph_heading_rendered"])
        self.assertFalse(diagnostics["topology_text_rendered"])
        self.assertFalse(diagnostics["panorama_rendered"])

    def test_plain_map_reuses_original_image_without_rendered_aids(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            map_path = Path(tmpdir) / "original.png"
            Image.new("RGB", (320, 240), (210, 210, 210)).save(map_path)
            diagnostics = describe_plain_heading_map(
                base_map_path=map_path,
                current_room_id="Room A",
                target_room_id="Room B",
            )
            experiment = MapGroundedHeadingExperiment(response_client=lambda _: {})
            request, prompt = experiment.build_request_body(
                current_room_id="Room A",
                target_room_id="Room B",
                semantic_map_path=map_path,
                map_mode="plain",
            )

        self.assertEqual(diagnostics["path"], str(map_path))
        self.assertEqual(diagnostics["map_mode"], "plain")
        self.assertFalse(diagnostics["room_highlights_rendered"])
        self.assertFalse(diagnostics["compass_sidebar_rendered"])
        content = request["input"][0]["content"]
        self.assertEqual(sum(block.get("type") == "input_image" for block in content), 1)
        request_text = request["instructions"] + "\n" + prompt
        self.assertIn("current_room_id: Room A", request_text)
        self.assertIn("target_room_id: Room B", request_text)
        self.assertIn("original unmodified", request_text)
        for forbidden in ("blue", "green", "highlighted"):
            self.assertNotIn(forbidden, request_text.lower())

    def test_request_contains_exactly_one_image_and_no_ground_truth_spatial_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            map_path = Path(tmpdir) / "semantic.png"
            Image.new("RGB", (24, 24), (100, 120, 140)).save(map_path)
            experiment = MapGroundedHeadingExperiment(response_client=lambda _: {})
            request, _ = experiment.build_request_body(
                current_room_id="Room A",
                target_room_id="Room B",
                semantic_map_path=map_path,
            )
        content = request["input"][0]["content"]
        self.assertEqual(sum(block.get("type") == "input_image" for block in content), 1)
        self.assertEqual(sum(block.get("type") == "input_text" for block in content), 1)
        prompt = str(content[0]["text"])
        self.assertIn("current_room_id: Room A", prompt)
        self.assertIn("target_room_id: Room B", prompt)
        serialized = json.dumps(request)
        for forbidden in (
            "expected_heading_deg",
            "allocentric_heading_deg",
            "geocentric_heading_deg",
            "target_pano_id",
            "candidate_actions",
            "topology_signature",
        ):
            self.assertNotIn(forbidden, serialized)

    def test_direction_heading_mapping_and_local_validation(self) -> None:
        self.assertEqual(
            DIRECTION_TO_HEADING,
            {"N": 0.0, "NE": 45.0, "E": 90.0, "SE": 135.0, "S": 180.0, "SW": 225.0, "W": 270.0, "NW": 315.0},
        )
        validate_heading_output(valid_output(), current_room_id="Room A", target_room_id="Room B")
        invalid = valid_output()
        invalid["heading_deg"] = 45.0
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            validate_heading_output(invalid, current_room_id="Room A", target_room_id="Room B")

    def test_schema_locks_requested_room_ids_and_eight_headings(self) -> None:
        schema = build_heading_schema(current_room_id="Room A", target_room_id="Room B")
        properties = schema["properties"]
        self.assertEqual(properties["current_room_id"]["enum"], ["Room A"])
        self.assertEqual(properties["target_room_id"]["enum"], ["Room B"])
        self.assertEqual(properties["passage_direction"]["enum"], list(DIRECTION_TO_HEADING))
        self.assertEqual(properties["heading_deg"]["enum"], list(DIRECTION_TO_HEADING.values()))

    def test_mock_vlm_completes_single_map_only_call(self) -> None:
        requests = []

        def fake_response(request: dict) -> dict:
            requests.append(request)
            return {"output_text": json.dumps(valid_output())}

        with tempfile.TemporaryDirectory() as tmpdir:
            map_path = Path(tmpdir) / "semantic.png"
            Image.new("RGB", (24, 24), (100, 120, 140)).save(map_path)
            experiment = MapGroundedHeadingExperiment(response_client=fake_response)
            wrapped = experiment.run_once(
                current_room_id="Room A",
                target_room_id="Room B",
                semantic_map_path=map_path,
            )
        self.assertEqual(len(requests), 1)
        self.assertEqual(wrapped["result"]["passage_direction"], "E")
        self.assertEqual(wrapped["logical_vlm_call_count"], 1)


if __name__ == "__main__":
    unittest.main()
