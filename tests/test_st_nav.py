from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from st_nav import (
    GroundingIndex,
    InstructionRoutePlanner,
    LLMInstructionParser,
    ManifestPerceptionProvider,
    Observation,
    PerceptionPipeline,
    PanoramaRenderer,
    SourcePanoResolver,
    SourcePerceptionWorkflow,
    SpatialEngine,
    ViewDetector,
    build_grounding_template,
    normalize_pano_graph,
    normalize_room_graph,
)
from st_nav.normalize import (
    BRITISH_MUSEUM_DIRECTION_OVERRIDES,
    BRITISH_MUSEUM_EXCLUDED_EDGES,
    BRITISH_MUSEUM_ROOM_CANONICAL_IDS,
    BRITISH_MUSEUM_TRANSITION_OVERRIDES,
)
from st_nav.prompts import (
    build_view_detection_input,
    build_view_detection_instructions,
    build_view_detection_schema,
)


class STNavTests(unittest.TestCase):
    def setUp(self) -> None:
        self.explicit_map = {
            "Room 8": {
                "name": "Room 8",
                "Level": 0,
                "category": "Middle East",
                "title": "Assyria: Nimrud",
                "links": [
                    {"direction": "left", "name": "Room 23"},
                    {"direction": "up", "name": "Room 9"},
                ],
            },
            "Room 9": {
                "name": "Room 9",
                "Level": 0,
                "category": "Middle East",
                "title": "Assyria: Nineveh",
                "links": [
                    {"direction": "down", "name": "Room 8"},
                ],
            },
            "Room 23": {
                "name": "Room 23",
                "Level": 0,
                "category": "Ancient Greece and Rome",
                "title": "Greek and Roman sculpture",
                "links": [
                    {"direction": "right", "name": "Room 8"},
                ],
            },
        }
        self.pano_graph = {
            "pano-8": {
                "panoID": "pano-8",
                "floor": "0",
                "lat": 1.0,
                "lng": 1.0,
                "links": [
                    {"panoID": "pano-23", "heading": 240.0, "description": None},
                    {"panoID": "pano-9", "heading": 330.0, "description": None},
                ],
            },
            "pano-23": {
                "panoID": "pano-23",
                "floor": "0",
                "lat": 1.1,
                "lng": 1.1,
                "links": [
                    {"panoID": "pano-8", "heading": 60.0, "description": None},
                ],
            },
            "pano-9": {
                "panoID": "pano-9",
                "floor": "0",
                "lat": 1.2,
                "lng": 1.2,
                "links": [
                    {"panoID": "pano-8", "heading": 150.0, "description": None},
                ],
            },
        }

    def test_normalize_room_graph_preserves_direction_metadata(self) -> None:
        room_graph = normalize_room_graph(self.explicit_map)
        self.assertIn("Room 8", room_graph)
        left_edge = next(
            edge for edge in room_graph["Room 8"]["neighbors"] if edge["target_room_id"] == "Room 23"
        )
        self.assertEqual(left_edge["allocentric_direction"], "west")
        self.assertEqual(left_edge["allocentric_heading_deg"], 270.0)

    def test_normalize_room_graph_only_keeps_gallery_rooms_up_to_33(self) -> None:
        explicit_map = dict(self.explicit_map)
        explicit_map["East Stairs"] = {
            "name": "East Stairs",
            "Level": 0,
            "category": None,
            "title": None,
            "links": [],
        }
        explicit_map["Room 40"] = {
            "name": "Room 40",
            "Level": 3,
            "category": "Other",
            "title": "Other",
            "links": [],
        }

        room_graph = normalize_room_graph(explicit_map)
        self.assertNotIn("East Stairs", room_graph)
        self.assertNotIn("Room 40", room_graph)

    def test_normalize_room_graph_can_filter_to_experiment_subset(self) -> None:
        explicit_map = {
            **self.explicit_map,
            "Room 6": {
                "name": "Room 6",
                "Level": 0,
                "category": "Middle East",
                "title": "Assyrian sculpture and Balawat Gates",
                "links": [
                    {"direction": "up", "name": "Room 4"},
                    {"direction": "down", "name": "Room 6 bottom"},
                ],
            },
            "Room 6 bottom": {
                "name": "Room 6",
                "Level": 0,
                "category": "Ancient Greece and Rome",
                "title": "Early Greece",
                "links": [
                    {"direction": "up", "name": "Room 6"},
                    {"direction": "left", "name": "Room 12"},
                ],
            },
            "Room 12": {
                "name": "Room 12",
                "Level": 0,
                "category": "Ancient Greece and Rome",
                "title": "Greece: Minoans and Mycenaeans",
                "links": [
                    {"direction": "right", "name": "Room 6 bottom"},
                ],
            },
            "Room 40": {
                "name": "Room 40",
                "Level": 3,
                "category": "Other",
                "title": "Other",
                "links": [],
            },
        }

        room_graph = normalize_room_graph(
            explicit_map,
            allowed_room_ids={"Room 4", "Room 6", "Room 8", "Room 9", "Room 12", "Room 23"},
            canonical_room_ids=BRITISH_MUSEUM_ROOM_CANONICAL_IDS,
        )
        self.assertNotIn("Room 6 bottom", room_graph)
        self.assertNotIn("Room 40", room_graph)
        self.assertIn("Room 6", room_graph)
        right_edge = next(edge for edge in room_graph["Room 12"]["neighbors"] if edge["target_room_id"] == "Room 6")
        self.assertEqual(right_edge["allocentric_direction"], "east")

    def test_normalize_room_graph_can_fill_missing_reverse_edges(self) -> None:
        explicit_map = {
            "Room 17": {
                "name": "Room 17",
                "Level": 0,
                "category": "Ancient Greece and Rome",
                "title": "Nereid Monument",
                "links": [
                    {"direction": "left", "name": "Room 18a"},
                ],
            },
            "Room 18a": {
                "name": "Room 18",
                "Level": 0,
                "category": "Ancient Greece and Rome",
                "title": "Greece: Parthenon",
                "links": [
                    {"direction": "right", "name": "Room 17"},
                ],
            },
            "Room 23": {
                "name": "Room 23",
                "Level": 0,
                "category": "Ancient Greece and Rome",
                "title": "Greek and Roman sculpture",
                "links": [
                    {"direction": "left", "name": "Room 17"},
                ],
            },
        }
        room_graph = normalize_room_graph(explicit_map, ensure_bidirectional=True)
        reverse_edge = next(edge for edge in room_graph["Room 17"]["neighbors"] if edge["target_room_id"] == "Room 23")
        self.assertEqual(reverse_edge["allocentric_direction"], "east")
        self.assertEqual(reverse_edge["allocentric_heading_deg"], 90.0)

    def test_normalize_room_graph_can_override_experiment_directions(self) -> None:
        explicit_map = {
            "Room 18a": {
                "name": "Room 18",
                "Level": 0,
                "category": "Ancient Greece and Rome",
                "title": "Greece: Parthenon",
                "links": [
                    {"direction": "down", "name": "Room 18b"},
                ],
            },
            "Room 18b": {
                "name": "Room 18",
                "Level": 0,
                "category": "Ancient Greece and Rome",
                "title": "Greece: Parthenon",
                "links": [
                    {"direction": "up", "name": "Room 18a"},
                ],
            },
        }
        room_graph = normalize_room_graph(
            explicit_map,
            direction_overrides=BRITISH_MUSEUM_DIRECTION_OVERRIDES,
        )
        edge_18a = room_graph["Room 18a"]["neighbors"][0]
        edge_18b = room_graph["Room 18b"]["neighbors"][0]
        self.assertEqual(edge_18a["allocentric_direction"], "north")
        self.assertEqual(edge_18a["allocentric_heading_deg"], 360.0)
        self.assertEqual(edge_18b["allocentric_direction"], "south")
        self.assertEqual(edge_18b["allocentric_heading_deg"], 180.0)

    def test_normalize_room_graph_can_override_transition_type(self) -> None:
        explicit_map = {
            "Room 22": {
                "name": "Room 22",
                "Level": 0,
                "category": "Ancient Greece and Rome",
                "title": "The world of Alexander",
                "links": [{"direction": "down", "name": "Room 23"}],
            },
            "Room 23": {
                "name": "Room 23",
                "Level": 0,
                "category": "Ancient Greece and Rome",
                "title": "Greek and Roman sculpture",
                "links": [{"direction": "up", "name": "Room 22"}],
            },
        }
        room_graph = normalize_room_graph(
            explicit_map,
            transition_overrides=BRITISH_MUSEUM_TRANSITION_OVERRIDES,
        )
        edge_22 = room_graph["Room 22"]["neighbors"][0]
        edge_23 = room_graph["Room 23"]["neighbors"][0]
        self.assertEqual(edge_22["transition_type"], "stairs")
        self.assertEqual(edge_23["transition_type"], "stairs")

    def test_normalize_room_graph_can_exclude_experiment_edges(self) -> None:
        explicit_map = {
            "Room 9": {
                "name": "Room 9",
                "Level": 0,
                "category": "Middle East",
                "title": "Assyria: Nineveh",
                "links": [{"direction": "right", "name": "Room 23"}],
            },
            "Room 23": {
                "name": "Room 23",
                "Level": 0,
                "category": "Ancient Greece and Rome",
                "title": "Greek and Roman sculpture",
                "links": [{"direction": "left", "name": "Room 9"}],
            },
        }
        room_graph = normalize_room_graph(
            explicit_map,
            excluded_edges=BRITISH_MUSEUM_EXCLUDED_EDGES,
        )
        self.assertEqual(room_graph["Room 9"]["neighbors"], [])
        self.assertEqual(room_graph["Room 23"]["neighbors"], [])

    def test_manifest_perception_loads_entities(self) -> None:
        pano_graph = normalize_pano_graph(self.pano_graph)
        provider = ManifestPerceptionProvider(pano_graph)
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            manifest_path = tmp / "pano-8_manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "pano_id": "pano-8",
                        "captures": [
                            {"label": "north", "heading": 330.0, "path": "north.png"},
                            {"label": "west", "heading": 240.0, "path": "west.png"},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            manifest_path.with_name("pano-8_manifest_detections.json").write_text(
                json.dumps(
                    {
                        "entities": [
                            {
                                "capture_label": "west",
                                "name": "Lamassu",
                                "confidence": 0.95,
                                "kind": "artwork",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            observation = provider.observe(manifest_path, current_heading=330.0)
            self.assertEqual(observation.pano_id, "pano-8")
            self.assertEqual(len(observation.entities), 1)

    def test_llm_instruction_parser_maps_instruction_to_rooms(self) -> None:
        room_graph = normalize_room_graph(self.explicit_map)
        captured = {}

        def fake_response_client(body: dict) -> dict:
            captured["body"] = body
            return {
                "output_text": json.dumps(
                    {
                        "task_type": "gallery_goal_navigation",
                        "source_room_id": "Room 8",
                        "source_entity": {
                            "name": "Room 8",
                            "entity_type": "gallery",
                            "predicted_room_id": "Room 8",
                            "confidence": 1.0,
                        },
                        "goal_entities": [
                            {
                                "name": "Room 23",
                                "entity_type": "gallery",
                                "predicted_room_id": "Room 23",
                                "confidence": 1.0,
                            }
                        ],
                        "waypoint_entities": [],
                    }
                )
            }

        parser = LLMInstructionParser(
            room_graph=room_graph,
            api_key="test-key",
            response_client=fake_response_client,
        )
        task = parser.parse("由 Room 8 到 Room 23")
        self.assertEqual(task.source_room_id, "Room 8")
        self.assertEqual(task.source_entity.name, "Room 8")
        self.assertEqual(task.goal_room_ids, ["Room 23"])
        self.assertEqual(task.task_type, "gallery_goal_navigation")
        self.assertIn("museum_navigation_parse", json.dumps(captured["body"]))
        self.assertIn("Classify the instruction into exactly one task type.", captured["body"]["instructions"])
        self.assertIn("Allowed room ids:", captured["body"]["input"])
        self.assertIn("Task:", captured["body"]["input"])
        self.assertEqual(parser.last_request_body, captured["body"])
        self.assertEqual(
            parser.last_response_payload["output_text"],
            json.dumps(
                {
                    "task_type": "gallery_goal_navigation",
                    "source_room_id": "Room 8",
                    "source_entity": {
                        "name": "Room 8",
                        "entity_type": "gallery",
                        "predicted_room_id": "Room 8",
                        "confidence": 1.0,
                    },
                    "goal_entities": [
                        {
                            "name": "Room 23",
                            "entity_type": "gallery",
                            "predicted_room_id": "Room 23",
                            "confidence": 1.0,
                        }
                    ],
                    "waypoint_entities": [],
                }
            ),
        )

    def test_llm_instruction_parser_supports_artwork_instruction_following(self) -> None:
        room_graph = normalize_room_graph(
            {
                **self.explicit_map,
                "Room 14": {
                    "name": "Room 14",
                    "Level": 0,
                    "category": "Ancient Greece and Rome",
                    "title": "Greek vases",
                    "links": [],
                },
                "Room 17": {
                    "name": "Room 17",
                    "Level": 0,
                    "category": "Ancient Greece and Rome",
                    "title": "Nereid Monument",
                    "links": [{"direction": "left", "name": "Room 23"}],
                },
            }
        )
        parser = LLMInstructionParser(
            room_graph=room_graph,
            api_key="test-key",
            response_client=lambda body: {
                "output_text": json.dumps(
                    {
                        "task_type": "artwork_instruction_following_navigation",
                        "source_room_id": "Room 14",
                        "source_entity": {
                            "name": "Bronze Container for Cosmetic Items",
                            "entity_type": "artwork",
                            "predicted_room_id": "Room 14",
                            "confidence": 0.76,
                        },
                        "goal_entities": [
                            {
                                "name": "Townley Venus",
                                "entity_type": "artwork",
                                "predicted_room_id": "Room 23",
                                "confidence": 0.93,
                            }
                        ],
                        "waypoint_entities": [
                            {
                                "name": "Lamassu",
                                "entity_type": "artwork",
                                "predicted_room_id": "Room 8",
                                "confidence": 0.88,
                            },
                            {
                                "name": "Nereid Monument",
                                "entity_type": "artwork",
                                "predicted_room_id": "Room 17",
                                "confidence": 0.99,
                            },
                        ],
                    }
                )
            },
        )
        task = parser.parse("Find the way from Bronze Container for Cosmetic Items, passing the Lamassu, the Nereid Monument, to the Townley Venus.")
        self.assertEqual(task.task_type, "artwork_instruction_following_navigation")
        self.assertEqual(task.source_room_id, "Room 14")
        self.assertEqual(task.source_entity.entity_type, "artwork")
        self.assertEqual(task.goal_room_ids, ["Room 23"])
        self.assertEqual(task.waypoint_room_ids, ["Room 8", "Room 17"])
        self.assertEqual(task.goal_entities[0].name, "Townley Venus")
        self.assertAlmostEqual(task.goal_entities[0].confidence, 0.93)

    def test_llm_instruction_parser_requires_api_key(self) -> None:
        room_graph = normalize_room_graph(self.explicit_map)
        parser = LLMInstructionParser(room_graph=room_graph, api_key=None)
        with self.assertRaises(RuntimeError):
            parser.parse("由 Room 8 到 Room 23")

    def test_llm_instruction_parser_raises_on_invalid_llm_output(self) -> None:
        room_graph = normalize_room_graph(self.explicit_map)
        parser = LLMInstructionParser(
            room_graph=room_graph,
            api_key="test-key",
            response_client=lambda body: {"output_text": json.dumps(
                {
                    "task_type": "gallery_goal_navigation",
                    "source_room_id": "Room 8",
                    "source_entity": {
                        "name": "Room 8",
                        "entity_type": "gallery",
                        "predicted_room_id": "Room 8",
                        "confidence": 1.0,
                    },
                    "goal_entities": [],
                    "waypoint_entities": [],
                }
            )},
        )
        with self.assertRaises(ValueError):
            parser.parse("由 Room 8 到 Room 23")

    def test_perception_pipeline_exposes_render_detect_aggregate_flow(self) -> None:
        pano_graph = normalize_pano_graph(self.pano_graph)
        downloads = []

        def fake_downloader(url: str, output_path: Path) -> None:
            downloads.append((url, output_path))
            output_path.write_bytes(b"fake-image")

        pipeline = PerceptionPipeline(
            pano_graph=pano_graph,
            renderer=PanoramaRenderer(pano_graph, image_downloader=fake_downloader),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            manifest = pipeline.render_views(
                pano_id="pano-8",
                api_key="test-key",
                output_dir=output_dir,
                heading_mode="museum",
            )
            manifest_path = Path(manifest["manifest_path"])
            manifest_path.with_name(f"{manifest_path.stem}_detections.json").write_text(
                json.dumps(
                    {
                        "entities": [
                            {
                                "capture_label": "north",
                                "name": "Lamassu",
                                "confidence": 0.9,
                                "kind": "artwork",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            detections = pipeline.detect_views(manifest_path)
            observation = pipeline.aggregate_observation(
                manifest_path,
                current_heading=330.0,
                view_detections=detections,
            )
            self.assertEqual(len(downloads), 4)
            self.assertEqual(len(detections), 1)
            self.assertEqual(observation.entities[0].name, "Lamassu")
            self.assertEqual(observation.entities[0].metadata["view_count"], 1)

    def test_view_detector_can_use_vlm_response_client(self) -> None:
        pano_graph = normalize_pano_graph(self.pano_graph)
        downloads = []
        captured_bodies = []

        def fake_downloader(url: str, output_path: Path) -> None:
            downloads.append((url, output_path))
            output_path.write_bytes(b"fake-image")

        detector = ViewDetector(
            api_key="test-key",
            response_client=lambda body: captured_bodies.append(body) or {
                "output_text": json.dumps(
                    {
                        "entities": [
                            {"name": "Lamassu", "kind": "artwork", "confidence": 0.95},
                        ]
                    }
                )
            },
            use_detection_files=False,
        )
        pipeline = PerceptionPipeline(
            pano_graph=pano_graph,
            renderer=PanoramaRenderer(pano_graph, image_downloader=fake_downloader),
            detector=detector,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = pipeline.render_views(
                pano_id="pano-8",
                api_key="render-key",
                output_dir=Path(tmpdir),
                heading_mode="museum",
            )
            observation = pipeline.observe_from_manifest(
                manifest["manifest_path"],
                current_heading=330.0,
            )

        self.assertEqual(len(observation.entities), 1)
        self.assertEqual(observation.entities[0].name, "Lamassu")
        self.assertEqual(observation.entities[0].metadata["view_count"], 4)
        self.assertEqual(len(captured_bodies), 4)
        self.assertIn(
            "Use a specific official exhibit name only when the identity is visually unique",
            captured_bodies[0]["instructions"],
        )
        self.assertIn(
            "This is the north view from a 4-view panorama.",
            captured_bodies[0]["input"][0]["content"][0]["text"],
        )
        self.assertEqual(detector.last_traces[0]["capture_label"], "north")
        self.assertEqual(
            detector.last_traces[0]["request"]["input"][0]["content"][1]["image_url"],
            "<IMAGE_DATA_URL_OMITTED>",
        )
        self.assertEqual(
            detector.last_traces[0]["response"]["output_text"],
            json.dumps(
                {
                    "entities": [
                        {"name": "Lamassu", "kind": "artwork", "confidence": 0.95},
                    ]
                }
            ),
        )

    def test_view_detection_prompt_and_schema_include_passage(self) -> None:
        schema = build_view_detection_schema()
        kinds = schema["properties"]["entities"]["items"]["properties"]["kind"]["enum"]
        instructions = build_view_detection_instructions()
        view_input = build_view_detection_input("north")

        self.assertIn("passage", kinds)
        self.assertIn("salient passages or doorways", instructions)
        self.assertIn("visually grounded entities", view_input)

    def test_view_detector_can_preserve_passage_kind(self) -> None:
        pano_graph = normalize_pano_graph(self.pano_graph)
        downloads = []

        def fake_downloader(url: str, output_path: Path) -> None:
            downloads.append((url, output_path))
            output_path.write_bytes(b"fake-image")

        detector = ViewDetector(
            api_key="test-key",
            response_client=lambda body: {
                "output_text": json.dumps(
                    {
                        "entities": [
                            {"name": "archway to next room", "kind": "passage", "confidence": 0.88},
                        ]
                    }
                )
            },
            use_detection_files=False,
        )
        pipeline = PerceptionPipeline(
            pano_graph=pano_graph,
            renderer=PanoramaRenderer(pano_graph, image_downloader=fake_downloader),
            detector=detector,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = pipeline.render_views(
                pano_id="pano-8",
                api_key="render-key",
                output_dir=Path(tmpdir),
                heading_mode="museum",
            )
            observation = pipeline.observe_from_manifest(
                manifest["manifest_path"],
                current_heading=330.0,
            )

        self.assertEqual(len(downloads), 4)
        self.assertEqual(len(observation.entities), 1)
        self.assertEqual(observation.entities[0].kind, "passage")
        self.assertEqual(observation.entities[0].name, "archway to next room")
        self.assertEqual(observation.entities[0].metadata["view_count"], 4)

    def test_multiview_aggregator_merges_same_entity_across_views(self) -> None:
        pano_graph = normalize_pano_graph(self.pano_graph)
        provider = ManifestPerceptionProvider(pano_graph)
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            manifest_path = tmp / "pano-8_manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "pano_id": "pano-8",
                        "captures": [
                            {"label": "north", "heading": 330.0, "path": "north.png"},
                            {"label": "west", "heading": 240.0, "path": "west.png"},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            manifest_path.with_name("pano-8_manifest_detections.json").write_text(
                json.dumps(
                    {
                        "entities": [
                            {
                                "capture_label": "north",
                                "name": "Lamassu",
                                "confidence": 0.75,
                                "kind": "artwork",
                            },
                            {
                                "capture_label": "west",
                                "name": "Lamassu",
                                "confidence": 0.95,
                                "kind": "artwork",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            observation = provider.observe(manifest_path, current_heading=330.0)

        self.assertEqual(len(observation.entities), 1)
        self.assertEqual(observation.entities[0].name, "Lamassu")
        self.assertEqual(observation.entities[0].confidence, 0.95)
        self.assertEqual(observation.entities[0].metadata["view_count"], 2)

    def test_panorama_renderer_writes_manifest_from_perception_layer(self) -> None:
        pano_graph = normalize_pano_graph(self.pano_graph)
        downloads = []

        def fake_downloader(url: str, output_path: Path) -> None:
            downloads.append((url, output_path))
            output_path.write_bytes(b"fake-image")

        renderer = PanoramaRenderer(pano_graph, image_downloader=fake_downloader)
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            manifest = renderer.render(
                pano_id="pano-8",
                api_key="test-key",
                output_dir=output_dir,
                heading_mode="museum",
                graph_path="dataset/sites/british_museum/pano_graph/processed/panos.json",
            )

            manifest_path = Path(manifest["manifest_path"])
            self.assertTrue(manifest_path.exists())
            self.assertEqual(len(manifest["captures"]), 4)
            self.assertEqual(len(downloads), 4)
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["pano_id"], "pano-8")
            self.assertEqual(payload["heading_mode"], "museum")

    def test_panorama_renderer_can_render_pano_missing_from_graph_in_non_graph_mode(self) -> None:
        pano_graph = normalize_pano_graph(self.pano_graph)
        downloads = []

        def fake_downloader(url: str, output_path: Path) -> None:
            downloads.append((url, output_path))
            output_path.write_bytes(b"fake-image")

        renderer = PanoramaRenderer(pano_graph, image_downloader=fake_downloader)
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            manifest = renderer.render(
                pano_id="missing-pano-id",
                api_key="test-key",
                output_dir=output_dir,
                heading_mode="museum",
            )

            self.assertEqual(manifest["pano_id"], "missing-pano-id")
            self.assertEqual(len(manifest["captures"]), 4)
            self.assertEqual(len(downloads), 4)
            self.assertIsNone(manifest["floor"])

    def test_spatial_engine_can_compute_shortest_room_route(self) -> None:
        room_graph = normalize_room_graph(self.explicit_map)
        pano_graph = normalize_pano_graph(self.pano_graph)
        grounding = build_grounding_template(room_graph)
        spatial = SpatialEngine(
            room_graph=room_graph,
            pano_graph=pano_graph,
            grounding_index=GroundingIndex(grounding),
        )
        self.assertEqual(
            spatial.shortest_room_route("Room 9", "Room 23"),
            ["Room 9", "Room 8", "Room 23"],
        )

    def test_spatial_update_does_not_map_pano_to_room_via_grounding(self) -> None:
        room_graph = normalize_room_graph(self.explicit_map)
        pano_graph = normalize_pano_graph(self.pano_graph)
        grounding = build_grounding_template(room_graph)
        grounding["Room 23"]["pano_ids"] = ["pano-23"]
        spatial = SpatialEngine(
            room_graph=room_graph,
            pano_graph=pano_graph,
            grounding_index=GroundingIndex(grounding),
        )
        state = spatial.initialize(start_pano_id="pano-8", start_room_id="Room 8")

        updated = spatial.update(
            state,
            Observation(
                pano_id="pano-23",
                heading_estimate=60.0,
                metadata={},
            ),
        )
        self.assertEqual(updated.current_room_id, "Room 8")

    def test_spatial_update_accepts_localized_room_from_observation_metadata(self) -> None:
        room_graph = normalize_room_graph(self.explicit_map)
        pano_graph = normalize_pano_graph(self.pano_graph)
        grounding = build_grounding_template(room_graph)
        spatial = SpatialEngine(
            room_graph=room_graph,
            pano_graph=pano_graph,
            grounding_index=GroundingIndex(grounding),
        )
        state = spatial.initialize(start_pano_id="pano-8", start_room_id="Room 8")

        updated = spatial.update(
            state,
            Observation(
                pano_id="pano-23",
                heading_estimate=60.0,
                metadata={"localized_room_id": "Room 23", "localization_confidence": 0.9},
            ),
        )
        self.assertEqual(updated.current_room_id, "Room 23")
        self.assertEqual(updated.room_belief, {"Room 23": 0.9})

    def test_instruction_route_planner_runs_parse_then_shortest_path(self) -> None:
        room_graph = normalize_room_graph(self.explicit_map)
        pano_graph = normalize_pano_graph(self.pano_graph)
        grounding = build_grounding_template(room_graph)

        parser = LLMInstructionParser(
            room_graph=room_graph,
            api_key="test-key",
            response_client=lambda body: {
                "output_text": json.dumps(
                    {
                        "task_type": "gallery_goal_navigation",
                        "source_room_id": "Room 9",
                        "source_entity": {
                            "name": "Room 9",
                            "entity_type": "gallery",
                            "predicted_room_id": "Room 9",
                            "confidence": 1.0,
                        },
                        "goal_entities": [
                            {
                                "name": "Room 23",
                                "entity_type": "gallery",
                                "predicted_room_id": "Room 23",
                                "confidence": 1.0,
                            }
                        ],
                        "waypoint_entities": [],
                    }
                )
            },
        )
        planner = InstructionRoutePlanner(
            instruction_parser=parser,
            spatial_engine=SpatialEngine(
                room_graph=room_graph,
                pano_graph=pano_graph,
                grounding_index=GroundingIndex(grounding),
            ),
        )

        plan = planner.plan("由 Room 9 到 Room 23")
        self.assertEqual(plan.source_room_id, "Room 9")
        self.assertEqual(plan.target_room_id, "Room 23")
        self.assertEqual(plan.shortest_path, ["Room 9", "Room 8", "Room 23"])

    def test_source_pano_resolver_returns_representative_pano(self) -> None:
        room_graph = normalize_room_graph(self.explicit_map)
        grounding = build_grounding_template(room_graph)
        grounding["Room 8"]["pano_ids"] = ["pano-8", "pano-8b"]

        resolver = SourcePanoResolver(GroundingIndex(grounding))
        resolution = resolver.resolve("Room 8")
        self.assertEqual(resolution.source_room_id, "Room 8")
        self.assertEqual(resolution.pano_id, "pano-8")
        self.assertEqual(resolution.candidate_pano_ids, ["pano-8", "pano-8b"])

    def test_source_perception_workflow_runs_parse_resolve_render_aggregate(self) -> None:
        room_graph = normalize_room_graph(self.explicit_map)
        pano_graph = normalize_pano_graph(self.pano_graph)
        grounding = build_grounding_template(room_graph)
        grounding["Room 8"]["pano_ids"] = ["pano-8"]
        downloads = []

        def fake_downloader(url: str, output_path: Path) -> None:
            downloads.append((url, output_path))
            output_path.write_bytes(b"fake-image")

        parser = LLMInstructionParser(
            room_graph=room_graph,
            api_key="test-key",
            response_client=lambda body: {
                "output_text": json.dumps(
                    {
                        "task_type": "gallery_goal_navigation",
                        "source_room_id": "Room 8",
                        "source_entity": {
                            "name": "Room 8",
                            "entity_type": "gallery",
                            "predicted_room_id": "Room 8",
                            "confidence": 1.0,
                        },
                        "goal_entities": [
                            {
                                "name": "Room 23",
                                "entity_type": "gallery",
                                "predicted_room_id": "Room 23",
                                "confidence": 1.0,
                            }
                        ],
                        "waypoint_entities": [],
                    }
                )
            },
        )
        workflow = SourcePerceptionWorkflow(
            instruction_parser=parser,
            source_pano_resolver=SourcePanoResolver(GroundingIndex(grounding)),
            perception_pipeline=PerceptionPipeline(
                pano_graph=pano_graph,
                renderer=PanoramaRenderer(pano_graph, image_downloader=fake_downloader),
            ),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            result = workflow.run(
                "Find the way from Room 8 to Room 23.",
                api_key="test-key",
                output_dir=tmpdir,
            )

        self.assertEqual(result.task.source_room_id, "Room 8")
        self.assertEqual(result.source_pano.pano_id, "pano-8")
        self.assertEqual(result.observation.pano_id, "pano-8")
        self.assertEqual(len(result.observation.views), 4)
        self.assertEqual(len(downloads), 4)

    def test_grounding_index_can_resolve_primary_pano(self) -> None:
        room_graph = normalize_room_graph(self.explicit_map)
        grounding = build_grounding_template(room_graph)
        grounding["Room 8"]["pano_ids"] = ["pano-8"]
        grounding_index = GroundingIndex(grounding)
        self.assertEqual(grounding_index.primary_pano_for_room("Room 8"), "pano-8")
        self.assertIsNone(grounding_index.primary_pano_for_room("Room 23"))


if __name__ == "__main__":
    unittest.main()
