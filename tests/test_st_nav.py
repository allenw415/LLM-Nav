from __future__ import annotations

import io
import json
import os
import random
import tempfile
import unittest
import urllib.error
from unittest import mock
from pathlib import Path

from st_nav import (
    CandidateAction,
    EntityDetection,
    EpisodeRunner,
    GroundingIndex,
    InstructionRoutePlanner,
    LLMActionPolicy,
    LLMRoomLocalizer,
    LLMSpatialAlignmentLocalizer,
    LLMInstructionParser,
    ManifestPerceptionProvider,
    ModelEnvironment,
    ModelResponseClient,
    NavigationPipeline,
    Observation,
    PerceptionPipeline,
    PanoramaRenderer,
    PolicyOutput,
    ReasoningInput,
    RenderedView,
    RoomLocalizer,
    SourcePanoResolver,
    SourceResolutionWorkflow,
    SpatialEngine,
    TaskSpec,
    ViewDetector,
    build_grounding_template,
    build_spatial_context_extraction_instructions,
    build_view_detection_input,
    build_view_detection_instructions,
    build_view_detection_schema,
    extract_output_text,
    load_dotenv,
    resolve_model_environment,
)
from st_nav_data.normalize import (
    BRITISH_MUSEUM_DIRECTION_OVERRIDES,
    BRITISH_MUSEUM_EXCLUDED_EDGES,
    BRITISH_MUSEUM_ROOM_CANONICAL_IDS,
    BRITISH_MUSEUM_TRANSITION_OVERRIDES,
    normalize_pano_graph,
    normalize_room_graph,
)
MUSEUM_CAPTURE_LABELS = [
    "north",
    "north_to_east",
    "east",
    "east_to_south",
    "south",
    "south_to_west",
    "west",
    "west_to_north",
]


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

    def test_model_response_client_can_convert_responses_payload_to_chat_completions(self) -> None:
        request_body = {
            "model": "demo-model",
            "instructions": "Return JSON only.",
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Describe this image."},
                        {
                            "type": "input_image",
                            "image_url": "data:image/png;base64,AAA",
                            "detail": "high",
                        },
                    ],
                }
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "demo_schema",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {"answer": {"type": "string"}},
                        "required": ["answer"],
                        "additionalProperties": False,
                    },
                }
            },
        }

        payload = ModelResponseClient._responses_to_chat_completions_payload(request_body)

        self.assertEqual(payload["model"], "demo-model")
        self.assertEqual(payload["messages"][0], {"role": "system", "content": "Return JSON only."})
        self.assertEqual(payload["messages"][1]["role"], "user")
        self.assertEqual(payload["messages"][1]["content"][0], {"type": "text", "text": "Describe this image."})
        self.assertEqual(
            payload["messages"][1]["content"][1],
            {
                "type": "image_url",
                "image_url": {
                    "url": "data:image/png;base64,AAA",
                    "detail": "high",
                },
            },
        )
        self.assertEqual(payload["response_format"]["type"], "json_schema")
        self.assertEqual(payload["response_format"]["json_schema"]["name"], "demo_schema")

    def test_model_response_client_preserves_string_input_for_chat_completions(self) -> None:
        payload = ModelResponseClient._responses_to_chat_completions_payload(
            {
                "model": "demo-model",
                "instructions": "Return JSON only.",
                "input": "Instruction: Find the way from Room 4 to Room 23.",
            }
        )

        self.assertEqual(payload["messages"][0], {"role": "system", "content": "Return JSON only."})
        self.assertEqual(
            payload["messages"][1],
            {"role": "user", "content": "Instruction: Find the way from Room 4 to Room 23."},
        )

    def test_extract_output_text_can_read_chat_completions_payload(self) -> None:
        payload = {
            "choices": [
                {
                    "message": {
                        "content": [
                            {"type": "output_text", "text": '{"answer":"ok"}'},
                        ]
                    }
                }
            ]
        }

        self.assertEqual(extract_output_text(payload), '{"answer":"ok"}')

    def test_extract_output_text_can_read_ollama_native_payload(self) -> None:
        payload = {
            "message": {
                "role": "assistant",
                "content": '{"answer":"ok"}',
            }
        }

        self.assertEqual(extract_output_text(payload), '{"answer":"ok"}')

    def test_extract_output_text_can_read_gemini_payload(self) -> None:
        payload = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": '{"answer":"ok"}'},
                        ]
                    }
                }
            ]
        }

        self.assertEqual(extract_output_text(payload), '{"answer":"ok"}')

    def test_load_dotenv_lets_later_file_values_override_earlier_file_values(self) -> None:
        original = os.environ.pop("ST_NAV_API_KIND", None)
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                dotenv_path = Path(tmpdir) / ".env"
                dotenv_path.write_text(
                    "ST_NAV_API_KIND=responses\nST_NAV_API_KIND=chat_completions\n",
                    encoding="utf-8",
                )

                load_dotenv(dotenv_path)

            self.assertEqual(os.environ.get("ST_NAV_API_KIND"), "chat_completions")
        finally:
            if original is None:
                os.environ.pop("ST_NAV_API_KIND", None)
            else:
                os.environ["ST_NAV_API_KIND"] = original

    def test_resolve_model_environment_can_use_active_profile(self) -> None:
        managed_keys = {
            "ST_NAV_ACTIVE_PROFILE": os.environ.get("ST_NAV_ACTIVE_PROFILE"),
            "ST_NAV_PROFILE_OLLAMA_MODEL_PROVIDER": os.environ.get("ST_NAV_PROFILE_OLLAMA_MODEL_PROVIDER"),
            "ST_NAV_PROFILE_OLLAMA_MODEL_NAME": os.environ.get("ST_NAV_PROFILE_OLLAMA_MODEL_NAME"),
            "ST_NAV_PROFILE_OLLAMA_API_BASE": os.environ.get("ST_NAV_PROFILE_OLLAMA_API_BASE"),
            "ST_NAV_PROFILE_OLLAMA_API_KEY": os.environ.get("ST_NAV_PROFILE_OLLAMA_API_KEY"),
            "ST_NAV_PROFILE_OLLAMA_API_KIND": os.environ.get("ST_NAV_PROFILE_OLLAMA_API_KIND"),
            "ST_NAV_PROFILE_OLLAMA_NUM_CTX": os.environ.get("ST_NAV_PROFILE_OLLAMA_NUM_CTX"),
            "ST_NAV_PROFILE_OLLAMA_TEMPERATURE": os.environ.get("ST_NAV_PROFILE_OLLAMA_TEMPERATURE"),
        }
        try:
            os.environ["ST_NAV_ACTIVE_PROFILE"] = "ollama"
            os.environ["ST_NAV_PROFILE_OLLAMA_MODEL_PROVIDER"] = "ollama"
            os.environ["ST_NAV_PROFILE_OLLAMA_MODEL_NAME"] = "gemma4:26b"
            os.environ["ST_NAV_PROFILE_OLLAMA_API_BASE"] = "http://127.0.0.1:11434/v1"
            os.environ["ST_NAV_PROFILE_OLLAMA_API_KEY"] = "ollama"
            os.environ["ST_NAV_PROFILE_OLLAMA_API_KIND"] = "chat_completions"
            os.environ["ST_NAV_PROFILE_OLLAMA_NUM_CTX"] = "4096"
            os.environ["ST_NAV_PROFILE_OLLAMA_TEMPERATURE"] = "0"

            resolved = resolve_model_environment(
                default_model="gpt-5-mini",
                default_api_base="https://api.openai.com/v1",
                default_api_kind="responses",
            )

            self.assertEqual(
                resolved,
                ModelEnvironment(
                    provider="ollama",
                    model_name="gemma4:26b",
                    api_key="ollama",
                    api_base="http://127.0.0.1:11434/v1",
                    api_kind="chat_completions",
                    request_timeout=None,
                    num_ctx=4096,
                    temperature=0.0,
                    active_profile="ollama",
                ),
            )
        finally:
            for key, value in managed_keys.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def test_resolve_model_environment_can_use_gemini_api_key_fallback(self) -> None:
        managed_keys = {
            "ST_NAV_ACTIVE_PROFILE": os.environ.get("ST_NAV_ACTIVE_PROFILE"),
            "ST_NAV_PROFILE_GEMINI_MODEL_PROVIDER": os.environ.get("ST_NAV_PROFILE_GEMINI_MODEL_PROVIDER"),
            "ST_NAV_PROFILE_GEMINI_MODEL_NAME": os.environ.get("ST_NAV_PROFILE_GEMINI_MODEL_NAME"),
            "ST_NAV_PROFILE_GEMINI_API_KEY": os.environ.get("ST_NAV_PROFILE_GEMINI_API_KEY"),
        }
        try:
            os.environ["ST_NAV_ACTIVE_PROFILE"] = "gemini"
            os.environ["ST_NAV_PROFILE_GEMINI_MODEL_PROVIDER"] = "gemini"
            os.environ["ST_NAV_PROFILE_GEMINI_MODEL_NAME"] = "gemma-4-26b-a4b-it"
            os.environ["ST_NAV_PROFILE_GEMINI_API_KEY"] = "gemini-test-key"

            resolved = resolve_model_environment(
                default_model="gpt-5-mini",
                default_api_base="https://api.openai.com/v1",
                default_api_kind="responses",
            )

            self.assertEqual(resolved.provider, "gemini")
            self.assertEqual(resolved.model_name, "gemma-4-26b-a4b-it")
            self.assertEqual(resolved.api_key, "gemini-test-key")
            self.assertEqual(resolved.api_base, "https://generativelanguage.googleapis.com/v1beta")
        finally:
            for key, value in managed_keys.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def test_model_response_client_can_convert_responses_payload_to_ollama_chat(self) -> None:
        client = ModelResponseClient(
            provider="ollama",
            api_base="http://127.0.0.1:11434/v1",
            api_kind="chat_completions",
        )
        request_body = {
            "model": "gemma4:26b",
            "instructions": "Return JSON only.",
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Describe this image."},
                        {
                            "type": "input_image",
                            "image_url": "data:image/png;base64,AAA",
                            "detail": "high",
                        },
                    ],
                }
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "demo_schema",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {"answer": {"type": "string"}},
                        "required": ["answer"],
                        "additionalProperties": False,
                    },
                }
            },
        }

        payload = client._responses_to_ollama_chat_payload(request_body)

        self.assertEqual(payload["model"], "gemma4:26b")
        self.assertEqual(payload["messages"][0], {"role": "system", "content": "Return JSON only."})
        self.assertEqual(payload["messages"][1]["role"], "user")
        self.assertEqual(payload["messages"][1]["content"], "Describe this image.")
        self.assertEqual(payload["messages"][1]["images"], ["AAA"])
        self.assertEqual(payload["format"]["required"], ["answer"])
        self.assertEqual(payload["options"]["num_ctx"], 4096)
        self.assertEqual(payload["options"]["temperature"], 0)

    def test_model_response_client_preserves_string_input_for_ollama_chat(self) -> None:
        client = ModelResponseClient(
            provider="ollama",
            api_base="http://127.0.0.1:11434/v1",
            api_kind="chat_completions",
        )

        payload = client._responses_to_ollama_chat_payload(
            {
                "model": "gemma4:26b",
                "instructions": "Return JSON only.",
                "input": "Instruction: Find the way from Room 4 to Room 23.",
            }
        )

        self.assertEqual(payload["messages"][0], {"role": "system", "content": "Return JSON only."})
        self.assertEqual(
            payload["messages"][1],
            {"role": "user", "content": "Instruction: Find the way from Room 4 to Room 23."},
        )

    def test_model_response_client_can_convert_responses_payload_to_gemini(self) -> None:
        client = ModelResponseClient(
            provider="gemini",
            api_base="https://generativelanguage.googleapis.com/v1beta",
            api_kind="responses",
        )
        request_body = {
            "model": "gemma-4-26b-a4b-it",
            "instructions": "Return JSON only.",
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Describe this image."},
                        {
                            "type": "input_image",
                            "image_url": "data:image/png;base64,AAA",
                            "detail": "high",
                        },
                    ],
                }
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "demo_schema",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {"answer": {"type": "string"}},
                        "required": ["answer"],
                        "additionalProperties": False,
                    },
                }
            },
        }

        payload = client._responses_to_gemini_generate_content_payload(request_body)

        self.assertEqual(payload["contents"][0], {"role": "user", "parts": [{"text": "Return JSON only."}]})
        self.assertEqual(payload["contents"][1]["role"], "user")
        self.assertEqual(payload["contents"][1]["parts"][0], {"text": "Describe this image."})
        self.assertEqual(
            payload["contents"][1]["parts"][1],
            {"inline_data": {"mime_type": "image/png", "data": "AAA"}},
        )
        self.assertEqual(payload["generationConfig"]["responseMimeType"], "application/json")
        self.assertEqual(payload["generationConfig"]["responseJsonSchema"]["required"], ["answer"])

    def test_model_response_client_preserves_string_input_for_gemini(self) -> None:
        client = ModelResponseClient(
            provider="gemini",
            api_base="https://generativelanguage.googleapis.com/v1beta",
            api_kind="responses",
        )

        payload = client._responses_to_gemini_generate_content_payload(
            {
                "model": "gemma-4-31b-it",
                "instructions": "Return JSON only.",
                "input": "Instruction: Find the way from the Lamassu to the Townley Venus.",
            }
        )

        self.assertEqual(payload["contents"][0], {"role": "user", "parts": [{"text": "Return JSON only."}]})
        self.assertEqual(
            payload["contents"][1],
            {
                "role": "user",
                "parts": [{"text": "Instruction: Find the way from the Lamassu to the Townley Venus."}],
            },
        )

    def test_model_response_client_retries_transient_http_errors(self) -> None:
        client = ModelResponseClient(
            api_key="test-key",
            api_base="https://example.test/v1",
            api_kind="responses",
        )
        request_body = {"model": "demo-model"}

        transient_error = urllib.error.HTTPError(
            url="https://example.test/v1/responses",
            code=503,
            msg="Service Unavailable",
            hdrs=None,
            fp=io.BytesIO(b'{"error":"backend busy"}'),
        )

        class FakeResponse:
            def __enter__(self) -> "FakeResponse":
                return self

            def __exit__(self, exc_type, exc, tb) -> None:
                return None

            def read(self) -> bytes:
                return b'{"output_text":"{\\"answer\\":\\"ok\\"}"}'

        with (
            mock.patch("st_nav.common.model_client.time.sleep") as sleep_mock,
            mock.patch(
                "st_nav.common.model_client.urllib.request.urlopen",
                side_effect=[transient_error, FakeResponse()],
            ) as urlopen_mock,
        ):
            payload = client.create(request_body)

        self.assertEqual(payload["output_text"], '{"answer":"ok"}')
        self.assertEqual(urlopen_mock.call_count, 2)
        sleep_mock.assert_called_once_with(1.0)

    def test_model_response_client_http_error_includes_response_body(self) -> None:
        client = ModelResponseClient(
            api_key="test-key",
            api_base="https://example.test/v1",
            api_kind="responses",
        )
        request_body = {"model": "demo-model"}

        def build_request_error() -> urllib.error.HTTPError:
            return urllib.error.HTTPError(
                url="https://example.test/v1/responses",
                code=400,
                msg="Bad Request",
                hdrs=None,
                fp=io.BytesIO(b'{"error":{"message":"bad schema"}}'),
            )

        with mock.patch(
            "st_nav.common.model_client.urllib.request.urlopen",
            side_effect=build_request_error(),
        ):
            with self.assertRaisesRegex(RuntimeError, "HTTP 400"):
                client.create(request_body)

        with self.assertRaisesRegex(RuntimeError, "bad schema"):
            with mock.patch(
                "st_nav.common.model_client.urllib.request.urlopen",
                side_effect=build_request_error(),
            ):
                client.create(request_body)

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
            self.assertEqual(len(downloads), 8)
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
                            {
                                "name": "Lamassu",
                                "kind": "artwork",
                                "confidence": 0.95,
                                "source_views": list(MUSEUM_CAPTURE_LABELS),
                            },
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
        self.assertEqual(observation.entities[0].metadata["view_count"], 8)
        self.assertEqual(observation.entities[0].metadata["source_views"], MUSEUM_CAPTURE_LABELS)
        self.assertEqual(len(captured_bodies), 1)
        self.assertIn(
            "Use a specific official exhibit name only when the identity is visually unique",
            captured_bodies[0]["instructions"],
        )
        self.assertIn(
            "These are 8 overlapping views from the same panorama.",
            captured_bodies[0]["input"][0]["content"][0]["text"],
        )
        self.assertEqual(detector.last_traces[0]["capture_label"], "multiview")
        self.assertEqual(detector.last_traces[0]["capture_labels"], MUSEUM_CAPTURE_LABELS)
        self.assertEqual(
            detector.last_traces[0]["request"]["input"][0]["content"][2]["image_url"],
            "<IMAGE_DATA_URL_OMITTED>",
        )
        self.assertEqual(
            detector.last_traces[0]["response"]["output_text"],
            json.dumps(
                {
                    "entities": [
                        {
                            "name": "Lamassu",
                            "kind": "artwork",
                            "confidence": 0.95,
                            "source_views": list(MUSEUM_CAPTURE_LABELS),
                        },
                    ]
                }
            ),
        )

    def test_view_detection_prompt_and_schema_include_passage(self) -> None:
        schema = build_view_detection_schema()
        kinds = schema["properties"]["entities"]["items"]["properties"]["kind"]["enum"]
        instructions = build_view_detection_instructions()
        view_input = build_view_detection_input(
            [
                {"label": "north", "heading": 330.0},
                {"label": "north_to_east", "heading": 15.0},
            ]
        )

        self.assertIn("passage", kinds)
        self.assertIn("salient passages or doorways", instructions)
        self.assertIn("first identify the full set of distinct navigable openings", instructions)
        self.assertIn("include each distinct opening as its own entity", instructions)
        self.assertIn("same physical entity", instructions)
        self.assertIn("Do not merge different entities just because they share the same type", instructions)
        self.assertIn("north-facing passage and a south-facing passage should usually be separate entities", instructions)
        self.assertIn("Do not combine passages seen in opposite or non-contiguous views into one entity", instructions)
        self.assertIn("Do not omit side openings just because they are partially occluded", instructions)
        self.assertIn("Treat room or gallery labels as signage by default", instructions)
        self.assertIn("Only mention a room id or destination in a passage name", instructions)
        self.assertIn("First inventory the distinct navigable openings visible across all views", view_input)
        self.assertIn("separate instances of the same kind", view_input)
        self.assertIn("prefer direction-aware names such as north passage, south doorway, east corridor", view_input)
        self.assertIn("include every clearly visible opening even if some are less central or partly occluded", view_input)
        self.assertIn("doorway beside the Assyria Nimrud sign", view_input)
        self.assertIn("Do not rename a passage as an entrance to Room 8", view_input)
        self.assertIn("neighboring and show the same opening continuously", view_input)
        self.assertIn("Aggregate all visible evidence across the full panorama", view_input)
        self.assertIn("source_views", schema["properties"]["entities"]["items"]["properties"])

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
                            {
                                "name": "archway to next room",
                                "kind": "passage",
                                "confidence": 0.88,
                                "source_views": list(MUSEUM_CAPTURE_LABELS),
                            },
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

        self.assertEqual(len(downloads), 8)
        self.assertEqual(len(observation.entities), 1)
        self.assertEqual(observation.entities[0].kind, "passage")
        self.assertEqual(observation.entities[0].name, "archway to next room")
        self.assertEqual(observation.entities[0].metadata["view_count"], 8)

    def test_view_detector_writes_and_reuses_detection_cache(self) -> None:
        calls = []
        detector = ViewDetector(
            api_key="test-key",
            response_client=lambda body: calls.append(body) or {
                "output_text": json.dumps(
                    {
                        "entities": [
                            {
                                "name": "north doorway",
                                "kind": "passage",
                                "confidence": 0.91,
                                "source_views": ["north", "north_to_east"],
                            }
                        ]
                    }
                )
            },
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            image_path = tmp / "north.png"
            image_path.write_bytes(b"fake-image")
            manifest_path = tmp / "pano-8_manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "pano_id": "pano-8",
                        "captures": [
                            {"label": "north", "heading": 330.0, "path": str(image_path)},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            first = detector.detect(manifest_path)
            second = detector.detect(manifest_path)

            detection_path = manifest_path.with_name("pano-8_manifest_detections.json")
            trace_path = manifest_path.with_name("pano-8_manifest_detections_trace.json")

            self.assertEqual(len(calls), 1)
            self.assertEqual(len(first), 1)
            self.assertEqual(len(second), 1)
            self.assertTrue(detection_path.exists())
            self.assertTrue(trace_path.exists())

            detection_payload = json.loads(detection_path.read_text(encoding="utf-8"))
            self.assertEqual(
                detection_payload["entities"][0]["source_views"],
                ["north", "north_to_east"],
            )
            trace_payload = json.loads(trace_path.read_text(encoding="utf-8"))
            self.assertEqual(
                trace_payload["requests_and_responses"][0]["capture_labels"],
                ["north"],
            )
            self.assertEqual(
                detector.last_traces[0]["capture_labels"],
                ["north"],
            )

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

        renderer = PanoramaRenderer(pano_graph, image_downloader=fake_downloader, rng=random.Random(0))
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
            self.assertEqual(len(manifest["captures"]), 8)
            self.assertEqual(len(downloads), 8)
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["pano_id"], "pano-8")
            self.assertEqual(payload["heading_mode"], "museum")
            self.assertEqual(payload["size"], {"width": 512, "height": 512})
            self.assertEqual([capture["label"] for capture in payload["captures"]], MUSEUM_CAPTURE_LABELS)
            self.assertEqual(payload["captures"][0]["heading"], 330.0)
            self.assertTrue(330.0 < payload["captures"][1]["heading"] or payload["captures"][1]["heading"] < 60.0)
            self.assertEqual(payload["captures"][2]["heading"], 60.0)
            self.assertTrue(60.0 < payload["captures"][3]["heading"] < 150.0)
            self.assertEqual(payload["captures"][4]["heading"], 150.0)
            self.assertTrue(150.0 < payload["captures"][5]["heading"] < 240.0)
            self.assertEqual(payload["captures"][6]["heading"], 240.0)
            self.assertTrue(240.0 < payload["captures"][7]["heading"] < 330.0)

    def test_panorama_renderer_grounding_mode_uses_four_museum_headings(self) -> None:
        pano_graph = normalize_pano_graph(self.pano_graph)
        downloads = []

        def fake_downloader(url: str, output_path: Path) -> None:
            downloads.append((url, output_path))
            output_path.write_bytes(b"fake-image")

        renderer = PanoramaRenderer(pano_graph, image_downloader=fake_downloader, rng=random.Random(0))
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            manifest = renderer.render(
                pano_id="pano-8",
                api_key="test-key",
                output_dir=output_dir,
                heading_mode="grounding",
            )

            self.assertEqual(len(manifest["captures"]), 4)
            self.assertEqual(len(downloads), 4)
            self.assertEqual([capture["label"] for capture in manifest["captures"]], ["north", "east", "south", "west"])
            self.assertEqual([capture["heading"] for capture in manifest["captures"]], [330.0, 60.0, 150.0, 240.0])

    def test_panorama_renderer_can_render_pano_missing_from_graph_in_non_graph_mode(self) -> None:
        pano_graph = normalize_pano_graph(self.pano_graph)
        downloads = []

        def fake_downloader(url: str, output_path: Path) -> None:
            downloads.append((url, output_path))
            output_path.write_bytes(b"fake-image")

        renderer = PanoramaRenderer(pano_graph, image_downloader=fake_downloader, rng=random.Random(0))
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            manifest = renderer.render(
                pano_id="missing-pano-id",
                api_key="test-key",
                output_dir=output_dir,
                heading_mode="museum",
            )

            self.assertEqual(manifest["pano_id"], "missing-pano-id")
            self.assertEqual(len(manifest["captures"]), 8)
            self.assertEqual(len(downloads), 8)
            self.assertIsNone(manifest["floor"])

    def test_panorama_renderer_reuses_cached_manifest_and_images(self) -> None:
        pano_graph = normalize_pano_graph(self.pano_graph)
        downloads = []

        def fake_downloader(url: str, output_path: Path) -> None:
            downloads.append((url, output_path))
            output_path.write_bytes(b"fake-image")

        renderer = PanoramaRenderer(pano_graph, image_downloader=fake_downloader, rng=random.Random(0))
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            first = renderer.render(
                pano_id="pano-8",
                api_key="test-key",
                output_dir=output_dir,
                heading_mode="museum",
            )
            second = renderer.render(
                pano_id="pano-8",
                api_key="test-key",
                output_dir=output_dir,
                heading_mode="museum",
            )

            self.assertEqual(len(downloads), 8)
            self.assertEqual(first["manifest_path"], second["manifest_path"])
            self.assertEqual(first["captures"], second["captures"])

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
        self.assertIsNone(updated.grounded_room_id)

    def test_spatial_update_stabilizes_room_when_localizer_switches_room_without_pano_change(self) -> None:
        room_graph = normalize_room_graph(self.explicit_map)
        pano_graph = normalize_pano_graph(self.pano_graph)
        grounding = build_grounding_template(room_graph)

        class FakeLocalizer:
            def localize(self, **kwargs):
                return {
                    "predicted_room_id": "Room 23",
                    "confidence": 0.88,
                    "room_belief": {"Room 23": 0.88, "Room 8": 0.12},
                    "transition_support": {"Room 8": 0.5, "Room 23": 0.5},
                    "evidence": ["Greek and Roman sculpture"],
                    "spatial_alignment": {"view_0_allocentric_direction": "south"},
                }

        spatial = SpatialEngine(
            room_graph=room_graph,
            pano_graph=pano_graph,
            grounding_index=GroundingIndex(grounding),
            localizer=FakeLocalizer(),
        )
        state = spatial.initialize(start_pano_id="pano-8", start_room_id="Room 8")

        observation = Observation(
            pano_id="pano-8",
            heading_estimate=330.0,
            metadata={},
        )
        updated = spatial.update(state, observation)

        self.assertEqual(updated.current_room_id, "Room 23")
        self.assertEqual(updated.room_belief, {"Room 23": 1.0})
        self.assertEqual(observation.metadata["localized_room_id"], "Room 23")
        self.assertNotIn("localized_room_id_raw", observation.metadata)
        self.assertNotIn("localization_stabilized", observation.metadata)

    def test_renderer_can_render_explicit_candidate_captures(self) -> None:
        pano_graph = normalize_pano_graph(
            {
                "pano-start": {
                    "panoID": "pano-start",
                    "floor": "0",
                    "lat": 1.0,
                    "lng": 1.0,
                    "links": [
                        {"panoID": "pano-a", "heading": 323.0, "description": None},
                        {"panoID": "pano-b", "heading": 54.0, "description": None},
                        {"panoID": "pano-c", "heading": 146.0, "description": None},
                    ],
                }
            }
        )

        def fake_downloader(_: str, output_path: Path) -> None:
            output_path.write_bytes(b"fake-image")

        renderer = PanoramaRenderer(pano_graph, image_downloader=fake_downloader, rng=random.Random(0))
        custom_captures = [
            ("candidate_00_pano-b", 54.0),
            ("candidate_01_pano-c", 146.0),
            ("candidate_02_pano-a", 323.0),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = renderer.render(
                pano_id="pano-start",
                api_key="test-key",
                output_dir=tmpdir,
                heading_mode="explicit",
                custom_captures=custom_captures,
                fov=90,
            )

        self.assertEqual(manifest["heading_mode"], "explicit")
        self.assertEqual([capture["label"] for capture in manifest["captures"]], [label for label, _ in custom_captures])
        self.assertEqual([capture["heading"] for capture in manifest["captures"]], [heading for _, heading in custom_captures])

    def test_generate_candidates_prioritizes_route_subgoal_without_known_heading(self) -> None:
        room_graph = normalize_room_graph(self.explicit_map)
        pano_graph = normalize_pano_graph(self.pano_graph)
        grounding = build_grounding_template(room_graph)
        spatial = SpatialEngine(
            room_graph=room_graph,
            pano_graph=pano_graph,
            grounding_index=GroundingIndex(grounding),
        )
        state = spatial.initialize(start_pano_id="pano-8", start_room_id="Room 8", start_heading=0.0)
        observation = Observation(
            pano_id="pano-8",
            views=[
                RenderedView(label="north", heading=330.0, path="/tmp/sector_0.png"),
                RenderedView(label="east", heading=60.0, path="/tmp/sector_1.png"),
                RenderedView(label="south", heading=150.0, path="/tmp/sector_2.png"),
                RenderedView(label="west", heading=240.0, path="/tmp/sector_3.png"),
            ],
            metadata={"spatial_alignment": {"view_0_allocentric_direction": "north"}},
        )

        candidates = spatial.generate_candidates(state, ["Room 8", "Room 23"], observation=observation)

        self.assertEqual(candidates[0].target_pano_id, "pano-23")
        self.assertEqual(candidates[0].target_room_id, "Room 23")
        self.assertEqual(candidates[0].route_step_index, 1)
        self.assertIn("matches_subgoal", candidates[0].reason)
        self.assertIsNotNone(candidates[0].relative_heading)
        self.assertEqual(candidates[0].metadata["matching_strategy"], "spatial_alignment_direction")

    def test_generate_candidates_prefers_grounded_room_for_target_pano(self) -> None:
        room_graph = normalize_room_graph(self.explicit_map)
        pano_graph = normalize_pano_graph(self.pano_graph)
        grounding = build_grounding_template(room_graph)
        grounding["Room 8"]["pano_ids"] = ["pano-8", "pano-23"]
        grounding["Room 23"]["pano_ids"] = []
        spatial = SpatialEngine(
            room_graph=room_graph,
            pano_graph=pano_graph,
            grounding_index=GroundingIndex(grounding),
        )
        state = spatial.initialize(start_pano_id="pano-8", start_room_id="Room 8", start_heading=0.0)
        observation = Observation(
            pano_id="pano-8",
            views=[
                RenderedView(label="north", heading=330.0, path="/tmp/sector_0.png"),
                RenderedView(label="east", heading=60.0, path="/tmp/sector_1.png"),
                RenderedView(label="south", heading=150.0, path="/tmp/sector_2.png"),
                RenderedView(label="west", heading=240.0, path="/tmp/sector_3.png"),
            ],
            metadata={"spatial_alignment": {"view_0_allocentric_direction": "north"}},
        )

        candidates = spatial.generate_candidates(state, ["Room 8", "Room 23"], observation=observation)

        grounded = next(candidate for candidate in candidates if candidate.target_pano_id == "pano-23")
        self.assertEqual(grounded.target_room_id, "Room 8")
        self.assertEqual(grounded.metadata["grounded_target_room_id"], "Room 8")
        self.assertEqual(grounded.metadata["inferred_target_room_id"], "Room 23")

    def test_extract_visible_passages_uses_spatial_alignment_instead_of_agent_heading(self) -> None:
        room_graph = normalize_room_graph(self.explicit_map)
        pano_graph = normalize_pano_graph(self.pano_graph)
        grounding = build_grounding_template(room_graph)
        spatial = SpatialEngine(
            room_graph=room_graph,
            pano_graph=pano_graph,
            grounding_index=GroundingIndex(grounding),
        )
        state = spatial.initialize(start_pano_id="pano-8", start_room_id="Room 8", start_heading=0.0)
        observation = Observation(
            pano_id="pano-8",
            views=[
                RenderedView(label="sector_0", heading=0.0, path="/tmp/sector_0.png"),
                RenderedView(label="sector_1", heading=90.0, path="/tmp/sector_1.png"),
                RenderedView(label="sector_2", heading=180.0, path="/tmp/sector_2.png"),
                RenderedView(label="sector_3", heading=270.0, path="/tmp/sector_3.png"),
            ],
            entities=[
                EntityDetection(
                    name="west doorway",
                    confidence=0.92,
                    kind="passage",
                    source_view="sector_0",
                    metadata={"source_views": ["sector_0"]},
                )
            ],
            metadata={"spatial_alignment": {"view_0_allocentric_direction": "west"}},
        )

        passages = spatial.extract_visible_passages(state, observation)

        self.assertEqual(len(passages), 1)
        self.assertEqual(passages[0]["allocentric_directions"], ["west"])
        self.assertEqual(passages[0]["matched_room_ids"], ["Room 23"])

    def test_generate_candidates_matches_closest_heading_after_fixed_museum_offset(self) -> None:
        room_graph = normalize_room_graph(
            {
                "Room 8": {
                    "name": "Room 8",
                    "Level": 0,
                    "category": "Middle East",
                    "title": "Assyria: Nimrud",
                    "links": [{"direction": "left", "name": "Room 23"}],
                },
                "Room 23": {
                    "name": "Room 23",
                    "Level": 0,
                    "category": "Ancient Greece and Rome",
                    "title": "Greek and Roman sculpture",
                    "links": [{"direction": "right", "name": "Room 8"}],
                },
            }
        )
        pano_graph = normalize_pano_graph(
            {
                "pano-8": {
                    "panoID": "pano-8",
                    "floor": "0",
                    "lat": 0.0,
                    "lng": 0.0,
                    "links": [
                        {"panoID": "pano-90", "heading": 180.0, "description": None},
                        {"panoID": "pano-225", "heading": 315.0, "description": None},
                        {"panoID": "pano-315", "heading": 45.0, "description": None},
                    ],
                },
                "pano-90": {"panoID": "pano-90", "floor": "0", "lat": 0.0, "lng": 0.0, "links": []},
                "pano-225": {"panoID": "pano-225", "floor": "0", "lat": 0.0, "lng": 0.0, "links": []},
                "pano-315": {"panoID": "pano-315", "floor": "0", "lat": 0.0, "lng": 0.0, "links": []},
            }
        )
        grounding = build_grounding_template(room_graph)
        spatial = SpatialEngine(
            room_graph=room_graph,
            pano_graph=pano_graph,
            grounding_index=GroundingIndex(grounding),
        )
        state = spatial.initialize(start_pano_id="pano-8", start_room_id="Room 8", start_heading=0.0)
        observation = Observation(
            pano_id="pano-8",
            views=[RenderedView(label="view_0", heading=90.0, path="/tmp/view_0.png")],
            metadata={"spatial_alignment": {"view_0_allocentric_direction": "north"}},
        )

        candidates = spatial.generate_candidates(state, ["Room 8", "Room 23"], observation=observation)

        self.assertEqual(candidates[0].target_pano_id, "pano-90")
        self.assertAlmostEqual(candidates[0].metadata["candidate_allocentric_heading_deg"], 210.0)
        self.assertAlmostEqual(candidates[0].metadata["target_relative_heading_deg"], 270.0)
        self.assertAlmostEqual(candidates[0].metadata["target_relative_diff_deg"], 60.0)

    def test_generate_candidates_no_longer_uses_sector_alignment_to_override_candidate_heading(self) -> None:
        room_graph = normalize_room_graph(self.explicit_map)
        pano_graph = normalize_pano_graph(
            {
                "pano-8": {
                    "panoID": "pano-8",
                    "floor": "0",
                    "lat": 0.0,
                    "lng": 0.0,
                    "links": [
                        {"panoID": "pano-north", "heading": 323.3, "description": None},
                        {"panoID": "pano-east", "heading": 54.2, "description": None},
                        {"panoID": "pano-southwest", "heading": 146.5, "description": None},
                    ],
                },
                "pano-north": {"panoID": "pano-north", "floor": "0", "lat": 0.0, "lng": 0.0, "links": []},
                "pano-east": {"panoID": "pano-east", "floor": "0", "lat": 0.0, "lng": 0.0, "links": []},
                "pano-southwest": {
                    "panoID": "pano-southwest",
                    "floor": "0",
                    "lat": 0.0,
                    "lng": 0.0,
                    "links": [],
                },
            }
        )
        grounding = build_grounding_template(room_graph)
        spatial = SpatialEngine(
            room_graph=room_graph,
            pano_graph=pano_graph,
            grounding_index=GroundingIndex(grounding),
        )
        state = spatial.initialize(start_pano_id="pano-8", start_room_id="Room 8", start_heading=0.0)
        observation = Observation(
            pano_id="pano-8",
            views=[
                RenderedView(label="north", heading=330.0, path="/tmp/view_0.png"),
                RenderedView(label="north_to_east", heading=16.9, path="/tmp/view_1.png"),
                RenderedView(label="east", heading=60.0, path="/tmp/view_2.png"),
                RenderedView(label="east_to_south", heading=132.9, path="/tmp/view_3.png"),
                RenderedView(label="south", heading=150.0, path="/tmp/view_4.png"),
                RenderedView(label="south_to_west", heading=201.1, path="/tmp/view_5.png"),
                RenderedView(label="west", heading=240.0, path="/tmp/view_6.png"),
                RenderedView(label="west_to_north", heading=315.4, path="/tmp/view_7.png"),
            ],
            metadata={
                "spatial_alignment": {
                    "view_0_allocentric_direction": "north",
                    "sector_alignment": [
                        {
                            "view_id": "view_5",
                            "allocentric_direction": "west",
                            "matched_room_id": "Room 23",
                            "matched_theme": "Greek and Roman sculpture",
                            "rationale": "The subgoal room is visible through this sector.",
                        }
                    ],
                }
            },
        )

        candidates = spatial.generate_candidates(state, ["Room 8", "Room 23"], observation=observation)

        self.assertEqual(candidates[0].target_pano_id, "pano-north")
        self.assertEqual(candidates[0].metadata["target_heading_source"], "allocentric_subgoal")
        self.assertAlmostEqual(candidates[0].metadata["candidate_allocentric_heading_deg"], 353.3, places=1)

    def test_generate_candidates_attaches_spatial_context_per_candidate(self) -> None:
        room_graph = normalize_room_graph(self.explicit_map)
        pano_graph = normalize_pano_graph(self.pano_graph)
        grounding = build_grounding_template(room_graph)
        spatial = SpatialEngine(
            room_graph=room_graph,
            pano_graph=pano_graph,
            grounding_index=GroundingIndex(grounding),
        )
        state = spatial.initialize(start_pano_id="pano-8", start_room_id="Room 8", start_heading=0.0)
        observation = Observation(
            pano_id="pano-8",
            views=[
                RenderedView(label="north", heading=330.0, path="/tmp/view_0.png"),
                RenderedView(label="east", heading=60.0, path="/tmp/view_1.png"),
                RenderedView(label="south", heading=150.0, path="/tmp/view_2.png"),
                RenderedView(label="west", heading=240.0, path="/tmp/view_3.png"),
            ],
            entities=[
                EntityDetection(
                    name="Greek statue",
                    confidence=0.92,
                    kind="artwork",
                    source_view="west",
                    metadata={"source_views": ["west"]},
                ),
                EntityDetection(
                    name="west passage",
                    confidence=0.9,
                    kind="passage",
                    source_view="west",
                    metadata={"source_views": ["west"]},
                ),
            ],
            metadata={
                "spatial_alignment": {
                    "view_0_allocentric_direction": "north",
                    "ego_context_views": [
                        {"view_id": "view_3", "themes": [{"label": "Greek and Roman sculpture", "confidence": 0.88}]}
                    ],
                }
            },
        )

        candidates = spatial.generate_candidates(state, ["Room 8", "Room 23"], observation=observation)

        target_candidate = next(candidate for candidate in candidates if candidate.target_pano_id == "pano-23")
        spatial_context = target_candidate.metadata["spatial_context"]
        self.assertEqual(spatial_context["supporting_views"][0]["label"], "west")
        self.assertEqual(spatial_context["salient_entities"][0]["name"], "Greek statue")
        self.assertEqual(spatial_context["theme_hints"][0]["label"], "Greek and Roman sculpture")

    def test_generate_candidates_blocks_immediate_backtrack_when_alternative_exists(self) -> None:
        room_graph = normalize_room_graph(self.explicit_map)
        pano_graph = normalize_pano_graph(self.pano_graph)
        grounding = build_grounding_template(room_graph)
        spatial = SpatialEngine(
            room_graph=room_graph,
            pano_graph=pano_graph,
            grounding_index=GroundingIndex(grounding),
        )
        state = spatial.initialize(start_pano_id="pano-8", start_room_id="Room 8", start_heading=330.0)
        state.previous_pano_id = "pano-9"
        observation = Observation(
            pano_id="pano-8",
            views=[
                RenderedView(label="north", heading=330.0, path="/tmp/view_0.png"),
                RenderedView(label="east", heading=60.0, path="/tmp/view_1.png"),
                RenderedView(label="south", heading=150.0, path="/tmp/view_2.png"),
                RenderedView(label="west", heading=240.0, path="/tmp/view_3.png"),
            ],
            metadata={"spatial_alignment": {"view_0_allocentric_direction": "north"}},
        )

        candidates = spatial.generate_candidates(state, ["Room 8", "Room 23"], observation=observation)

        self.assertNotIn("pano-9", [candidate.target_pano_id for candidate in candidates])
        self.assertIn("pano-23", [candidate.target_pano_id for candidate in candidates])

    def test_generate_candidates_keeps_immediate_backtrack_when_it_is_only_option(self) -> None:
        room_graph = normalize_room_graph(
            {
                "Room A": {
                    "name": "Room A",
                    "Level": 0,
                    "category": "Test",
                    "title": "Room A",
                    "links": [{"direction": "right", "name": "Room B"}],
                },
                "Room B": {
                    "name": "Room B",
                    "Level": 0,
                    "category": "Test",
                    "title": "Room B",
                    "links": [{"direction": "left", "name": "Room A"}],
                },
            }
        )
        pano_graph = normalize_pano_graph(
            {
                "pano-a": {
                    "panoID": "pano-a",
                    "floor": "0",
                    "lat": 0.0,
                    "lng": 0.0,
                    "links": [{"panoID": "pano-b", "heading": 90.0, "description": None}],
                },
                "pano-b": {
                    "panoID": "pano-b",
                    "floor": "0",
                    "lat": 0.0,
                    "lng": 0.0,
                    "links": [{"panoID": "pano-a", "heading": 270.0, "description": None}],
                },
            }
        )
        grounding = build_grounding_template(room_graph)
        spatial = SpatialEngine(
            room_graph=room_graph,
            pano_graph=pano_graph,
            grounding_index=GroundingIndex(grounding),
        )
        state = spatial.initialize(start_pano_id="pano-b", start_room_id="Room B", start_heading=330.0)
        state.previous_pano_id = "pano-a"
        observation = Observation(
            pano_id="pano-b",
            views=[RenderedView(label="north", heading=330.0, path="/tmp/view_0.png")],
            metadata={"spatial_alignment": {"view_0_allocentric_direction": "north"}},
        )

        candidates = spatial.generate_candidates(state, ["Room B", "Room A"], observation=observation)

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].target_pano_id, "pano-a")
        self.assertTrue(candidates[0].metadata["is_immediate_backtrack"])

    def test_generate_candidates_uses_context_observation_without_reusing_direction_alignment(self) -> None:
        room_graph = normalize_room_graph(self.explicit_map)
        pano_graph = normalize_pano_graph(self.pano_graph)
        grounding = build_grounding_template(room_graph)
        spatial = SpatialEngine(
            room_graph=room_graph,
            pano_graph=pano_graph,
            grounding_index=GroundingIndex(grounding),
        )
        state = spatial.initialize(start_pano_id="pano-8", start_room_id="Room 8", start_heading=330.0)
        main_observation = Observation(
            pano_id="pano-8",
            views=[
                RenderedView(label="north", heading=330.0, path="/tmp/view_0.png"),
                RenderedView(label="east", heading=60.0, path="/tmp/view_1.png"),
                RenderedView(label="south", heading=150.0, path="/tmp/view_2.png"),
                RenderedView(label="west", heading=240.0, path="/tmp/view_3.png"),
            ],
            metadata={
                "spatial_alignment": {
                    "view_0_allocentric_direction": "north",
                }
            },
        )
        context_observation = Observation(
            pano_id="pano-8",
            views=[
                RenderedView(label="candidate_00_pano-23", heading=240.0, path="/tmp/candidate_0.png"),
                RenderedView(label="candidate_01_pano-9", heading=330.0, path="/tmp/candidate_1.png"),
            ],
            entities=[
                EntityDetection(
                    name="Greek statue",
                    confidence=0.92,
                    kind="artwork",
                    source_view="candidate_00_pano-23",
                    metadata={"source_views": ["candidate_00_pano-23"]},
                ),
                EntityDetection(
                    name="Assyrian reliefs",
                    confidence=0.95,
                    kind="artwork",
                    source_view="candidate_01_pano-9",
                    metadata={"source_views": ["candidate_01_pano-9"]},
                ),
            ],
            metadata={
                "ego_spatial_context": {
                    "views": [
                        {
                            "view_id": "view_0",
                            "themes": [{"label": "Greek and Roman sculpture", "confidence": 0.88}],
                        },
                        {
                            "view_id": "view_1",
                            "themes": [{"label": "Assyria: Nineveh", "confidence": 0.91}],
                        },
                    ]
                }
            },
        )

        candidates = spatial.generate_candidates(
            state,
            ["Room 8", "Room 23"],
            observation=main_observation,
            context_observation=context_observation,
        )

        target_candidate = next(candidate for candidate in candidates if candidate.target_pano_id == "pano-23")
        spatial_context = target_candidate.metadata["spatial_context"]
        self.assertEqual(len(spatial_context["supporting_views"]), 1)
        self.assertEqual(spatial_context["supporting_views"][0]["label"], "candidate_00_pano-23")
        self.assertEqual(spatial_context["theme_hints"][0]["label"], "Greek and Roman sculpture")

    def test_describe_view_contexts_uses_ego_spatial_context_metadata_when_alignment_views_missing(self) -> None:
        room_graph = normalize_room_graph(self.explicit_map)
        pano_graph = normalize_pano_graph(self.pano_graph)
        grounding = build_grounding_template(room_graph)
        spatial = SpatialEngine(
            room_graph=room_graph,
            pano_graph=pano_graph,
            grounding_index=GroundingIndex(grounding),
        )
        observation = Observation(
            pano_id="pano-8",
            views=[
                RenderedView(label="north", heading=330.0, path="/tmp/view_0.png"),
                RenderedView(label="east", heading=60.0, path="/tmp/view_1.png"),
            ],
            metadata={
                "spatial_alignment": {"view_0_allocentric_direction": "north"},
                "ego_spatial_context": {
                    "views": [
                        {
                            "view_id": "view_1",
                            "themes": [{"label": "Greek and Roman sculpture", "confidence": 0.88}],
                        }
                    ]
                },
            },
        )

        view_contexts = spatial.describe_view_contexts(observation)

        self.assertEqual(view_contexts[1]["themes"][0]["label"], "Greek and Roman sculpture")
        self.assertEqual(view_contexts[1]["allocentric_direction"], "east")

    def test_spatial_context_extraction_instructions_allow_multiple_themes_per_view(self) -> None:
        instructions = build_spatial_context_extraction_instructions()

        self.assertIn("multiple adjacent gallery themes", instructions)
        self.assertIn("Do not collapse the answer to only one dominant theme", instructions)

    def test_build_current_room_context_includes_subgoal_theme_labels(self) -> None:
        room_graph = normalize_room_graph(self.explicit_map)
        pano_graph = normalize_pano_graph(self.pano_graph)
        grounding = build_grounding_template(room_graph)
        spatial = SpatialEngine(
            room_graph=room_graph,
            pano_graph=pano_graph,
            grounding_index=GroundingIndex(grounding),
        )
        state = spatial.initialize(start_pano_id="pano-8", start_room_id="Room 8", start_heading=330.0)

        context = spatial.build_current_room_context(state, ["Room 8", "Room 23"])

        self.assertEqual(context["subgoal_room_id"], "Room 23")
        self.assertIn("Greek and Roman sculpture", context["subgoal_theme_labels"])
        self.assertIn("Ancient Greece and Rome", context["subgoal_theme_labels"])

    def test_llm_action_policy_can_choose_semantically_better_candidate_than_angle_nearest(self) -> None:
        policy = LLMActionPolicy(
            api_key="test-key",
            response_client=lambda body: {
                "output_text": json.dumps(
                    {
                        "selected_target_pano_id": "pano-southwest",
                        "rationale": "The southwest-facing candidate shows Greek/Roman context and is more likely to connect toward Room 23 than the angle-nearest Assyria-facing candidate.",
                    }
                )
            },
        )
        reasoning_input = ReasoningInput(
            task=TaskSpec(
                task_type="gallery_goal_navigation",
                raw_instruction="Find the way from Room 8 to Room 23.",
                source_room_id="Room 8",
                goal_room_ids=["Room 23"],
            ),
            route=["Room 8", "Room 23"],
            current_room_id="Room 8",
            subgoal_room_id="Room 23",
            current_room_context={
                "room_id": "Room 8",
                "title": "Assyria: Nimrud",
                "subgoal_room_id": "Room 23",
                "subgoal_title": "Greek and Roman sculpture",
                "subgoal_theme_labels": ["Greek and Roman sculpture", "Ancient Greece and Rome"],
                "neighbors": [
                    {"target_room_id": "Room 23", "allocentric_direction": "west", "allocentric_heading_deg": 270.0}
                ],
            },
            view_contexts=[
                {"view_id": "view_0", "label": "north", "heading": 330.0, "allocentric_direction": "north"},
                {"view_id": "view_2", "label": "east", "heading": 60.0, "allocentric_direction": "east"},
                {"view_id": "view_4", "label": "south", "heading": 150.0, "allocentric_direction": "south"},
                {"view_id": "view_6", "label": "west", "heading": 240.0, "allocentric_direction": "west"},
            ],
            candidates=[
                CandidateAction(
                    target_pano_id="pano-north",
                    target_room_id="Room 9",
                    absolute_heading=323.0,
                    relative_heading=353.0,
                    relative_label="front",
                    route_step_index=None,
                    score=5.0,
                    reason="angle-nearest",
                    metadata={
                        "target_relative_diff_deg": 37.0,
                        "spatial_context": {
                            "supporting_views": [{"label": "north", "heading": 330.0}],
                            "salient_entities": [{"name": "Assyrian reliefs", "kind": "artwork", "confidence": 0.95}],
                            "theme_hints": [{"label": "Assyria: Nineveh", "confidence": 0.9}],
                        },
                    },
                ),
                CandidateAction(
                    target_pano_id="pano-southwest",
                    target_room_id="Room 8",
                    absolute_heading=146.0,
                    relative_heading=176.0,
                    relative_label="back",
                    route_step_index=0,
                    score=3.0,
                    reason="semantic-progress",
                    metadata={
                        "target_relative_diff_deg": 124.0,
                        "spatial_context": {
                            "supporting_views": [{"label": "south", "heading": 150.0}],
                            "salient_entities": [{"name": "Greek statue", "kind": "artwork", "confidence": 0.92}],
                            "theme_hints": [{"label": "Greek and Roman sculpture", "confidence": 0.88}],
                        },
                    },
                ),
            ],
        )

        output = policy.choose_next_action(reasoning_input)

        self.assertIsNotNone(output.action)
        self.assertEqual(output.action.target_pano_id, "pano-southwest")
        self.assertIn("Greek/Roman", output.rationale)

    def test_llm_action_policy_request_includes_all_candidates_without_directional_prefilter(self) -> None:
        captured = {}

        def response_client(body: dict) -> dict:
            captured["body"] = body
            return {
                "output_text": json.dumps(
                    {
                        "selected_target_pano_id": "pano-a",
                        "rationale": "Choose a directionally competitive candidate with plausible semantic support.",
                    }
                )
            }

        policy = LLMActionPolicy(api_key="test-key", response_client=response_client)
        reasoning_input = ReasoningInput(
            task=TaskSpec(
                task_type="gallery_goal_navigation",
                raw_instruction="Find the way from Room 8 to Room 23.",
                source_room_id="Room 8",
                goal_room_ids=["Room 23"],
            ),
            route=["Room 8", "Room 23"],
            current_room_id="Room 8",
            subgoal_room_id="Room 23",
            current_room_context={
                "room_id": "Room 8",
                "title": "Assyria: Nimrud",
                "category": "Middle East",
                "subgoal_room_id": "Room 23",
                "subgoal_title": "Greek and Roman sculpture",
                "subgoal_theme_labels": ["Greek and Roman sculpture", "Ancient Greece and Rome"],
                "remaining_route": ["Room 8", "Room 23"],
                "neighbors": [
                    {
                        "target_room_id": "Room 23",
                        "target_title": "Greek and Roman sculpture",
                        "allocentric_direction": "west",
                        "allocentric_heading_deg": 270.0,
                    }
                ],
            },
            visible_passages=[
                {
                    "name": "west opening",
                    "confidence": 0.9,
                    "source_views": ["west"],
                    "allocentric_directions": ["west"],
                    "matched_room_ids": ["Room 23"],
                }
            ],
            spatial_alignment={
                "view_0_allocentric_direction": "south",
                "alignment_summary": "Should be omitted from the LLM prompt.",
            },
            view_contexts=[
                {
                    "view_id": "view_0",
                    "label": "candidate_00_pano-a",
                    "heading": 146.0,
                    "themes": [{"label": "Greek and Roman sculpture", "confidence": 0.8}],
                }
            ],
            candidates=[
                CandidateAction(
                    target_pano_id="pano-a",
                    target_room_id="Room 8",
                    absolute_heading=146.0,
                    relative_heading=176.0,
                    relative_label="back",
                    route_step_index=0,
                    score=3.0,
                    reason="competitive",
                    metadata={
                        "target_relative_diff_deg": 20.0,
                        "inferred_target_room_id": "Room 23",
                        "spatial_context": {},
                    },
                ),
                CandidateAction(
                    target_pano_id="pano-b",
                    target_room_id="Room 9",
                    absolute_heading=323.0,
                    relative_heading=353.0,
                    relative_label="front",
                    route_step_index=None,
                    score=2.5,
                    reason="competitive",
                    metadata={"target_relative_diff_deg": 35.0, "spatial_context": {}},
                ),
                CandidateAction(
                    target_pano_id="pano-c",
                    target_room_id="Room 8",
                    absolute_heading=54.0,
                    relative_heading=84.0,
                    relative_label="right",
                    route_step_index=0,
                    score=1.0,
                    reason="noncompetitive",
                    metadata={"target_relative_diff_deg": 140.0, "spatial_context": {}},
                ),
            ],
        )

        policy.choose_next_action(reasoning_input)
        payload_text = captured["body"]["input"]
        self.assertNotIn("directional_guidance", payload_text)
        self.assertIn("pano-a", payload_text)
        self.assertIn("pano-b", payload_text)
        self.assertIn("pano-c", payload_text)
        self.assertNotIn("heuristic_score", payload_text)
        self.assertNotIn("inferred_target_room_id", payload_text)
        self.assertNotIn("target_room_id", payload_text)
        self.assertNotIn("grounded_target_room_id", payload_text)
        self.assertNotIn("target_matched_room_id", payload_text)
        self.assertNotIn("relative_label", payload_text)
        self.assertNotIn("relative_heading_deg", payload_text)
        self.assertNotIn("absolute_heading_deg", payload_text)
        self.assertNotIn("candidate_absolute_heading_deg", payload_text)
        self.assertIn("candidate_geocentric_heading_deg", payload_text)
        self.assertIn("target_allocentric_heading_deg", payload_text)
        self.assertNotIn("\"neighbors\"", payload_text)
        self.assertNotIn("matched_room_ids", payload_text)
        self.assertNotIn("allocentric_directions", payload_text)
        self.assertNotIn("egocentric_allocentric_alignment", payload_text)
        self.assertNotIn("view_0_allocentric_direction", payload_text)
        self.assertNotIn("alignment_summary", payload_text)
        self.assertNotIn("\"view_contexts\"", payload_text)

    def test_llm_action_policy_request_keeps_theme_labels_without_priority_subset(self) -> None:
        captured = {}

        def response_client(body: dict) -> dict:
            captured["body"] = body
            return {
                "output_text": json.dumps(
                    {
                        "selected_target_pano_id": "pano-best-angle",
                        "rationale": "stub",
                    }
                )
            }

        policy = LLMActionPolicy(api_key="test-key", response_client=response_client)
        reasoning_input = ReasoningInput(
            task=TaskSpec(
                task_type="gallery_goal_navigation",
                raw_instruction="Find the way from Room 8 to Room 23.",
                source_room_id="Room 8",
                goal_room_ids=["Room 23"],
            ),
            route=["Room 8", "Room 23"],
            current_room_id="Room 8",
            subgoal_room_id="Room 23",
            current_room_context={
                "subgoal_room_id": "Room 23",
                "subgoal_title": "Greek and Roman sculpture",
                "subgoal_theme_labels": ["Greek and Roman sculpture", "Ancient Greece and Rome"],
            },
            candidates=[
                CandidateAction(
                    target_pano_id="pano-best-angle",
                    target_room_id="Room 8",
                    absolute_heading=323.0,
                    relative_heading=353.0,
                    relative_label="front",
                    route_step_index=0,
                    score=4.0,
                    reason="best-angle",
                    metadata={
                        "target_relative_diff_deg": 53.0,
                        "spatial_context": {"theme_hints": [{"label": "Assyria: Nineveh", "confidence": 0.9}]},
                    },
                ),
                CandidateAction(
                    target_pano_id="pano-theme-match",
                    target_room_id="Room 8",
                    absolute_heading=146.0,
                    relative_heading=176.0,
                    relative_label="back",
                    route_step_index=0,
                    score=3.2,
                    reason="theme-match",
                    metadata={
                        "target_relative_diff_deg": 123.0,
                        "spatial_context": {
                            "theme_hints": [{"label": "Greek and Roman sculpture", "confidence": 0.88}]
                        },
                    },
                ),
                CandidateAction(
                    target_pano_id="pano-bad",
                    target_room_id="Room 8",
                    absolute_heading=54.0,
                    relative_heading=84.0,
                    relative_label="right",
                    route_step_index=0,
                    score=2.0,
                    reason="bad",
                    metadata={
                        "target_relative_diff_deg": 144.0,
                        "spatial_context": {
                            "theme_hints": [{"label": "Greek and Roman sculpture", "confidence": 0.55}]
                        },
                    },
                ),
            ],
        )

        policy.choose_next_action(reasoning_input)
        payload_text = captured["body"]["input"]
        self.assertIn("Greek and Roman sculpture", payload_text)
        self.assertIn("pano-theme-match", payload_text)

    def test_room_localizer_combines_transition_and_entity_evidence(self) -> None:
        explicit_map = {
            "Room 7": {
                "name": "Room 7",
                "Level": 0,
                "category": "Middle East",
                "title": "Assyria",
                "links": [{"direction": "right", "name": "Room 10"}],
            },
            "Room 10": {
                "name": "Room 10",
                "Level": 0,
                "category": "Middle East",
                "title": "Assyria: Lion hunts",
                "links": [
                    {"direction": "left", "name": "Room 7"},
                    {"direction": "up", "name": "Room 23"},
                ],
            },
            "Room 18": {
                "name": "Room 18",
                "Level": 0,
                "category": "Ancient Greece and Rome",
                "title": "Greek sculpture",
                "links": [{"direction": "up", "name": "Room 19"}],
            },
            "Room 19": {
                "name": "Room 19",
                "Level": 0,
                "category": "Ancient Greece and Rome",
                "title": "Greek marble sculpture",
                "links": [{"direction": "down", "name": "Room 18"}],
            },
            "Room 20": {
                "name": "Room 20",
                "Level": 0,
                "category": "Ancient Greece and Rome",
                "title": "Roman sculpture",
                "links": [{"direction": "up", "name": "Room 21"}],
            },
            "Room 23": {
                "name": "Room 23",
                "Level": 0,
                "category": "Ancient Greece and Rome",
                "title": "Greek and Roman sculpture",
                "links": [{"direction": "down", "name": "Room 10"}],
            },
        }
        room_graph = normalize_room_graph(explicit_map)
        grounding = build_grounding_template(room_graph)
        localizer = RoomLocalizer(
            room_graph=room_graph,
            grounding_index=GroundingIndex(grounding),
        )

        observation = Observation(
            pano_id="pano-unknown",
            entities=[
                EntityDetection(
                    name="Greek Roman statue",
                    confidence=0.95,
                    kind="artwork",
                    source_view="north",
                ),
                EntityDetection(
                    name="marble sculpture",
                    confidence=0.9,
                    kind="landmark",
                    source_view="east",
                ),
                EntityDetection(
                    name="stone relief",
                    confidence=0.75,
                    kind="artwork",
                    source_view="south",
                ),
            ],
            metadata={"floor": "0"},
        )
        localization = localizer.localize(
            observation=observation,
            prior_room_belief={"Room 10": 1.0},
            fallback_room_id="Room 10",
        )

        self.assertEqual(localization["predicted_room_id"], "Room 23")
        self.assertGreater(localization["room_belief"]["Room 23"], localization["room_belief"]["Room 7"])
        self.assertEqual(localization["room_belief"]["Room 18"], 0.0)
        self.assertEqual(localization["room_belief"]["Room 19"], 0.0)
        self.assertEqual(localization["room_belief"]["Room 20"], 0.0)

    def test_spatial_update_can_localize_room_without_explicit_metadata(self) -> None:
        explicit_map = {
            "Room 7": {
                "name": "Room 7",
                "Level": 0,
                "category": "Middle East",
                "title": "Assyria",
                "links": [{"direction": "right", "name": "Room 10"}],
            },
            "Room 10": {
                "name": "Room 10",
                "Level": 0,
                "category": "Middle East",
                "title": "Assyria: Lion hunts",
                "links": [
                    {"direction": "left", "name": "Room 7"},
                    {"direction": "up", "name": "Room 23"},
                ],
            },
            "Room 23": {
                "name": "Room 23",
                "Level": 0,
                "category": "Ancient Greece and Rome",
                "title": "Greek and Roman sculpture",
                "links": [{"direction": "down", "name": "Room 10"}],
            },
        }
        room_graph = normalize_room_graph(explicit_map)
        pano_graph = normalize_pano_graph(self.pano_graph)
        grounding = build_grounding_template(room_graph)
        spatial = SpatialEngine(
            room_graph=room_graph,
            pano_graph=pano_graph,
            grounding_index=GroundingIndex(grounding),
        )
        state = spatial.initialize(start_pano_id="pano-8", start_room_id="Room 10")

        updated = spatial.update(
            state,
            Observation(
                pano_id="pano-23",
                entities=[
                    EntityDetection(
                        name="Greek Roman statue",
                        confidence=0.95,
                        kind="artwork",
                        source_view="north",
                    ),
                    EntityDetection(
                        name="marble sculpture",
                        confidence=0.9,
                        kind="landmark",
                        source_view="east",
                    ),
                ],
                metadata={"floor": "0"},
            ),
        )

        self.assertEqual(updated.current_room_id, "Room 23")
        self.assertGreater(updated.room_belief["Room 23"], updated.room_belief["Room 7"])

    def test_llm_room_localizer_combines_llm_scores_with_transition_prior(self) -> None:
        room_graph = normalize_room_graph(self.explicit_map)
        grounding = build_grounding_template(room_graph)
        localizer = LLMRoomLocalizer(
            room_graph=room_graph,
            grounding_index=GroundingIndex(grounding),
            response_client=lambda body: {
                "output_text": json.dumps(
                    {
                        "predicted_room_id": "Room 23",
                        "confidence": 0.91,
                        "evidence": ["marble statue on pedestal", "classical sculpture hall"],
                        "room_scores": [
                            {"room_id": "Room 7", "score": 0.2},
                            {"room_id": "Room 8", "score": 0.2},
                            {"room_id": "Room 9", "score": 0.2},
                            {"room_id": "Room 23", "score": 0.9},
                        ],
                        "summary": "Observation best matches the Greek and Roman sculpture room.",
                    }
                )
            },
        )

        localization = localizer.localize(
            observation=Observation(
                pano_id="pano-23",
                entities=[
                    EntityDetection(
                        name="marble statue on pedestal",
                        confidence=0.95,
                        kind="artwork",
                        source_view="north",
                    ),
                ],
                metadata={"floor": "0"},
            ),
            prior_room_belief={"Room 8": 1.0},
            fallback_room_id="Room 8",
        )

        self.assertEqual(localization["predicted_room_id"], "Room 23")
        self.assertGreater(localization["observation_likelihood"]["Room 23"], localization["observation_likelihood"]["Room 8"])
        self.assertGreater(localization["room_belief"]["Room 23"], localization["room_belief"]["Room 8"])

    def test_llm_spatial_alignment_localizer_a_uses_rotation_aware_view_ids(self) -> None:
        explicit_map = {
            "Room 7": {
                "name": "Room 7",
                "Level": 0,
                "category": "Middle East",
                "title": "Assyria: Nimrud",
                "links": [{"direction": "up", "name": "Room 8"}],
            },
            "Room 8": {
                "name": "Room 8",
                "Level": 0,
                "category": "Middle East",
                "title": "Assyria: Nimrud",
                "links": [
                    {"direction": "up", "name": "Room 9"},
                    {"direction": "down", "name": "Room 7"},
                    {"direction": "left", "name": "Room 23"},
                ],
            },
            "Room 9": {
                "name": "Room 9",
                "Level": 0,
                "category": "Middle East",
                "title": "Assyria: Nineveh",
                "links": [{"direction": "down", "name": "Room 8"}],
            },
            "Room 10": {
                "name": "Room 10",
                "Level": 0,
                "category": "Middle East",
                "title": "Assyria: Lion hunts, Siege of Lachish and Khorsabad",
                "links": [{"direction": "up", "name": "Room 23"}],
            },
            "Room 23": {
                "name": "Room 23",
                "Level": 0,
                "category": "Ancient Greece and Rome",
                "title": "Greek and Roman sculpture",
                "links": [
                    {"direction": "right", "name": "Room 8"},
                    {"direction": "down", "name": "Room 10"},
                ],
            },
        }
        room_graph = normalize_room_graph(explicit_map)
        grounding = build_grounding_template(room_graph)

        responses = [
            {
                "output_text": json.dumps(
                    {
                        "views": [
                            {
                                "view_id": "view_0",
                                "themes": [{"label": "Greek and Roman sculpture", "confidence": 0.82}],
                                "summary": "Greek and Roman sculpture is visible.",
                            },
                            {
                                "view_id": "view_1",
                                "themes": [{"label": "Assyria: Nimrud", "confidence": 0.83}],
                                "summary": "Assyria Nimrud appears here.",
                            },
                        ],
                        "summary": "Two panorama sectors were identified.",
                    }
                )
            },
            {
                "output_text": json.dumps(
                    {
                        "predicted_room_id": "Room 8",
                        "confidence": 0.9,
                        "view_0_allocentric_direction": "west",
                        "evidence": ["view_0 matches Greek and Roman sculpture", "view_1 matches Assyria: Nimrud"],
                        "room_distribution": [
                            {"room_id": "Room 7", "score": 0.0},
                            {"room_id": "Room 8", "score": 0.9},
                            {"room_id": "Room 9", "score": 0.0},
                            {"room_id": "Room 10", "score": 0.1},
                            {"room_id": "Room 23", "score": 0.0},
                        ],
                        "summary": "Room 8 best aligns after rotation.",
                    }
                )
            },
        ]

        def response_client(_: dict) -> dict:
            return responses.pop(0)

        with tempfile.TemporaryDirectory() as tmpdir:
            image_paths = []
            for index in range(2):
                path = Path(tmpdir) / f"view_{index}.png"
                path.write_bytes(b"fake-image")
                image_paths.append(path)

            localizer = LLMSpatialAlignmentLocalizer(
                room_graph=room_graph,
                grounding_index=GroundingIndex(grounding),
                alignment_mode="text_from_images",
                response_client=response_client,
            )
            localization = localizer.localize(
                observation=Observation(
                    pano_id="pano-8",
                    views=[
                        RenderedView(label="north", heading=330.0, path=str(image_paths[0])),
                        RenderedView(label="east", heading=60.0, path=str(image_paths[1])),
                    ],
                    entities=[
                        EntityDetection(
                            name="Assyria: Nimrud sign",
                            confidence=0.95,
                            kind="signage",
                            source_view="north",
                        )
                    ],
                    metadata={"floor": "0"},
                ),
                prior_room_belief={"Room 23": 1.0},
                fallback_room_id="Room 23",
            )

        self.assertEqual(localization["predicted_room_id"], "Room 8")
        self.assertEqual(localization["spatial_alignment"]["view_0_allocentric_direction"], "west")
        self.assertIsNotNone(localizer.last_ego_spatial_context)
        alignment_input = localizer.last_alignment_request_body["input"]
        self.assertIn("view_0", alignment_input)
        self.assertNotIn("Front", alignment_input)
        self.assertNotIn("front", alignment_input)

    def test_llm_spatial_alignment_localizer_b_can_align_directly_from_images(self) -> None:
        explicit_map = {
            "Room 8": {
                "name": "Room 8",
                "Level": 0,
                "category": "Middle East",
                "title": "Assyria: Nimrud",
                "links": [
                    {"direction": "up", "name": "Room 9"},
                    {"direction": "left", "name": "Room 23"},
                ],
            },
            "Room 9": {
                "name": "Room 9",
                "Level": 0,
                "category": "Middle East",
                "title": "Assyria: Nineveh",
                "links": [{"direction": "down", "name": "Room 8"}],
            },
            "Room 10": {
                "name": "Room 10",
                "Level": 0,
                "category": "Middle East",
                "title": "Assyria: Lion hunts, Siege of Lachish and Khorsabad",
                "links": [{"direction": "up", "name": "Room 23"}],
            },
            "Room 23": {
                "name": "Room 23",
                "Level": 0,
                "category": "Ancient Greece and Rome",
                "title": "Greek and Roman sculpture",
                "links": [
                    {"direction": "right", "name": "Room 8"},
                    {"direction": "down", "name": "Room 10"},
                ],
            },
        }
        room_graph = normalize_room_graph(explicit_map)
        grounding = build_grounding_template(room_graph)
        localizer = LLMSpatialAlignmentLocalizer(
            room_graph=room_graph,
            grounding_index=GroundingIndex(grounding),
            alignment_mode="direct_images",
            response_client=lambda _: {
                "output_text": json.dumps(
                    {
                        "predicted_room_id": "Room 10",
                        "confidence": 0.87,
                        "view_0_allocentric_direction": "south",
                        "sector_alignment": [
                            {
                                "view_id": "view_0",
                                "allocentric_direction": "south",
                                "matched_room_id": "Room 10",
                                "matched_theme": "Assyria: Lion hunts, Siege of Lachish and Khorsabad",
                                "rationale": "The dominant theme best matches the Room 10 gallery itself.",
                            }
                        ],
                        "evidence": ["The panorama matches Room 10 after rotation."],
                        "room_distribution": [
                            {"room_id": "Room 8", "score": 0.13},
                            {"room_id": "Room 9", "score": 0.0},
                            {"room_id": "Room 10", "score": 0.87},
                            {"room_id": "Room 23", "score": 0.0},
                        ],
                        "summary": "Direct visual alignment favors Room 10.",
                    }
                )
            },
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "view_0.png"
            image_path.write_bytes(b"fake-image")
            localization = localizer.localize(
                observation=Observation(
                    pano_id="pano-10",
                    views=[RenderedView(label="north", heading=330.0, path=str(image_path))],
                    entities=[
                        EntityDetection(
                            name="Lion hunts relief",
                            confidence=0.95,
                            kind="artwork",
                            source_view="north",
                        )
                    ],
                    metadata={"floor": "0"},
                ),
                prior_room_belief={"Room 23": 1.0},
                fallback_room_id="Room 23",
            )

        self.assertEqual(localization["predicted_room_id"], "Room 10")
        self.assertEqual(localization["spatial_alignment"]["mode"], "direct_images")
        self.assertEqual(localization["spatial_alignment"]["view_0_allocentric_direction"], "south")
        self.assertEqual(localization["spatial_alignment"]["sector_alignment"][0]["view_id"], "view_0")
        direct_input = localizer.last_alignment_request_body["input"][0]["content"][0]["text"]
        self.assertIn("view_0", direct_input)
        self.assertIn("global heading is unknown", direct_input)
        self.assertIn("sector_alignment", direct_input)
        self.assertIn("do not jump straight to a room prediction from one sign", localizer.last_alignment_request_body["instructions"])

    def test_llm_spatial_alignment_localizer_reuses_alignment_cache(self) -> None:
        explicit_map = {
            "Room 8": {
                "name": "Room 8",
                "Level": 0,
                "category": "Middle East",
                "title": "Assyria: Nimrud",
                "links": [{"direction": "left", "name": "Room 23"}],
            },
            "Room 23": {
                "name": "Room 23",
                "Level": 0,
                "category": "Ancient Greece and Rome",
                "title": "Greek and Roman sculpture",
                "links": [{"direction": "right", "name": "Room 8"}],
            },
        }
        room_graph = normalize_room_graph(explicit_map)
        grounding = build_grounding_template(room_graph)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            image_path = tmp_path / "view_0.png"
            image_path.write_bytes(b"fake-image")
            manifest_path = tmp_path / "pano-8_manifest.json"
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

            request_count = 0

            def response_client(_: dict) -> dict:
                nonlocal request_count
                request_count += 1
                return {
                    "output_text": json.dumps(
                        {
                            "predicted_room_id": "Room 8",
                            "confidence": 0.91,
                            "view_0_allocentric_direction": "west",
                            "sector_alignment": [
                                {
                                    "view_id": "view_0",
                                    "allocentric_direction": "west",
                                    "matched_room_id": "Room 8",
                                    "matched_theme": "Assyria: Nimrud",
                                    "rationale": "The sector best matches the current gallery.",
                                }
                            ],
                            "evidence": ["Room 8 aligns best after rotation."],
                            "room_distribution": [
                                {"room_id": "Room 8", "score": 0.91},
                                {"room_id": "Room 23", "score": 0.09},
                            ],
                            "summary": "Cached direct visual alignment favors Room 8.",
                        }
                    )
                }

            localizer = LLMSpatialAlignmentLocalizer(
                room_graph=room_graph,
                grounding_index=GroundingIndex(grounding),
                alignment_mode="direct_images",
                response_client=response_client,
            )
            observation = Observation(
                pano_id="pano-8",
                views=[RenderedView(label="north", heading=330.0, path=str(image_path))],
                entities=[
                    EntityDetection(
                        name="Assyria: Nimrud sign",
                        confidence=0.95,
                        kind="signage",
                        source_view="north",
                    )
                ],
                metadata={"floor": "0", "manifest_path": str(manifest_path)},
            )

            first = localizer.localize(
                observation=observation,
                prior_room_belief={"Room 8": 1.0},
                fallback_room_id="Room 8",
            )
            second = localizer.localize(
                observation=observation,
                prior_room_belief={"Room 8": 1.0},
                fallback_room_id="Room 8",
            )

            self.assertEqual(request_count, 1)
            self.assertEqual(first["predicted_room_id"], "Room 8")
            self.assertEqual(second["predicted_room_id"], "Room 8")
            self.assertIsNone(localizer.last_alignment_request_body)
            cache_path = manifest_path.with_name("pano-8_manifest_spatial_alignment_direct_images.json")
            self.assertTrue(cache_path.exists())

    def test_llm_spatial_alignment_localizer_fuses_entity_and_alignment_likelihoods_when_entity_transition_is_ambiguous(self) -> None:
        explicit_map = {
            "Room 8": {
                "name": "Room 8",
                "Level": 0,
                "category": "Middle East",
                "title": "Assyria: Nimrud",
                "links": [{"direction": "left", "name": "Room 23"}],
            },
            "Room 23": {
                "name": "Room 23",
                "Level": 0,
                "category": "Ancient Greece and Rome",
                "title": "Greek and Roman sculpture",
                "links": [{"direction": "right", "name": "Room 8"}],
            },
        }
        room_graph = normalize_room_graph(explicit_map)
        grounding = build_grounding_template(room_graph)
        localizer = LLMSpatialAlignmentLocalizer(
            room_graph=room_graph,
            grounding_index=GroundingIndex(grounding),
            alignment_mode="direct_images",
            response_client=lambda _: {
                "output_text": json.dumps(
                    {
                        "predicted_room_id": "Room 23",
                        "confidence": 0.95,
                        "view_0_allocentric_direction": "west",
                        "room_distribution": [
                            {"room_id": "Room 8", "score": 0.05},
                            {"room_id": "Room 23", "score": 0.95},
                        ],
                        "summary": "Alignment favors Room 23.",
                    }
                )
            },
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "view_0.png"
            image_path.write_bytes(b"fake-image")
            localization = localizer.localize(
                observation=Observation(
                    pano_id="pano-8",
                    views=[RenderedView(label="north", heading=330.0, path=str(image_path))],
                    entities=[
                        EntityDetection(
                            name="stone statue",
                            confidence=0.55,
                            kind="artwork",
                            source_view="north",
                        )
                    ],
                    metadata={"floor": "0"},
                ),
                prior_room_belief={"Room 8": 1.0},
                fallback_room_id="Room 8",
            )

        self.assertEqual(localization["predicted_room_id"], "Room 23")
        self.assertTrue(localization["alignment_fusion_applied"])
        self.assertGreater(localization["observation_likelihood"]["Room 23"], localization["observation_likelihood"]["Room 8"])
        self.assertGreater(
            localization["observation_likelihood"]["Room 23"],
            localization["entity_observation_distribution"]["Room 23"],
        )
        self.assertLess(
            localization["observation_likelihood"]["Room 8"],
            localization["entity_observation_distribution"]["Room 8"],
        )
        self.assertEqual(localization["spatial_alignment"]["alignment_predicted_room_id"], "Room 23")

    def test_llm_spatial_alignment_localizer_skips_alignment_when_entity_transition_is_decisive(self) -> None:
        explicit_map = {
            "Room 8": {
                "name": "Room 8",
                "Level": 0,
                "category": "Middle East",
                "title": "Assyria: Nimrud",
                "links": [{"direction": "left", "name": "Room 23"}],
            },
            "Room 23": {
                "name": "Room 23",
                "Level": 0,
                "category": "Ancient Greece and Rome",
                "title": "Greek and Roman sculpture",
                "links": [{"direction": "right", "name": "Room 8"}],
            },
        }
        room_graph = normalize_room_graph(explicit_map)
        grounding = build_grounding_template(room_graph)
        localizer = LLMSpatialAlignmentLocalizer(
            room_graph=room_graph,
            grounding_index=GroundingIndex(grounding),
            alignment_mode="direct_images",
            response_client=lambda _: {
                "output_text": json.dumps(
                    {
                        "predicted_room_id": "Room 23",
                        "confidence": 0.95,
                        "view_0_allocentric_direction": "west",
                        "room_distribution": [
                            {"room_id": "Room 8", "score": 0.05},
                            {"room_id": "Room 23", "score": 0.95},
                        ],
                        "summary": "Alignment favors Room 23.",
                    }
                )
            },
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "view_0.png"
            image_path.write_bytes(b"fake-image")
            localization = localizer.localize(
                observation=Observation(
                    pano_id="pano-8",
                    views=[RenderedView(label="north", heading=330.0, path=str(image_path))],
                    entities=[
                        EntityDetection(
                            name="Assyria: Nimrud sign",
                            confidence=0.95,
                            kind="signage",
                            source_view="north",
                            metadata={"predicted_room_id": "Room 8"},
                        )
                    ],
                    metadata={"floor": "0"},
                ),
                prior_room_belief={"Room 8": 1.0},
                fallback_room_id="Room 8",
            )

        self.assertEqual(localization["predicted_room_id"], "Room 8")
        self.assertFalse(localization["alignment_fusion_applied"])
        self.assertEqual(
            localization["observation_likelihood"]["Room 8"],
            localization["entity_observation_distribution"]["Room 8"],
        )

    def test_llm_spatial_alignment_localizer_uses_top_two_room_subgraph_for_alignment(self) -> None:
        explicit_map = {
            "Room 7": {
                "name": "Room 7",
                "Level": 0,
                "category": "Middle East",
                "title": "Assyria: Nimrud",
                "links": [{"direction": "up", "name": "Room 8"}],
            },
            "Room 8": {
                "name": "Room 8",
                "Level": 0,
                "category": "Middle East",
                "title": "Assyria: Nimrud",
                "links": [
                    {"direction": "down", "name": "Room 7"},
                    {"direction": "right", "name": "Room 23"},
                ],
            },
            "Room 23": {
                "name": "Room 23",
                "Level": 0,
                "category": "Ancient Greece and Rome",
                "title": "Greek and Roman sculpture",
                "links": [{"direction": "left", "name": "Room 8"}],
            },
        }
        room_graph = normalize_room_graph(explicit_map)
        grounding = build_grounding_template(room_graph)
        responses = [
            {
                "output_text": json.dumps(
                    {
                        "predicted_room_id": "Room 7",
                        "confidence": 0.9,
                        "room_scores": [
                            {"room_id": "Room 7", "score": 0.8},
                            {"room_id": "Room 8", "score": 0.2},
                            {"room_id": "Room 23", "score": 0.0},
                        ],
                        "summary": "Entity-only view prefers Room 7.",
                    }
                )
            },
            {
                "output_text": json.dumps(
                    {
                        "predicted_room_id": "Room 8",
                        "confidence": 0.9,
                        "view_0_allocentric_direction": "north",
                        "room_distribution": [
                            {"room_id": "Room 7", "score": 0.1},
                            {"room_id": "Room 8", "score": 0.9},
                        ],
                        "summary": "Alignment favors Room 8.",
                    }
                )
            },
        ]

        def response_client(_: dict) -> dict:
            return responses.pop(0)

        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "view_0.png"
            image_path.write_bytes(b"fake-image")
            localizer = LLMSpatialAlignmentLocalizer(
                room_graph=room_graph,
                grounding_index=GroundingIndex(grounding),
                alignment_mode="direct_images",
                entity_distribution_mode="llm",
                response_client=response_client,
            )
            localization = localizer.localize(
                observation=Observation(
                    pano_id="pano-8",
                    views=[RenderedView(label="north", heading=330.0, path=str(image_path))],
                    entities=[
                        EntityDetection(
                            name="Assyria: Nimrud sign",
                            confidence=0.95,
                            kind="signage",
                            source_view="north",
                        )
                    ],
                    metadata={"floor": "0"},
                ),
                prior_room_belief={"Room 8": 1.0},
                fallback_room_id="Room 8",
            )

        self.assertTrue(localization["alignment_fusion_applied"])
        self.assertEqual(localization["predicted_room_id"], "Room 8")
        request_text = localizer.last_alignment_request_body["input"][0]["content"][0]["text"]
        self.assertIn("Candidate room Room 7", request_text)
        self.assertIn("Candidate room Room 8", request_text)
        self.assertNotIn("Candidate room Room 23", request_text)

    def test_llm_spatial_alignment_localizer_can_use_llm_entity_distribution(self) -> None:
        room_graph = normalize_room_graph(self.explicit_map)
        grounding = build_grounding_template(room_graph)
        responses = [
            {
                "output_text": json.dumps(
                    {
                        "predicted_room_id": "Room 23",
                        "confidence": 0.9,
                        "room_scores": [
                            {"room_id": "Room 7", "score": 0.0},
                            {"room_id": "Room 8", "score": 0.2},
                            {"room_id": "Room 9", "score": 0.0},
                            {"room_id": "Room 23", "score": 0.8},
                        ],
                        "summary": "Detected entities best match Room 23.",
                    }
                )
            },
            {
                "output_text": json.dumps(
                    {
                        "predicted_room_id": "Room 23",
                        "confidence": 0.75,
                        "view_0_allocentric_direction": "west",
                        "room_distribution": [
                            {"room_id": "Room 7", "score": 0.0},
                            {"room_id": "Room 8", "score": 0.25},
                            {"room_id": "Room 9", "score": 0.0},
                            {"room_id": "Room 23", "score": 0.75},
                        ],
                        "summary": "Alignment favors Room 23.",
                    }
                )
            },
        ]

        def response_client(_: dict) -> dict:
            return responses.pop(0)

        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "view_0.png"
            image_path.write_bytes(b"fake-image")
            localizer = LLMSpatialAlignmentLocalizer(
                room_graph=room_graph,
                grounding_index=GroundingIndex(grounding),
                alignment_mode="direct_images",
                entity_distribution_mode="llm",
                response_client=response_client,
            )
            localization = localizer.localize(
                observation=Observation(
                    pano_id="pano-23",
                    views=[RenderedView(label="north", heading=330.0, path=str(image_path))],
                    entities=[
                        EntityDetection(
                            name="Greek marble statue",
                            confidence=0.95,
                            kind="artwork",
                            source_view="north",
                        )
                    ],
                    metadata={"floor": "0"},
                ),
                prior_room_belief={"Room 8": 1.0},
                fallback_room_id="Room 8",
            )

        self.assertEqual(localization["entity_distribution_mode"], "llm")
        self.assertEqual(localization["predicted_room_id"], "Room 23")
        self.assertIsNotNone(localizer.last_entity_request_body)
        self.assertIsNotNone(localizer.last_entity_response_payload)
        self.assertNotIn("transition_support=", localizer.last_entity_request_body["input"])
        self.assertGreater(localization["entity_observation_distribution"]["Room 23"], localization["entity_observation_distribution"]["Room 8"])

    def test_spatial_engine_can_use_injected_llm_localizer(self) -> None:
        room_graph = normalize_room_graph(self.explicit_map)
        pano_graph = normalize_pano_graph(self.pano_graph)
        grounding = build_grounding_template(room_graph)
        localizer = LLMRoomLocalizer(
            room_graph=room_graph,
            grounding_index=GroundingIndex(grounding),
            response_client=lambda body: {
                "output_text": json.dumps(
                    {
                        "predicted_room_id": "Room 23",
                        "confidence": 0.88,
                        "evidence": ["Greek and Roman sculpture sign"],
                        "room_scores": [
                            {"room_id": "Room 7", "score": 0.2},
                            {"room_id": "Room 8", "score": 0.2},
                            {"room_id": "Room 9", "score": 0.2},
                            {"room_id": "Room 23", "score": 0.9},
                        ],
                        "summary": "Observation favors Room 23.",
                    }
                )
            },
        )
        spatial = SpatialEngine(
            room_graph=room_graph,
            pano_graph=pano_graph,
            grounding_index=GroundingIndex(grounding),
            localizer=localizer,
        )
        state = spatial.initialize(start_pano_id="pano-8", start_room_id="Room 8")
        updated = spatial.update(
            state,
            Observation(
                pano_id="pano-23",
                entities=[
                    EntityDetection(
                        name="Greek and Roman sculpture sign",
                        confidence=0.95,
                        kind="signage",
                        source_view="north",
                    )
                ],
                metadata={"floor": "0"},
            ),
        )

        self.assertEqual(updated.current_room_id, "Room 23")
        self.assertEqual(updated.room_belief["Room 23"], max(updated.room_belief.values()))

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

    def test_source_resolution_workflow_runs_parse_and_resolve_source_pano(self) -> None:
        room_graph = normalize_room_graph(self.explicit_map)
        grounding = build_grounding_template(room_graph)
        grounding["Room 8"]["pano_ids"] = ["pano-8"]

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
        workflow = SourceResolutionWorkflow(
            instruction_parser=parser,
            source_pano_resolver=SourcePanoResolver(GroundingIndex(grounding)),
        )

        result = workflow.run("Find the way from Room 8 to Room 23.")

        self.assertEqual(result.task.source_room_id, "Room 8")
        self.assertEqual(result.source_pano.pano_id, "pano-8")

    def test_episode_runner_passes_subgoal_and_alignment_context_to_policy(self) -> None:
        room_graph = normalize_room_graph(self.explicit_map)
        pano_graph = normalize_pano_graph(self.pano_graph)
        grounding = build_grounding_template(room_graph)

        class FakePerceptionProvider:
            def observe_from_manifest(self, manifest_path, *, current_heading):
                return Observation(
                    pano_id="pano-8",
                    views=[
                        RenderedView(label="sector_0", heading=0.0, path="/tmp/sector_0.png"),
                        RenderedView(label="sector_1", heading=90.0, path="/tmp/sector_1.png"),
                        RenderedView(label="sector_2", heading=180.0, path="/tmp/sector_2.png"),
                        RenderedView(label="sector_3", heading=270.0, path="/tmp/sector_3.png"),
                    ],
                    entities=[
                        EntityDetection(
                            name="west doorway",
                            confidence=0.92,
                            kind="passage",
                            source_view="sector_0",
                            metadata={"source_views": ["sector_0"]},
                        )
                    ],
                    metadata={
                        "localized_room_id": "Room 8",
                        "localization_confidence": 0.95,
                        "spatial_alignment": {"view_0_allocentric_direction": "west"},
                    },
                )

        class RecordingPolicy:
            def __init__(self) -> None:
                self.last_reasoning_input = None

            def choose_next_action(self, reasoning_input):
                self.last_reasoning_input = reasoning_input
                return PolicyOutput(action=None, rationale="stop after inspection")

        policy = RecordingPolicy()
        runner = EpisodeRunner(
            perception_provider=FakePerceptionProvider(),
            spatial_engine=SpatialEngine(
                room_graph=room_graph,
                pano_graph=pano_graph,
                grounding_index=GroundingIndex(grounding),
            ),
            policy=policy,
        )

        final_state, traces = runner.run(
            task=TaskSpec(
                task_type="gallery_goal_navigation",
                raw_instruction="Find the way from Room 8 to Room 23.",
                source_room_id="Room 8",
                goal_room_ids=["Room 23"],
            ),
            start_pano_id="pano-8",
            start_room_id="Room 8",
            manifest_paths={"pano-8": "/tmp/pano-8_manifest.json"},
            step_budget=1,
        )

        self.assertEqual(final_state.current_room_id, "Room 8")
        self.assertEqual(len(traces), 1)
        self.assertIsNotNone(policy.last_reasoning_input)
        self.assertEqual(policy.last_reasoning_input.subgoal_room_id, "Room 23")
        self.assertEqual(policy.last_reasoning_input.current_room_context["room_id"], "Room 8")
        self.assertEqual(policy.last_reasoning_input.visible_passages[0]["matched_room_ids"], ["Room 23"])
        self.assertEqual(
            policy.last_reasoning_input.spatial_alignment["view_0_allocentric_direction"],
            "west",
        )

    def test_serialize_trace_includes_localization_distributions(self) -> None:
        observation = Observation(
            pano_id="pano-8",
            heading_estimate=330.0,
            metadata={
                "localized_room_id": "Room 8",
                "grounded_room_id": "Room 8",
                "transition_support": {"Room 9": 0.4, "Room 8": 0.6},
                "entity_observation_distribution": {"Room 9": 0.2, "Room 8": 0.8},
                "alignment_observation_distribution": {"Room 9": 0.1, "Room 8": 0.9},
                "observation_likelihood": {"Room 9": 0.02, "Room 8": 0.72},
                "room_belief": {"Room 9": 0.1, "Room 8": 0.9},
                "spatial_alignment": {"alignment_predicted_room_id": "Room 8"},
            },
        )
        trace = mock.Mock()
        trace.step_index = 0
        trace.pano_id = "pano-8"
        trace.room_id = "Room 8"
        trace.route = ["Room 8", "Room 23"]
        trace.subgoal_room_id = "Room 23"
        trace.current_room_context = {"room_id": "Room 8"}
        trace.visible_passages = []
        trace.view_contexts = []
        trace.candidates = []
        trace.observation = observation
        trace.policy_output = PolicyOutput(action=None, rationale="done")
        trace.policy_request = {"model": "test-model"}
        trace.policy_response = {"output_text": "{}"}

        payload = EpisodeRunner._serialize_trace_payload(trace)

        self.assertEqual(payload["observation"]["transition_support"]["Room 8"], 0.6)
        self.assertEqual(payload["observation"]["entity_observation_distribution"]["Room 8"], 0.8)
        self.assertEqual(payload["observation"]["alignment_observation_distribution"]["Room 8"], 0.9)
        self.assertEqual(payload["observation"]["observation_likelihood"]["Room 8"], 0.72)
        self.assertEqual(payload["observation"]["room_belief"]["Room 8"], 0.9)

    def test_goal_reached_uses_grounded_pano_room_not_localized_room(self) -> None:
        room_graph = normalize_room_graph(self.explicit_map)
        pano_graph = normalize_pano_graph(self.pano_graph)
        grounding = build_grounding_template(room_graph)
        grounding["Room 8"]["pano_ids"] = ["pano-8", "pano-23"]
        grounding["Room 23"]["pano_ids"] = []
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
        self.assertEqual(updated.grounded_room_id, "Room 8")
        self.assertFalse(
            spatial.goal_reached(
                TaskSpec(
                    task_type="gallery_goal_navigation",
                    raw_instruction="Find the way from Room 8 to Room 23.",
                    source_room_id="Room 8",
                    goal_room_ids=["Room 23"],
                ),
                updated,
            )
        )

    def test_navigation_pipeline_runs_source_resolution_then_episode_runner(self) -> None:
        room_graph = normalize_room_graph(self.explicit_map)
        grounding = build_grounding_template(room_graph)
        grounding["Room 8"]["pano_ids"] = ["pano-8"]

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

        class FakeEpisodeRunner:
            def __init__(self) -> None:
                self.last_call = None

            def run(self, **kwargs):
                self.last_call = kwargs
                return {"final_pano_id": kwargs["start_pano_id"]}, ["trace-0"]

        runner = FakeEpisodeRunner()
        pipeline = NavigationPipeline(
            source_resolution_workflow=SourceResolutionWorkflow(
                instruction_parser=parser,
                source_pano_resolver=SourcePanoResolver(GroundingIndex(grounding)),
            ),
            episode_runner=runner,
        )

        result = pipeline.run("Find the way from Room 8 to Room 23.", step_budget=3)

        self.assertEqual(result.task.source_room_id, "Room 8")
        self.assertEqual(result.source.source_pano.pano_id, "pano-8")
        self.assertEqual(result.final_state["final_pano_id"], "pano-8")
        self.assertEqual(result.traces, ["trace-0"])
        self.assertEqual(runner.last_call["start_pano_id"], "pano-8")
        self.assertEqual(runner.last_call["start_room_id"], "Room 8")
        self.assertEqual(runner.last_call["step_budget"], 3)

    def test_grounding_index_can_resolve_primary_pano(self) -> None:
        room_graph = normalize_room_graph(self.explicit_map)
        grounding = build_grounding_template(room_graph)
        grounding["Room 8"]["pano_ids"] = ["pano-8"]
        grounding_index = GroundingIndex(grounding, pano_to_room={"mappings": {"pano-23": "Room 23"}})
        self.assertEqual(grounding_index.primary_pano_for_room("Room 8"), "pano-8")
        self.assertIsNone(grounding_index.primary_pano_for_room("Room 23"))
        self.assertEqual(grounding_index.room_for_pano("pano-8"), "Room 8")
        self.assertEqual(grounding_index.room_for_pano("pano-23"), "Room 23")
        self.assertIsNone(grounding_index.room_for_pano("missing"))

    def test_model_response_client_retries_timeout_errors(self) -> None:
        client = ModelResponseClient(
            provider="openai",
            api_key="test-key",
            api_base="https://api.example.com/v1",
            request_timeout=0.01,
        )

        attempts = {"count": 0}

        def flaky_urlopen(*args, **kwargs):
            attempts["count"] += 1
            if attempts["count"] < 3:
                raise TimeoutError("timed out")

            class _Response:
                def __enter__(self_inner):
                    return self_inner

                def __exit__(self_inner, exc_type, exc, tb):
                    return False

                def read(self_inner):
                    return b'{"output_text":"{\\"ok\\": true}"}'

            return _Response()

        with mock.patch("urllib.request.urlopen", side_effect=flaky_urlopen):
            payload = client.create({"model": "test-model", "input": "hello"})

        self.assertEqual(attempts["count"], 3)
        self.assertEqual(payload["output_text"], '{"ok": true}')


if __name__ == "__main__":
    unittest.main()
