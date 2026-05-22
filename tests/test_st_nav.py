from __future__ import annotations

import io
import importlib.util
import json
import os
import random
import subprocess
import sys
import tempfile
import unittest
import urllib.error
from unittest import mock
from pathlib import Path

from st_nav import (
    CandidateAction,
    EvidenceScoreLocalizer,
    EntityDetection,
    EpisodeRunner,
    GroundingIndex,
    InstructionRoutePlanner,
    LLMActionPolicy,
    LLMInstructionParser,
    ManifestPerceptionProvider,
    ModelEnvironment,
    ModelResponseClient,
    MultiViewAggregator,
    NavigationPipeline,
    NavigationPipelineConfig,
    Observation,
    PerceptionPipeline,
    PanoramaRenderer,
    PolicyOutput,
    ReasoningInput,
    RenderedView,
    SpatialAlignmentRefiner,
    SourcePanoResolver,
    SourceResolutionWorkflow,
    SpatialEngine,
    TaskSpec,
    ViewDetector,
    build_spatial_context_extraction_instructions,
    build_spatial_context_extraction_schema,
    build_visual_detection_localization_input,
    build_visual_detection_localization_instructions,
    build_visual_detection_localization_schema,
    build_navigation_pipeline,
    extract_output_text,
    load_dotenv,
    resolve_model_environment,
    resolve_task_num_ctx,
)
from st_nav.common.room_profiles import room_candidate_payload
from st_nav_data.normalize import (
    BRITISH_MUSEUM_DIRECTION_OVERRIDES,
    BRITISH_MUSEUM_EXCLUDED_EDGES,
    BRITISH_MUSEUM_EXPERIMENT_ROOM_IDS,
    BRITISH_MUSEUM_ROOM_CANONICAL_IDS,
    BRITISH_MUSEUM_TRANSITION_OVERRIDES,
    normalize_pano_graph,
    normalize_room_graph,
)
from st_nav_data.pano_room_grounding import build_room_grounding_from_pano_room_mapping


def build_test_grounding(room_graph: dict, pano_room_grounding: dict | None = None) -> dict:
    return build_room_grounding_from_pano_room_mapping(room_graph, pano_room_grounding or {})


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
    @staticmethod
    def _load_pano_perception_eval_module():
        module_path = Path(__file__).resolve().parents[1] / "tools/evaluation/localization/eval_localization.py"
        script_dir = str(module_path.parent)
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)
        spec = importlib.util.spec_from_file_location("eval_localization_test", module_path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module

    @staticmethod
    def _load_parse_instruction_eval_module():
        module_path = Path(__file__).resolve().parents[1] / "tools/evaluation/parse_instruction/eval_parse_instruction.py"
        script_dir = str(module_path.parent)
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)
        spec = importlib.util.spec_from_file_location("eval_parse_instruction_test", module_path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module

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
            "ST_NAV_REQUEST_TIMEOUT": os.environ.get("ST_NAV_REQUEST_TIMEOUT"),
            "ST_NAV_PROFILE_OLLAMA_REQUEST_TIMEOUT": os.environ.get("ST_NAV_PROFILE_OLLAMA_REQUEST_TIMEOUT"),
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
            os.environ.pop("ST_NAV_REQUEST_TIMEOUT", None)
            os.environ.pop("ST_NAV_PROFILE_OLLAMA_REQUEST_TIMEOUT", None)
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

    def test_model_response_client_retries_http_500_errors(self) -> None:
        client = ModelResponseClient(
            api_key="test-key",
            api_base="https://example.test/v1",
            api_kind="responses",
        )
        request_body = {"model": "demo-model"}

        transient_error = urllib.error.HTTPError(
            url="https://example.test/v1/responses",
            code=500,
            msg="Internal Server Error",
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

    def test_british_museum_experiment_rooms_include_grounded_floor_zero_theme_rooms(self) -> None:
        required_room_ids = {
            "Room 1",
            "Room 2",
            "Room 11",
            "Room 24",
            "Room 26",
            "Room 27",
            "Room 29a",
            "Room 29b",
        }
        self.assertTrue(required_room_ids.issubset(BRITISH_MUSEUM_EXPERIMENT_ROOM_IDS))

    def test_normalize_room_graph_includes_room_11_between_room_6_and_room_12(self) -> None:
        explicit_map = {
            "Room 6 bottom": {
                "name": "Room 6",
                "Level": 0,
                "category": "Ancient Greece and Rome",
                "title": "Early Greece",
                "links": [{"direction": "left", "name": "Room 11"}],
            },
            "Room 11": {
                "name": "Room 11",
                "Level": 0,
                "category": "Ancient Greece and Rome",
                "title": "Greece: Cycladic Islands",
                "links": [
                    {"direction": "right", "name": "Room 6 bottom"},
                    {"direction": "left", "name": "Room 12"},
                ],
            },
            "Room 12": {
                "name": "Room 12",
                "Level": 0,
                "category": "Ancient Greece and Rome",
                "title": "Greece: Minoans and Mycenaeans",
                "links": [{"direction": "right", "name": "Room 11"}],
            },
        }

        room_graph = normalize_room_graph(
            explicit_map,
            allowed_room_ids={"Room 6", "Room 11", "Room 12"},
            canonical_room_ids=BRITISH_MUSEUM_ROOM_CANONICAL_IDS,
        )
        self.assertEqual(room_graph["Room 11"]["category"], "Ancient Greece and Rome")
        self.assertEqual(room_graph["Room 11"]["title"], "Greece: Cycladic Islands")
        east_edge = next(edge for edge in room_graph["Room 11"]["neighbors"] if edge["target_room_id"] == "Room 6")
        west_edge = next(edge for edge in room_graph["Room 11"]["neighbors"] if edge["target_room_id"] == "Room 12")
        self.assertEqual(east_edge["allocentric_direction"], "east")
        self.assertEqual(west_edge["allocentric_direction"], "west")

    def test_normalize_room_graph_preserves_list_titles_as_theme_aliases(self) -> None:
        room_graph = normalize_room_graph(
            {
                "Room 24": {
                    "name": "Room 24",
                    "Level": 0,
                    "category": "Themes",
                    "title": ["Living and Dying", "The Wellcome Trust Gallery"],
                    "links": [],
                }
            }
        )

        self.assertEqual(room_graph["Room 24"]["title"], "Living and Dying; The Wellcome Trust Gallery")
        self.assertIn("Living and Dying", room_graph["Room 24"]["aliases"])
        self.assertIn("The Wellcome Trust Gallery", room_graph["Room 24"]["aliases"])

    def test_normalize_room_graph_includes_room_29_india_candidates(self) -> None:
        room_graph = normalize_room_graph(
            {
                "Room 29a": {
                    "name": "Room 29a",
                    "Level": 0,
                    "category": "Asia",
                    "title": "India",
                    "links": [{"direction": "right", "name": "Room 29b"}],
                },
                "Room 29b": {
                    "name": "Room 29b",
                    "Level": 0,
                    "category": "Asia",
                    "title": "India",
                    "links": [
                        {"direction": "left", "name": "Room 29a"},
                        {"direction": "right", "name": "Room 24"},
                    ],
                },
                "Room 24": {
                    "name": "Room 24",
                    "Level": 0,
                    "category": "Themes",
                    "title": "Living and Dying",
                    "links": [{"direction": "left", "name": "Room 29b"}],
                },
            },
            allowed_room_ids={"Room 24", "Room 29a", "Room 29b"},
        )

        self.assertEqual(room_graph["Room 29a"]["category"], "Asia")
        self.assertEqual(room_graph["Room 29a"]["title"], "India")
        self.assertEqual(room_graph["Room 29b"]["category"], "Asia")
        self.assertEqual(room_graph["Room 29b"]["title"], "India")
        edge_29a_to_29b = next(edge for edge in room_graph["Room 29a"]["neighbors"] if edge["target_room_id"] == "Room 29b")
        edge_29b_to_29a = next(edge for edge in room_graph["Room 29b"]["neighbors"] if edge["target_room_id"] == "Room 29a")
        edge_29b_to_24 = next(edge for edge in room_graph["Room 29b"]["neighbors"] if edge["target_room_id"] == "Room 24")
        edge_24_to_29b = next(edge for edge in room_graph["Room 24"]["neighbors"] if edge["target_room_id"] == "Room 29b")
        self.assertEqual(edge_29a_to_29b["allocentric_direction"], "east")
        self.assertEqual(edge_29b_to_29a["allocentric_direction"], "west")
        self.assertEqual(edge_29b_to_24["allocentric_direction"], "east")
        self.assertEqual(edge_24_to_29b["allocentric_direction"], "west")

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

    def test_llm_instruction_parser_supports_artwork_gallery_instruction_following(self) -> None:
        room_graph = normalize_room_graph(
            {
                **self.explicit_map,
                "Room 6": {
                    "name": "Room 6",
                    "Level": 0,
                    "category": "Middle East",
                    "title": "Assyrian sculpture and Balawat Gates",
                    "links": [],
                },
            }
        )
        parser = LLMInstructionParser(
            room_graph=room_graph,
            api_key="test-key",
            response_client=lambda body: {
                "output_text": json.dumps(
                    {
                        "task_type": "artwork_gallery_instruction_following_navigation",
                        "source_room_id": "Room 6",
                        "source_entity": {
                            "name": "Room 6",
                            "entity_type": "gallery",
                            "predicted_room_id": "Room 6",
                            "confidence": 1.0,
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
                            }
                        ],
                    }
                )
            },
        )

        task = parser.parse("Find the way from Room 6, passing the Lamassu, to the Townley Venus.")

        self.assertEqual(task.task_type, "artwork_gallery_instruction_following_navigation")
        self.assertEqual(task.source_room_id, "Room 6")
        self.assertEqual(task.source_entity.entity_type, "gallery")
        self.assertEqual(task.goal_room_ids, ["Room 23"])
        self.assertEqual(task.waypoint_room_ids, ["Room 8"])
        self.assertEqual(task.waypoint_entities[0].entity_type, "artwork")

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
        room_graph = normalize_room_graph(self.explicit_map)
        grounding = build_test_grounding(room_graph)
        downloads = []
        captured_bodies = []
        response_payload = {
            "entities": [
                {
                    "name": "Lamassu",
                    "kind": "artwork",
                    "confidence": 0.95,
                    "source_views": list(MUSEUM_CAPTURE_LABELS),
                    "location_scope": "inside",
                },
            ],
            "visual_localization": {
                "predicted_room_id": "Room 8",
                "room_scores": [
                    {
                        "room_id": "Room 8",
                        "evidence_type": "room_specific",
                        "score": 8.0,
                        "reason": "Lamassu supports Nimrud.",
                    }
                ],
                "evidence_entities": ["Lamassu"],
                "summary": "Room 8 likely.",
            },
        }

        def fake_downloader(url: str, output_path: Path) -> None:
            downloads.append((url, output_path))
            output_path.write_bytes(b"fake-image")

        detector = ViewDetector(
            api_key="test-key",
            room_graph=room_graph,
            grounding_index=GroundingIndex(grounding),
            response_client=lambda body: captured_bodies.append(body) or {
                "output_text": json.dumps(response_payload)
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
            "Return independent evidence scores for every candidate room",
            captured_bodies[0]["instructions"],
        )
        self.assertIn(
            "These are 8 overlapping views from the same panorama.",
            captured_bodies[0]["input"][0]["content"][0]["text"],
        )
        self.assertIn("Candidate rooms:", captured_bodies[0]["input"][0]["content"][0]["text"])
        self.assertEqual(detector.last_traces[0]["capture_label"], "multiview")
        self.assertEqual(detector.last_traces[0]["capture_labels"], MUSEUM_CAPTURE_LABELS)
        self.assertEqual(
            detector.last_traces[0]["request"]["input"][0]["content"][2]["image_url"],
            "<IMAGE_DATA_URL_OMITTED>",
        )
        self.assertEqual(
            detector.last_traces[0]["response"]["output_text"],
            json.dumps(response_payload),
        )

    def test_visual_detection_localization_prompt_and_schema_include_scope_and_distribution(self) -> None:
        schema = build_visual_detection_localization_schema(["Room 8", "Room 23"])
        instructions = build_visual_detection_localization_instructions()
        visual_input = build_visual_detection_localization_input(
            captures=[{"label": "north", "heading": 330.0}],
            candidates=[
                {
                    "room_id": "Room 8",
                    "title": "Assyria: Nimrud",
                    "category": "Middle East",
                    "aliases": ["Room 8"],
                    "anchor_entities": ["Assyria: Nimrud"],
                    "visual_cues": ["lamassu reliefs", "Nimrud palace sculpture"],
                    "possible_text_labels": ["Assyria: Nimrud"],
                    "negative_cues": ["Do not prefer Room 10 for generic Assyrian sculpture alone."],
                }
            ],
        )

        entity_schema = schema["properties"]["entities"]["items"]
        self.assertIn("location_scope", entity_schema["properties"])
        self.assertEqual(entity_schema["properties"]["location_scope"]["enum"], ["inside", "outside", "unknown"])
        self.assertIn("visual_localization", schema["properties"])
        room_score_schema = schema["properties"]["visual_localization"]["properties"]["room_scores"]["items"]
        self.assertIn("evidence_type", room_score_schema["properties"])
        self.assertIn("reason", room_score_schema["properties"])
        self.assertIn("Reason internally before answering", instructions)
        self.assertIn("use inside entities as the primary evidence", instructions)
        self.assertIn("direct_room_label", instructions)
        self.assertIn("shared_theme", instructions)
        self.assertIn("not probabilities", instructions)
        self.assertIn("Candidate rooms:", visual_input)
        self.assertIn("location_scope=inside", visual_input)
        self.assertIn("visual_cues=lamassu reliefs, Nimrud palace sculpture", visual_input)
        self.assertNotIn("text_labels=Assyria: Nimrud", visual_input)
        self.assertNotIn("negative_cues=Do not prefer Room 10", visual_input)
        self.assertIn("room_scores", visual_input)

    def test_room_visual_profiles_feed_grounding_and_candidate_payloads(self) -> None:
        room_graph = {
            "Room 8": {
                "room_id": "Room 8",
                "display_name": "Room 8",
                "floor": "0",
                "category": "Middle East",
                "title": "Assyria: Nimrud",
                "aliases": ["Room 8"],
                "neighbors": [],
                "visual_profile": {
                    "short_description": "Nimrud gallery with Assyrian palace reliefs.",
                    "visual_cues": ["lamassu reliefs", "carved palace panels"],
                    "possible_text_labels": ["Assyria: Nimrud"],
                    "negative_cues": ["A Lachish siege label points toward Room 10."],
                },
            }
        }
        grounding = build_test_grounding(room_graph)
        self.assertIn("lamassu reliefs", grounding["Room 8"]["anchor_entities"])

        candidate = room_candidate_payload(room_id="Room 8", node=room_graph["Room 8"], entry=grounding["Room 8"])
        visual_input = build_visual_detection_localization_input(captures=[], candidates=[candidate])
        self.assertIn("short_description=Nimrud gallery", visual_input)
        self.assertIn("visual_cues=lamassu reliefs, carved palace panels", visual_input)
        self.assertNotIn("anchors=", visual_input)
        self.assertNotIn("text_labels=Assyria: Nimrud", visual_input)

    def test_room_candidate_payload_uses_anchor_entities_without_visual_profile(self) -> None:
        room_graph = {
            "Room 8": {
                "room_id": "Room 8",
                "display_name": "Room 8",
                "floor": "0",
                "category": "Middle East",
                "title": "Assyria: Nimrud",
                "aliases": ["Room 8"],
                "neighbors": [],
            }
        }
        grounding = build_test_grounding(room_graph)

        candidate = room_candidate_payload(room_id="Room 8", node=room_graph["Room 8"], entry=grounding["Room 8"])
        visual_input = build_visual_detection_localization_input(captures=[], candidates=[candidate])
        self.assertIn("anchors=Assyria: Nimrud, Middle East", visual_input)

    def test_normalized_artifact_loader_prefers_visual_profile_room_graph(self) -> None:
        from st_nav.cli._common import load_normalized_artifacts

        with tempfile.TemporaryDirectory() as tmpdir:
            artifacts_dir = Path(tmpdir)
            (artifacts_dir / "room_graph.json").write_text(
                json.dumps({"Room 8": {"title": "Old title"}}),
                encoding="utf-8",
            )
            (artifacts_dir / "room_graph_with_visual_profiles.json").write_text(
                json.dumps(
                    {
                        "Room 8": {
                            "title": "Assyria: Nimrud",
                            "visual_profile": {"visual_cues": ["lamassu reliefs"]},
                        }
                    }
                ),
                encoding="utf-8",
            )

            artifacts = load_normalized_artifacts(artifacts_dir, room_graph=True)

        self.assertEqual(artifacts.room_graph["Room 8"]["title"], "Assyria: Nimrud")
        self.assertEqual(artifacts.room_graph["Room 8"]["visual_profile"]["visual_cues"], ["lamassu reliefs"])

    def test_view_detector_extracts_view_themes_without_heading_labels(self) -> None:
        room_graph = normalize_room_graph(self.explicit_map)
        grounding = build_test_grounding(room_graph)
        calls = []

        def fake_response(body):
            calls.append(body)
            if body["text"]["format"]["name"] == "view_theme_extraction":
                return {
                    "output_text": json.dumps(
                        {
                            "view_theme_observations": [
                                {
                                    "view_id": "view_0",
                                    "observed_theme": "Assyria: Nimrud",
                                    "confidence": 0.82,
                                    "visible_room_label": None,
                                    "evidence": ["relief panels"],
                                    "current_or_adjacent": "ambiguous",
                                    "reason": "Assyrian reliefs are visible.",
                                }
                            ],
                            "summary": "Assyrian themes visible.",
                        }
                    )
                }
            return {
                "output_text": json.dumps(
                    {
                        "entities": [],
                        "visual_localization": {
                            "predicted_room_id": "Room 8",
                            "room_scores": [{"room_id": "Room 8", "evidence_type": "room_specific", "score": 8.0, "reason": "Nimrud theme."}],
                            "evidence_entities": [],
                            "summary": "Room 8 likely.",
                        },
                    }
                )
            }

        detector = ViewDetector(
            api_key="test-key",
            room_graph=room_graph,
            grounding_index=GroundingIndex(grounding),
            response_client=fake_response,
            enable_view_themes=True,
            use_detection_files=False,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            image_path = tmp / "view.png"
            image_path.write_bytes(b"fake-image")
            manifest_path = tmp / "pano-8_manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "pano_id": "pano-8",
                        "floor": "0",
                        "captures": [{"label": "east_to_south", "heading": 78.0, "path": str(image_path)}],
                    }
                ),
                encoding="utf-8",
            )

            detections = detector.detect(manifest_path)
            observation = MultiViewAggregator({}).aggregate(
                manifest_path,
                current_heading=330.0,
                view_detections=detections,
            )

        self.assertEqual(len(calls), 2)
        theme_request = next(body for body in calls if body["text"]["format"]["name"] == "view_theme_extraction")
        theme_text = json.dumps(theme_request)
        self.assertNotIn("east_to_south", theme_text)
        self.assertNotIn("78.0", theme_text)
        self.assertEqual(observation.metadata["view_theme_observations"][0]["view_id"], "view_0")
        self.assertEqual(observation.metadata["view_theme_observations"][0]["observed_theme"], "Assyria: Nimrud")

    def test_view_detector_can_parse_integrated_visual_localization(self) -> None:
        room_graph = normalize_room_graph(self.explicit_map)
        grounding = build_test_grounding(room_graph)
        captured_bodies = []
        detector = ViewDetector(
            api_key="test-key",
            room_graph=room_graph,
            grounding_index=GroundingIndex(grounding),
            response_client=lambda body: captured_bodies.append(body) or {
                "output_text": json.dumps(
                    {
                        "entities": [
                            {
                                "name": "Lamassu",
                                "kind": "artwork",
                                "confidence": 0.95,
                                "source_views": ["north"],
                                "location_scope": "inside",
                            },
                            {
                                "name": "Greek sculpture glimpsed through doorway",
                                "kind": "artwork",
                                "confidence": 0.75,
                                "source_views": ["west"],
                                "location_scope": "outside",
                            },
                        ],
                        "visual_localization": {
                            "predicted_room_id": "Room 8",
                            "room_scores": [
                                {
                                    "room_id": "Room 7",
                                    "evidence_type": "shared_theme",
                                    "score": 5.0,
                                    "reason": "Assyrian theme is plausible but weaker here.",
                                },
                                {
                                    "room_id": "Room 8",
                                    "evidence_type": "room_specific",
                                    "score": 8.0,
                                    "reason": "Lamassu is strong support for this candidate.",
                                },
                                {
                                    "room_id": "Room 9",
                                    "evidence_type": "shared_theme",
                                    "score": 5.0,
                                    "reason": "Assyrian theme is plausible but weaker here.",
                                },
                                {
                                    "room_id": "Room 23",
                                    "evidence_type": "weak_generic",
                                    "score": 5.0,
                                    "reason": "Generic sculpture overlap only.",
                                },
                            ],
                            "evidence_entities": ["Lamassu"],
                            "summary": "Inside evidence matches Assyria: Nimrud.",
                        },
                    }
                )
            },
            use_detection_files=False,
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
                        "floor": "0",
                        "captures": [{"label": "north", "heading": 330.0, "path": str(image_path)}],
                    }
                ),
                encoding="utf-8",
            )

            detections = detector.detect(manifest_path)
            observation = MultiViewAggregator({}).aggregate(
                manifest_path,
                current_heading=330.0,
                view_detections=detections,
            )

        self.assertEqual(len(captured_bodies), 1)
        self.assertIn("visual_detection_localization", json.dumps(captured_bodies[0]))
        self.assertEqual(observation.entities[0].location_scope, "inside")
        self.assertEqual(observation.entities[1].location_scope, "outside")
        self.assertEqual(len(observation.metadata["inside_entities"]), 1)
        self.assertEqual(len(observation.metadata["outside_entities"]), 1)
        self.assertEqual(observation.metadata["visual_localization"]["predicted_room_id"], "Room 8")
        self.assertIn("room_scores", observation.metadata["visual_localization"])
        room_score = next(
            record
            for record in observation.metadata["visual_localization"]["room_scores"]
            if record["room_id"] == "Room 8"
        )
        self.assertEqual(room_score["evidence_type"], "room_specific")
        self.assertEqual(room_score["reason"], "Lamassu is strong support for this candidate.")
        self.assertIn("room_distribution", observation.metadata["visual_localization"])
        distribution = {
            record["room_id"]: record["score"]
            for record in observation.metadata["visual_localization"]["room_distribution"]
        }
        self.assertGreater(distribution["Room 8"], distribution["Room 9"])
        self.assertLess(distribution["Room 8"], 0.9)

    def test_pano_perception_eval_samples_up_to_limit_per_room(self) -> None:
        eval_module = self._load_pano_perception_eval_module()
        grounding_payload = {
            "mappings": {
                "r8-a": "Room 8",
                "r8-b": "Room 8",
                "r8-c": "Room 8",
                "r9-a": "Room 9",
                "r9-b": "Room 9",
                "r10-a": "Room 10",
                "missing": "null",
            }
        }

        samples = eval_module.room_samples_from_grounding(
            grounding_payload,
            room_ids=None,
            samples_per_room=2,
            seed=0,
        )

        counts = {}
        for sample in samples:
            counts[sample["room_id"]] = counts.get(sample["room_id"], 0) + 1
        self.assertEqual(counts, {"Room 10": 1, "Room 8": 2, "Room 9": 2})

    def test_pano_perception_eval_ranking_and_summary_metrics(self) -> None:
        eval_module = self._load_pano_perception_eval_module()

        ranking = eval_module.ranking_payload(
            {"Room 10": 0.4, "Room 7": 0.3, "Room 8": 0.2, "Room 9": 0.1},
            "Room 8",
        )
        self.assertEqual(ranking["rank"], 3)
        self.assertEqual(ranking["top1"], "Room 10")
        self.assertEqual(ranking["top3"], ["Room 10", "Room 7", "Room 8"])
        self.assertEqual(ranking["top5"], ["Room 10", "Room 7", "Room 8", "Room 9"])

        records = [
            {
                "status": "scored",
                "expected_room_id": "Room 8",
                "observation_only": {"rank": 1, "top1": "Room 8", "top3": ["Room 8"], "top5": ["Room 8"]},
            },
            {
                "status": "scored",
                "expected_room_id": "Room 9",
                "observation_only": {
                    "rank": 3,
                    "top1": "Room 10",
                    "top3": ["Room 10", "Room 8", "Room 9"],
                    "top5": ["Room 10", "Room 8", "Room 9"],
                },
            },
            {
                "status": "scored",
                "expected_room_id": "Room 7",
                "observation_only": {
                    "rank": None,
                    "top1": "Room 10",
                    "top3": ["Room 10", "Room 8", "Room 9"],
                    "top5": ["Room 10", "Room 8", "Room 9"],
                },
            },
        ]

        metrics = eval_module.rank_metrics(records, "observation_only")
        self.assertEqual(metrics["sample_count"], 3)
        self.assertEqual(metrics["scored_count"], 3)
        self.assertEqual(metrics["ranked_count"], 2)
        self.assertAlmostEqual(metrics["top1_accuracy"], 1 / 3)
        self.assertAlmostEqual(metrics["top3_accuracy"], 2 / 3)
        self.assertAlmostEqual(metrics["top5_accuracy"], 2 / 3)
        self.assertAlmostEqual(metrics["mrr"], (1.0 + 1 / 3) / 2)
        self.assertAlmostEqual(metrics["mean_rank"], 2.0)

        summary = eval_module.score_results(
            [
                {
                    **record,
                    "prior_fused": record["observation_only"],
                }
                for record in records
            ]
        )
        self.assertIn("observation_only", summary)
        self.assertIn("prior_fused", summary)
        self.assertIn("per_room", summary)
        self.assertAlmostEqual(summary["observation_only"]["top3_accuracy"], 2 / 3)
        self.assertIn("Room 8", summary["per_room"]["observation_only"])

        spatial_summary = eval_module.score_results(
            [
                {
                    **record,
                    "prior_fused": record["observation_only"],
                    "spatial_aligned": record["observation_only"],
                }
                for record in records
            ]
        )
        self.assertIn("spatial_aligned", spatial_summary)
        self.assertIn("spatial_aligned", spatial_summary["per_room"])

    def test_integrated_visual_eval_uses_existing_localizer_with_gt_prior(self) -> None:
        eval_module = self._load_pano_perception_eval_module()
        room_graph = normalize_room_graph(self.explicit_map)
        grounding = build_test_grounding(room_graph)
        captured = {}

        class FakeEvidenceScoreLocalizer:
            def __init__(self, **kwargs):
                captured["room_graph"] = kwargs["room_graph"]
                captured["grounding_index"] = kwargs["grounding_index"]

            def localize(self, *, observation, prior_room_belief, fallback_room_id):
                captured["visual_localization"] = observation.metadata.get("visual_localization")
                captured["prior_room_belief"] = dict(prior_room_belief)
                captured["fallback_room_id"] = fallback_room_id
                return {
                    "observation_distribution": {"Room 10": 0.5, "Room 8": 0.4, "Room 9": 0.1},
                    "transition_support": {"Room 8": 0.5, "Room 9": 0.5},
                    "room_belief": {"Room 8": 0.7, "Room 9": 0.2, "Room 10": 0.1},
                }

        payload = {
            "pano_id": "pano-8",
            "floor": "0",
            "current_heading": 330.0,
            "entities": [
                {
                    "name": "Lamassu",
                    "kind": "artwork",
                    "confidence": 0.95,
                    "source_views": ["north"],
                    "location_scope": "inside",
                }
            ],
            "visual_localization": {
                "room_scores": [
                    {
                        "room_id": "Room 8",
                        "evidence_type": "direct_room_label",
                        "score": 10.0,
                        "reason": "Legible Room 8 sign.",
                    }
                ],
                "room_distribution": [{"room_id": "Room 8", "score": 1.0}],
            },
        }

        with mock.patch.object(eval_module, "EvidenceScoreLocalizer", FakeEvidenceScoreLocalizer):
            localization = eval_module.localize_integrated_visual(
                payload=payload,
                expected_room_id="Room 8",
                room_graph=room_graph,
                grounding_index=GroundingIndex(grounding),
            )

        self.assertEqual(captured["prior_room_belief"], {"Room 8": 1.0})
        self.assertEqual(captured["fallback_room_id"], "Room 8")
        self.assertEqual(captured["visual_localization"], payload["visual_localization"])
        self.assertEqual(localization["room_belief"]["Room 8"], 0.7)

    def test_integrated_visual_eval_record_contains_observation_and_prior_rankings(self) -> None:
        eval_module = self._load_pano_perception_eval_module()
        room_graph = normalize_room_graph(self.explicit_map)
        grounding = build_test_grounding(room_graph)

        class FakeEvidenceScoreLocalizer:
            def __init__(self, **kwargs):
                pass

            def localize(self, *, observation, prior_room_belief, fallback_room_id):
                return {
                    "observation_distribution": {"Room 10": 0.5, "Room 8": 0.4, "Room 9": 0.1},
                    "transition_support": {"Room 8": 0.5, "Room 9": 0.5},
                    "room_belief": {"Room 8": 0.7, "Room 9": 0.2, "Room 10": 0.1},
                }

        args = eval_module.argparse.Namespace(
            pipeline="integrated-visual",
            reuse_existing_output=True,
            force=False,
        )
        sample = {"room_id": "Room 8", "pano_id": "pano-8"}
        payload = {
            "pano_id": "pano-8",
            "floor": "0",
            "current_heading": 330.0,
            "manifest_path": "/tmp/pano-8_manifest.json",
            "entities": [],
            "visual_localization": {
                "room_scores": [
                    {
                        "room_id": "Room 8",
                        "evidence_type": "direct_room_label",
                        "score": 10.0,
                        "reason": "Legible Room 8 sign.",
                    }
                ],
                "room_distribution": [{"room_id": "Room 8", "score": 1.0}],
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            manifest_path = output_dir / "pano-8_manifest.json"
            manifest_path.write_text("{}", encoding="utf-8")
            payload["manifest_path"] = str(manifest_path)
            output_path = eval_module.output_path_for_sample(output_dir, sample)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(payload), encoding="utf-8")
            with mock.patch.object(eval_module, "EvidenceScoreLocalizer", FakeEvidenceScoreLocalizer):
                record = eval_module.evaluate_sample(
                    args,
                    sample=sample,
                    output_dir=output_dir,
                    room_graph=room_graph,
                    grounding_index=GroundingIndex(grounding),
                )

        self.assertEqual(record["status"], "scored")
        self.assertEqual(record["observation_only"]["rank"], 2)
        self.assertEqual(record["observation_only"]["top1"], "Room 10")
        self.assertEqual(record["prior_fused"]["rank"], 1)
        self.assertEqual(record["prior_fused"]["top1"], "Room 8")
        self.assertEqual(record["observation_distribution"], {"Room 10": 0.5, "Room 8": 0.4, "Room 9": 0.1})
        self.assertEqual(record["transition_support"], {"Room 8": 0.5, "Room 9": 0.5})
        self.assertEqual(record["posterior_room_belief"], {"Room 8": 0.7, "Room 9": 0.2, "Room 10": 0.1})
        self.assertEqual(record["room_scores"][0]["evidence_type"], "direct_room_label")

    def test_pano_perception_eval_passes_detector_model_fov_and_no_detection_cache(self) -> None:
        eval_module = self._load_pano_perception_eval_module()
        args = eval_module.argparse.Namespace(
            artifacts_dir="dataset/sites/british_museum/normalized",
            render_output_dir="renders/test_eval",
            heading_mode="museum",
            pitch=0.0,
            fov=45,
            width=512,
            height=512,
            current_heading=330.0,
            render_api_key=None,
            llm_api_key=None,
            detector_model="gemma-4-31b-it",
            detector_api_kind=None,
            detector_api_base=None,
            vlm_timeout=None,
            no_detection_cache=True,
            pipeline="integrated-visual",
        )
        completed = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "sample.json"
            with mock.patch.object(eval_module.subprocess, "run", return_value=completed) as run_mock:
                result = eval_module.run_perception(
                    args,
                    sample={"room_id": "Room 8", "pano_id": "pano-8"},
                    output_path=output_path,
                )

        command = run_mock.call_args.args[0]
        self.assertEqual(result["returncode"], 0)
        self.assertIn("--fov", command)
        self.assertEqual(command[command.index("--fov") + 1], "45")
        self.assertIn("--detector-model", command)
        self.assertEqual(command[command.index("--detector-model") + 1], "gemma-4-31b-it")
        self.assertIn("--no-detection-cache", command)

    def test_spatial_ranking_from_localization_prepends_alignment_top_k(self) -> None:
        eval_module = self._load_pano_perception_eval_module()

        ranking = eval_module.spatial_ranking_from_localization(
            {
                "alignment_top_k": [
                    {"room_id": "Room 8", "score": 0.9},
                    {"room_id": "Room 23", "score": 0.4},
                ]
            },
            {"Room 23": 0.5, "Room 8": 0.3, "Room 9": 0.2},
        )

        self.assertEqual(ranking, ["Room 8", "Room 23", "Room 9"])

    def test_integrated_visual_eval_record_uses_runtime_spatial_refiner(self) -> None:
        eval_module = self._load_pano_perception_eval_module()
        room_graph = normalize_room_graph(self.explicit_map)
        grounding = build_test_grounding(room_graph)
        refine_calls = []

        class FakeSpatialAlignmentRefiner:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def refine(self, *, observation, candidate_room_ids):
                refine_calls.append({"view_count": len(observation.views), "candidate_room_ids": list(candidate_room_ids)})
                return {
                    "applied": True,
                    "alignment_top_k": [
                        {"room_id": "Room 8", "score": 0.92},
                        {"room_id": "Room 23", "score": 0.41},
                    ],
                    "alignment_predicted_room_id": "Room 8",
                    "alignment_evidence": ["Runtime refiner selected Room 8."],
                    "alignment_summary": "Room 8 is spatially best aligned.",
                    "spatial_alignment": {"mode": "text_from_images"},
                    "ego_spatial_context": {"summary": "views parsed"},
                }

        args = eval_module.argparse.Namespace(
            pipeline="integrated-visual",
            reuse_existing_output=True,
            force=False,
            enable_spatial_alignment=True,
            render_output_dir="unused",
            alignment_model="gemma-4-31b-it",
            alignment_timeout=None,
            alignment_candidate_ratio_threshold=0.5,
            alignment_candidate_max=5,
            llm_api_key="test-key",
            detector_api_base=None,
            detector_api_kind=None,
            vlm_timeout=None,
        )
        sample = {"room_id": "Room 8", "pano_id": "pano-8"}

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            image_path = output_dir / "view_0.png"
            image_path.write_bytes(b"fake-image")
            manifest_path = output_dir / "pano-8_manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "pano_id": "pano-8",
                        "floor": "0",
                        "captures": [{"label": "north", "heading": 330.0, "path": str(image_path)}],
                    }
                ),
                encoding="utf-8",
            )
            payload = {
                "pano_id": "pano-8",
                "floor": "0",
                "current_heading": 330.0,
                "manifest_path": str(manifest_path),
                "entities": [],
                "visual_localization": {
                    "room_scores": [
                        {"room_id": "Room 23", "score": 8.0},
                        {"room_id": "Room 8", "score": 7.9},
                    ]
                },
            }
            output_path = eval_module.output_path_for_sample(output_dir, sample)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(payload), encoding="utf-8")
            with mock.patch.object(eval_module, "SpatialAlignmentRefiner", FakeSpatialAlignmentRefiner):
                record = eval_module.evaluate_sample(
                    args,
                    sample=sample,
                    output_dir=output_dir,
                    room_graph=room_graph,
                    grounding_index=GroundingIndex(grounding),
                )

        self.assertEqual(record["status"], "scored")
        self.assertEqual(refine_calls[0]["view_count"], 1)
        self.assertGreaterEqual(len(refine_calls[0]["candidate_room_ids"]), 2)
        self.assertEqual(record["prior_fused"]["top1"], record["base_predicted_room_id"])
        self.assertEqual(record["posterior_room_belief"], eval_module.compact_distribution(record["posterior_room_belief"]))
        self.assertEqual(record["spatial_aligned"]["top1"], "Room 8")
        self.assertEqual(record["alignment_predicted_room_id"], "Room 8")
        self.assertEqual(record["spatial_alignment"]["mode"], "text_from_images")

    def test_view_detector_writes_and_reuses_integrated_detection_cache(self) -> None:
        room_graph = normalize_room_graph(self.explicit_map)
        grounding = build_test_grounding(room_graph)
        calls = []
        detector = ViewDetector(
            api_key="test-key",
            room_graph=room_graph,
            grounding_index=GroundingIndex(grounding),
            response_client=lambda body: calls.append(body) or {
                "output_text": json.dumps(
                    {
                        "entities": [
                            {
                                "name": "Lamassu",
                                "kind": "artwork",
                                "confidence": 0.95,
                                "source_views": ["north"],
                                "location_scope": "inside",
                            }
                        ],
                        "visual_localization": {
                            "predicted_room_id": "Room 8",
                            "confidence": 0.9,
                            "room_distribution": [
                                {"room_id": "Room 8", "score": 0.9},
                                {"room_id": "Room 9", "score": 0.05},
                                {"room_id": "Room 23", "score": 0.05},
                            ],
                            "evidence_entities": ["Lamassu"],
                            "summary": "Inside evidence matches Assyria: Nimrud.",
                        },
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
                        "floor": "0",
                        "captures": [{"label": "north", "heading": 330.0, "path": str(image_path)}],
                    }
                ),
                encoding="utf-8",
            )

            first = detector.detect(manifest_path)
            second = detector.detect(manifest_path)
            cache_payload = json.loads(
                manifest_path.with_name("pano-8_manifest_detections.json").read_text(encoding="utf-8")
            )

        self.assertEqual(len(calls), 1)
        self.assertEqual(first[0].entities[0].location_scope, "inside")
        self.assertEqual(second[0].entities[0].location_scope, "inside")
        self.assertEqual(cache_payload["cache_version"], 3)
        self.assertEqual(cache_payload["entities"][0]["location_scope"], "inside")
        self.assertEqual(cache_payload["visual_localization"]["predicted_room_id"], "Room 8")

    def test_view_detector_writes_and_reuses_view_theme_cache(self) -> None:
        room_graph = normalize_room_graph(self.explicit_map)
        grounding = build_test_grounding(room_graph)
        calls = []

        def fake_response(body):
            calls.append(body)
            if body["text"]["format"]["name"] == "view_theme_extraction":
                return {
                    "output_text": json.dumps(
                        {
                            "view_theme_observations": [
                                {
                                    "view_id": "view_0",
                                    "observed_theme": "Assyria: Nimrud",
                                    "confidence": 0.9,
                                    "visible_room_label": None,
                                    "evidence": ["reliefs"],
                                    "current_or_adjacent": "current",
                                    "reason": "Clear Assyrian reliefs.",
                                }
                            ],
                            "summary": "Nimrud theme.",
                        }
                    )
                }
            return {
                "output_text": json.dumps(
                    {
                        "entities": [],
                        "visual_localization": {
                            "predicted_room_id": "Room 8",
                            "room_scores": [{"room_id": "Room 8", "evidence_type": "room_specific", "score": 8.0, "reason": "Nimrud."}],
                            "evidence_entities": [],
                            "summary": "Room 8.",
                        },
                    }
                )
            }

        detector = ViewDetector(
            api_key="test-key",
            room_graph=room_graph,
            grounding_index=GroundingIndex(grounding),
            response_client=fake_response,
            enable_view_themes=True,
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
                        "floor": "0",
                        "captures": [{"label": "north", "heading": 330.0, "path": str(image_path)}],
                    }
                ),
                encoding="utf-8",
            )

            first = detector.detect(manifest_path)
            second = detector.detect(manifest_path)
            cache_payload = json.loads(
                manifest_path.with_name("pano-8_manifest_detections.json").read_text(encoding="utf-8")
            )

        self.assertEqual(len(calls), 2)
        self.assertEqual(first[0].metadata["view_theme_observations"][0]["observed_theme"], "Assyria: Nimrud")
        self.assertEqual(second[0].metadata["view_theme_observations"][0]["observed_theme"], "Assyria: Nimrud")
        self.assertEqual(cache_payload["view_theme_observations"][0]["view_id"], "view_0")

    def test_view_detector_refreshes_old_cache_when_view_themes_are_enabled(self) -> None:
        room_graph = normalize_room_graph(self.explicit_map)
        grounding = build_test_grounding(room_graph)
        calls = []

        def fake_response(body):
            calls.append(body)
            if body["text"]["format"]["name"] == "view_theme_extraction":
                return {
                    "output_text": json.dumps(
                        {
                            "view_theme_observations": [
                                {
                                    "view_id": "view_0",
                                    "observed_theme": "Assyria: Nimrud",
                                    "confidence": 0.9,
                                    "visible_room_label": None,
                                    "evidence": ["reliefs"],
                                    "current_or_adjacent": "current",
                                    "reason": "Clear Assyrian reliefs.",
                                }
                            ],
                            "summary": "Nimrud theme.",
                        }
                    )
                }
            return {
                "output_text": json.dumps(
                    {
                        "entities": [],
                        "visual_localization": {
                            "predicted_room_id": "Room 8",
                            "room_scores": [{"room_id": "Room 8", "evidence_type": "room_specific", "score": 8.0, "reason": "Nimrud."}],
                            "evidence_entities": [],
                            "summary": "Room 8.",
                        },
                    }
                )
            }

        detector = ViewDetector(
            api_key="test-key",
            room_graph=room_graph,
            grounding_index=GroundingIndex(grounding),
            response_client=fake_response,
            enable_view_themes=True,
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
                        "floor": "0",
                        "captures": [{"label": "north", "heading": 330.0, "path": str(image_path)}],
                    }
                ),
                encoding="utf-8",
            )
            manifest_path.with_name("pano-8_manifest_detections.json").write_text(
                json.dumps(
                    {
                        "cache_version": 2,
                        "entities": [],
                        "candidate_room_ids": ["Room 8", "Room 9", "Room 23"],
                        "visual_localization": {"predicted_room_id": "Room 8"},
                    }
                ),
                encoding="utf-8",
            )

            detections = detector.detect(manifest_path)
            cache_payload = json.loads(
                manifest_path.with_name("pano-8_manifest_detections.json").read_text(encoding="utf-8")
            )

        self.assertEqual(len(calls), 2)
        self.assertEqual(detections[0].metadata["view_theme_observations"][0]["view_id"], "view_0")
        self.assertIn("view_theme_observations", cache_payload)

    def test_view_detector_can_preserve_passage_kind(self) -> None:
        pano_graph = normalize_pano_graph(self.pano_graph)
        room_graph = normalize_room_graph(self.explicit_map)
        grounding = build_test_grounding(room_graph)
        downloads = []

        def fake_downloader(url: str, output_path: Path) -> None:
            downloads.append((url, output_path))
            output_path.write_bytes(b"fake-image")

        detector = ViewDetector(
            api_key="test-key",
            room_graph=room_graph,
            grounding_index=GroundingIndex(grounding),
            response_client=lambda body: {
                "output_text": json.dumps(
                    {
                        "entities": [
                            {
                                "name": "archway to next room",
                                "kind": "passage",
                                "confidence": 0.88,
                                "source_views": list(MUSEUM_CAPTURE_LABELS),
                                "location_scope": "inside",
                            },
                        ],
                        "visual_localization": {
                            "predicted_room_id": "Room 8",
                            "room_scores": [
                                {
                                    "room_id": "Room 8",
                                    "evidence_type": "weak_generic",
                                    "score": 4.0,
                                    "reason": "Passage evidence is generic.",
                                }
                            ],
                            "evidence_entities": ["archway to next room"],
                            "summary": "Passage preserved as entity evidence.",
                        },
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
        room_graph = normalize_room_graph(self.explicit_map)
        grounding = build_test_grounding(room_graph)
        calls = []
        detector = ViewDetector(
            api_key="test-key",
            room_graph=room_graph,
            grounding_index=GroundingIndex(grounding),
            response_client=lambda body: calls.append(body) or {
                "output_text": json.dumps(
                    {
                        "entities": [
                            {
                                "name": "north doorway",
                                "kind": "passage",
                                "confidence": 0.91,
                                "source_views": ["north", "north_to_east"],
                                "location_scope": "inside",
                            }
                        ],
                        "visual_localization": {
                            "predicted_room_id": "Room 8",
                            "room_scores": [
                                {
                                    "room_id": "Room 8",
                                    "evidence_type": "weak_generic",
                                    "score": 4.0,
                                    "reason": "Doorway evidence is generic.",
                                }
                            ],
                            "evidence_entities": ["north doorway"],
                            "summary": "Doorway preserved as passage evidence.",
                        },
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
                        "floor": "0",
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
        grounding = build_test_grounding(room_graph)
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
        grounding = build_test_grounding(room_graph)
        grounding["Room 9"]["pano_ids"] = ["pano-23"]
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
        self.assertNotEqual(updated.current_room_id, "Room 9")
        self.assertEqual(updated.grounded_room_id, "Room 9")

    def test_spatial_update_accepts_localized_room_from_observation_metadata(self) -> None:
        room_graph = normalize_room_graph(self.explicit_map)
        pano_graph = normalize_pano_graph(self.pano_graph)
        grounding = build_test_grounding(room_graph)
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
        grounding = build_test_grounding(room_graph)

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
        grounding = build_test_grounding(room_graph)
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
        grounding = build_test_grounding(room_graph)
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
        grounding = build_test_grounding(room_graph)
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
        grounding = build_test_grounding(room_graph)
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
        grounding = build_test_grounding(room_graph)
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
        grounding = build_test_grounding(room_graph)
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
        grounding = build_test_grounding(room_graph)
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
        grounding = build_test_grounding(room_graph)
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
        grounding = build_test_grounding(room_graph)
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
        grounding = build_test_grounding(room_graph)
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
        self.assertIn("Do not decide the current room", instructions)
        self.assertIn("return an empty themes list", instructions)
        self.assertIn("Do not guess a room theme from weak, cropped, or partial evidence", instructions)

        schema = build_spatial_context_extraction_schema(["view_0"])
        theme_properties = schema["properties"]["views"]["items"]["properties"]["themes"]["items"]["properties"]
        self.assertIn("label", theme_properties)
        self.assertNotIn("matched_theme", theme_properties)

    def test_build_current_room_context_includes_subgoal_theme_labels(self) -> None:
        room_graph = normalize_room_graph(self.explicit_map)
        pano_graph = normalize_pano_graph(self.pano_graph)
        grounding = build_test_grounding(room_graph)
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

    def test_evidence_score_localizer_combines_room_scores_with_transition_prior(self) -> None:
        room_graph = normalize_room_graph(self.explicit_map)
        grounding = build_test_grounding(room_graph)
        localizer = EvidenceScoreLocalizer(
            room_graph=room_graph,
            grounding_index=GroundingIndex(grounding),
            spatial_refiner=None,
        )

        localization = localizer.localize(
            observation=Observation(
                pano_id="pano-23",
                metadata={
                    "floor": "0",
                    "visual_localization": {
                        "room_scores": [
                            {"room_id": "Room 8", "score": 1.0},
                            {"room_id": "Room 9", "score": 0.0},
                            {"room_id": "Room 23", "score": 9.0},
                        ],
                        "evidence_entities": ["Greek marble statue"],
                        "summary": "Inside evidence matches Room 23.",
                    },
                },
            ),
            prior_room_belief={"Room 8": 1.0},
            fallback_room_id="Room 8",
        )

        self.assertEqual(localization["predicted_room_id"], "Room 23")
        self.assertGreater(localization["observation_likelihood"]["Room 23"], localization["observation_likelihood"]["Room 8"])
        self.assertGreater(localization["room_belief"]["Room 23"], localization["room_belief"]["Room 8"])
        self.assertEqual(localization["evidence"], ["Greek marble statue"])

    def test_evidence_score_localizer_selects_ratio_based_alignment_candidates(self) -> None:
        room_graph = normalize_room_graph(self.explicit_map)
        grounding = build_test_grounding(room_graph)
        localizer = EvidenceScoreLocalizer(
            room_graph=room_graph,
            grounding_index=GroundingIndex(grounding),
            alignment_candidate_ratio_threshold=0.5,
            alignment_candidate_max=2,
            spatial_refiner=None,
        )

        localization = localizer.localize(
            observation=Observation(
                pano_id="pano-8",
                metadata={
                    "floor": "0",
                    "visual_localization": {
                        "room_scores": [
                            {"room_id": "Room 8", "score": 8.0},
                            {"room_id": "Room 9", "score": 7.5},
                            {"room_id": "Room 23", "score": 1.0},
                        ]
                    },
                },
            ),
            prior_room_belief={},
            fallback_room_id=None,
        )

        self.assertEqual(localization["alignment_candidate_room_ids"], ["Room 8", "Room 9"])
        self.assertFalse(localization["alignment_applied"])
        self.assertEqual(localization["alignment_skipped_reason"], "missing_spatial_refiner")

    def test_evidence_score_localizer_skips_alignment_with_single_candidate(self) -> None:
        room_graph = normalize_room_graph(self.explicit_map)
        grounding = build_test_grounding(room_graph)
        localizer = EvidenceScoreLocalizer(
            room_graph=room_graph,
            grounding_index=GroundingIndex(grounding),
            alignment_candidate_ratio_threshold=0.5,
            spatial_refiner=None,
        )

        localization = localizer.localize(
            observation=Observation(
                pano_id="pano-8",
                metadata={
                    "floor": "0",
                    "visual_localization": {
                        "room_scores": [
                            {"room_id": "Room 8", "score": 10.0},
                            {"room_id": "Room 9", "score": 0.0},
                        ]
                    },
                },
            ),
            prior_room_belief={"Room 8": 1.0},
            fallback_room_id="Room 8",
        )

        self.assertEqual(localization["predicted_room_id"], "Room 8")
        self.assertEqual(localization["alignment_candidate_room_ids"], ["Room 8"])
        self.assertEqual(localization["alignment_skipped_reason"], "insufficient_alignment_candidates")

    def test_spatial_alignment_refiner_uses_rotation_aware_view_ids(self) -> None:
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
            "Room 23": {
                "name": "Room 23",
                "Level": 0,
                "category": "Ancient Greece and Rome",
                "title": "Greek and Roman sculpture",
                "links": [{"direction": "right", "name": "Room 8"}],
            },
        }
        room_graph = normalize_room_graph(explicit_map)
        grounding = build_test_grounding(room_graph)
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
                        "room_ranking": [
                            {"room_id": "Room 8", "score": 0.9},
                            {"room_id": "Room 23", "score": 0.1},
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

            refiner = SpatialAlignmentRefiner(
                room_graph=room_graph,
                grounding_index=GroundingIndex(grounding),
                response_client=response_client,
            )
            refinement = refiner.refine(
                observation=Observation(
                    pano_id="pano-8",
                    views=[
                        RenderedView(label="north", heading=330.0, path=str(image_paths[0])),
                        RenderedView(label="east", heading=60.0, path=str(image_paths[1])),
                    ],
                    metadata={"floor": "0"},
                ),
                candidate_room_ids=["Room 8", "Room 23"],
            )

        self.assertTrue(refinement["applied"])
        self.assertEqual(refinement["alignment_predicted_room_id"], "Room 8")
        self.assertEqual(refinement["alignment_top_k"][0]["room_id"], "Room 8")
        self.assertEqual(refinement["spatial_alignment"]["view_0_allocentric_direction"], "west")
        alignment_input = refiner.last_alignment_request_body["input"]
        self.assertIn("view_0", alignment_input)
        self.assertNotIn("Front", alignment_input)
        self.assertNotIn("front", alignment_input)

    def test_spatial_alignment_refiner_candidate_theme_labels_exclude_neighbor_visual_profiles(self) -> None:
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
        room_graph["Room 8"]["visual_profile"] = {
            "short_description": "Nimrud gallery with Assyrian palace reliefs.",
            "visual_cues": ["lamassu reliefs"],
        }
        room_graph["Room 23"]["visual_profile"] = {
            "short_description": "Sculpture court with marble statues visible through the opening.",
            "visual_cues": ["marble statues", "pedestal sculpture displays"],
        }
        grounding = build_test_grounding(room_graph)
        refiner = SpatialAlignmentRefiner(
            room_graph=room_graph,
            grounding_index=GroundingIndex(grounding),
            response_client=lambda _: {},
        )

        labels = refiner._candidate_theme_labels(["Room 8"])

        self.assertIn("Assyria: Nimrud", labels)
        self.assertIn("Nimrud gallery with Assyrian palace reliefs.", labels)
        self.assertIn("lamassu reliefs", labels)
        self.assertIn("Greek and Roman sculpture", labels)
        self.assertNotIn("Sculpture court with marble statues visible through the opening.", labels)
        self.assertNotIn("marble statues", labels)
        self.assertNotIn("pedestal sculpture displays", labels)

    def test_evidence_score_localizer_uses_alignment_top1_without_rewriting_room_belief(self) -> None:
        room_graph = normalize_room_graph(self.explicit_map)
        grounding = build_test_grounding(room_graph)
        responses = [
            {
                "output_text": json.dumps(
                    {
                        "views": [
                            {"view_id": "view_0", "themes": [{"label": "Room 23", "confidence": 0.8}], "summary": "Room 23 visible."}
                        ],
                        "summary": "Context extracted.",
                    }
                )
            },
            {
                "output_text": json.dumps(
                    {
                        "predicted_room_id": "Room 23",
                        "confidence": 0.82,
                        "view_0_allocentric_direction": "west",
                        "evidence": ["Spatial pattern favors Room 23."],
                        "room_ranking": [
                            {"room_id": "Room 8", "score": 0.2},
                            {"room_id": "Room 23", "score": 0.8},
                        ],
                        "summary": "Alignment reranks Room 23 first.",
                    }
                )
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "view_0.png"
            image_path.write_bytes(b"fake-image")
            localizer = EvidenceScoreLocalizer(
                room_graph=room_graph,
                grounding_index=GroundingIndex(grounding),
                alignment_candidate_ratio_threshold=0.5,
                spatial_refiner=SpatialAlignmentRefiner(
                    room_graph=room_graph,
                    grounding_index=GroundingIndex(grounding),
                    response_client=lambda _: responses.pop(0),
                ),
            )
            localization = localizer.localize(
                observation=Observation(
                    pano_id="pano-8",
                    views=[RenderedView(label="north", heading=330.0, path=str(image_path))],
                    metadata={
                        "floor": "0",
                        "visual_localization": {
                            "room_scores": [
                                {"room_id": "Room 8", "score": 8.0},
                                {"room_id": "Room 23", "score": 7.8},
                            ]
                        },
                    },
                ),
                prior_room_belief={},
                fallback_room_id=None,
            )

        self.assertEqual(localization["base_predicted_room_id"], "Room 8")
        self.assertEqual(localization["predicted_room_id"], "Room 23")
        self.assertEqual(localization["room_belief"], localization["base_room_belief"])
        self.assertEqual(localization["alignment_top_k"][0]["room_id"], "Room 23")
        self.assertTrue(localization["alignment_applied"])

    def test_evidence_score_localizer_records_skip_reason_when_alignment_lacks_views(self) -> None:
        room_graph = normalize_room_graph(self.explicit_map)
        grounding = build_test_grounding(room_graph)
        localizer = EvidenceScoreLocalizer(
            room_graph=room_graph,
            grounding_index=GroundingIndex(grounding),
            spatial_refiner=SpatialAlignmentRefiner(
                room_graph=room_graph,
                grounding_index=GroundingIndex(grounding),
                response_client=lambda _: {},
            ),
        )

        localization = localizer.localize(
            observation=Observation(
                pano_id="pano-8",
                metadata={
                    "floor": "0",
                    "visual_localization": {
                        "room_scores": [
                            {"room_id": "Room 8", "score": 8.0},
                            {"room_id": "Room 23", "score": 7.8},
                        ]
                    },
                },
            ),
            prior_room_belief={},
            fallback_room_id=None,
        )

        self.assertFalse(localization["alignment_applied"])
        self.assertEqual(localization["alignment_skipped_reason"], "missing_panorama_views")
        self.assertEqual(localization["predicted_room_id"], localization["base_predicted_room_id"])

    def test_spatial_engine_can_use_injected_evidence_score_localizer(self) -> None:
        room_graph = normalize_room_graph(self.explicit_map)
        pano_graph = normalize_pano_graph(self.pano_graph)
        grounding = build_test_grounding(room_graph)
        localizer = EvidenceScoreLocalizer(
            room_graph=room_graph,
            grounding_index=GroundingIndex(grounding),
            spatial_refiner=None,
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
                metadata={
                    "floor": "0",
                    "visual_localization": {
                        "room_scores": [
                            {"room_id": "Room 8", "score": 1.0},
                            {"room_id": "Room 23", "score": 9.0},
                        ]
                    },
                },
            ),
        )

        self.assertEqual(updated.current_room_id, "Room 23")
        self.assertEqual(updated.room_belief["Room 23"], max(updated.room_belief.values()))

    def test_instruction_route_planner_runs_parse_then_shortest_path(self) -> None:
        room_graph = normalize_room_graph(self.explicit_map)
        pano_graph = normalize_pano_graph(self.pano_graph)
        grounding = build_test_grounding(room_graph)

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
        grounding = build_test_grounding(room_graph)
        grounding["Room 8"]["pano_ids"] = ["pano-8", "pano-8b"]

        resolver = SourcePanoResolver(GroundingIndex(grounding), rng=random.Random(0))
        resolution = resolver.resolve("Room 8")
        self.assertEqual(resolution.source_room_id, "Room 8")
        self.assertIn(resolution.pano_id, ["pano-8", "pano-8b"])
        self.assertEqual(resolution.candidate_pano_ids, ["pano-8", "pano-8b"])
        self.assertEqual(resolution.resolution_method, "random_room_grounding")

    def test_source_pano_resolver_uses_compact_pano_room_grounding(self) -> None:
        resolver = SourcePanoResolver(
            GroundingIndex(
                pano_to_room={
                    "mappings": {
                        "pano-8": "Room 8",
                        "pano-8b": "Room 8",
                        "pano-23": "Room 23",
                        "pano-null": "null",
                    }
                }
            ),
            rng=random.Random(0),
        )

        resolution = resolver.resolve("Room 8")

        self.assertEqual(resolution.source_room_id, "Room 8")
        self.assertIn(resolution.pano_id, ["pano-8", "pano-8b"])
        self.assertEqual(resolution.candidate_pano_ids, ["pano-8", "pano-8b"])

    def test_source_resolution_workflow_runs_parse_and_resolve_source_pano(self) -> None:
        room_graph = normalize_room_graph(self.explicit_map)
        grounding = build_test_grounding(room_graph)
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
        grounding = build_test_grounding(room_graph)

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
                "evidence_distribution": {"Room 9": 0.2, "Room 8": 0.8},
                "base_room_belief": {"Room 9": 0.1, "Room 8": 0.9},
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
        self.assertEqual(payload["observation"]["evidence_distribution"]["Room 8"], 0.8)
        self.assertEqual(payload["observation"]["base_room_belief"]["Room 8"], 0.9)
        self.assertEqual(payload["observation"]["observation_likelihood"]["Room 8"], 0.72)
        self.assertEqual(payload["observation"]["room_belief"]["Room 8"], 0.9)

    def test_goal_reached_uses_grounded_pano_room_not_localized_room(self) -> None:
        room_graph = normalize_room_graph(self.explicit_map)
        pano_graph = normalize_pano_graph(self.pano_graph)
        grounding = build_test_grounding(room_graph)
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
        grounding = build_test_grounding(room_graph)
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

    def test_build_navigation_pipeline_assembles_runtime_components(self) -> None:
        room_graph = normalize_room_graph(self.explicit_map)
        pano_graph = normalize_pano_graph(
            {
                "pano-8": {
                    "panoID": "pano-8",
                    "floor": "0",
                    "lat": 0.0,
                    "lng": 0.0,
                    "links": [],
                }
            }
        )
        grounding = build_test_grounding(room_graph)
        grounding["Room 8"]["pano_ids"] = ["pano-8"]

        pipeline = build_navigation_pipeline(
            room_graph=room_graph,
            pano_graph=pano_graph,
            grounding_payload=grounding,
            config=NavigationPipelineConfig(
                llm_model="test-model",
                llm_api_key="test-key",
                llm_api_kind="responses",
                llm_api_base="https://example.test/v1",
                llm_timeout=1.0,
            ),
        )

        self.assertIsInstance(pipeline, NavigationPipeline)
        self.assertIsInstance(pipeline.episode_runner, EpisodeRunner)
        self.assertIsInstance(pipeline.source_resolution_workflow, SourceResolutionWorkflow)
        self.assertIsInstance(pipeline.episode_runner.spatial_engine.state_estimator.localizer, EvidenceScoreLocalizer)

    def test_grounding_index_can_resolve_primary_pano(self) -> None:
        room_graph = normalize_room_graph(self.explicit_map)
        grounding = build_test_grounding(room_graph)
        grounding["Room 8"]["pano_ids"] = ["pano-8"]
        grounding_index = GroundingIndex(
            grounding,
            pano_to_room={"mappings": {"pano-23": "Room 23", "pano-23b": "Room 23"}},
        )
        self.assertEqual(grounding_index.primary_pano_for_room("Room 8"), "pano-8")
        self.assertEqual(grounding_index.primary_pano_for_room("Room 23"), "pano-23")
        self.assertEqual(grounding_index.pano_ids_for_room("Room 23"), ["pano-23", "pano-23b"])
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


    def test_resolve_task_num_ctx_prefers_explicit_then_task_env_then_fallback(self) -> None:
        original = os.environ.get("ST_NAV_PARSE_INSTRUCTION_NUM_CTX")
        try:
            os.environ["ST_NAV_PARSE_INSTRUCTION_NUM_CTX"] = "2048"
            self.assertEqual(
                resolve_task_num_ctx(
                    "parse_instruction",
                    fallback_num_ctx=4096,
                    default_num_ctx=1024,
                ),
                2048,
            )
            self.assertEqual(
                resolve_task_num_ctx(
                    "parse_instruction",
                    explicit_num_ctx=1024,
                    fallback_num_ctx=4096,
                    default_num_ctx=2048,
                ),
                1024,
            )
            os.environ.pop("ST_NAV_PARSE_INSTRUCTION_NUM_CTX", None)
            self.assertEqual(
                resolve_task_num_ctx(
                    "parse_instruction",
                    fallback_num_ctx=4096,
                    default_num_ctx=2048,
                ),
                4096,
            )
            self.assertEqual(
                resolve_task_num_ctx("parse_instruction", default_num_ctx=2048),
                2048,
            )
        finally:
            if original is None:
                os.environ.pop("ST_NAV_PARSE_INSTRUCTION_NUM_CTX", None)
            else:
                os.environ["ST_NAV_PARSE_INSTRUCTION_NUM_CTX"] = original

    def test_instruction_parser_uses_task_specific_num_ctx_with_explicit_override(self) -> None:
        managed_keys = {
            "ST_NAV_ACTIVE_PROFILE": os.environ.get("ST_NAV_ACTIVE_PROFILE"),
            "ST_NAV_PROFILE_OLLAMA_MODEL_PROVIDER": os.environ.get("ST_NAV_PROFILE_OLLAMA_MODEL_PROVIDER"),
            "ST_NAV_PROFILE_OLLAMA_MODEL_NAME": os.environ.get("ST_NAV_PROFILE_OLLAMA_MODEL_NAME"),
            "ST_NAV_PROFILE_OLLAMA_API_BASE": os.environ.get("ST_NAV_PROFILE_OLLAMA_API_BASE"),
            "ST_NAV_PROFILE_OLLAMA_API_KEY": os.environ.get("ST_NAV_PROFILE_OLLAMA_API_KEY"),
            "ST_NAV_PROFILE_OLLAMA_API_KIND": os.environ.get("ST_NAV_PROFILE_OLLAMA_API_KIND"),
            "ST_NAV_PARSE_INSTRUCTION_NUM_CTX": os.environ.get("ST_NAV_PARSE_INSTRUCTION_NUM_CTX"),
        }
        try:
            os.environ["ST_NAV_ACTIVE_PROFILE"] = "ollama"
            os.environ["ST_NAV_PROFILE_OLLAMA_MODEL_PROVIDER"] = "ollama"
            os.environ["ST_NAV_PROFILE_OLLAMA_MODEL_NAME"] = "gemma4:26b"
            os.environ["ST_NAV_PROFILE_OLLAMA_API_BASE"] = "http://127.0.0.1:11434/v1"
            os.environ["ST_NAV_PROFILE_OLLAMA_API_KEY"] = "ollama"
            os.environ["ST_NAV_PROFILE_OLLAMA_API_KIND"] = "chat_completions"
            os.environ["ST_NAV_PARSE_INSTRUCTION_NUM_CTX"] = "2048"

            room_graph = normalize_room_graph(self.explicit_map)
            parser = LLMInstructionParser(room_graph=room_graph)
            payload = parser.model_client._responses_to_ollama_chat_payload(
                parser._build_request_body("Find the way from Room 8 to Room 23.")
            )
            self.assertEqual(parser.num_ctx, 2048)
            self.assertEqual(payload["options"]["num_ctx"], 2048)

            override_parser = LLMInstructionParser(room_graph=room_graph, num_ctx=1024)
            override_payload = override_parser.model_client._responses_to_ollama_chat_payload(
                override_parser._build_request_body("Find the way from Room 8 to Room 23.")
            )
            self.assertEqual(override_parser.num_ctx, 1024)
            self.assertEqual(override_payload["options"]["num_ctx"], 1024)
        finally:
            for key, value in managed_keys.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def test_spatial_alignment_refiner_uses_task_specific_num_ctx_with_explicit_override(self) -> None:
        managed_keys = {
            "ST_NAV_ACTIVE_PROFILE": os.environ.get("ST_NAV_ACTIVE_PROFILE"),
            "ST_NAV_PROFILE_OLLAMA_MODEL_PROVIDER": os.environ.get("ST_NAV_PROFILE_OLLAMA_MODEL_PROVIDER"),
            "ST_NAV_PROFILE_OLLAMA_MODEL_NAME": os.environ.get("ST_NAV_PROFILE_OLLAMA_MODEL_NAME"),
            "ST_NAV_PROFILE_OLLAMA_API_BASE": os.environ.get("ST_NAV_PROFILE_OLLAMA_API_BASE"),
            "ST_NAV_PROFILE_OLLAMA_API_KEY": os.environ.get("ST_NAV_PROFILE_OLLAMA_API_KEY"),
            "ST_NAV_PROFILE_OLLAMA_API_KIND": os.environ.get("ST_NAV_PROFILE_OLLAMA_API_KIND"),
            "ST_NAV_LOCALIZATION_NUM_CTX": os.environ.get("ST_NAV_LOCALIZATION_NUM_CTX"),
        }
        try:
            os.environ["ST_NAV_ACTIVE_PROFILE"] = "ollama"
            os.environ["ST_NAV_PROFILE_OLLAMA_MODEL_PROVIDER"] = "ollama"
            os.environ["ST_NAV_PROFILE_OLLAMA_MODEL_NAME"] = "gemma4:26b"
            os.environ["ST_NAV_PROFILE_OLLAMA_API_BASE"] = "http://127.0.0.1:11434/v1"
            os.environ["ST_NAV_PROFILE_OLLAMA_API_KEY"] = "ollama"
            os.environ["ST_NAV_PROFILE_OLLAMA_API_KIND"] = "chat_completions"
            os.environ["ST_NAV_LOCALIZATION_NUM_CTX"] = "16384"

            room_graph = normalize_room_graph(self.explicit_map)
            grounding = build_test_grounding(room_graph)
            refiner = SpatialAlignmentRefiner(
                room_graph=room_graph,
                grounding_index=GroundingIndex(grounding),
                response_client=lambda _: {},
            )
            self.assertEqual(refiner.num_ctx, 16384)
            self.assertEqual(refiner.model_client.num_ctx, 16384)

            override_refiner = SpatialAlignmentRefiner(
                room_graph=room_graph,
                grounding_index=GroundingIndex(grounding),
                num_ctx=8192,
                response_client=lambda _: {},
            )
            self.assertEqual(override_refiner.num_ctx, 8192)
            self.assertEqual(override_refiner.model_client.num_ctx, 8192)
        finally:
            for key, value in managed_keys.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def test_parse_instruction_eval_reports_token_usage_and_report_columns(self) -> None:
        module = self._load_parse_instruction_eval_module()

        ollama_usage = module.extract_token_usage({"prompt_eval_count": 120, "eval_count": 30})
        self.assertEqual(ollama_usage["input_tokens"], 120)
        self.assertEqual(ollama_usage["output_tokens"], 30)
        self.assertEqual(ollama_usage["total_tokens"], 150)
        self.assertIsNone(ollama_usage["reasoning_tokens"])

        hosted_usage = module.extract_token_usage(
            {
                "usage": {
                    "input_tokens": 50,
                    "output_tokens": 20,
                    "total_tokens": 70,
                    "output_tokens_details": {"reasoning_tokens": 7},
                }
            }
        )
        self.assertEqual(hosted_usage["reasoning_tokens"], 7)

        results = [
            {
                "instruction": "Find the way from Room 8 to Room 23.",
                "elapsed_seconds": 1.25,
                "result": {
                    "task_type": "gallery_goal_navigation",
                    "source_room_id": "Room 8",
                    "goal_room_ids": ["Room 23"],
                    "waypoint_room_ids": [],
                },
                "input_tokens": 120,
                "output_tokens": 30,
                "total_tokens": 150,
                "reasoning_tokens": None,
                "raw_usage": {"prompt_eval_count": 120, "eval_count": 30},
            },
            {
                "instruction": "Find the way from the Lamassu to the Townley Venus.",
                "elapsed_seconds": 1.75,
                "error": "parse failed",
                "input_tokens": 50,
                "output_tokens": 20,
                "total_tokens": 70,
                "reasoning_tokens": 7,
                "raw_usage": {"input_tokens": 50, "output_tokens": 20, "total_tokens": 70},
            },
        ]
        summary = module.summarize_results(results)
        self.assertEqual(summary["total_input_tokens"], 170)
        self.assertEqual(summary["total_output_tokens"], 50)
        self.assertEqual(summary["total_tokens"], 220)
        self.assertEqual(summary["total_reasoning_tokens"], 7)
        self.assertEqual(summary["average_input_tokens"], 85.0)
        self.assertEqual(summary["average_reasoning_tokens"], 7.0)

        report = module.render_report(
            {
                "parser": "runtime",
                "generated_at": "2026-05-21 00:00:00 +0000",
                "case_count": 2,
                "config": {
                    "active_profile": "ollama",
                    "model": "gemma4:31b",
                    "api_base": "http://127.0.0.1:11434/v1",
                    "api_kind": "chat_completions",
                    "effective_num_ctx": 8192,
                    "artifacts_dir": "dataset/sites/british_museum/normalized",
                },
                "summary": summary,
                "results": results,
            }
        )
        self.assertIn("Average input tokens", report)
        self.assertIn("Reasoning Tokens", report)
        self.assertIn("Effective num ctx", report)


if __name__ == "__main__":
    unittest.main()
