from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest import mock

from st_nav import (
    AnthropicMessageBatchClient,
    ModelResponseClient,
    PersistentAnthropicBatchBroker,
    atomic_write_batch_checkpoint,
    anthropic_billing_pause_reason,
    extract_output_text,
    normalize_message_batch_result,
    prepare_message_batch_request,
    resolve_model_environment,
    serialized_batch_bytes,
    split_message_batch_requests,
)
from st_nav.common.anthropic_batch_gateway import AnthropicBatchGatewayHTTPServer


class FakeHTTPResponse:
    def __init__(self, body: bytes, *, lines: list[bytes] | None = None) -> None:
        self.body = body
        self.lines = lines or []

    def __enter__(self) -> "FakeHTTPResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def read(self) -> bytes:
        return self.body

    def __iter__(self):
        return iter(self.lines)


class FakeMessageBatchClient:
    def __init__(self) -> None:
        self.created: list[list[dict]] = []
        self.requests_by_batch_id: dict[str, list[dict]] = {}
        self._lock = threading.Lock()

    def create(self, requests: list[dict]) -> dict:
        with self._lock:
            batch_id = f"msgbatch_fake_{len(self.created) + 1}"
            cloned = json.loads(json.dumps(requests))
            self.created.append(cloned)
            self.requests_by_batch_id[batch_id] = cloned
        return {"id": batch_id, "processing_status": "in_progress"}

    def wait_until_ended(self, batch_id: str, **kwargs) -> dict:
        return {"id": batch_id, "processing_status": "ended"}

    def results(self, batch_id: str):
        for request in self.requests_by_batch_id[batch_id]:
            custom_id = request["custom_id"]
            yield {
                "custom_id": custom_id,
                "result": {
                    "type": "succeeded",
                    "message": {
                        "id": f"msg_{custom_id[:8]}",
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps({"fingerprint": custom_id}),
                            }
                        ],
                    },
                },
            }


class AnthropicTransportTests(unittest.TestCase):
    def test_local_batch_gateway_accepts_campaign_worker_burst(self) -> None:
        self.assertGreaterEqual(
            AnthropicBatchGatewayHTTPServer.request_queue_size,
            32,
        )

    def test_billing_pause_classifier_distinguishes_billing_from_rate_limits(self) -> None:
        billing = RuntimeError(
            'HTTP 402: {"type":"error","error":{"type":"billing_error"}}'
        )
        spend_limit = RuntimeError(
            'HTTP 429: {"details":{"error_code":"enforced_spend_limit_reached"}}'
        )
        ordinary_rate_limit = RuntimeError("HTTP 429 rate_limit_error retry after 10")

        self.assertIsNotNone(anthropic_billing_pause_reason(billing))
        self.assertIsNotNone(anthropic_billing_pause_reason(spend_limit))
        self.assertIsNone(anthropic_billing_pause_reason(ordinary_rate_limit))

    def test_anthropic_profile_uses_native_defaults_and_api_key_fallback(self) -> None:
        managed = {
            name: os.environ.get(name)
            for name in (
                "ST_NAV_ACTIVE_PROFILE",
                "ST_NAV_PROFILE_CLAUDE_MODEL_PROVIDER",
                "ST_NAV_PROFILE_CLAUDE_MODEL_NAME",
                "ST_NAV_PROFILE_CLAUDE_API_KEY",
                "ST_NAV_PROFILE_CLAUDE_API_BASE",
                "ST_NAV_PROFILE_CLAUDE_API_KIND",
                "ST_NAV_API_KEY",
                "OPENAI_API_KEY",
                "ANTHROPIC_API_KEY",
            )
        }
        try:
            for name in managed:
                os.environ.pop(name, None)
            os.environ["ST_NAV_ACTIVE_PROFILE"] = "claude"
            os.environ["ST_NAV_PROFILE_CLAUDE_MODEL_PROVIDER"] = "anthropic"
            os.environ["ST_NAV_PROFILE_CLAUDE_MODEL_NAME"] = "claude-sonnet-5"
            os.environ["OPENAI_API_KEY"] = "must-not-be-used-for-anthropic"
            os.environ["ANTHROPIC_API_KEY"] = "anthropic-test-key"

            resolved = resolve_model_environment()

            self.assertEqual(resolved.provider, "anthropic")
            self.assertEqual(resolved.model_name, "claude-sonnet-5")
            self.assertEqual(resolved.api_key, "anthropic-test-key")
            self.assertEqual(resolved.api_base, "https://api.anthropic.com/v1")
            self.assertEqual(resolved.api_kind, "anthropic_messages")
        finally:
            for name, value in managed.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value

    def test_responses_payload_converts_to_native_anthropic_messages(self) -> None:
        jpeg_data = base64.b64encode(b"\xff\xd8\xfffake-jpeg").decode("ascii")
        request_body = {
            "model": "claude-sonnet-5",
            "max_output_tokens": 2048,
            "instructions": "Return JSON only.",
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Inspect this view."},
                        {
                            "type": "input_image",
                            # The renderer currently has JPEG bytes behind some
                            # .png paths. Conversion must trust the bytes.
                            "image_url": f"data:image/png;base64,{jpeg_data}",
                            "detail": "high",
                        },
                    ],
                }
            ],
            "reasoning": {"effort": "low"},
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "answer",
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

        payload = ModelResponseClient._responses_to_anthropic_messages_payload(request_body)

        self.assertEqual(payload["model"], "claude-sonnet-5")
        self.assertEqual(payload["max_tokens"], 2048)
        self.assertEqual(payload["system"], "Return JSON only.")
        self.assertEqual(payload["messages"][0]["content"][0]["type"], "text")
        image = payload["messages"][0]["content"][1]
        self.assertEqual(image["type"], "image")
        self.assertEqual(image["source"]["media_type"], "image/jpeg")
        self.assertEqual(image["source"]["data"], jpeg_data)
        self.assertEqual(payload["output_config"]["effort"], "low")
        self.assertEqual(payload["output_config"]["format"]["type"], "json_schema")
        self.assertEqual(
            payload["output_config"]["format"]["schema"]["required"],
            ["answer"],
        )

    def test_anthropic_payload_uses_safe_default_max_tokens(self) -> None:
        payload = ModelResponseClient._responses_to_anthropic_messages_payload(
            {
                "model": "claude-sonnet-5",
                "input": "Return OK.",
            }
        )
        self.assertEqual(payload["max_tokens"], 4096)
        self.assertEqual(payload["messages"], [{"role": "user", "content": "Return OK."}])

    def test_anthropic_schema_splits_nullable_enum_into_any_of(self) -> None:
        original_schema = {
            "type": "object",
            "properties": {
                "predicted_room_id": {
                    "type": ["string", "null"],
                    "enum": ["Room 1", "East stairs", None],
                }
            },
            "required": ["predicted_room_id"],
            "additionalProperties": False,
        }
        request_body = {
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "nullable_enum",
                    "schema": original_schema,
                }
            }
        }

        transformed = ModelResponseClient._anthropic_response_schema_from_request_body(
            request_body
        )

        self.assertIsNotNone(transformed)
        field = transformed["properties"]["predicted_room_id"]
        self.assertNotIn("type", field)
        self.assertNotIn("enum", field)
        self.assertEqual(
            field["anyOf"],
            [
                {"type": "string", "enum": ["Room 1", "East stairs"]},
                {"type": "null", "enum": [None]},
            ],
        )
        # The source schema is shared by other providers and must stay untouched.
        self.assertEqual(
            original_schema["properties"]["predicted_room_id"]["type"],
            ["string", "null"],
        )

    def test_anthropic_schema_moves_unsupported_constraints_to_descriptions(self) -> None:
        transformed = ModelResponseClient._transform_schema_for_anthropic(
            {
                "type": "object",
                "properties": {
                    "confidence": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                    },
                    "evidence": {
                        "type": "array",
                        "minItems": 2,
                        "maxItems": 8,
                        "items": {
                            "type": "object",
                            "properties": {
                                "reason": {"type": "string", "maxLength": 160}
                            },
                            "required": ["reason"],
                        },
                    },
                },
                "required": ["confidence", "evidence"],
            }
        )

        self.assertFalse(transformed["additionalProperties"])
        confidence = transformed["properties"]["confidence"]
        self.assertNotIn("minimum", confidence)
        self.assertNotIn("maximum", confidence)
        self.assertIn("minimum: 0.0", confidence["description"])
        evidence = transformed["properties"]["evidence"]
        self.assertNotIn("minItems", evidence)
        self.assertNotIn("maxItems", evidence)
        self.assertIn("minItems: 2", evidence["description"])
        nested_object = evidence["items"]
        self.assertFalse(nested_object["additionalProperties"])
        reason = nested_object["properties"]["reason"]
        self.assertNotIn("maxLength", reason)
        self.assertIn("maxLength: 160", reason["description"])

    def test_extract_output_text_skips_thinking_blocks(self) -> None:
        payload = {
            "content": [
                {"type": "thinking", "thinking": "private reasoning"},
                {"type": "text", "text": '{"answer":"ok"}'},
            ]
        }
        self.assertEqual(extract_output_text(payload), '{"answer":"ok"}')

    def test_native_create_uses_anthropic_endpoint_headers_and_payload(self) -> None:
        response = FakeHTTPResponse(
            json.dumps(
                {
                    "id": "msg_test",
                    "type": "message",
                    "content": [{"type": "text", "text": '{"answer":"ok"}'}],
                }
            ).encode("utf-8")
        )
        client = ModelResponseClient(
            provider="anthropic",
            api_key="test-key",
            api_base="https://api.anthropic.com/v1",
            api_kind="anthropic_messages",
        )

        with mock.patch(
            "st_nav.common.model_client.urllib.request.urlopen",
            return_value=response,
        ) as urlopen_mock:
            result = client.create(
                {
                    "model": "claude-sonnet-5",
                    "input": "Return JSON.",
                    "max_output_tokens": 32,
                }
            )

        request = urlopen_mock.call_args.args[0]
        headers = {key.lower(): value for key, value in request.header_items()}
        self.assertEqual(request.full_url, "https://api.anthropic.com/v1/messages")
        self.assertEqual(headers["x-api-key"], "test-key")
        self.assertEqual(headers["anthropic-version"], "2023-06-01")
        body = json.loads(request.data.decode("utf-8"))
        self.assertEqual(body["max_tokens"], 32)
        self.assertEqual(result["output_text"], '{"answer":"ok"}')

    def test_prepare_split_and_normalize_batch_requests(self) -> None:
        requests = [
            prepare_message_batch_request(
                custom_id=f"TEST{index:03d}_query_s000_a01",
                request_body={
                    "model": "claude-sonnet-5",
                    "input": f"request {index}",
                    "max_output_tokens": 64,
                },
            )
            for index in range(1, 4)
        ]
        one_request_size = serialized_batch_bytes(requests[:1])
        batches = split_message_batch_requests(
            requests,
            max_requests=2,
            target_bytes=one_request_size * 2,
        )
        self.assertEqual([len(batch) for batch in batches], [2, 1])

        normalized = normalize_message_batch_result(
            {
                "custom_id": "TEST001_query_s000_a01",
                "result": {
                    "type": "succeeded",
                    "message": {
                        "content": [
                            {"type": "thinking", "thinking": "hidden"},
                            {"type": "text", "text": '{"answer":"ok"}'},
                        ]
                    },
                },
            }
        )
        self.assertEqual(normalized["result"]["message"]["output_text"], '{"answer":"ok"}')

    def test_batch_client_create_and_stream_results(self) -> None:
        created = FakeHTTPResponse(
            b'{"id":"msgbatch_test","processing_status":"in_progress"}'
        )
        result_line = json.dumps(
            {
                "custom_id": "TEST001_query_s000_a01",
                "result": {
                    "type": "succeeded",
                    "message": {"content": [{"type": "text", "text": "OK"}]},
                },
            }
        ).encode("utf-8") + b"\n"
        streamed = FakeHTTPResponse(b"", lines=[result_line])
        client = AnthropicMessageBatchClient(api_key="test-key")
        prepared = prepare_message_batch_request(
            custom_id="TEST001_query_s000_a01",
            request_body={"model": "claude-sonnet-5", "input": "OK"},
        )

        with mock.patch(
            "st_nav.common.anthropic_batch.urllib.request.urlopen",
            side_effect=[created, streamed],
        ) as urlopen_mock:
            batch = client.create([prepared])
            results = list(client.results(batch["id"]))

        create_request = urlopen_mock.call_args_list[0].args[0]
        results_request = urlopen_mock.call_args_list[1].args[0]
        self.assertEqual(create_request.full_url, "https://api.anthropic.com/v1/messages/batches")
        self.assertEqual(
            results_request.full_url,
            "https://api.anthropic.com/v1/messages/batches/msgbatch_test/results",
        )
        self.assertEqual(results[0]["result"]["message"]["output_text"], "OK")

    def test_batch_custom_ids_must_be_unique(self) -> None:
        request = prepare_message_batch_request(
            custom_id="TEST001_query_s000_a01",
            request_body={"model": "claude-sonnet-5", "input": "OK"},
        )
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            split_message_batch_requests([request, request])

        client = AnthropicMessageBatchClient(api_key="test-key")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            client.create([request, request])

    def test_batch_wait_polls_until_ended(self) -> None:
        client = AnthropicMessageBatchClient(api_key="test-key")
        with (
            mock.patch.object(
                client,
                "retrieve",
                side_effect=[
                    {"id": "msgbatch_test", "processing_status": "in_progress"},
                    {"id": "msgbatch_test", "processing_status": "ended"},
                ],
            ) as retrieve_mock,
            mock.patch("st_nav.common.anthropic_batch.time.sleep") as sleep_mock,
        ):
            ended = client.wait_until_ended(
                "msgbatch_test",
                poll_interval_seconds=0.01,
            )

        self.assertEqual(ended["processing_status"], "ended")
        self.assertEqual(retrieve_mock.call_count, 2)
        sleep_mock.assert_called_once_with(0.01)

    def test_batch_checkpoint_write_is_atomic_and_secret_free_by_caller(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nested" / "checkpoint.json"
            atomic_write_batch_checkpoint(
                path,
                {
                    "batch_id": "msgbatch_test",
                    "custom_ids": ["TEST001_query_s000_a01"],
                },
            )
            saved = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(saved["batch_id"], "msgbatch_test")
        self.assertNotIn("api_key", saved)

    def test_persistent_batch_broker_groups_routes_and_caches_requests(self) -> None:
        fake_client = FakeMessageBatchClient()
        with tempfile.TemporaryDirectory() as tmpdir:
            broker = PersistentAnthropicBatchBroker(
                client=fake_client,
                state_dir=tmpdir,
                flush_interval_seconds=0.05,
                poll_interval_seconds=0.0,
                max_concurrent_batches=1,
            )
            broker.start()
            params = [
                {
                    "model": "claude-sonnet-5",
                    "max_tokens": 32,
                    "messages": [{"role": "user", "content": f"request {index}"}],
                }
                for index in range(2)
            ]
            responses: list[dict | None] = [None, None]
            errors: list[BaseException] = []
            barrier = threading.Barrier(3)

            def submit(index: int) -> None:
                try:
                    barrier.wait()
                    responses[index] = broker.submit(params[index], timeout_seconds=5.0)
                except BaseException as error:  # pragma: no cover - assertion aid
                    errors.append(error)

            threads = [
                threading.Thread(target=submit, args=(index,))
                for index in range(2)
            ]
            for thread in threads:
                thread.start()
            barrier.wait()
            for thread in threads:
                thread.join(timeout=5.0)

            self.assertFalse(errors)
            self.assertTrue(all(isinstance(response, dict) for response in responses))
            self.assertEqual(len(fake_client.created), 1)
            self.assertEqual(len(fake_client.created[0]), 2)

            cached = broker.submit(params[0], timeout_seconds=1.0)
            self.assertEqual(cached, responses[0])
            self.assertEqual(len(fake_client.created), 1)
            status = broker.snapshot()
            self.assertEqual(status["submitted_request_count"], 2)
            self.assertEqual(status["response_cache_hit_count"], 1)
            broker.close()

            checkpoints = list((Path(tmpdir) / "batches").glob("*.json"))
            self.assertEqual(len(checkpoints), 1)
            checkpoint = json.loads(checkpoints[0].read_text(encoding="utf-8"))
            self.assertEqual(checkpoint["status"], "complete")
            self.assertNotIn("params", checkpoint)
            self.assertNotIn("api_key", checkpoint)


if __name__ == "__main__":
    unittest.main()
