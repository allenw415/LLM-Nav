from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest import mock

from st_nav.common.prompts import build_visual_detection_localization_schema
from st_nav.common.request_cache import model_transport_identity, request_fingerprint
from st_nav.decision.instruction_parser import LLMInstructionParser
from st_nav.perception.detector import VISUAL_CACHE_VERSION, ViewDetector
from tools.evaluation.navigation import evaluate_navigation_batch as evaluator
from tools.evaluation.navigation import repair_qwen3_vl_8b_no_manifest as qwen_repair
from tools.evaluation.navigation import run_gemma4_12b_with_trajectory_715 as campaign
from tools.evaluation.navigation import run_qwen3_vl_8b_earth_resume_715 as qwen_campaign


MODEL_ENV = {
    "ST_NAV_ACTIVE_PROFILE": "cachetest",
    "ST_NAV_PROFILE_CACHETEST_MODEL_PROVIDER": "ollama",
    "ST_NAV_PROFILE_CACHETEST_MODEL_NAME": "gemma4:test",
    "ST_NAV_PROFILE_CACHETEST_API_BASE": "http://127.0.0.1:11434/v1",
    "ST_NAV_PROFILE_CACHETEST_API_KIND": "chat_completions",
    "ST_NAV_PROFILE_CACHETEST_NUM_CTX": "16384",
    "ST_NAV_PROFILE_CACHETEST_TEMPERATURE": "0",
    "ST_NAV_PROFILE_CACHETEST_THINK": "false",
}


def parser_response(room_id: str = "Room 8") -> dict:
    return {
        "output_text": json.dumps(
            {
                "goal_entities": [
                    {
                        "name": "Assyria: Nimrud",
                        "entity_type": "gallery",
                        "predicted_room_id": room_id,
                        "confidence": 0.95,
                    }
                ],
                "waypoint_entities": [],
            }
        )
    }


def visual_response(room_ids: list[str]) -> dict:
    return {
        "output_text": json.dumps(
            {
                "entities": [],
                "visual_localization": {
                    "predicted_room_id": room_ids[0],
                    "room_scores": [
                        {
                            "room_id": room_id,
                            "evidence_type": "room_specific" if index == 0 else "weak_generic",
                            "score": 8.0 if index == 0 else 1.0,
                            "reason": "test evidence",
                        }
                        for index, room_id in enumerate(room_ids)
                    ],
                    "evidence_entities": [],
                    "summary": "test localization",
                },
            }
        )
    }


class RequestCacheTests(unittest.TestCase):
    def test_visual_localization_schema_bounds_runaway_arrays(self) -> None:
        schema = build_visual_detection_localization_schema(["Room 8", "Room 9"])
        entities = schema["properties"]["entities"]
        localization = schema["properties"]["visual_localization"]["properties"]
        self.assertEqual(entities["maxItems"], 8)
        self.assertEqual(entities["items"]["properties"]["source_views"]["maxItems"], 8)
        self.assertEqual(localization["room_scores"]["minItems"], 2)
        self.assertEqual(localization["room_scores"]["maxItems"], 2)
        self.assertEqual(localization["evidence_entities"]["maxItems"], 6)
        self.assertEqual(localization["room_scores"]["items"]["properties"]["reason"]["maxLength"], 160)
        self.assertEqual(localization["summary"]["maxLength"], 240)

    def test_prior_successful_visual_schema_can_omit_transport_bounds(self) -> None:
        schema = build_visual_detection_localization_schema(
            ["Room 8", "Room 9"],
            include_bounds=False,
        )
        entities = schema["properties"]["entities"]
        localization = schema["properties"]["visual_localization"]["properties"]
        self.assertNotIn("maxItems", entities)
        self.assertNotIn("maxLength", entities["items"]["properties"]["name"])
        self.assertNotIn("maxItems", entities["items"]["properties"]["source_views"])
        self.assertNotIn("minItems", localization["room_scores"])
        self.assertNotIn("maxItems", localization["room_scores"])
        self.assertNotIn(
            "maxLength",
            localization["room_scores"]["items"]["properties"]["reason"],
        )
        self.assertNotIn("maxItems", localization["evidence_entities"])
        self.assertNotIn("maxLength", localization["summary"])

    def test_fingerprint_changes_with_response_affecting_transport_settings(self) -> None:
        client = type(
            "Client",
            (),
            {
                "provider": "ollama",
                "api_base": "http://localhost:11434/v1",
                "api_kind": "chat_completions",
                "num_ctx": 16384,
                "temperature": 0,
                "think": False,
            },
        )()
        identity = model_transport_identity(model="gemma4:test", model_client=client)
        first = request_fingerprint(
            namespace="test",
            cache_version=1,
            request_body={"input": "same"},
            transport_identity=identity,
        )
        changed = dict(identity, think=True)
        second = request_fingerprint(
            namespace="test",
            cache_version=1,
            request_body={"input": "same"},
            transport_identity=changed,
        )
        self.assertNotEqual(first, second)

    def test_query_parser_reuses_only_valid_fingerprinted_response(self) -> None:
        calls: list[dict] = []
        room_graph = {
            "Room 8": {"title": "Assyria: Nimrud", "floor": "0", "neighbors": []}
        }
        with mock.patch.dict(os.environ, MODEL_ENV, clear=False):
            with tempfile.TemporaryDirectory() as tmpdir:
                cache_dir = Path(tmpdir) / "query"
                parser = LLMInstructionParser(
                    room_graph=room_graph,
                    model="gemma4:test",
                    api_base="http://127.0.0.1:11434/v1",
                    api_kind="chat_completions",
                    num_ctx=16384,
                    cache_dir=cache_dir,
                    response_client=lambda body: calls.append(body) or parser_response(),
                )
                first = parser.parse("Take me to the Assyria: Nimrud gallery.")
                telemetry = parser.model_client.telemetry_snapshot()
                second = parser.parse("Take me to the Assyria: Nimrud gallery.")
                self.assertEqual(parser.model_client.telemetry_snapshot(), telemetry)
                self.assertEqual(first.goal_room_ids, ["Room 8"])
                self.assertEqual(second.goal_room_ids, ["Room 8"])
                self.assertEqual(len(calls), 1)
                self.assertIs(parser.last_cache_hit, True)

                cache_path = next(cache_dir.rglob("*.json"))
                entry = json.loads(cache_path.read_text(encoding="utf-8"))
                entry["response"] = {"output_text": "not json"}
                cache_path.write_text(json.dumps(entry), encoding="utf-8")
                parser.parse("Take me to the Assyria: Nimrud gallery.")
                self.assertEqual(len(calls), 2)
                self.assertIs(parser.last_cache_hit, False)

    def test_visual_cache_ignores_legacy_sibling_and_serializes_concurrent_fill(self) -> None:
        room_graph = {
            "Room 8": {"title": "Assyria: Nimrud", "floor": "0", "neighbors": []},
            "Room 9": {"title": "Assyria: Nineveh", "floor": "0", "neighbors": []},
        }
        calls = 0
        calls_lock = threading.Lock()

        def response_client(body: dict) -> dict:
            nonlocal calls
            with calls_lock:
                calls += 1
            time.sleep(0.05)
            return visual_response(["Room 8", "Room 9"])

        with mock.patch.dict(os.environ, MODEL_ENV, clear=False):
            with tempfile.TemporaryDirectory() as tmpdir:
                root = Path(tmpdir)
                image_path = root / "north.png"
                image_path.write_bytes(b"model-specific-image")
                manifest_path = root / "pano-8_manifest.json"
                manifest_path.write_text(
                    json.dumps(
                        {
                            "pano_id": "pano-8",
                            "floor": "0",
                            "captures": [
                                {"label": "north", "heading": 330.0, "path": str(image_path)}
                            ],
                        }
                    ),
                    encoding="utf-8",
                )
                legacy_path = root / "pano-8_manifest_detections.json"
                legacy_path.write_text(
                    json.dumps({"cache_version": 3, "entities": []}),
                    encoding="utf-8",
                )
                cache_dir = root / "visual"

                def build_detector() -> ViewDetector:
                    return ViewDetector(
                        model="gemma4:test",
                        api_base="http://127.0.0.1:11434/v1",
                        api_kind="chat_completions",
                        room_graph=room_graph,
                        cache_dir=cache_dir,
                        response_client=response_client,
                    )

                first_detector = build_detector()
                second_detector = build_detector()
                with ThreadPoolExecutor(max_workers=2) as executor:
                    first_future = executor.submit(first_detector.detect, manifest_path)
                    second_future = executor.submit(second_detector.detect, manifest_path)
                    first = first_future.result()
                    second = second_future.result()

                self.assertEqual(calls, 1)
                self.assertEqual(
                    sorted(
                        [first[0].metadata["visual_cache_hit"], second[0].metadata["visual_cache_hit"]]
                    ),
                    [False, True],
                )
                self.assertTrue(
                    all(
                        item[0].metadata["visual_cache_version"] == VISUAL_CACHE_VERSION
                        for item in [first, second]
                    )
                )
                self.assertNotEqual(first[0].metadata["visual_cache_path"], str(legacy_path))
                self.assertEqual(len(list(cache_dir.rglob("*.json"))), 1)

                telemetry = second_detector.model_client.telemetry_snapshot()
                third = second_detector.detect(manifest_path)
                self.assertIs(third[0].metadata["visual_cache_hit"], True)
                self.assertEqual(second_detector.model_client.telemetry_snapshot(), telemetry)

                migrated_detector = ViewDetector(
                    model="gemma4:test",
                    api_base="http://127.0.0.1:11434/v1",
                    api_kind="chat_completions",
                    room_graph=room_graph,
                    cache_dir=cache_dir,
                    schema_validation_retries=1,
                    response_client=response_client,
                )
                migrated = migrated_detector.detect(manifest_path)
                self.assertEqual(calls, 1)
                self.assertIs(migrated[0].metadata["visual_cache_hit"], True)
                self.assertEqual(len(list(cache_dir.rglob("*.json"))), 2)

    def test_visual_schema_failure_retries_once_and_stores_diagnostics_only(self) -> None:
        room_graph = {
            "Room 8": {"title": "Assyria: Nimrud", "floor": "0", "neighbors": []},
            "Room 9": {"title": "Assyria: Nineveh", "floor": "0", "neighbors": []},
        }
        calls: list[dict] = []

        def response_client(body: dict) -> dict:
            calls.append(body)
            if len(calls) == 1:
                return visual_response(["Room 8", "Room 8"])
            return visual_response(["Room 8", "Room 9"])

        with mock.patch.dict(os.environ, MODEL_ENV, clear=False):
            with tempfile.TemporaryDirectory() as tmpdir:
                root = Path(tmpdir)
                image_path = root / "north.png"
                image_path.write_bytes(b"schema-retry-image")
                manifest_path = root / "pano-retry_manifest.json"
                manifest_path.write_text(
                    json.dumps(
                        {
                            "pano_id": "pano-retry",
                            "floor": "0",
                            "captures": [
                                {"label": "north", "heading": 330.0, "path": str(image_path)}
                            ],
                        }
                    ),
                    encoding="utf-8",
                )
                cache_dir = root / "visual"
                detector = ViewDetector(
                    model="gemma4:test",
                    api_base="http://127.0.0.1:11434/v1",
                    api_kind="chat_completions",
                    room_graph=room_graph,
                    cache_dir=cache_dir,
                    schema_validation_retries=1,
                    max_output_tokens=3072,
                    response_client=response_client,
                )
                detections = detector.detect(manifest_path)
                self.assertEqual(len(calls), 2)
                self.assertTrue(all(call.get("max_output_tokens") == 3072 for call in calls))
                self.assertIn('Required room_id sequence: ["Room 8", "Room 9"]', calls[1]["instructions"])
                self.assertIn('missing=["Room 9"]', calls[1]["instructions"])
                self.assertIn('duplicates=["Room 8"]', calls[1]["instructions"])
                self.assertEqual(detections[0].metadata["visual_validation_attempt_count"], 2)
                diagnostics = list((cache_dir / "_failures").rglob("*.json"))
                self.assertEqual(len(diagnostics), 1)
                diagnostic = json.loads(diagnostics[0].read_text(encoding="utf-8"))
                self.assertEqual(diagnostic["failures"][0]["validation_attempt"], 1)
                self.assertIn('missing=["Room 9"]', diagnostic["failures"][0]["error_message"])
                self.assertIn('duplicates=["Room 8"]', diagnostic["failures"][0]["error_message"])
                self.assertNotIn("data:image", json.dumps(diagnostic["failures"][0]["request"]))
                valid_entries = [
                    path for path in cache_dir.rglob("*.json") if "_failures" not in path.parts
                ]
                self.assertEqual(len(valid_entries), 1)
                entry = json.loads(valid_entries[0].read_text(encoding="utf-8"))
                self.assertEqual(entry["validation_attempt_count"], 2)
                self.assertEqual(len(entry["traces"]), 1)
                detector.detect(manifest_path)
                self.assertEqual(len(calls), 2)


class CampaignTests(unittest.TestCase):
    def test_alternating_split_is_358_357_without_overlap(self) -> None:
        tests = [
            {"test_id": f"TEST{index:03d}", "path_id": f"NP{index:06d}"}
            for index in range(1, 716)
        ]
        left, right = campaign.split_tests(tests)
        self.assertEqual((len(left), len(right)), (358, 357))
        self.assertFalse({item["test_id"] for item in left} & {item["test_id"] for item in right})

    def test_merge_preserves_dataset_order_and_rejects_overlap(self) -> None:
        tests = [
            {"test_id": f"TEST{index:03d}", "path_id": f"NP{index:06d}"}
            for index in range(1, 5)
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            first_path = root / "first.jsonl"
            second_path = root / "second.jsonl"
            first_path.write_text(
                "".join(json.dumps(tests[index]) + "\n" for index in [0, 2]),
                encoding="utf-8",
            )
            second_path.write_text(
                "".join(json.dumps(tests[index]) + "\n" for index in [1, 3]),
                encoding="utf-8",
            )
            merged = campaign.merge_shard_results(
                tests=tests,
                shard_paths=[first_path, second_path],
            )
            self.assertEqual([item["test_id"] for item in merged], [item["test_id"] for item in tests])

            second_path.write_text(json.dumps(tests[0]) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "Duplicate result ID"):
                campaign.merge_shard_results(
                    tests=tests,
                    shard_paths=[first_path, second_path],
                )

    def test_formal_command_locks_requested_experiment_settings(self) -> None:
        command = campaign.build_evaluator_command(
            tests=[{"test_id": "TEST001"}],
            test_set=Path("/tmp/test_set.jsonl"),
            output_dir=Path("/tmp/output"),
            cache_root=Path("/tmp/cache"),
            render_root=Path("/tmp/render"),
            max_attempts=3,
        )
        joined = " ".join(command)
        self.assertIn("gemma4:12b-it-bf16", joined)
        self.assertIn("--heading-alignment-mode initial-spatial", joined)
        self.assertIn("--direction-policy geometry-edge-visit", joined)
        self.assertIn("--localization-interval 3", joined)
        self.assertIn("--llm-timeout 600", joined)
        self.assertIn("--max-http-retries 1", joined)
        self.assertIn("--allow-infrastructure-resume", command)
        self.assertIn("--visual-schema-validation-retries 1", joined)
        self.assertIn("--spatial-schema-validation-retries 1", joined)
        self.assertIn("--spatial-context-max-output-tokens 2048", joined)
        self.assertIn("--spatial-alignment-max-output-tokens 1024", joined)
        self.assertNotIn("--force-initial-visual-refresh", command)

    def test_resume_compatibility_allows_only_infrastructure_changes(self) -> None:
        previous = {
            "schema": "llm_nav_floor0_batch_config",
            "schema_version": 7,
            "dataset_sha256": "same",
            "llm": {"model": "gemma4:test", "timeout": 300.0},
        }
        current = {
            "schema": "llm_nav_floor0_batch_config",
            "schema_version": 8,
            "dataset_sha256": "same",
            "llm": {
                "model": "gemma4:test",
                "timeout": 600.0,
                "max_http_retries": 1,
            },
        }
        current["spatial_alignment_reliability"] = {
            "schema_validation_retries": 1,
            "context_max_output_tokens": 2048,
            "alignment_max_output_tokens": 1024,
        }
        self.assertTrue(evaluator._infrastructure_resume_compatible(previous, current))
        render_previous = json.loads(json.dumps(current))
        render_previous["render"] = {
            "render_missing": False,
            "heading_mode": "museum",
        }
        render_current = json.loads(json.dumps(render_previous))
        render_current["render"]["render_missing"] = True
        self.assertTrue(
            evaluator._infrastructure_resume_compatible(render_previous, render_current)
        )
        render_current["render"]["heading_mode"] = "cardinal"
        self.assertFalse(
            evaluator._infrastructure_resume_compatible(render_previous, render_current)
        )
        changed_model = json.loads(json.dumps(current))
        changed_model["llm"]["model"] = "different"
        self.assertFalse(evaluator._infrastructure_resume_compatible(previous, changed_model))
        root_previous = {"schema": "run", "schema_version": 1, "model": "same"}
        root_current = {
            "schema": "run",
            "schema_version": 2,
            "model": "same",
            "llm_transport": {"request_timeout_seconds": 600},
            "runtime_provenance": {"source_tree_sha256": "new"},
        }
        self.assertTrue(campaign._run_config_resume_compatible(root_previous, root_current))
        root_current["model"] = "different"
        self.assertFalse(campaign._run_config_resume_compatible(root_previous, root_current))


class QwenEarthResumeTests(unittest.TestCase):
    def test_snapshot_is_partitioned_without_losing_attempt_history(self) -> None:
        tests = [
            {"test_id": f"TEST{index:03d}", "path_id": f"NP{index:06d}"}
            for index in range(1, 7)
        ]
        snapshot_results = [
            dict(tests[index], stop_reason="goal_reached", attempt_number=1)
            for index in range(4)
        ]
        snapshot_attempts = list(snapshot_results) + [
            dict(tests[1], stop_reason="error", attempt_number=1)
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            snapshot = root / "snapshot"
            output = root / "output"
            snapshot.mkdir()
            (snapshot / "results.jsonl").write_text(
                "".join(json.dumps(record) + "\n" for record in snapshot_results),
                encoding="utf-8",
            )
            (snapshot / "attempts.jsonl").write_text(
                "".join(json.dumps(record) + "\n" for record in snapshot_attempts),
                encoding="utf-8",
            )
            manifest = qwen_campaign.prepare_shards(
                tests=tests,
                snapshot_dir=snapshot,
                output_root=output,
            )
            worker0 = qwen_campaign.load_jsonl(output / "shards/worker0/results.jsonl")
            worker1 = qwen_campaign.load_jsonl(output / "shards/worker1/results.jsonl")
            worker1_attempts = qwen_campaign.load_jsonl(output / "shards/worker1/attempts.jsonl")
            self.assertEqual([record["test_id"] for record in worker0], ["TEST001", "TEST003"])
            self.assertEqual([record["test_id"] for record in worker1], ["TEST002", "TEST004"])
            self.assertEqual(len(worker1_attempts), 3)
            self.assertEqual(manifest["snapshot_result_count"], 4)
            self.assertEqual(manifest["shards"]["worker0"]["test_count"], 3)
            self.assertEqual(manifest["shards"]["worker1"]["test_count"], 3)
            qwen_campaign.prepare_shards(
                tests=tests,
                snapshot_dir=snapshot,
                output_root=output,
            )

    def test_qwen_command_preserves_neptune_settings(self) -> None:
        command = qwen_campaign.build_evaluator_command(
            tests=[{"test_id": "TEST001"}],
            test_set=Path("/tmp/test_set.jsonl"),
            output_dir=Path("/tmp/output"),
            cache_root=Path("/tmp/cache"),
            render_root=Path("/tmp/render"),
            max_attempts=3,
        )
        joined = " ".join(command)
        self.assertIn("qwen3-vl:8b-instruct-bf16", joined)
        self.assertIn("--llm-timeout 180", joined)
        self.assertIn("--visual-max-output-tokens 3072", joined)
        self.assertIn("--spatial-context-max-output-tokens 2048", joined)
        self.assertIn("--spatial-alignment-max-output-tokens 768", joined)
        self.assertIn("--max-attempts-per-case 3", joined)
        self.assertNotIn("--render-missing", command)

        repair_command = qwen_campaign.build_evaluator_command(
            tests=[{"test_id": "TEST001"}],
            test_set=Path("/tmp/test_set.jsonl"),
            output_dir=Path("/tmp/output"),
            cache_root=Path("/tmp/cache"),
            render_root=Path("/tmp/render"),
            max_attempts=3,
            render_missing=True,
            retry_stop_reasons=("stop_no_manifest",),
        )
        self.assertIn("--render-missing", repair_command)
        self.assertIn("--retry-stop-reason", repair_command)
        self.assertIn("stop_no_manifest", repair_command)

    def test_no_manifest_repair_partition_keeps_shared_panos_together(self) -> None:
        records = [
            {
                "test_id": "TEST001",
                "elapsed_seconds": 100,
                "render": {"cache_miss_pano_ids": ["pano-a"]},
            },
            {
                "test_id": "TEST002",
                "elapsed_seconds": 90,
                "render": {"cache_miss_pano_ids": ["pano-a"]},
            },
            {
                "test_id": "TEST003",
                "elapsed_seconds": 180,
                "render": {"cache_miss_pano_ids": ["pano-b"]},
            },
            {
                "test_id": "TEST004",
                "elapsed_seconds": 10,
                "render": {"cache_miss_pano_ids": ["pano-c"]},
            },
        ]
        partitions = qwen_repair.partition_repairs(records)
        self.assertEqual(set(partitions[0]) | set(partitions[1]), {f"TEST{i:03d}" for i in range(1, 5)})
        self.assertFalse(set(partitions[0]) & set(partitions[1]))
        self.assertTrue(
            {"TEST001", "TEST002"}.issubset(set(partitions[0]))
            or {"TEST001", "TEST002"}.issubset(set(partitions[1]))
        )


if __name__ == "__main__":
    unittest.main()
