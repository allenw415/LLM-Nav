from __future__ import annotations

from pathlib import Path
import unittest
from unittest import mock

from tools.evaluation.navigation import run_gemma4_31b_visual_theme_baseline_715 as campaign
from tools.evaluation.navigation import retry_gemma4_31b_visual_theme_baseline_errors as retry_campaign


class VisualThemeBaselineCampaignTests(unittest.TestCase):
    def test_error_retry_selection_and_merge_preserve_nonerrors(self) -> None:
        tests = [{"test_id": f"TEST{index:03d}"} for index in range(1, 716)]
        original = [
            {
                "test_id": test["test_id"],
                "stop_reason": "error" if index in {2, 5} else "goal_reached",
                "attempt_number": 1,
                "marker": "original",
            }
            for index, test in enumerate(tests, start=1)
        ]
        selected = retry_campaign.select_retry_tests(tests, original)
        self.assertEqual([test["test_id"] for test in selected], ["TEST002", "TEST005"])

        replacements = [
            {
                "test_id": test_id,
                "stop_reason": "goal_reached",
                "attempt_number": 2,
                "marker": "retry",
            }
            for test_id in ("TEST002", "TEST005")
        ]
        merged = retry_campaign.merge_retry_results(
            tests=tests,
            original_results=original,
            retry_results=replacements,
        )
        self.assertEqual(len(merged), 715)
        self.assertEqual(merged[0]["marker"], "original")
        self.assertEqual(merged[1]["marker"], "retry")
        self.assertEqual(merged[4]["attempt_number"], 2)

    def test_each_worker_receives_only_its_runtime_api_key(self) -> None:
        base = {
            "GEMMA4_API_KEY_WORKER0": "first-secret",
            "GEMMA4_API_KEY_WORKER1": "second-secret",
            "GMAPS_API_KEY": "maps-secret",
            "ST_NAV_ACTIVE_PROFILE": "ollama",
            "ST_NAV_PROFILE": "ollama",
        }

        worker0 = campaign.worker_environment(base, key_slot=0)
        worker1 = campaign.worker_environment(base, key_slot=1)

        self.assertEqual(worker0["ST_NAV_API_KEY"], "first-secret")
        self.assertEqual(worker1["ST_NAV_API_KEY"], "second-secret")
        self.assertEqual(worker0["ST_NAV_MODEL_PROVIDER"], "gemini")
        self.assertEqual(worker1["ST_NAV_MODEL_NAME"], "gemma-4-31b-it")
        self.assertEqual(worker0["ST_NAV_ACTIVE_PROFILE"], "gemma4_worker")
        self.assertEqual(worker1["ST_NAV_PROFILE"], "gemma4_worker")
        self.assertEqual(
            worker0["ST_NAV_HTTP_RETRY_BACKOFF_SECONDS"],
            str(campaign.HTTP_RETRY_BACKOFF_SECONDS),
        )
        for source_name in campaign.WORKER_KEY_ENV_NAMES:
            self.assertNotIn(source_name, worker0)
            self.assertNotIn(source_name, worker1)

    def test_evaluator_command_locks_baseline_ablation_settings(self) -> None:
        command = campaign.build_evaluator_command(
            tests=[{"test_id": "TEST001"}, {"test_id": "TEST003"}],
            test_set=Path("/tmp/test_set.jsonl"),
            output_dir=Path("/tmp/output/worker0"),
            cache_root=Path("/tmp/cache"),
            render_root=Path("/tmp/renders"),
            max_attempts=3,
            render_missing=False,
            direction_cache_dir=Path("/tmp/cache/direction_selection"),
        )
        joined = " ".join(command)
        self.assertIn("--llm-model gemma-4-31b-it", joined)
        self.assertIn("--localization-interval 3", joined)
        self.assertIn("--localization-effective-stage visual", joined)
        self.assertIn("--heading-alignment-mode none", joined)
        self.assertIn("--direction-policy visual-theme", joined)
        self.assertIn("--direction-cache-dir /tmp/cache/direction_selection", joined)
        self.assertNotIn("first-secret", joined)
        self.assertNotIn("second-secret", joined)
        self.assertNotIn("--render-missing", command)
        self.assertEqual(command.count("--test-id"), 2)
        self.assertIn("no_trajectory", campaign.DEFAULT_OUTPUT_ROOT.name)

    def test_evaluator_command_can_disable_only_direction_cache(self) -> None:
        command = campaign.build_evaluator_command(
            tests=[{"test_id": "TEST001"}],
            test_set=Path("/tmp/test_set.jsonl"),
            output_dir=Path("/tmp/output/worker0"),
            cache_root=Path("/tmp/cache"),
            render_root=Path("/tmp/renders"),
            max_attempts=3,
            render_missing=False,
            direction_cache_dir=None,
        )

        joined = " ".join(command)
        self.assertNotIn("--direction-cache-dir", command)
        self.assertIn("--query-cache-dir /tmp/cache/query_parser", joined)
        self.assertIn("--visual-cache-dir /tmp/cache/visual_localization", joined)

    def test_worker_environment_requires_both_named_slots_independently(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "GEMMA4_API_KEY_WORKER1"):
            campaign.worker_environment(
                {"GEMMA4_API_KEY_WORKER0": "only-first"},
                key_slot=1,
            )

    def test_campaign_rejects_reusing_one_key_in_both_slots(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "must contain different keys"):
            campaign.validate_distinct_worker_keys(
                {
                    "GEMMA4_API_KEY_WORKER0": "same-secret",
                    "GEMMA4_API_KEY_WORKER1": "same-secret",
                }
            )

    def test_generation_preflight_checks_both_workers_before_launch(self) -> None:
        checked: list[str] = []

        class FakeClient:
            def __init__(self, *, api_key: str, **kwargs) -> None:
                self.api_key = api_key

            def create(self, body: dict) -> dict:
                checked.append(self.api_key)
                if self.api_key == "depleted-key":
                    raise RuntimeError("HTTP 402 Payment Required")
                return {"output_text": "OK"}

        with mock.patch.object(campaign, "ModelResponseClient", FakeClient):
            with self.assertRaisesRegex(RuntimeError, "worker1"):
                campaign.validate_worker_generation_access(
                    {
                        "worker0": {"ST_NAV_API_KEY": "working-key"},
                        "worker1": {"ST_NAV_API_KEY": "depleted-key"},
                    }
                )
        self.assertEqual(checked, ["working-key", "depleted-key"])


if __name__ == "__main__":
    unittest.main()
