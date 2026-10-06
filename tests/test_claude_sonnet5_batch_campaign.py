from __future__ import annotations

from pathlib import Path
import unittest

from tools.evaluation.navigation import run_claude_sonnet5_batch_715 as campaign


class ClaudeSonnet5BatchCampaignTests(unittest.TestCase):
    def test_worker_shards_cover_cases_without_overlap(self) -> None:
        tests = [{"test_id": f"TEST{index:03d}"} for index in range(1, 11)]
        shards = campaign.split_worker_shards(tests, 4)

        flattened = [record["test_id"] for shard in shards for record in shard]
        self.assertEqual(len(shards), 4)
        self.assertEqual(len(flattened), len(set(flattened)))
        self.assertEqual(set(flattened), {record["test_id"] for record in tests})
        self.assertEqual([len(shard) for shard in shards], [3, 3, 2, 2])

    def test_evaluator_command_preserves_main_method_and_shared_caches(self) -> None:
        tests = [{"test_id": "TEST006"}, {"test_id": "TEST008"}]
        command = campaign.build_evaluator_command(
            tests=tests,
            test_set=Path("/tmp/test_set.jsonl"),
            output_dir=Path("/tmp/output"),
            cache_root=Path("/tmp/cache"),
            render_root=Path("/tmp/renders"),
            gateway_base_url="http://127.0.0.1:8765/v1",
            max_attempts=3,
            render_missing=True,
            spatial_context_max_output_tokens=2048,
            spatial_alignment_max_output_tokens=3072,
        )

        self.assertIn("claude-sonnet-5", command)
        self.assertIn("anthropic_messages", command)
        self.assertIn("geometry-edge-visit", command)
        self.assertIn("initial-spatial", command)
        self.assertIn("spatial", command)
        self.assertIn("--force-initial-visual-refresh", command)
        self.assertIn("--query-cache-dir", command)
        self.assertIn("/tmp/cache/query_parser", command)
        self.assertIn("--visual-cache-dir", command)
        self.assertIn("/tmp/cache/visual_localization", command)
        self.assertEqual(
            command[command.index("--spatial-context-max-output-tokens") + 1],
            "2048",
        )
        self.assertEqual(
            command[command.index("--spatial-alignment-max-output-tokens") + 1],
            "3072",
        )
        self.assertIn("--render-missing", command)
        self.assertEqual(command.count("--test-id"), 2)
        self.assertNotIn("ANTHROPIC_API_KEY", " ".join(command))
        self.assertNotIn("sk-ant-", " ".join(command))

    def test_worker_environment_pins_anthropic_provider(self) -> None:
        environment = campaign.worker_environment({"KEEP": "yes"})

        self.assertEqual(environment["KEEP"], "yes")
        self.assertEqual(environment["ST_NAV_MODEL_PROVIDER"], "anthropic")
        self.assertEqual(environment["ST_NAV_MODEL_NAME"], "claude-sonnet-5")
        self.assertEqual(environment["ST_NAV_API_KIND"], "anthropic_messages")


if __name__ == "__main__":
    unittest.main()
