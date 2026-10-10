from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sa_nav.cli._common import load_json
from sa_nav.common.room_profiles import preferred_room_graph_path


def load_eval_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "tools/evaluation/navigation/eval_map_grounded_heading.py"
    )
    spec = importlib.util.spec_from_file_location("map_grounded_heading_eval_test", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load evaluator: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MapGroundedHeadingEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_eval_module()
        root = Path(__file__).resolve().parents[1]
        cls.root = root
        cls.artifacts_dir = root / "dataset/sites/british_museum/normalized"
        cls.room_graph = load_json(preferred_room_graph_path(cls.artifacts_dir))
        cls.base_room_graph = load_json(cls.artifacts_dir / "room_graph.json")
        cls.explicit_map = load_json(
            root / "dataset/sites/british_museum/explicit_map/explicit_map.json"
        )
        cls.region_payload = load_json(
            root / "dataset/sites/british_museum/maps/level_0_room_regions.json"
        )

    def test_real_catalog_contains_76_unique_directed_edges_and_reverse_pairs(self) -> None:
        cases = self.module.build_heading_cases(self.room_graph, self.region_payload)
        edges = {(case["current_room_id"], case["target_room_id"]) for case in cases}
        self.assertEqual(len(cases), 76)
        self.assertEqual(len(edges), 76)
        self.assertIn(("Room 6", "Room 7"), edges)
        self.assertIn(("Room 7", "Room 6"), edges)
        self.assertEqual(
            {case["expected_heading_deg"] for case in cases},
            {0.0, 90.0, 180.0, 270.0},
        )

    def test_room4_room9_direction_is_aligned_across_graph_sources(self) -> None:
        source_edge = next(
            edge
            for edge in self.explicit_map["Room 9"]["links"]
            if edge.get("name") == "Room 4"
        )
        self.assertEqual(source_edge["direction"], "right")

        for graph in (self.base_room_graph, self.room_graph):
            room4_to_9 = next(
                edge
                for edge in graph["Room 4"]["neighbors"]
                if edge["target_room_id"] == "Room 9"
            )
            room9_to_4 = next(
                edge
                for edge in graph["Room 9"]["neighbors"]
                if edge["target_room_id"] == "Room 4"
            )
            self.assertEqual(
                (room4_to_9["allocentric_direction"], room4_to_9["allocentric_heading_deg"]),
                ("west", 270.0),
            )
            self.assertEqual(
                (room9_to_4["allocentric_direction"], room9_to_4["allocentric_heading_deg"]),
                ("east", 90.0),
            )

    def test_visual_profile_graph_matches_base_graph_topology(self) -> None:
        visual_without_profiles = {
            room_id: {
                key: value
                for key, value in node.items()
                if key != "visual_profile"
            }
            for room_id, node in self.room_graph.items()
        }
        self.assertEqual(visual_without_profiles, self.base_room_graph)

    def test_circular_error_handles_zero_and_315_degrees(self) -> None:
        self.assertEqual(self.module.angular_error_deg(0.0, 360.0), 0.0)
        self.assertEqual(self.module.angular_error_deg(315.0, 0.0), 45.0)
        self.assertEqual(self.module.angular_error_deg(225.0, 45.0), 180.0)

    def test_three_different_predictions_are_unstable_without_tie_break(self) -> None:
        case = {
            "case_id": "Room A -> Room B",
            "current_room_id": "Room A",
            "target_room_id": "Room B",
            "expected_allocentric_direction": "east",
            "expected_heading_deg": 90.0,
        }
        records = [
            {
                "case_id": case["case_id"],
                "status": "completed",
                "predicted_direction": direction,
                "confidence": 0.7,
            }
            for direction in ("N", "E", "S")
        ]
        summary = self.module.aggregate_cases(records, [case], repeat=3)[0]
        self.assertFalse(summary["stable_majority"])
        self.assertIsNone(summary["majority_direction"])
        self.assertIsNone(summary["majority_angular_error_deg"])

    def test_metrics_and_acceptance_apply_declared_gates(self) -> None:
        case_summaries = [
            {
                "expected_allocentric_direction": "east",
                "pairwise_prediction_agreement": 1.0,
                "stable_majority": True,
                "unanimous_prediction": True,
                "majority_angular_error_deg": 45.0,
                "majority_within_45_correct": True,
            }
        ]
        records = [
            {
                "status": "completed",
                "angular_error_deg": 45.0,
                "latency_seconds": 1.0,
            }
            for _ in range(3)
        ]
        metrics = self.module.evaluation_metrics(records, case_summaries)
        acceptance = self.module.feasibility_acceptance(metrics)
        self.assertEqual(metrics["schema_valid_rate"], 1.0)
        self.assertEqual(metrics["stable_case_rate"], 1.0)
        self.assertEqual(metrics["majority_within_45_accuracy"], 1.0)
        self.assertTrue(acceptance["passed"])

    def test_mock_batch_runs_three_calls_and_writes_reports(self) -> None:
        module = self.module

        class FakeExperiment:
            def __init__(self, **kwargs):
                self.model_client = SimpleNamespace(is_configured=lambda: True)
                self.last_request_body = None
                self.last_response_payload = None

            def run_once(
                self,
                *,
                current_room_id,
                target_room_id,
                semantic_map_path,
                map_mode="highlighted",
            ):
                self.last_request_body = {
                    "instructions": "map-only",
                    "input": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_text",
                                    "text": f"current_room_id: {current_room_id}\ntarget_room_id: {target_room_id}",
                                },
                                {"type": "input_image", "image_url": "data:image/png;base64,AA=="},
                            ],
                        }
                    ],
                }
                result = {
                    "current_room_id": current_room_id,
                    "target_room_id": target_room_id,
                    "passage_direction": "N",
                    "heading_deg": 0.0,
                    "confidence": 0.9,
                    "ambiguity_level": "low",
                    "passage_description": "North-side passage.",
                    "summary": "Head north.",
                }
                return {"result": result, "response_payload": {"output_text": "{}"}}

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "output"
            args = module.build_parser().parse_args(
                [
                    "--max-cases",
                    "1",
                    "--repeat",
                    "3",
                    "--map-mode",
                    "plain",
                    "--output-dir",
                    str(output_dir),
                    "--quiet",
                ]
            )
            with patch.object(module, "MapGroundedHeadingExperiment", FakeExperiment):
                payload = module.evaluate(args)
            self.assertEqual(payload["case_count"], 1)
            self.assertEqual(payload["expected_run_count"], 3)
            self.assertEqual(payload["metrics"]["completed_run_count"], 3)
            self.assertEqual(payload["config"]["map_mode"], "plain")
            self.assertIsNone(payload["paths"]["maps"])
            self.assertFalse((output_dir / "maps").exists())
            run_records = [
                self.module.json.loads(line)
                for line in (output_dir / "runs.jsonl").read_text().splitlines()
            ]
            self.assertTrue(all(record["map_mode"] == "plain" for record in run_records))
            self.assertTrue(
                all(
                    Path(record["semantic_map_path"]).resolve()
                    == Path(payload["paths"]["input_map"]).resolve()
                    for record in run_records
                )
            )
            self.assertTrue(
                all(record["map"]["room_highlights_rendered"] is False for record in run_records)
            )
            self.assertTrue((output_dir / "runs.jsonl").is_file())
            self.assertTrue((output_dir / "summary.json").is_file())
            self.assertTrue((output_dir / "meeting_summary.md").is_file())
            self.assertEqual(len((output_dir / "runs.jsonl").read_text().splitlines()), 3)


if __name__ == "__main__":
    unittest.main()
