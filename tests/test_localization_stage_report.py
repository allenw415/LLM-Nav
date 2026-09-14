from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from st_nav.common.types import Observation
from st_nav.spatial.localization import EvidenceScoreLocalizer
from tools.evaluation.navigation.evaluate_navigation_batch import BatchProgress
from tools.evaluation.navigation.report_localization_stages import (
    apply_error_replacements,
    build_report,
    flatten_localization_events,
)


class _AppliedRefiner:
    def refine(self, *, observation, candidate_room_ids):
        return {
            "applied": True,
            "alignment_predicted_room_id": "B",
            "alignment_top_k": [{"room_id": "B", "rank": 1}],
        }


class LocalizationStageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.room_graph = {
            "A": {"floor": "0", "neighbors": [{"target_room_id": "B"}]},
            "B": {"floor": "0", "neighbors": [{"target_room_id": "A"}]},
        }
        self.observation = Observation(
            pano_id="p1",
            metadata={
                "floor": "0",
                "visual_localization": {
                    "room_scores": [
                        {"room_id": "A", "score": 1.0},
                        {"room_id": "B", "score": 9.0},
                    ]
                },
            },
        )

    def test_localizer_exposes_three_stage_aliases_and_spatial_fallback(self) -> None:
        localizer = EvidenceScoreLocalizer(
            room_graph=self.room_graph,
            grounding_index=None,
            self_transition_weight=8.0,
            neighbor_transition_weight=1.0,
            alignment_candidate_ratio_threshold=0.0,
            spatial_refiner=None,
        )

        result = localizer.localize(
            observation=self.observation,
            prior_room_belief={"A": 1.0},
            fallback_room_id="A",
        )

        self.assertEqual(result["visual_predicted_room_id"], "B")
        self.assertEqual(result["prior_room_belief"], {"A": 1.0})
        self.assertEqual(result["bayesian_room_belief"], result["base_room_belief"])
        self.assertEqual(result["bayesian_predicted_room_id"], result["base_predicted_room_id"])
        self.assertEqual(result["spatial_predicted_room_id"], result["bayesian_predicted_room_id"])
        self.assertFalse(result["alignment_applied"])

    def test_spatial_stage_uses_successful_alignment(self) -> None:
        localizer = EvidenceScoreLocalizer(
            room_graph=self.room_graph,
            grounding_index=None,
            alignment_candidate_ratio_threshold=0.0,
            spatial_refiner=_AppliedRefiner(),
        )

        result = localizer.localize(
            observation=self.observation,
            prior_room_belief={"A": 0.5, "B": 0.5},
            fallback_room_id="A",
        )

        self.assertTrue(result["alignment_applied"])
        self.assertEqual(result["alignment_predicted_room_id"], "B")
        self.assertEqual(result["spatial_predicted_room_id"], "B")

    def test_batch_progress_only_records_completed_localization_calls(self) -> None:
        progress = BatchProgress()
        progress({"event": "localization_skipped", "localization_event": {"step_index": 1}})
        progress(
            {
                "event": "localization_done",
                "localization_event": {"step_index": 2, "pano_id": "p2"},
            }
        )

        self.assertEqual(progress.localization_events, [{"step_index": 2, "pano_id": "p2"}])


class LocalizationStageReportTests(unittest.TestCase):
    @staticmethod
    def _event(step: int, *, visual: str, bayesian: str, spatial: str | None, gt: str = "A") -> dict:
        return {
            "step_index": step,
            "pano_id": f"p{step}",
            "ground_truth_room_id": gt,
            "visual_cache_hit": step % 2 == 0,
            "visual_cache_version": 3,
            "visual_predicted_room_id": visual,
            "visual_room_scores": {"A": 5.0, "B": 1.0},
            "visual_room_belief": {"A": 0.8, "B": 0.2},
            "prior_room_belief": {"A": 1.0},
            "transition_support": {"A": 0.8, "B": 0.2},
            "bayesian_predicted_room_id": bayesian,
            "bayesian_room_belief": {"A": 0.9, "B": 0.1},
            "alignment_applied": spatial is not None,
            "alignment_predicted_room_id": spatial,
            "spatial_predicted_room_id": spatial,
            "stabilized_room_id": bayesian,
        }

    @classmethod
    def _record(cls, test_id: str = "T1", *, error=None) -> dict:
        events = [
            cls._event(0, visual="B", bayesian="A", spatial="A"),
            cls._event(3, visual="A", bayesian="B", spatial=None),
        ]
        return {
            "test_id": test_id,
            "stop_reason": "error" if error else "goal_reached",
            "error": error,
            "localization_event_count": len(events),
            "localization_events": events,
        }

    @staticmethod
    def _write_jsonl(path: Path, records: list[dict]) -> None:
        path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")

    def test_report_counts_denominators_transitions_cache_and_spatial_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            results = root / "results.jsonl"
            self._write_jsonl(results, [self._record()])

            summary = build_report(results_path=results, output_dir=root / "report")

            stage = {record["stage"]: record for record in summary["stages"]}
            self.assertEqual(stage["visual"]["correct_count"], 1)
            self.assertEqual(stage["bayesian"]["correct_count"], 1)
            self.assertEqual(stage["spatial"]["comparable_count"], 2)
            self.assertEqual(stage["spatial"]["correct_count"], 1)
            visual_to_bayes = summary["transitions"][0]
            self.assertEqual(visual_to_bayes["fail_to_success"], 1)
            self.assertEqual(visual_to_bayes["success_to_fail"], 1)
            self.assertEqual(summary["visual_cache"], {"hit": 1, "miss": 1})
            self.assertEqual(summary["alignment"], {"applied": 1, "skipped": 1})
            for name in (
                "localization_events.csv",
                "stage_summary.csv",
                "by_ground_truth_room.csv",
                "summary.json",
                "replacement_audit.json",
                "effective_results.jsonl",
            ):
                self.assertTrue((root / "report" / name).is_file())

    def test_duplicate_test_id_and_duplicate_event_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "results.jsonl"
            record = self._record()
            self._write_jsonl(path, [record, record])
            with self.assertRaisesRegex(ValueError, "Duplicate test_id"):
                build_report(results_path=path, output_dir=Path(tmpdir) / "report")

        duplicate_event_record = self._record()
        duplicate_event_record["localization_events"].append(
            dict(duplicate_event_record["localization_events"][0])
        )
        duplicate_event_record["localization_event_count"] = 3
        with self.assertRaisesRegex(ValueError, "Duplicate localization event"):
            flatten_localization_events([duplicate_event_record])

    def test_replacement_must_target_original_error_and_be_completed(self) -> None:
        original_error = self._record(error={"type": "TimeoutError", "message": "timeout"})
        successful = self._record()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            successful_path = root / "successful.jsonl"
            self._write_jsonl(successful_path, [successful])
            effective, audit = apply_error_replacements([original_error], [successful_path])
            self.assertIsNone(effective[0]["error"])
            self.assertEqual(audit["replacement_count"], 1)

            with self.assertRaisesRegex(ValueError, "only allowed for an original error"):
                apply_error_replacements([successful], [successful_path])

            unknown_path = root / "unknown.jsonl"
            self._write_jsonl(unknown_path, [self._record("UNKNOWN")])
            with self.assertRaisesRegex(ValueError, "Unknown replacement test_id"):
                apply_error_replacements([original_error], [unknown_path])

            failed_path = root / "failed.jsonl"
            self._write_jsonl(failed_path, [original_error])
            with self.assertRaisesRegex(ValueError, "completed non-error rerun"):
                apply_error_replacements([original_error], [failed_path])

    def test_report_reads_ground_truth_offset_from_each_result(self) -> None:
        record = self._record()
        record["initial_heading_alignment"] = {"ground_truth_offset_deg": 0.0}
        record["localization_events"][0]["estimated_offset_deg"] = 0.0
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            results = root / "results.jsonl"
            self._write_jsonl(results, [record])

            summary = build_report(results_path=results, output_dir=root / "report")

        self.assertEqual(summary["initial_alignment"]["ground_truth_offset_deg"], 0.0)
        self.assertEqual(summary["initial_alignment"]["correct_count"], 1)


if __name__ == "__main__":
    unittest.main()
