from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from st_nav.common.types import (
    CandidateAction,
    ReasoningInput,
    RenderedView,
    TaskSpec,
)
from st_nav.decision import EdgeVisitActionPolicy, GreedyActionPolicy, VisualThemeActionPolicy
from st_nav.pipeline import NavigationPipelineConfig, build_navigation_pipeline
from st_nav.spatial import GroundingIndex, SpatialEngine


MODEL_ENV = {
    "ST_NAV_MODEL_PROVIDER": "gemini",
    "ST_NAV_API_KEY": "test-key",
    "ST_NAV_API_BASE": "https://generativelanguage.googleapis.com/v1beta",
    "ST_NAV_API_KIND": "responses",
    "ST_NAV_MODEL_NAME": "gemma-4-31b-it",
    "ST_NAV_TEMPERATURE": "0",
}


def direction_response(view_id: str, *, confidence: float = 0.8) -> dict:
    return {
        "output_text": json.dumps(
            {
                "selected_view_id": view_id,
                "confidence": confidence,
                "visual_evidence": "The doorway reveals matching reliefs.",
            }
        )
    }


def candidate(target_pano_id: str, heading: float, index: int) -> CandidateAction:
    return CandidateAction(
        target_pano_id=target_pano_id,
        absolute_heading=heading,
        relative_heading=None,
        relative_label=None,
        target_room_id=None,
        score=0.0,
        reason="test",
        metadata={
            "candidate_index": index,
            "turn_error_deg": None,
            "history_return_error_deg": None,
        },
    )


class VisualThemeBaselineTests(unittest.TestCase):
    def _reasoning_input(self, root: Path) -> ReasoningInput:
        views = []
        for index in range(8):
            image_path = root / f"view_{index}.jpg"
            image_path.write_bytes(f"image-{index}".encode())
            views.append(
                RenderedView(
                    label=f"museum_{index}",
                    heading=float(index * 45),
                    path=str(image_path),
                )
            )
        return ReasoningInput(
            task=TaskSpec(task_type="gallery_goal_navigation", raw_instruction="Navigate."),
            route=["Room 8", "Room 10"],
            candidates=[candidate("pano-a", 0.0, 0), candidate("pano-b", 90.0, 1)],
            current_pano_id="pano-current",
            current_room_id="Room 8",
            subgoal_room_id="Room 10",
            views=views,
        )

    @staticmethod
    def _room_graph() -> dict[str, dict]:
        return {
            "Room 8": {
                "title": "Assyria: Nimrud",
                "category": "gallery",
                "visual_profile": {
                    "short_description": "Large stone guardian figures.",
                    "visual_cues": ["lamassu", "stone reliefs"],
                },
            },
            "Room 10": {
                "title": "Assyria: Lion hunts",
                "category": "gallery",
                "visual_profile": {
                    "short_description": "Narrative palace reliefs showing royal hunts.",
                    "visual_cues": [
                        "lion hunt reliefs",
                        "long carved stone walls",
                        "royal hunting scenes",
                        "Assyrian palace panels",
                        "must be truncated",
                    ],
                },
            },
        }

    def test_prompt_contains_only_subgoal_theme_and_neutral_view_ids(self) -> None:
        calls: list[dict] = []
        with mock.patch.dict(os.environ, MODEL_ENV, clear=False), tempfile.TemporaryDirectory() as tmpdir:
            reasoning_input = self._reasoning_input(Path(tmpdir))
            policy = VisualThemeActionPolicy(
                room_graph=self._room_graph(),
                response_client=lambda body: calls.append(body) or direction_response("view_2"),
            )

            output = policy.choose_next_action(reasoning_input)

        self.assertIsInstance(policy, GreedyActionPolicy)
        self.assertNotIsInstance(policy, EdgeVisitActionPolicy)
        self.assertEqual(output.action.target_pano_id, "pano-b")
        self.assertEqual(len(calls), 1)
        text_parts = [
            part["text"]
            for part in calls[0]["input"][0]["content"]
            if part.get("type") == "input_text"
        ]
        prompt_text = "\n".join(text_parts)
        self.assertIn("Assyria: Lion hunts", prompt_text)
        self.assertIn("lion hunt reliefs", prompt_text)
        self.assertNotIn("must be truncated", prompt_text)
        self.assertNotIn("Room 10", prompt_text)
        self.assertNotIn("Room 8", prompt_text)
        self.assertNotIn("pano-a", prompt_text)
        self.assertNotIn("90.0", prompt_text)
        self.assertEqual(
            [
                part["text"]
                for part in calls[0]["input"][0]["content"]
                if part.get("type") == "input_text" and part["text"].startswith("View id:")
            ],
            [f"View id: view_{index}." for index in range(8)],
        )

    def test_revisit_does_not_use_trajectory_memory(self) -> None:
        calls: list[dict] = []
        with mock.patch.dict(os.environ, MODEL_ENV, clear=False), tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            reasoning_input = self._reasoning_input(root)
            policy = VisualThemeActionPolicy(
                room_graph=self._room_graph(),
                cache_dir=root / "cache",
                response_client=lambda body: calls.append(body) or direction_response("view_0"),
            )

            first = policy.choose_next_action(reasoning_input)
            self.assertEqual(first.action.target_pano_id, "pano-a")
            second = policy.choose_next_action(reasoning_input)

        self.assertFalse(hasattr(policy, "record_transition"))
        self.assertEqual(second.action.target_pano_id, "pano-a")
        self.assertIn("trajectory_memory_enabled=false", second.rationale)
        self.assertFalse(second.action.metadata["visual_theme_trajectory_memory_enabled"])
        self.assertNotIn("edge_visit_count", second.action.metadata)
        self.assertEqual(len(calls), 1, "The same validated view selection should be cached.")

    def test_invalid_direction_output_retries_once_without_geometry_fallback(self) -> None:
        calls: list[dict] = []

        def response_client(body: dict) -> dict:
            calls.append(body)
            if len(calls) == 1:
                return direction_response("view_99")
            return direction_response("view_2")

        with mock.patch.dict(os.environ, MODEL_ENV, clear=False), tempfile.TemporaryDirectory() as tmpdir:
            policy = VisualThemeActionPolicy(
                room_graph=self._room_graph(),
                schema_validation_retries=1,
                response_client=response_client,
            )
            output = policy.choose_next_action(self._reasoning_input(Path(tmpdir)))

        self.assertEqual(output.action.target_pano_id, "pano-b")
        self.assertEqual(len(calls), 2)
        self.assertIn("Do not return a list, ranking, repeated choices", calls[1]["instructions"])
        self.assertEqual(policy.last_validation_attempt_count, 2)

    def test_none_heading_alignment_leaves_offset_disabled(self) -> None:
        engine = SpatialEngine(
            room_graph={},
            pano_graph={"pano": {"neighbors": []}},
            grounding_index=GroundingIndex({}),
            heading_alignment_mode="none",
        )
        state = engine.initialize(start_pano_id="pano")
        self.assertIsNone(state.geocentric_to_allocentric_offset_deg)
        self.assertEqual(state.heading_alignment_status, "disabled")
        self.assertTrue(engine.initialize_heading_alignment(state, object(), step_index=0))

    def test_factory_enforces_no_alignment_visual_localization_ablation(self) -> None:
        common = {
            "room_graph": {},
            "pano_graph": {"pano": {"neighbors": []}},
            "grounding_payload": {},
            "pano_room_grounding": {},
        }
        with self.assertRaisesRegex(ValueError, "heading_alignment_mode='none'"):
            build_navigation_pipeline(
                **common,
                config=NavigationPipelineConfig(
                    direction_policy="visual-theme",
                    localization_effective_stage="visual",
                    heading_alignment_mode="fixed",
                ),
            )
        with self.assertRaisesRegex(ValueError, "localization_effective_stage='visual'"):
            build_navigation_pipeline(
                **common,
                config=NavigationPipelineConfig(
                    direction_policy="visual-theme",
                    localization_effective_stage="spatial",
                    heading_alignment_mode="none",
                ),
            )


if __name__ == "__main__":
    unittest.main()
