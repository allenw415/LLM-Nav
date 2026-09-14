from __future__ import annotations

import unittest

from st_nav.common.types import BeliefState, CandidateAction, Observation, PolicyOutput, TaskSpec
from st_nav.execution.episode_runner import EpisodeRunner
from st_nav.pipeline.factory import NavigationPipelineConfig
from st_nav.spatial.state import StateEstimator


class _GroundingIndex:
    @staticmethod
    def room_for_pano(pano_id: str) -> str:
        return "Room A"


class _ManifestFreeEpisodeRunner(EpisodeRunner):
    def _ensure_manifest(self, *, pano_id, **kwargs):
        return pano_id


class LocalizationIntervalTests(unittest.TestCase):
    def test_runner_localizes_after_interval_and_immediately_confirms_pending_room(self) -> None:
        next_panos = {"A": "B", "B": "C", "C": "D", "D": "E", "E": "F"}

        class FakePerceptionProvider:
            def __init__(self) -> None:
                self.localization_calls = []
                self.tracking_calls = []

            def observe(self, manifest_path, *, current_heading):
                pano_id = str(manifest_path)
                self.localization_calls.append(pano_id)
                return Observation(
                    pano_id=pano_id,
                    heading_estimate=current_heading,
                    metadata={"localized_room_id": "Room A", "localization_confidence": 1.0},
                )

            def observe_tracking(self, manifest_path, *, current_heading):
                pano_id = str(manifest_path)
                self.tracking_calls.append(pano_id)
                return Observation(pano_id=pano_id, heading_estimate=current_heading)

        class FakeSpatialEngine:
            def __init__(self) -> None:
                self.grounding_index = _GroundingIndex()
                self.localization_updates = []
                self.tracking_updates = []

            def initialize(self, *, start_pano_id, start_room_id, start_heading):
                return BeliefState(
                    current_pano_id=start_pano_id,
                    current_room_id=start_room_id,
                    grounded_room_id="Room A",
                    current_heading=start_heading,
                    room_belief={"Room A": 1.0},
                )

            def update(self, state, observation):
                self.localization_updates.append(observation.pano_id)
                state.current_pano_id = observation.pano_id
                if observation.pano_id == "D":
                    state.pending_room_id = "Room B"
                    state.pending_room_confirmation_count = 1
                    state.pending_room_last_pano_id = "D"
                elif observation.pano_id == "E":
                    state.pending_room_id = None
                    state.pending_room_confirmation_count = 0
                    state.pending_room_last_pano_id = None
                observation.metadata["localization_performed"] = True
                observation.metadata["localized_room_id"] = state.current_room_id
                observation.metadata["grounded_room_id"] = "Room A"
                observation.metadata["room_belief"] = dict(state.room_belief)
                return state

            def update_tracking(self, state, observation):
                self.tracking_updates.append(observation.pano_id)
                state.current_pano_id = observation.pano_id
                observation.metadata["localization_performed"] = False
                observation.metadata["localized_room_id"] = state.current_room_id
                observation.metadata["grounded_room_id"] = "Room A"
                observation.metadata["room_belief"] = dict(state.room_belief)
                return state

            def route_to_goal(self, task, state):
                return [state.current_room_id, task.goal_room_ids[-1]]

            def next_subgoal_room_id(self, state, route):
                return route[-1]

            def build_current_room_context(self, state, route):
                return {"room_id": state.current_room_id}

            def extract_visible_passages(self, state, observation):
                return []

            def goal_reached(self, task, state):
                return False

            def describe_view_contexts(self, observation):
                return []

            def generate_candidates(self, state, route, **kwargs):
                target_pano_id = next_panos.get(state.current_pano_id)
                if target_pano_id is None:
                    return []
                return [
                    CandidateAction(
                        target_pano_id=target_pano_id,
                        absolute_heading=0.0,
                        relative_heading=0.0,
                        relative_label="front",
                        target_room_id="Room A",
                        score=1.0,
                        reason="test",
                    )
                ]

        class FirstCandidatePolicy:
            def choose_next_action(self, reasoning_input):
                action = reasoning_input.candidates[0] if reasoning_input.candidates else None
                return PolicyOutput(action=action, rationale="test")

        perception = FakePerceptionProvider()
        spatial = FakeSpatialEngine()
        runner = _ManifestFreeEpisodeRunner(
            perception_provider=perception,
            spatial_engine=spatial,
            policy=FirstCandidatePolicy(),
            localization_interval=3,
        )

        _, traces = runner.run(
            task=TaskSpec(
                task_type="test",
                raw_instruction="test",
                source_room_id="Room A",
                goal_room_ids=["Room Z"],
            ),
            start_pano_id="A",
            start_room_id="Room A",
            step_budget=5,
        )

        self.assertEqual(perception.localization_calls, ["A", "D", "E"])
        self.assertEqual(perception.tracking_calls, ["B", "C"])
        self.assertEqual(spatial.localization_updates, ["A", "D", "E"])
        self.assertEqual(spatial.tracking_updates, ["B", "C"])
        self.assertEqual(
            [trace.observation.metadata["localization_performed"] for trace in traces],
            [True, False, False, True, True],
        )
        self.assertEqual(
            [
                trace.observation.metadata["movement_steps_since_last_localization"]
                for trace in traces
            ],
            [0, 1, 2, 3, 1],
        )

    def test_tracking_update_preserves_room_belief_without_calling_localizer(self) -> None:
        class FailingLocalizer:
            def localize(self, **kwargs):
                raise AssertionError("tracking update must not call localizer")

        estimator = StateEstimator(
            room_graph={"Room A": {"neighbors": []}},
            pano_graph={"A": {"neighbors": []}, "B": {"neighbors": []}},
            grounding_index=_GroundingIndex(),
            localizer=FailingLocalizer(),
        )
        state = estimator.initialize(start_pano_id="A", start_room_id="Room A")
        observation = Observation(pano_id="B", heading_estimate=90.0)

        estimator.update(state, observation, localize=False)

        self.assertEqual(state.room_belief, {"Room A": 1.0})
        self.assertFalse(observation.metadata["localization_performed"])
        self.assertEqual(observation.metadata["localization_skipped_reason"], "localization_interval")

    def test_pipeline_defaults_to_three_action_localization_interval(self) -> None:
        self.assertEqual(NavigationPipelineConfig().localization_interval, 3)

    def test_pipeline_defaults_to_spatial_effective_localization_stage(self) -> None:
        self.assertEqual(
            NavigationPipelineConfig().localization_effective_stage,
            "spatial",
        )


if __name__ == "__main__":
    unittest.main()
