from __future__ import annotations

from dataclasses import dataclass, field

from ..common.types import TaskSpec
from ..execution.episode_runner import EpisodeTrace
from .source_resolution import SourceResolutionResult, SourceResolutionWorkflow


@dataclass
class NavigationPipelineResult:
    instruction: str
    task: TaskSpec
    source: SourceResolutionResult
    final_state: object
    traces: list[EpisodeTrace] = field(default_factory=list)


class NavigationPipeline:
    """
    End-to-end navigation pipeline that connects parsing, source resolution,
    and episode execution.
    """

    def __init__(self, *, source_resolution_workflow: SourceResolutionWorkflow, episode_runner):
        self.source_resolution_workflow = source_resolution_workflow
        self.episode_runner = episode_runner

    def run(self, instruction: str, **episode_kwargs) -> NavigationPipelineResult:
        source = self.source_resolution_workflow.run(instruction)
        final_state, traces = self.episode_runner.run(
            task=source.task,
            start_pano_id=source.source_pano.pano_id,
            start_room_id=source.task.source_room_id,
            **episode_kwargs,
        )
        return NavigationPipelineResult(
            instruction=instruction,
            task=source.task,
            source=source,
            final_state=final_state,
            traces=list(traces),
        )
