from __future__ import annotations

from dataclasses import dataclass, field
import time

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

    def __init__(
        self,
        *,
        source_resolution_workflow: SourceResolutionWorkflow,
        episode_runner,
    ):
        self.source_resolution_workflow = source_resolution_workflow
        self.episode_runner = episode_runner

    def run(
        self,
        instruction: str | None = None,
        progress_callback=None,
        *,
        start_pano_id: str | None = None,
        target_room_id: str | None = None,
        waypoint_room_ids: list[str] | None = None,
        **episode_kwargs,
    ) -> NavigationPipelineResult:
        pipeline_started = time.perf_counter()
        has_instruction = isinstance(instruction, str) and bool(instruction.strip())
        has_start = isinstance(start_pano_id, str) and bool(start_pano_id.strip())
        has_target = isinstance(target_room_id, str) and bool(target_room_id.strip())
        has_explicit_waypoints = bool(waypoint_room_ids)

        if not has_start:
            raise ValueError("Navigation requires start_pano_id.")
        if has_instruction and has_target:
            raise ValueError(
                "Instruction navigation cannot be combined with target_room_id."
            )
        if has_instruction and has_explicit_waypoints:
            raise ValueError(
                "Instruction navigation cannot be combined with explicit waypoint_room_ids."
            )
        if not has_instruction and not has_target:
            raise ValueError(
                "Provide instruction + start_pano_id or start_pano_id + target_room_id."
            )
        if has_explicit_waypoints and not has_target:
            raise ValueError(
                "Explicit waypoint_room_ids require target_room_id."
            )

        resolution_mode = "query" if has_instruction else "explicit"
        if progress_callback is not None:
            progress_callback({"event": "pipeline_start", "instruction": instruction})
            progress_callback(
                {
                    "event": "source_resolution_start",
                    "instruction": instruction,
                    "resolution_mode": resolution_mode,
                }
            )

        source_started = time.perf_counter()
        if has_instruction:
            source = self.source_resolution_workflow.run(
                instruction.strip(),
                start_pano_id=start_pano_id.strip(),
            )
        else:
            source = self.source_resolution_workflow.run_explicit(
                start_pano_id=start_pano_id.strip(),
                target_room_id=target_room_id.strip(),
                waypoint_room_ids=waypoint_room_ids,
            )
        source_total_ms = (time.perf_counter() - source_started) * 1000.0
        parser_client = getattr(
            self.source_resolution_workflow.instruction_parser,
            "model_client",
            None,
        )
        parser_vlm_ms = (
            getattr(parser_client, "last_call_wall_ms", 0.0)
            if has_instruction
            else 0.0
        )
        if progress_callback is not None:
            progress_callback(
                {
                    "event": "module_timing",
                    "module": "source_resolution",
                    "total_wall_ms": source_total_ms,
                    "vlm_response_ms": (
                        float(parser_vlm_ms)
                        if isinstance(parser_vlm_ms, (int, float))
                        else 0.0
                    ),
                    "program_module_ms": max(
                        0.0,
                        source_total_ms
                        - (
                            float(parser_vlm_ms)
                            if isinstance(parser_vlm_ms, (int, float))
                            else 0.0
                        ),
                    ),
                    "call_count": 1,
                }
            )

        if progress_callback is not None:
            progress_callback(
                {
                    "event": "source_resolution_done",
                    "source_room_id": source.task.source_room_id,
                    "source_pano_id": source.source_pano.pano_id,
                    "goal_room_ids": list(source.task.goal_room_ids),
                    "waypoint_room_ids": list(source.task.waypoint_room_ids),
                    "resolution_method": source.source_pano.resolution_method,
                }
            )
        final_state, traces = self.episode_runner.run(
            task=source.task,
            start_pano_id=source.source_pano.pano_id,
            start_room_id=source.task.source_room_id,
            progress_callback=progress_callback,
            **episode_kwargs,
        )
        if progress_callback is not None:
            progress_callback(
                {
                    "event": "pipeline_done",
                    "final_pano_id": getattr(final_state, "current_pano_id", None),
                    "final_room_id": getattr(final_state, "current_room_id", None),
                    "trace_count": len(traces),
                    "end_to_end_wall_ms": (
                        time.perf_counter() - pipeline_started
                    )
                    * 1000.0,
                }
            )
        return NavigationPipelineResult(
            instruction=source.instruction,
            task=source.task,
            source=source,
            final_state=final_state,
            traces=list(traces),
        )
