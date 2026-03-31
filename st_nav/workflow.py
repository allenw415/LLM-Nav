from __future__ import annotations

from dataclasses import dataclass, field

from .models import Observation, SourcePanoResolution, TaskSpec


@dataclass
class ParsedRoutePlan:
    instruction: str
    source_room_id: str | None
    target_room_id: str | None
    waypoint_room_ids: list[str] = field(default_factory=list)
    shortest_path: list[str] = field(default_factory=list)
    task: TaskSpec | None = None


@dataclass
class SourcePerceptionResult:
    instruction: str
    task: TaskSpec
    source_pano: SourcePanoResolution
    manifest_path: str
    observation: Observation


class InstructionRoutePlanner:
    """
    Minimal planning flow:
    instruction -> parser -> source/target -> shortest room route
    """

    def __init__(self, *, instruction_parser, spatial_engine):
        self.instruction_parser = instruction_parser
        self.spatial_engine = spatial_engine

    def plan(self, instruction: str) -> ParsedRoutePlan:
        task = self.instruction_parser.parse(instruction)
        target_room_id = task.goal_room_ids[-1] if task.goal_room_ids else None
        shortest_path: list[str] = []
        if task.source_room_id and target_room_id:
            shortest_path = self.spatial_engine.shortest_room_route(
                task.source_room_id,
                target_room_id,
                task.waypoint_room_ids,
            )
        return ParsedRoutePlan(
            instruction=instruction,
            source_room_id=task.source_room_id,
            target_room_id=target_room_id,
            waypoint_room_ids=list(task.waypoint_room_ids),
            shortest_path=shortest_path,
            task=task,
        )


class SourcePerceptionWorkflow:
    """
    Minimal start-of-episode flow:
    instruction -> parse -> source room -> source pano -> render -> detect -> aggregate
    """

    def __init__(self, *, instruction_parser, source_pano_resolver, perception_pipeline):
        self.instruction_parser = instruction_parser
        self.source_pano_resolver = source_pano_resolver
        self.perception_pipeline = perception_pipeline

    def run(
        self,
        instruction: str,
        *,
        api_key: str,
        output_dir: str,
        heading_mode: str = "museum",
        pitch: float = 0.0,
        fov: int = 45,
        width: int = 640,
        height: int = 640,
        current_heading: float = 330.0,
        graph_path: str | None = None,
    ) -> SourcePerceptionResult:
        task = self.instruction_parser.parse(instruction)
        if not task.source_room_id:
            raise RuntimeError("Instruction did not resolve a source room.")

        source_pano = self.source_pano_resolver.resolve(task.source_room_id)
        if not source_pano.pano_id:
            raise RuntimeError(f"No representative pano configured for {task.source_room_id}.")

        manifest = self.perception_pipeline.render_views(
            pano_id=source_pano.pano_id,
            api_key=api_key,
            output_dir=output_dir,
            heading_mode=heading_mode,
            pitch=pitch,
            fov=fov,
            width=width,
            height=height,
            graph_path=graph_path,
        )
        observation = self.perception_pipeline.observe_from_manifest(
            manifest["manifest_path"],
            current_heading=current_heading,
        )
        return SourcePerceptionResult(
            instruction=instruction,
            task=task,
            source_pano=source_pano,
            manifest_path=str(manifest["manifest_path"]),
            observation=observation,
        )
