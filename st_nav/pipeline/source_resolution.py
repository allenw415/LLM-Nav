from __future__ import annotations

from dataclasses import dataclass

from ..common.types import SourcePanoResolution, TaskSpec


@dataclass
class SourceResolutionResult:
    instruction: str
    task: TaskSpec
    source_pano: SourcePanoResolution


class SourceResolutionWorkflow:
    """
    Resolve the source panorama for an instruction.
    """

    def __init__(self, *, instruction_parser, source_pano_resolver):
        self.instruction_parser = instruction_parser
        self.source_pano_resolver = source_pano_resolver

    def run(self, instruction: str) -> SourceResolutionResult:
        task = self.instruction_parser.parse(instruction)
        if not task.source_room_id:
            raise RuntimeError("Instruction did not resolve a source room.")

        source_pano = self.source_pano_resolver.resolve(task.source_room_id)
        if not source_pano.pano_id:
            raise RuntimeError(f"No representative pano configured for {task.source_room_id}.")

        return SourceResolutionResult(
            instruction=instruction,
            task=task,
            source_pano=source_pano,
        )
