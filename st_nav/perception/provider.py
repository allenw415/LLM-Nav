from __future__ import annotations

from pathlib import Path

from ..common.types import Observation
from .detector import PerceptionPipeline


class ManifestPerceptionProvider:
    """
    Backward-compatible wrapper over the explicit perception pipeline.
    """

    def __init__(
        self,
        pano_graph: dict[str, dict],
        *,
        room_graph: dict[str, dict] | None = None,
        grounding_index=None,
    ):
        self.pipeline = PerceptionPipeline(
            pano_graph=pano_graph,
            room_graph=room_graph,
            grounding_index=grounding_index,
        )

    def observe(self, manifest_path: str | Path, *, current_heading: float) -> Observation:
        return self.pipeline.observe_from_manifest(manifest_path, current_heading=current_heading)
