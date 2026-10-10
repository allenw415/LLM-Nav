from __future__ import annotations

from pathlib import Path

from ..common.types import Observation
from .detector import PerceptionPipeline, ViewDetector


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
        use_detection_files: bool = True,
        model: str | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
        api_kind: str | None = None,
        request_timeout: float | None = None,
        cache_dir: str | Path | None = None,
        schema_validation_retries: int = 0,
        max_output_tokens: int | None = None,
    ):
        self.pipeline = PerceptionPipeline(
            pano_graph=pano_graph,
            room_graph=room_graph,
            grounding_index=grounding_index,
            detector=ViewDetector(
                model=model,
                api_key=api_key,
                api_base=api_base,
                api_kind=api_kind,
                request_timeout=request_timeout,
                room_graph=room_graph,
                grounding_index=grounding_index,
                use_detection_files=use_detection_files,
                cache_dir=cache_dir,
                schema_validation_retries=schema_validation_retries,
                max_output_tokens=max_output_tokens,
            ),
        )

    def observe(
        self,
        manifest_path: str | Path,
        *,
        current_heading: float,
        force_refresh: bool = False,
    ) -> Observation:
        return self.pipeline.observe_from_manifest(
            manifest_path,
            current_heading=current_heading,
            force_refresh=force_refresh,
        )

    def observe_from_manifest(
        self,
        manifest_path: str | Path,
        *,
        current_heading: float,
        force_refresh: bool = False,
    ) -> Observation:
        return self.pipeline.observe_from_manifest(
            manifest_path,
            current_heading=current_heading,
            force_refresh=force_refresh,
        )

    def observe_tracking(self, manifest_path: str | Path, *, current_heading: float) -> Observation:
        """Build an observation without running VLM detection/localization."""
        return self.pipeline.aggregate_observation(
            manifest_path,
            current_heading=current_heading,
            view_detections=[],
        )
