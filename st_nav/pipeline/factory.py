from __future__ import annotations

from dataclasses import dataclass

from ..decision import LLMActionPolicy, LLMInstructionParser
from ..execution import EpisodeRunner
from ..perception import ManifestPerceptionProvider, PanoramaRenderer
from ..spatial import (
    EvidenceScoreLocalizer,
    GroundingIndex,
    SourcePanoResolver,
    SpatialAlignmentRefiner,
    SpatialEngine,
)
from .navigation import NavigationPipeline
from .source_resolution import SourceResolutionWorkflow


@dataclass(frozen=True)
class NavigationPipelineConfig:
    llm_model: str | None = None
    llm_api_key: str | None = None
    llm_api_kind: str | None = None
    llm_api_base: str | None = None
    llm_timeout: float | None = None
    alignment_candidate_ratio_threshold: float = 0.5
    alignment_candidate_max: int = 5


def build_navigation_pipeline(
    *,
    room_graph: dict[str, dict],
    pano_graph: dict[str, dict],
    grounding_payload: dict | None = None,
    pano_room_grounding: dict | None = None,
    config: NavigationPipelineConfig | None = None,
) -> NavigationPipeline:
    config = config or NavigationPipelineConfig()
    grounding_index = GroundingIndex(
        grounding_payload or {},
        pano_to_room=pano_room_grounding or {},
    )
    instruction_parser = LLMInstructionParser(
        room_graph=room_graph,
        model=config.llm_model,
        api_key=config.llm_api_key,
        api_base=config.llm_api_base,
        api_kind=config.llm_api_kind,
        request_timeout=config.llm_timeout,
    )
    spatial_engine = SpatialEngine(
        room_graph=room_graph,
        pano_graph=pano_graph,
        grounding_index=grounding_index,
        localizer=build_evidence_score_localizer(
            config,
            room_graph=room_graph,
            grounding_index=grounding_index,
        ),
    )
    return NavigationPipeline(
        source_resolution_workflow=SourceResolutionWorkflow(
            instruction_parser=instruction_parser,
            source_pano_resolver=SourcePanoResolver(grounding_index),
        ),
        episode_runner=EpisodeRunner(
            perception_provider=ManifestPerceptionProvider(
                pano_graph=pano_graph,
                room_graph=room_graph,
                grounding_index=grounding_index,
            ),
            spatial_engine=spatial_engine,
            policy=LLMActionPolicy(
                model=config.llm_model,
                api_key=config.llm_api_key,
                api_base=config.llm_api_base,
                api_kind=config.llm_api_kind,
                request_timeout=config.llm_timeout,
            ),
            renderer=PanoramaRenderer(pano_graph),
        ),
    )


def build_evidence_score_localizer(
    config: NavigationPipelineConfig,
    *,
    room_graph: dict[str, dict],
    grounding_index: GroundingIndex,
):
    return EvidenceScoreLocalizer(
        room_graph=room_graph,
        grounding_index=grounding_index,
        alignment_candidate_ratio_threshold=config.alignment_candidate_ratio_threshold,
        alignment_candidate_max=config.alignment_candidate_max,
        spatial_refiner=SpatialAlignmentRefiner(
            room_graph=room_graph,
            grounding_index=grounding_index,
            model=config.llm_model,
            api_key=config.llm_api_key,
            api_base=config.llm_api_base,
            api_kind=config.llm_api_kind,
            request_timeout=config.llm_timeout,
        ),
    )
