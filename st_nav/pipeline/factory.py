from __future__ import annotations

from dataclasses import dataclass
import random

from ..decision import (
    EdgeVisitActionPolicy,
    GreedyActionPolicy,
    LLMActionPolicy,
    LLMInstructionParser,
)
from ..execution import EpisodeRunner
from ..perception import ManifestPerceptionProvider, PanoramaRenderer
from ..perception.renderer import streetview_metadata_available
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
    u_turn_threshold_deg: float = 150.0
    history_return_threshold_deg: float = 30.0
    localization_interval: int = 3
    localization_effective_stage: str = "spatial"
    direction_policy: str = "geometry"
    heading_alignment_mode: str = "fixed"
    fixed_heading_offset_deg: float = 30.0
    force_initial_visual_refresh: bool = False
    stop_on_initial_alignment_failure: bool = True
    use_detection_cache: bool = True
    renderer_seed: int | None = None
    render_metadata_preflight: bool = True


def build_navigation_pipeline(
    *,
    room_graph: dict[str, dict],
    pano_graph: dict[str, dict],
    grounding_payload: dict | None = None,
    pano_room_grounding: dict | None = None,
    config: NavigationPipelineConfig | None = None,
) -> NavigationPipeline:
    config = config or NavigationPipelineConfig()
    if config.direction_policy not in {"geometry", "geometry-edge-visit", "llm"}:
        raise ValueError(
            "direction_policy must be 'geometry', 'geometry-edge-visit', or 'llm'."
        )
    if config.heading_alignment_mode not in {"fixed", "initial-spatial"}:
        raise ValueError(
            "heading_alignment_mode must be 'fixed' or 'initial-spatial'."
        )
    if config.localization_effective_stage not in {"visual", "bayesian", "spatial"}:
        raise ValueError(
            "localization_effective_stage must be 'visual', 'bayesian', or 'spatial'."
        )
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
        heading_alignment_mode=config.heading_alignment_mode,
        fixed_heading_offset_deg=config.fixed_heading_offset_deg,
    )
    if config.direction_policy == "geometry":
        direction_policy = GreedyActionPolicy(
            u_turn_threshold_deg=config.u_turn_threshold_deg,
            history_return_threshold_deg=config.history_return_threshold_deg,
        )
    elif config.direction_policy == "geometry-edge-visit":
        direction_policy = EdgeVisitActionPolicy(
            u_turn_threshold_deg=config.u_turn_threshold_deg,
            history_return_threshold_deg=config.history_return_threshold_deg,
        )
    else:
        direction_policy = LLMActionPolicy(
            model=config.llm_model,
            api_key=config.llm_api_key,
            api_base=config.llm_api_base,
            api_kind=config.llm_api_kind,
            request_timeout=config.llm_timeout,
            fallback_policy=GreedyActionPolicy(
                u_turn_threshold_deg=config.u_turn_threshold_deg,
                history_return_threshold_deg=config.history_return_threshold_deg,
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
                use_detection_files=config.use_detection_cache,
            ),
            spatial_engine=spatial_engine,
            policy=direction_policy,
            renderer=PanoramaRenderer(
                pano_graph,
                rng=(
                    random.Random(config.renderer_seed)
                    if config.renderer_seed is not None
                    else None
                ),
                metadata_checker=(
                    streetview_metadata_available
                    if config.render_metadata_preflight
                    else None
                ),
            ),
            localization_interval=config.localization_interval,
            force_initial_visual_refresh=config.force_initial_visual_refresh,
            stop_on_initial_alignment_failure=config.stop_on_initial_alignment_failure,
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
        effective_stage=config.localization_effective_stage,
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
