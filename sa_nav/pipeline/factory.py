from __future__ import annotations

from dataclasses import dataclass
import random

from ..decision import (
    EdgeVisitActionPolicy,
    GreedyActionPolicy,
    LLMActionPolicy,
    LLMInstructionParser,
    VisualThemeActionPolicy,
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
    query_cache_dir: str | None = None
    visual_cache_dir: str | None = None
    visual_max_output_tokens: int | None = None
    visual_schema_validation_retries: int = 0
    direction_cache_dir: str | None = None
    direction_max_output_tokens: int | None = None
    direction_schema_validation_retries: int = 1
    spatial_schema_validation_retries: int = 0
    spatial_context_max_output_tokens: int | None = None
    spatial_alignment_max_output_tokens: int | None = None
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
    if config.direction_policy not in {
        "geometry",
        "geometry-edge-visit",
        "llm",
        "visual-theme",
    }:
        raise ValueError(
            "direction_policy must be 'geometry', 'geometry-edge-visit', 'llm', or 'visual-theme'."
        )
    if config.heading_alignment_mode not in {"none", "fixed", "initial-spatial"}:
        raise ValueError(
            "heading_alignment_mode must be 'none', 'fixed', or 'initial-spatial'."
        )
    if config.localization_effective_stage not in {"visual", "bayesian", "spatial"}:
        raise ValueError(
            "localization_effective_stage must be 'visual', 'bayesian', or 'spatial'."
        )
    if config.direction_policy == "visual-theme":
        if config.heading_alignment_mode != "none":
            raise ValueError(
                "visual-theme direction_policy requires heading_alignment_mode='none'."
            )
        if config.localization_effective_stage != "visual":
            raise ValueError(
                "visual-theme direction_policy requires localization_effective_stage='visual'."
            )
    if config.visual_schema_validation_retries < 0:
        raise ValueError("visual_schema_validation_retries must be non-negative.")
    if config.direction_schema_validation_retries < 0:
        raise ValueError("direction_schema_validation_retries must be non-negative.")
    if config.spatial_schema_validation_retries < 0:
        raise ValueError("spatial_schema_validation_retries must be non-negative.")
    for name, value in (
        ("visual_max_output_tokens", config.visual_max_output_tokens),
        ("direction_max_output_tokens", config.direction_max_output_tokens),
        ("spatial_context_max_output_tokens", config.spatial_context_max_output_tokens),
        ("spatial_alignment_max_output_tokens", config.spatial_alignment_max_output_tokens),
    ):
        if value is not None and value < 1:
            raise ValueError(f"{name} must be positive when set.")
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
        cache_dir=config.query_cache_dir,
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
    elif config.direction_policy == "llm":
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
    else:
        direction_policy = VisualThemeActionPolicy(
            room_graph=room_graph,
            model=config.llm_model,
            api_key=config.llm_api_key,
            api_base=config.llm_api_base,
            api_kind=config.llm_api_kind,
            request_timeout=config.llm_timeout,
            cache_dir=config.direction_cache_dir,
            schema_validation_retries=config.direction_schema_validation_retries,
            max_output_tokens=config.direction_max_output_tokens,
            u_turn_threshold_deg=config.u_turn_threshold_deg,
            history_return_threshold_deg=config.history_return_threshold_deg,
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
                model=config.llm_model,
                api_key=config.llm_api_key,
                api_base=config.llm_api_base,
                api_kind=config.llm_api_kind,
                request_timeout=config.llm_timeout,
                cache_dir=config.visual_cache_dir,
                schema_validation_retries=config.visual_schema_validation_retries,
                max_output_tokens=config.visual_max_output_tokens,
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
            schema_validation_retries=config.spatial_schema_validation_retries,
            context_max_output_tokens=config.spatial_context_max_output_tokens,
            alignment_max_output_tokens=config.spatial_alignment_max_output_tokens,
        ),
    )
