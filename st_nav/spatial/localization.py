from __future__ import annotations

import json
import mimetypes
import time
from base64 import b64encode
from pathlib import Path
from typing import Callable

from ..common.env import resolve_model_environment, resolve_task_num_ctx
from ..common.model_client import DEFAULT_OPENAI_API_BASE, ModelResponseClient, parse_json_output, resolve_api_kind
from ..common.prompts import (
    ALLOCENTRIC_DIRECTIONS,
    build_spatial_alignment_input,
    build_spatial_alignment_instructions,
    build_spatial_alignment_schema,
    build_spatial_context_extraction_input,
    build_spatial_context_extraction_instructions,
    build_spatial_context_extraction_schema,
)
from ..common.room_profiles import compact_visual_profile
from ..common.scoring import evidence_scores_to_distribution
from ..common.types import Observation
from ..perception.renderer import normalize_heading
from .grounding import GroundingIndex


DEFAULT_ALIGNMENT_CANDIDATE_RATIO_THRESHOLD = 0.5
DEFAULT_ALIGNMENT_CANDIDATE_MAX = 5


def _normalize_scores(scores: dict[str, float]) -> dict[str, float]:
    total = sum(value for value in scores.values() if value > 0.0)
    if total <= 0.0:
        return {key: 0.0 for key in scores}
    return {key: value / total for key, value in scores.items()}


def _ranked_room_ids(room_belief: dict[str, float]) -> list[str]:
    return [
        room_id
        for room_id, _ in sorted(
            (
                (room_id, float(probability))
                for room_id, probability in room_belief.items()
                if isinstance(room_id, str) and isinstance(probability, (int, float))
            ),
            key=lambda item: (-item[1], item[0]),
        )
    ]


def _candidate_room_ids(
    *,
    room_graph: dict[str, dict],
    observation: Observation,
    same_floor_only: bool,
) -> list[str]:
    if not same_floor_only:
        return sorted(room_graph.keys())
    floor = observation.metadata.get("floor")
    if floor is None:
        return sorted(room_graph.keys())
    floor_text = str(floor)
    return [
        room_id
        for room_id in sorted(room_graph.keys())
        if str(room_graph.get(room_id, {}).get("floor")) == floor_text
    ]


def _build_transition_support(
    *,
    room_graph: dict[str, dict],
    prior_room_belief: dict[str, float],
    candidate_room_ids: list[str],
    fallback_room_id: str | None,
    self_transition_weight: float,
    neighbor_transition_weight: float,
) -> dict[str, float]:
    candidate_set = set(candidate_room_ids)
    filtered_prior = {
        room_id: float(probability)
        for room_id, probability in prior_room_belief.items()
        if room_id in candidate_set and isinstance(probability, (int, float)) and probability > 0.0
    }
    if not filtered_prior:
        if fallback_room_id in candidate_set:
            return {room_id: 1.0 if room_id == fallback_room_id else 0.0 for room_id in candidate_room_ids}
        uniform = 1.0 / len(candidate_room_ids) if candidate_room_ids else 0.0
        return {room_id: uniform for room_id in candidate_room_ids}

    support = {room_id: 0.0 for room_id in candidate_room_ids}
    for source_room_id, source_probability in filtered_prior.items():
        targets = [source_room_id]
        for neighbor in room_graph.get(source_room_id, {}).get("neighbors", []):
            if not isinstance(neighbor, dict):
                continue
            target_room_id = neighbor.get("target_room_id")
            if target_room_id in candidate_set and target_room_id not in targets:
                targets.append(target_room_id)

        target_weights = {
            target_room_id: (
                self_transition_weight if target_room_id == source_room_id else neighbor_transition_weight
            )
            for target_room_id in targets
        }
        total_weight = sum(weight for weight in target_weights.values() if weight > 0.0)
        if total_weight <= 0.0:
            support[source_room_id] += source_probability
            continue
        for target_room_id, target_weight in target_weights.items():
            if target_weight > 0.0:
                support[target_room_id] += source_probability * (target_weight / total_weight)
    return support


def _room_scores_from_visual_localization(
    visual_localization: object,
    candidate_room_ids: list[str],
) -> dict[str, float]:
    scores = {room_id: 0.0 for room_id in candidate_room_ids}
    if not isinstance(visual_localization, dict):
        return scores
    raw_room_scores = visual_localization.get("room_scores")
    if not isinstance(raw_room_scores, list):
        return scores
    for record in raw_room_scores:
        if not isinstance(record, dict):
            continue
        room_id = record.get("room_id")
        score = record.get("score")
        if room_id in scores and isinstance(score, (int, float)):
            scores[room_id] = max(0.0, min(10.0, float(score)))
    return scores


def _alignment_candidate_room_ids(
    room_belief: dict[str, float],
    *,
    ratio_threshold: float,
    max_candidates: int,
) -> list[str]:
    ranked = [
        (room_id, float(probability))
        for room_id, probability in room_belief.items()
        if isinstance(room_id, str) and isinstance(probability, (int, float)) and float(probability) > 0.0
    ]
    if not ranked:
        return []
    ranked = sorted(ranked, key=lambda item: (-item[1], item[0]))
    top_probability = ranked[0][1]
    threshold = max(0.0, float(ratio_threshold)) * top_probability
    return [
        room_id
        for room_id, probability in ranked
        if probability >= threshold
    ][: max(0, int(max_candidates))]


class EvidenceScoreLocalizer:
    def __init__(
        self,
        *,
        room_graph: dict[str, dict],
        grounding_index: GroundingIndex,
        same_floor_only: bool = True,
        self_transition_weight: float = 1.0,
        neighbor_transition_weight: float = 1.0,
        alignment_candidate_ratio_threshold: float = DEFAULT_ALIGNMENT_CANDIDATE_RATIO_THRESHOLD,
        alignment_candidate_max: int = DEFAULT_ALIGNMENT_CANDIDATE_MAX,
        spatial_refiner: "SpatialAlignmentRefiner | None" = None,
        effective_stage: str = "spatial",
    ):
        if effective_stage not in {"visual", "bayesian", "spatial"}:
            raise ValueError(
                "effective_stage must be 'visual', 'bayesian', or 'spatial'."
            )
        self.room_graph = room_graph
        self.grounding_index = grounding_index
        self.same_floor_only = same_floor_only
        self.self_transition_weight = float(self_transition_weight)
        self.neighbor_transition_weight = float(neighbor_transition_weight)
        self.alignment_candidate_ratio_threshold = float(alignment_candidate_ratio_threshold)
        self.alignment_candidate_max = int(alignment_candidate_max)
        self.spatial_refiner = spatial_refiner
        self.effective_stage = effective_stage

    def localize(
        self,
        *,
        observation: Observation,
        prior_room_belief: dict[str, float] | None,
        fallback_room_id: str | None = None,
    ) -> dict:
        bayesian_started = time.perf_counter()
        candidate_room_ids = _candidate_room_ids(
            room_graph=self.room_graph,
            observation=observation,
            same_floor_only=self.same_floor_only,
        )
        if not candidate_room_ids:
            return self._empty_result()

        visual_localization = observation.metadata.get("visual_localization")
        raw_scores = _room_scores_from_visual_localization(visual_localization, candidate_room_ids)
        evidence_distribution = evidence_scores_to_distribution(raw_scores)
        if not any(value > 0.0 for value in evidence_distribution.values()):
            evidence_distribution = {room_id: 1.0 / len(candidate_room_ids) for room_id in candidate_room_ids}
        visual_predicted_room_id = self._predicted_room_id(evidence_distribution)
        prior_belief = {
            room_id: float(probability)
            for room_id, probability in (prior_room_belief or {}).items()
            if room_id in candidate_room_ids and isinstance(probability, (int, float))
        }

        transition_support = _build_transition_support(
            room_graph=self.room_graph,
            prior_room_belief=prior_room_belief or {},
            candidate_room_ids=candidate_room_ids,
            fallback_room_id=fallback_room_id,
            self_transition_weight=self.self_transition_weight,
            neighbor_transition_weight=self.neighbor_transition_weight,
        )
        base_scores = {
            room_id: float(transition_support.get(room_id, 0.0)) * float(evidence_distribution.get(room_id, 0.0))
            for room_id in candidate_room_ids
        }
        base_room_belief = _normalize_scores(base_scores)
        base_predicted_room_id = self._predicted_room_id(base_room_belief)
        evidence = self._visual_evidence(visual_localization)
        summary = self._visual_summary(visual_localization)
        result = {
            "predicted_room_id": base_predicted_room_id,
            "confidence": base_room_belief.get(base_predicted_room_id, 0.0) if base_predicted_room_id else 0.0,
            "room_belief": base_room_belief,
            "transition_support": transition_support,
            "observation_distribution": evidence_distribution,
            "observation_likelihood": evidence_distribution,
            "evidence_distribution": evidence_distribution,
            "raw_room_scores": raw_scores,
            "visual_room_scores": dict(raw_scores),
            "visual_room_belief": dict(evidence_distribution),
            "visual_predicted_room_id": visual_predicted_room_id,
            "prior_room_belief": prior_belief,
            "bayesian_room_belief": dict(base_room_belief),
            "bayesian_predicted_room_id": base_predicted_room_id,
            "base_predicted_room_id": base_predicted_room_id,
            "base_room_belief": dict(base_room_belief),
            "alignment_candidate_room_ids": [],
            "alignment_top_k": [],
            "alignment_applied": False,
            "alignment_predicted_room_id": None,
            "spatial_predicted_room_id": base_predicted_room_id,
            "evidence": evidence[:3],
            "summary": summary,
            "visual_localization": dict(visual_localization) if isinstance(visual_localization, dict) else {},
            "effective_localization_stage": self.effective_stage,
        }

        def select_effective_stage() -> None:
            if self.effective_stage == "visual":
                result["predicted_room_id"] = visual_predicted_room_id
                result["confidence"] = (
                    evidence_distribution.get(visual_predicted_room_id, 0.0)
                    if visual_predicted_room_id
                    else 0.0
                )
                result["room_belief"] = dict(evidence_distribution)
            elif self.effective_stage == "bayesian":
                result["predicted_room_id"] = base_predicted_room_id
                result["confidence"] = (
                    base_room_belief.get(base_predicted_room_id, 0.0)
                    if base_predicted_room_id
                    else 0.0
                )
                result["room_belief"] = dict(base_room_belief)

        select_effective_stage()

        bayesian_total_ms = (time.perf_counter() - bayesian_started) * 1000.0
        bayesian_timing = {
            "program_module_ms": bayesian_total_ms,
            "vlm_response_ms": 0.0,
            "stage_total_ms": bayesian_total_ms,
            "logical_vlm_call_count": 0,
        }
        visual_timing = observation.metadata.get("visual_timing")
        visual_timing = dict(visual_timing) if isinstance(visual_timing, dict) else {}

        def finalize(spatial_timing: dict) -> dict:
            visual_total = visual_timing.get("stage_total_ms")
            visual_total = float(visual_total) if isinstance(visual_total, (int, float)) else 0.0
            spatial_total = spatial_timing.get("stage_total_ms")
            spatial_total = float(spatial_total) if isinstance(spatial_total, (int, float)) else 0.0
            result["localization_timing"] = {
                "visual": visual_timing,
                "bayesian": bayesian_timing,
                "spatial": dict(spatial_timing),
                "three_stage_total_ms": visual_total + bayesian_total_ms + spatial_total,
                "effective_stage": self.effective_stage,
            }
            return result

        spatial_started = time.perf_counter()
        force_spatial_alignment = bool(
            observation.metadata.get("force_spatial_alignment", False)
        )
        if self.effective_stage != "spatial" and not force_spatial_alignment:
            result["alignment_skipped_reason"] = "disabled_by_effective_stage"
            result["spatial_predicted_room_id"] = None
            spatial_total_ms = (time.perf_counter() - spatial_started) * 1000.0
            return finalize(
                {
                    "program_module_ms": spatial_total_ms,
                    "vlm_response_ms": 0.0,
                    "stage_total_ms": spatial_total_ms,
                    "logical_vlm_call_count": 0,
                    "status": "disabled",
                }
            )
        alignment_candidate_ids = _alignment_candidate_room_ids(
            base_room_belief,
            ratio_threshold=(
                0.0 if force_spatial_alignment else self.alignment_candidate_ratio_threshold
            ),
            max_candidates=self.alignment_candidate_max,
        )
        if force_spatial_alignment and len(alignment_candidate_ids) < 2:
            fallback_ranked = sorted(
                candidate_room_ids,
                key=lambda room_id: (
                    -float(base_room_belief.get(room_id, 0.0)),
                    -float(evidence_distribution.get(room_id, 0.0)),
                    room_id,
                ),
            )
            for room_id in fallback_ranked:
                if room_id not in alignment_candidate_ids:
                    alignment_candidate_ids.append(room_id)
                if len(alignment_candidate_ids) >= min(2, len(candidate_room_ids)):
                    break
            alignment_candidate_ids = alignment_candidate_ids[
                : max(2, self.alignment_candidate_max)
            ]
        result["alignment_candidate_room_ids"] = list(alignment_candidate_ids)
        if len(alignment_candidate_ids) < 2:
            result["alignment_skipped_reason"] = "insufficient_alignment_candidates"
            spatial_total_ms = (time.perf_counter() - spatial_started) * 1000.0
            return finalize(
                {
                    "program_module_ms": spatial_total_ms,
                    "vlm_response_ms": 0.0,
                    "stage_total_ms": spatial_total_ms,
                    "logical_vlm_call_count": 0,
                    "status": "skipped",
                }
            )
        if self.spatial_refiner is None:
            result["alignment_skipped_reason"] = "missing_spatial_refiner"
            spatial_total_ms = (time.perf_counter() - spatial_started) * 1000.0
            return finalize(
                {
                    "program_module_ms": spatial_total_ms,
                    "vlm_response_ms": 0.0,
                    "stage_total_ms": spatial_total_ms,
                    "logical_vlm_call_count": 0,
                    "status": "skipped",
                }
            )

        refinement = self.spatial_refiner.refine(
            observation=observation,
            candidate_room_ids=alignment_candidate_ids,
        )
        if not refinement.get("applied"):
            result.update(refinement)
            spatial_timing = refinement.get("spatial_timing")
            if not isinstance(spatial_timing, dict):
                spatial_total_ms = (time.perf_counter() - spatial_started) * 1000.0
                spatial_timing = {
                    "program_module_ms": spatial_total_ms,
                    "vlm_response_ms": 0.0,
                    "stage_total_ms": spatial_total_ms,
                    "logical_vlm_call_count": 0,
                    "status": "failed",
                }
            return finalize(spatial_timing)

        alignment_predicted_room_id = refinement.get("alignment_predicted_room_id")
        if isinstance(alignment_predicted_room_id, str) and alignment_predicted_room_id in base_room_belief:
            result["predicted_room_id"] = alignment_predicted_room_id
            result["confidence"] = base_room_belief.get(alignment_predicted_room_id, 0.0)
        result.update(refinement)
        result["alignment_applied"] = True
        result["spatial_predicted_room_id"] = result.get("predicted_room_id")
        select_effective_stage()
        spatial_timing = refinement.get("spatial_timing")
        if not isinstance(spatial_timing, dict):
            spatial_total_ms = (time.perf_counter() - spatial_started) * 1000.0
            spatial_timing = {
                "program_module_ms": spatial_total_ms,
                "vlm_response_ms": 0.0,
                "stage_total_ms": spatial_total_ms,
                "logical_vlm_call_count": 0,
            }
        return finalize(spatial_timing)

    @staticmethod
    def _empty_result() -> dict:
        return {
            "predicted_room_id": None,
            "confidence": 0.0,
            "room_belief": {},
            "transition_support": {},
            "observation_distribution": {},
            "observation_likelihood": {},
            "evidence_distribution": {},
            "raw_room_scores": {},
            "visual_room_scores": {},
            "visual_room_belief": {},
            "visual_predicted_room_id": None,
            "prior_room_belief": {},
            "bayesian_room_belief": {},
            "bayesian_predicted_room_id": None,
            "base_predicted_room_id": None,
            "base_room_belief": {},
            "alignment_candidate_room_ids": [],
            "alignment_top_k": [],
            "alignment_applied": False,
            "alignment_predicted_room_id": None,
            "spatial_predicted_room_id": None,
            "evidence": [],
        }

    @staticmethod
    def _predicted_room_id(room_belief: dict[str, float]) -> str | None:
        if not room_belief:
            return None
        predicted_room_id = max(room_belief, key=room_belief.get)
        if room_belief.get(predicted_room_id, 0.0) <= 0.0:
            return None
        return predicted_room_id

    @staticmethod
    def _visual_evidence(visual_localization: object) -> list[str]:
        if not isinstance(visual_localization, dict):
            return []
        raw_evidence = visual_localization.get("evidence_entities")
        if not isinstance(raw_evidence, list):
            raw_evidence = visual_localization.get("evidence")
        return [value for value in raw_evidence if isinstance(value, str) and value] if isinstance(raw_evidence, list) else []

    @staticmethod
    def _visual_summary(visual_localization: object) -> str:
        if not isinstance(visual_localization, dict):
            return ""
        summary = visual_localization.get("summary")
        return summary if isinstance(summary, str) else ""


class SpatialAlignmentRefiner:
    def __init__(
        self,
        *,
        room_graph: dict[str, dict],
        grounding_index: GroundingIndex,
        model: str | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
        api_kind: str | None = None,
        request_timeout: float | None = None,
        num_ctx: int | None = None,
        response_client: Callable[[dict], dict] | None = None,
        max_http_retries: int | None = None,
    ):
        model_env = resolve_model_environment(
            default_model="gpt-5-mini",
            default_api_base=DEFAULT_OPENAI_API_BASE,
            default_api_kind="responses",
        )
        self.room_graph = room_graph
        self.grounding_index = grounding_index
        self.model = model or model_env.model_name or "gpt-5-mini"
        self.api_key = api_key if api_key is not None else model_env.api_key
        self.api_base = (api_base or model_env.api_base or DEFAULT_OPENAI_API_BASE).rstrip("/")
        self.api_kind = resolve_api_kind(api_kind or model_env.api_kind)
        self.request_timeout = float(request_timeout if request_timeout is not None else (model_env.request_timeout or 180.0))
        self.num_ctx = resolve_task_num_ctx(
            "localization",
            explicit_num_ctx=num_ctx,
            fallback_num_ctx=model_env.num_ctx,
            default_num_ctx=16384,
        )
        self.model_client = ModelResponseClient(
            provider=model_env.provider,
            api_key=self.api_key,
            api_base=self.api_base,
            api_kind=self.api_kind,
            request_timeout=self.request_timeout,
            num_ctx=self.num_ctx,
            temperature=model_env.temperature,
            response_client=response_client,
            max_http_retries=max_http_retries,
        )
        self.last_extraction_request_body: dict | None = None
        self.last_extraction_response_payload: dict | None = None
        self.last_alignment_request_body: dict | None = None
        self.last_alignment_response_payload: dict | None = None
        self.last_ego_spatial_context: dict | None = None
        self.last_single_call_schema_mode: str | None = None

    def refine(self, *, observation: Observation, candidate_room_ids: list[str]) -> dict:
        stage_started = time.perf_counter()
        if len(candidate_room_ids) < 2:
            return {"applied": False, "alignment_skipped_reason": "insufficient_alignment_candidates"}
        if not self.model_client.is_configured():
            return {"applied": False, "alignment_skipped_reason": "missing_model_configuration"}
        ordered_views = self._ordered_views(observation)
        if not ordered_views:
            return {"applied": False, "alignment_skipped_reason": "missing_panorama_views"}

        extraction_request = self._build_context_extraction_request_body(
            ordered_views=ordered_views,
            candidate_room_ids=candidate_room_ids,
        )
        self.last_extraction_request_body = self._clone_json(extraction_request)
        extraction_started = time.perf_counter()
        extraction_payload = self._create_response(extraction_request)
        extraction_vlm_ms = (time.perf_counter() - extraction_started) * 1000.0
        self.last_extraction_response_payload = self._clone_json(extraction_payload)
        extracted = self._parse_output_payload(extraction_payload)
        ego_spatial_context = self._format_ego_spatial_context(extracted, ordered_views)
        self.last_ego_spatial_context = self._clone_json(ego_spatial_context)

        candidate_context_text = self._build_candidate_context_text(candidate_room_ids)
        alignment_request = self._build_alignment_text_request_body(
            candidate_room_ids=candidate_room_ids,
            candidate_context_text=candidate_context_text,
            ego_context_text=str(ego_spatial_context.get("text", "")),
            ordered_views=ordered_views,
        )
        self.last_alignment_request_body = self._clone_json(alignment_request)
        alignment_started = time.perf_counter()
        alignment_payload = self._create_response(alignment_request)
        alignment_vlm_ms = (time.perf_counter() - alignment_started) * 1000.0
        self.last_alignment_response_payload = self._clone_json(alignment_payload)
        parsed = self._parse_output_payload(alignment_payload)
        alignment_top_k = self._alignment_top_k(parsed, candidate_room_ids)
        alignment_predicted_room_id = alignment_top_k[0]["room_id"] if alignment_top_k else parsed.get("predicted_room_id")
        evidence = parsed.get("evidence")
        summary = parsed.get("summary")
        view_0_direction_raw = parsed.get("view_0_allocentric_direction")
        view_0_direction = (
            str(view_0_direction_raw).strip().lower()
            if isinstance(view_0_direction_raw, str)
            else None
        )
        direction_headings = {
            "north": 0.0,
            "east": 90.0,
            "south": 180.0,
            "west": 270.0,
        }
        view_0_heading = float(ordered_views[0]["heading"])
        offset_deg = (
            normalize_heading(direction_headings[view_0_direction] - view_0_heading)
            if view_0_direction in direction_headings
            else None
        )
        stage_total_ms = (time.perf_counter() - stage_started) * 1000.0
        total_vlm_ms = extraction_vlm_ms + alignment_vlm_ms
        spatial_timing = {
            "program_module_ms": max(0.0, stage_total_ms - total_vlm_ms),
            "context_extraction_vlm_ms": extraction_vlm_ms,
            "alignment_vlm_ms": alignment_vlm_ms,
            "vlm_response_ms": total_vlm_ms,
            "stage_total_ms": stage_total_ms,
            "logical_vlm_call_count": 2,
            "status": "applied" if offset_deg is not None else "invalid_offset",
        }
        return {
            "applied": True,
            "logical_vlm_call_count": 2,
            "alignment_top_k": alignment_top_k,
            "alignment_predicted_room_id": alignment_predicted_room_id,
            "alignment_confidence": parsed.get("confidence"),
            "alignment_evidence": [value for value in evidence if isinstance(value, str) and value][:3]
            if isinstance(evidence, list)
            else [],
            "alignment_summary": summary if isinstance(summary, str) else "",
            "spatial_timing": spatial_timing,
            "spatial_alignment": {
                "mode": "text_from_images",
                "candidate_context_text": candidate_context_text,
                "ego_context_text": str(ego_spatial_context.get("text", "")),
                "view_0_geocentric_heading_deg": view_0_heading,
                "view_0_allocentric_direction": view_0_direction,
                "geocentric_to_allocentric_offset_deg": offset_deg,
                "offset_failure_reason": (
                    None if offset_deg is not None else "invalid_view_0_allocentric_direction"
                ),
                "alignment_top_k": alignment_top_k,
                "alignment_predicted_room_id": alignment_predicted_room_id,
                "alignment_confidence": parsed.get("confidence"),
                "alignment_evidence": [value for value in evidence if isinstance(value, str) and value][:3]
                if isinstance(evidence, list)
                else [],
                "alignment_summary": summary if isinstance(summary, str) else "",
                "timing": spatial_timing,
            },
            "ego_spatial_context": ego_spatial_context,
            "alignment_request_body": self._clone_json(self.last_alignment_request_body),
            "alignment_response_payload": self._clone_json(self.last_alignment_response_payload),
        }

    def refine_single_call(self, *, observation: Observation, candidate_room_ids: list[str]) -> dict:
        candidate_room_ids = list(dict.fromkeys(candidate_room_ids))
        if len(candidate_room_ids) < 2:
            raise ValueError("Single-call spatial alignment requires at least two candidate rooms.")
        unknown_room_ids = [room_id for room_id in candidate_room_ids if room_id not in self.room_graph]
        if unknown_room_ids:
            raise ValueError(f"Unknown candidate rooms: {', '.join(unknown_room_ids)}")
        if not self.model_client.is_configured():
            raise RuntimeError("Single-call spatial alignment requires configured VLM access.")

        ordered_views = self._ordered_views(observation)
        if len(ordered_views) != 8:
            raise ValueError(
                f"Single-call spatial alignment requires exactly 8 panorama views; received {len(ordered_views)}."
            )
        missing_image_paths = [view["path"] for view in ordered_views if not Path(view["path"]).is_file()]
        if missing_image_paths:
            raise FileNotFoundError(f"Missing panorama image: {missing_image_paths[0]}")

        candidate_context_text = self._build_candidate_context_text(candidate_room_ids)
        context_room_ids = self._context_room_ids(candidate_room_ids)
        candidate_neighbor_directions = self._candidate_neighbor_directions(candidate_room_ids)
        request_body, prompt_text = self._build_single_call_request_body(
            candidate_room_ids=candidate_room_ids,
            context_room_ids=context_room_ids,
            candidate_context_text=candidate_context_text,
            ordered_views=ordered_views,
        )
        self.last_extraction_request_body = None
        self.last_extraction_response_payload = None
        self.last_ego_spatial_context = None
        self.last_alignment_request_body = self._clone_json(request_body)
        response_payload = self._create_response(request_body)
        self.last_alignment_response_payload = self._clone_json(response_payload)
        parsed = self._parse_output_payload(response_payload)
        self._validate_single_call_output(
            parsed,
            candidate_room_ids=candidate_room_ids,
            context_room_ids=context_room_ids,
            view_ids=[view["view_id"] for view in ordered_views],
            candidate_neighbor_directions=candidate_neighbor_directions,
        )

        alignment_top_k = self._alignment_top_k(parsed, candidate_room_ids)
        alignment_predicted_room_id = (
            alignment_top_k[0]["room_id"] if alignment_top_k else parsed.get("predicted_room_id")
        )
        view_0_direction = str(parsed["view_0_allocentric_direction"])
        direction_headings = {"north": 0.0, "east": 90.0, "south": 180.0, "west": 270.0}
        offset_deg = normalize_heading(direction_headings[view_0_direction] - float(ordered_views[0]["heading"]))

        aligned_views = []
        aligned_directions: dict[str, str] = {}
        for view in ordered_views:
            allocentric_heading = normalize_heading(float(view["heading"]) + offset_deg)
            allocentric_direction = self._heading_to_allocentric_direction(allocentric_heading)
            aligned_directions[view["view_id"]] = allocentric_direction
            aligned_views.append(
                {
                    "view_id": view["view_id"],
                    "geocentric_heading_deg": float(view["heading"]),
                    "allocentric_heading_deg": allocentric_heading,
                    "allocentric_direction": allocentric_direction,
                }
            )

        sector_alignment = list(parsed.get("sector_alignment", []))
        sector_alignment_checks = []
        for record in sector_alignment:
            view_id = str(record.get("view_id"))
            claimed_direction = record.get("allocentric_direction")
            expected_direction = aligned_directions.get(view_id)
            sector_alignment_checks.append(
                {
                    "view_id": view_id,
                    "claimed_allocentric_direction": claimed_direction,
                    "rotation_derived_allocentric_direction": expected_direction,
                    "consistent": claimed_direction == expected_direction,
                }
            )

        evidence = parsed.get("evidence")
        summary = parsed.get("summary")
        return {
            "applied": True,
            "logical_vlm_call_count": 1,
            "alignment_top_k": alignment_top_k,
            "alignment_predicted_room_id": alignment_predicted_room_id,
            "alignment_confidence": parsed.get("confidence"),
            "alignment_evidence": [value for value in evidence if isinstance(value, str) and value]
            if isinstance(evidence, list)
            else [],
            "alignment_summary": summary if isinstance(summary, str) else "",
            "view_observations": list(parsed.get("view_observations", [])),
            "candidate_hypotheses": list(parsed.get("candidate_hypotheses", [])),
            "structured_output_mode": self.last_single_call_schema_mode,
            "spatial_alignment": {
                "mode": "direct_images_single_call",
                "candidate_context_text": candidate_context_text,
                "view_0_allocentric_direction": view_0_direction,
                "geocentric_to_allocentric_offset_deg": offset_deg,
                "aligned_views": aligned_views,
                "sector_alignment": sector_alignment,
                "sector_alignment_checks": sector_alignment_checks,
                "sector_alignment_consistent": all(
                    record["consistent"] for record in sector_alignment_checks
                ),
            },
            "alignment_prompt_text": prompt_text,
            "alignment_response_payload": self._clone_json(response_payload),
        }

    def _ordered_views(self, observation: Observation) -> list[dict]:
        ordered = []
        for index, view in enumerate(observation.views):
            if not isinstance(view.path, str) or not view.path:
                continue
            ordered.append(
                {
                    "view_id": f"view_{index}",
                    "path": view.path,
                    "heading": float(view.heading),
                }
            )
        return ordered

    def _build_candidate_context_text(self, candidate_room_ids: list[str]) -> str:
        lines: list[str] = []
        for room_id in candidate_room_ids:
            node = self.room_graph.get(room_id, {})
            title = node.get("title") or "unknown"
            category = node.get("category") or "unknown"
            lines.append(f"Candidate room {room_id}: title={title}; category={category}.")
            visual_profile_fields = compact_visual_profile(node)
            short_description = visual_profile_fields.get("short_description")
            if isinstance(short_description, str) and short_description:
                lines.append(f"- short description: {short_description}")
            visual_cues = visual_profile_fields.get("visual_cues")
            if isinstance(visual_cues, list) and visual_cues:
                lines.append("- visual cues: " + ", ".join(str(value) for value in visual_cues))
            for neighbor in node.get("neighbors", []):
                if not isinstance(neighbor, dict):
                    continue
                neighbor_room_id = neighbor.get("target_room_id")
                if not isinstance(neighbor_room_id, str) or not neighbor_room_id:
                    continue
                direction = neighbor.get("allocentric_direction") or "unknown"
                neighbor_node = self.room_graph.get(neighbor_room_id, {})
                neighbor_title = neighbor_node.get("title") or neighbor_room_id
                neighbor_category = neighbor_node.get("category") or "unknown"
                neighbor_profile = compact_visual_profile(neighbor_node)
                lines.append(
                    f"- Neighbor {neighbor_room_id} is {direction} of {room_id}: "
                    f"title={neighbor_title}; category={neighbor_category}."
                )
                neighbor_description = neighbor_profile.get("short_description")
                if isinstance(neighbor_description, str) and neighbor_description:
                    lines.append(f"  - neighbor short description: {neighbor_description}")
                neighbor_cues = neighbor_profile.get("visual_cues")
                if isinstance(neighbor_cues, list) and neighbor_cues:
                    lines.append(
                        "  - neighbor visual cues: "
                        + ", ".join(str(value) for value in neighbor_cues)
                    )
        return "\n".join(lines)

    def _context_room_ids(self, candidate_room_ids: list[str]) -> list[str]:
        room_ids = list(candidate_room_ids)
        for room_id in candidate_room_ids:
            for neighbor in self.room_graph.get(room_id, {}).get("neighbors", []):
                if not isinstance(neighbor, dict):
                    continue
                neighbor_room_id = neighbor.get("target_room_id")
                if (
                    isinstance(neighbor_room_id, str)
                    and neighbor_room_id in self.room_graph
                    and neighbor_room_id not in room_ids
                ):
                    room_ids.append(neighbor_room_id)
        return room_ids

    def _candidate_neighbor_directions(
        self,
        candidate_room_ids: list[str],
    ) -> dict[str, dict[str, str]]:
        result: dict[str, dict[str, str]] = {}
        for room_id in candidate_room_ids:
            neighbors: dict[str, str] = {}
            for neighbor in self.room_graph.get(room_id, {}).get("neighbors", []):
                if not isinstance(neighbor, dict):
                    continue
                neighbor_room_id = neighbor.get("target_room_id")
                direction = neighbor.get("allocentric_direction")
                if (
                    isinstance(neighbor_room_id, str)
                    and neighbor_room_id in self.room_graph
                    and direction in ALLOCENTRIC_DIRECTIONS
                ):
                    neighbors[neighbor_room_id] = str(direction)
            result[room_id] = neighbors
        return result

    def _candidate_theme_labels(self, candidate_room_ids: list[str]) -> list[str]:
        labels: list[str] = []

        def append_label(value: object) -> None:
            if isinstance(value, str) and value and value not in labels:
                labels.append(value)

        def append_visual_profile_labels(node: dict) -> None:
            visual_profile_fields = compact_visual_profile(node)
            for key in ("short_description", "visual_cues"):
                raw_values = visual_profile_fields.get(key)
                if isinstance(raw_values, str):
                    raw_values = [raw_values]
                if isinstance(raw_values, list):
                    for value in raw_values:
                        append_label(value)

        for room_id in candidate_room_ids:
            node = self.room_graph.get(room_id, {})
            for key in ("title", "category"):
                append_label(node.get(key))
            append_visual_profile_labels(node)
            for neighbor in node.get("neighbors", []):
                if not isinstance(neighbor, dict):
                    continue
                neighbor_node = self.room_graph.get(str(neighbor.get("target_room_id")), {})
                for key in ("title", "category"):
                    append_label(neighbor_node.get(key))
        return labels

    def _build_context_extraction_request_body(
        self,
        *,
        ordered_views: list[dict],
        candidate_room_ids: list[str],
    ) -> dict:
        content: list[dict] = [
            {
                "type": "input_text",
                "text": build_spatial_context_extraction_input(
                    view_ids=[view["view_id"] for view in ordered_views],
                    candidate_theme_labels=self._candidate_theme_labels(candidate_room_ids),
                ),
            }
        ]
        for view in ordered_views:
            content.extend(
                [
                    {
                        "type": "input_text",
                        "text": f"Panorama sector {view['view_id']}. The panorama's global heading is unknown.",
                    },
                    {
                        "type": "input_image",
                        "image_url": self._image_to_data_url(Path(view["path"])),
                        "detail": "high",
                    },
                ]
            )
        return {
            "model": self.model,
            "instructions": build_spatial_context_extraction_instructions(),
            "input": [{"role": "user", "content": content}],
            "reasoning": {"effort": "low"},
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "spatial_context_extraction",
                    "strict": True,
                    "schema": build_spatial_context_extraction_schema([view["view_id"] for view in ordered_views]),
                }
            },
        }

    def _format_ego_spatial_context(self, parsed: dict, ordered_views: list[dict]) -> dict:
        raw_views = parsed.get("views")
        by_id = {}
        if isinstance(raw_views, list):
            for record in raw_views:
                if not isinstance(record, dict):
                    continue
                view_id = record.get("view_id")
                if isinstance(view_id, str) and view_id:
                    by_id[view_id] = record

        formatted_views = []
        lines = []
        for view in ordered_views:
            record = by_id.get(view["view_id"], {})
            raw_themes = record.get("themes")
            themes = []
            if isinstance(raw_themes, list):
                for theme in raw_themes[:3]:
                    if not isinstance(theme, dict):
                        continue
                    label = theme.get("label")
                    confidence = theme.get("confidence")
                    if isinstance(label, str) and label:
                        themes.append(
                            {
                                "label": label,
                                "confidence": float(confidence) if isinstance(confidence, (int, float)) else 0.0,
                            }
                        )
            formatted_views.append({"view_id": view["view_id"], "themes": themes})
            if themes:
                theme_text = ", ".join(f"{theme['label']} ({theme['confidence']:.2f})" for theme in themes)
            else:
                theme_text = "(no reliable theme)"
            lines.append(f"- {view['view_id']}: {theme_text}")

        summary = parsed.get("summary")
        return {
            "views": formatted_views,
            "summary": summary if isinstance(summary, str) else "",
            "text": "\n".join(lines),
        }

    def _build_single_call_request_body(
        self,
        *,
        candidate_room_ids: list[str],
        context_room_ids: list[str],
        candidate_context_text: str,
        ordered_views: list[dict],
    ) -> tuple[dict, str]:
        view_ids = [view["view_id"] for view in ordered_views]
        prompt_text = build_spatial_alignment_input(
            candidate_context_text=candidate_context_text,
            ego_context_text="Inspect the eight attached panorama sector images directly.",
            view_ids=view_ids,
            direct_images=True,
            view_headings=[
                {"view_id": view["view_id"], "heading_deg": float(view["heading"])}
                for view in ordered_views
            ],
        )
        content: list[dict] = [{"type": "input_text", "text": prompt_text}]
        for view in ordered_views:
            content.extend(
                [
                    {
                        "type": "input_text",
                        "text": (
                            f"Panorama sector {view['view_id']}; "
                            f"geocentric heading={float(view['heading']):.6f} degrees."
                        ),
                    },
                    {
                        "type": "input_image",
                        "image_url": self._image_to_data_url(Path(view["path"])),
                        "detail": "high",
                    },
                ]
            )
        full_response_schema = build_spatial_alignment_schema(
            candidate_room_ids,
            view_ids=view_ids,
            include_sector_alignment=True,
            include_view_observations=True,
            include_candidate_hypotheses=True,
            context_room_ids=context_room_ids,
        )
        if (
            self.model_client.provider in {"gemini", "gemini_api", "google_gemma_api"}
            and self.model_client.response_client is None
        ):
            response_schema = self._relax_json_schema(full_response_schema)
            self.last_single_call_schema_mode = "gemini_relaxed_schema_with_local_validation"
        else:
            response_schema = full_response_schema
            self.last_single_call_schema_mode = "full_json_schema"
        return {
            "model": self.model,
            "instructions": build_spatial_alignment_instructions(direct_images=True),
            "input": [{"role": "user", "content": content}],
            "reasoning": {"effort": "low"},
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "single_call_spatial_alignment",
                    "strict": True,
                    "schema": response_schema,
                }
            },
        }, prompt_text

    def _build_alignment_text_request_body(
        self,
        *,
        candidate_room_ids: list[str],
        candidate_context_text: str,
        ego_context_text: str,
        ordered_views: list[dict],
    ) -> dict:
        return {
            "model": self.model,
            "instructions": build_spatial_alignment_instructions(),
            "input": build_spatial_alignment_input(
                candidate_context_text=candidate_context_text,
                ego_context_text=ego_context_text,
                view_ids=[view["view_id"] for view in ordered_views],
            ),
            "reasoning": {"effort": "low"},
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "spatial_alignment",
                    "strict": True,
                    "schema": build_spatial_alignment_schema(candidate_room_ids),
                }
            },
        }

    @staticmethod
    def _relax_json_schema(value: object) -> object:
        if isinstance(value, list):
            return [SpatialAlignmentRefiner._relax_json_schema(item) for item in value]
        if not isinstance(value, dict):
            return value
        ignored_keys = {"enum", "minItems", "maxItems", "additionalProperties"}
        return {
            key: SpatialAlignmentRefiner._relax_json_schema(item)
            for key, item in value.items()
            if key not in ignored_keys
        }

    @staticmethod
    def _validate_single_call_output(
        parsed: dict,
        *,
        candidate_room_ids: list[str],
        context_room_ids: list[str],
        view_ids: list[str],
        candidate_neighbor_directions: dict[str, dict[str, str]],
    ) -> None:
        predicted_room_id = parsed.get("predicted_room_id")
        if predicted_room_id not in candidate_room_ids:
            raise ValueError(f"VLM returned an invalid predicted room: {predicted_room_id}")
        if not isinstance(parsed.get("confidence"), (int, float)):
            raise ValueError("VLM confidence must be numeric.")
        evidence = parsed.get("evidence")
        if not isinstance(evidence, list) or any(not isinstance(value, str) for value in evidence):
            raise ValueError("VLM evidence must be an array of strings.")
        if not isinstance(parsed.get("summary"), str):
            raise ValueError("VLM summary must be a string.")
        view_0_direction = parsed.get("view_0_allocentric_direction")
        if view_0_direction not in ALLOCENTRIC_DIRECTIONS:
            raise ValueError("VLM did not return a valid allocentric direction for view_0.")

        raw_ranking = parsed.get("room_ranking")
        ranked_room_ids = []
        if isinstance(raw_ranking, list):
            for record in raw_ranking:
                if not isinstance(record, dict) or not isinstance(record.get("score"), (int, float)):
                    raise ValueError("Every VLM room_ranking item must contain room_id and numeric score.")
                ranked_room_ids.append(record.get("room_id"))
        if len(ranked_room_ids) != len(candidate_room_ids) or set(ranked_room_ids) != set(candidate_room_ids):
            raise ValueError("VLM room_ranking must contain every candidate room exactly once.")

        raw_hypotheses = parsed.get("candidate_hypotheses")
        hypothesis_room_ids = []
        if isinstance(raw_hypotheses, list):
            for record in raw_hypotheses:
                if not isinstance(record, dict) or not isinstance(record.get("score"), (int, float)):
                    raise ValueError(
                        "Every VLM candidate_hypotheses item must contain room_id and numeric score."
                    )
                hypothesis_room_ids.append(record.get("room_id"))
                for field_name in (
                    "current_room_support",
                    "neighbor_support",
                    "contradictions",
                ):
                    values = record.get(field_name)
                    if not isinstance(values, list) or any(
                        not isinstance(value, str) for value in values
                    ):
                        raise ValueError(
                            f"VLM candidate_hypotheses {field_name} must be an array of strings."
                        )
                room_id = str(record.get("room_id"))
                expected_neighbors = candidate_neighbor_directions.get(room_id, {})
                directional_checks = record.get("directional_neighbor_checks")
                checked_neighbor_ids: list[str] = []
                if not isinstance(directional_checks, list):
                    raise ValueError(
                        "VLM candidate_hypotheses directional_neighbor_checks must be an array."
                    )
                for check in directional_checks:
                    if not isinstance(check, dict):
                        raise ValueError(
                            "Every directional_neighbor_checks item must be an object."
                        )
                    neighbor_room_id = check.get("expected_neighbor_room_id")
                    checked_neighbor_ids.append(str(neighbor_room_id))
                    if neighbor_room_id not in expected_neighbors:
                        raise ValueError(
                            f"VLM directional_neighbor_checks used a non-neighbor for {room_id}: "
                            f"{neighbor_room_id}"
                        )
                    if check.get("allocentric_direction") != expected_neighbors[neighbor_room_id]:
                        raise ValueError(
                            f"VLM directional_neighbor_checks used the wrong direction for "
                            f"{room_id} -> {neighbor_room_id}."
                        )
                    inspected_view_ids = check.get("inspected_view_ids")
                    if (
                        not isinstance(inspected_view_ids, list)
                        or not inspected_view_ids
                        or any(view_id not in view_ids for view_id in inspected_view_ids)
                    ):
                        raise ValueError(
                            "VLM directional_neighbor_checks must inspect at least one valid view."
                        )
                    if check.get("status") not in {
                        "specific_match",
                        "broad_match",
                        "different_room",
                        "not_visible",
                        "uncertain",
                    }:
                        raise ValueError(
                            "VLM directional_neighbor_checks returned an invalid status."
                        )
                    if check.get("observed_room_hint") not in set(context_room_ids) | {None}:
                        raise ValueError(
                            "VLM directional_neighbor_checks returned an invalid room hint."
                        )
                    if not isinstance(check.get("rationale"), str):
                        raise ValueError(
                            "VLM directional_neighbor_checks rationale must be a string."
                        )
                if (
                    len(checked_neighbor_ids) != len(expected_neighbors)
                    or set(checked_neighbor_ids) != set(expected_neighbors)
                ):
                    raise ValueError(
                        f"VLM directional_neighbor_checks must cover every listed neighbor of {room_id}."
                    )
        if len(hypothesis_room_ids) != len(candidate_room_ids) or set(hypothesis_room_ids) != set(candidate_room_ids):
            raise ValueError("VLM candidate_hypotheses must contain every candidate room exactly once.")

        raw_observations = parsed.get("view_observations")
        observed_view_ids = []
        if isinstance(raw_observations, list):
            for record in raw_observations:
                if not isinstance(record, dict):
                    raise ValueError("Every VLM view_observations item must be an object.")
                observed_view_ids.append(record.get("view_id"))
                themes = record.get("themes")
                if not isinstance(themes, list) or any(not isinstance(value, str) for value in themes):
                    raise ValueError("Every VLM view_observations themes field must be an array of strings.")
                if record.get("location_scope") not in {
                    "current_room",
                    "adjacent_room",
                    "mixed",
                    "uncertain",
                }:
                    raise ValueError("VLM view_observations returned an invalid location_scope.")
                if record.get("room_hint") not in set(context_room_ids) | {None}:
                    raise ValueError("VLM view_observations returned an invalid room_hint.")
                if not isinstance(record.get("summary"), str):
                    raise ValueError("Every VLM view_observations summary must be a string.")
        if len(observed_view_ids) != len(view_ids) or set(observed_view_ids) != set(view_ids):
            raise ValueError("VLM view_observations must contain every panorama view exactly once.")

        sector_alignment = parsed.get("sector_alignment")
        if not isinstance(sector_alignment, list) or len(sector_alignment) < 2:
            raise ValueError("VLM sector_alignment must contain at least two sectors.")
        for record in sector_alignment:
            if not isinstance(record, dict):
                raise ValueError("Every VLM sector_alignment item must be an object.")
            if record.get("view_id") not in view_ids:
                raise ValueError("VLM sector_alignment referenced an unknown panorama view.")
            if record.get("allocentric_direction") not in ALLOCENTRIC_DIRECTIONS:
                raise ValueError("VLM sector_alignment returned an invalid allocentric direction.")
            if record.get("matched_room_id") not in set(context_room_ids) | {None}:
                raise ValueError("VLM sector_alignment returned an invalid matched_room_id.")
            if not isinstance(record.get("matched_theme"), str):
                raise ValueError("Every VLM sector_alignment matched_theme must be a string.")
            if not isinstance(record.get("rationale"), str):
                raise ValueError("Every VLM sector_alignment rationale must be a string.")

    @staticmethod
    def _heading_to_allocentric_direction(heading: float) -> str:
        candidates = (("north", 0.0), ("east", 90.0), ("south", 180.0), ("west", 270.0))
        normalized = normalize_heading(heading)
        return min(
            candidates,
            key=lambda item: abs(((normalized - item[1] + 180.0) % 360.0) - 180.0),
        )[0]

    @staticmethod
    def _alignment_top_k(parsed: dict, candidate_room_ids: list[str]) -> list[dict]:
        candidate_set = set(candidate_room_ids)
        seen: set[str] = set()
        ranked: list[dict] = []
        raw_ranking = parsed.get("room_ranking")
        if isinstance(raw_ranking, list):
            for record in raw_ranking:
                if not isinstance(record, dict):
                    continue
                room_id = record.get("room_id")
                score = record.get("score")
                if room_id not in candidate_set or room_id in seen:
                    continue
                ranked.append(
                    {
                        "room_id": room_id,
                        "score": float(score) if isinstance(score, (int, float)) else 0.0,
                    }
                )
                seen.add(room_id)
        predicted_room_id = parsed.get("predicted_room_id")
        if isinstance(predicted_room_id, str) and predicted_room_id in candidate_set and predicted_room_id not in seen:
            ranked.insert(0, {"room_id": predicted_room_id, "score": float(parsed.get("confidence", 1.0) or 1.0)})
        elif isinstance(predicted_room_id, str) and ranked and ranked[0]["room_id"] != predicted_room_id:
            ranked = sorted(ranked, key=lambda item: (item["room_id"] != predicted_room_id,))
        return ranked

    def _create_response(self, request_body: dict) -> dict:
        return self.model_client.create(request_body)

    @staticmethod
    def _parse_output_payload(payload: dict) -> dict:
        return parse_json_output(payload)

    @staticmethod
    def _clone_json(payload: dict | None) -> dict | None:
        if payload is None:
            return None
        return json.loads(json.dumps(payload))

    @staticmethod
    def _image_to_data_url(image_path: Path) -> str:
        mime_type, _ = mimetypes.guess_type(str(image_path))
        if not mime_type:
            mime_type = "image/png"
        return f"data:{mime_type};base64,{b64encode(image_path.read_bytes()).decode('ascii')}"
