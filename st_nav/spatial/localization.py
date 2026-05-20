from __future__ import annotations

import json
import mimetypes
from base64 import b64encode
from pathlib import Path
from typing import Callable

from ..common.env import resolve_model_environment
from ..common.model_client import DEFAULT_OPENAI_API_BASE, ModelResponseClient, parse_json_output, resolve_api_kind
from ..common.prompts import (
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
    ):
        self.room_graph = room_graph
        self.grounding_index = grounding_index
        self.same_floor_only = same_floor_only
        self.self_transition_weight = float(self_transition_weight)
        self.neighbor_transition_weight = float(neighbor_transition_weight)
        self.alignment_candidate_ratio_threshold = float(alignment_candidate_ratio_threshold)
        self.alignment_candidate_max = int(alignment_candidate_max)
        self.spatial_refiner = spatial_refiner

    def localize(
        self,
        *,
        observation: Observation,
        prior_room_belief: dict[str, float] | None,
        fallback_room_id: str | None = None,
    ) -> dict:
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
            "base_predicted_room_id": base_predicted_room_id,
            "base_room_belief": dict(base_room_belief),
            "alignment_candidate_room_ids": [],
            "alignment_top_k": [],
            "alignment_applied": False,
            "evidence": evidence[:3],
            "summary": summary,
            "visual_localization": dict(visual_localization) if isinstance(visual_localization, dict) else {},
        }

        alignment_candidate_ids = _alignment_candidate_room_ids(
            base_room_belief,
            ratio_threshold=self.alignment_candidate_ratio_threshold,
            max_candidates=self.alignment_candidate_max,
        )
        result["alignment_candidate_room_ids"] = list(alignment_candidate_ids)
        if len(alignment_candidate_ids) < 2:
            result["alignment_skipped_reason"] = "insufficient_alignment_candidates"
            return result
        if self.spatial_refiner is None:
            result["alignment_skipped_reason"] = "missing_spatial_refiner"
            return result

        refinement = self.spatial_refiner.refine(
            observation=observation,
            candidate_room_ids=alignment_candidate_ids,
        )
        if not refinement.get("applied"):
            result.update(refinement)
            return result

        alignment_predicted_room_id = refinement.get("alignment_predicted_room_id")
        if isinstance(alignment_predicted_room_id, str) and alignment_predicted_room_id in base_room_belief:
            result["predicted_room_id"] = alignment_predicted_room_id
            result["confidence"] = base_room_belief.get(alignment_predicted_room_id, 0.0)
        result.update(refinement)
        result["alignment_applied"] = True
        return result

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
            "alignment_candidate_room_ids": [],
            "alignment_top_k": [],
            "alignment_applied": False,
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
        response_client: Callable[[dict], dict] | None = None,
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
        self.model_client = ModelResponseClient(
            provider=model_env.provider,
            api_key=self.api_key,
            api_base=self.api_base,
            api_kind=self.api_kind,
            request_timeout=self.request_timeout,
            num_ctx=model_env.num_ctx,
            temperature=model_env.temperature,
            response_client=response_client,
        )
        self.last_extraction_request_body: dict | None = None
        self.last_extraction_response_payload: dict | None = None
        self.last_alignment_request_body: dict | None = None
        self.last_alignment_response_payload: dict | None = None
        self.last_ego_spatial_context: dict | None = None

    def refine(self, *, observation: Observation, candidate_room_ids: list[str]) -> dict:
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
        extraction_payload = self._create_response(extraction_request)
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
        alignment_payload = self._create_response(alignment_request)
        self.last_alignment_response_payload = self._clone_json(alignment_payload)
        parsed = self._parse_output_payload(alignment_payload)
        alignment_top_k = self._alignment_top_k(parsed, candidate_room_ids)
        alignment_predicted_room_id = alignment_top_k[0]["room_id"] if alignment_top_k else parsed.get("predicted_room_id")
        evidence = parsed.get("evidence")
        summary = parsed.get("summary")
        return {
            "applied": True,
            "alignment_top_k": alignment_top_k,
            "alignment_predicted_room_id": alignment_predicted_room_id,
            "alignment_confidence": parsed.get("confidence"),
            "alignment_evidence": [value for value in evidence if isinstance(value, str) and value][:3]
            if isinstance(evidence, list)
            else [],
            "alignment_summary": summary if isinstance(summary, str) else "",
            "spatial_alignment": {
                "mode": "text_from_images",
                "candidate_context_text": candidate_context_text,
                "ego_context_text": str(ego_spatial_context.get("text", "")),
                "view_0_allocentric_direction": parsed.get("view_0_allocentric_direction"),
                "alignment_top_k": alignment_top_k,
                "alignment_predicted_room_id": alignment_predicted_room_id,
                "alignment_confidence": parsed.get("confidence"),
                "alignment_evidence": [value for value in evidence if isinstance(value, str) and value][:3]
                if isinstance(evidence, list)
                else [],
                "alignment_summary": summary if isinstance(summary, str) else "",
            },
            "ego_spatial_context": ego_spatial_context,
            "alignment_request_body": self._clone_json(self.last_alignment_request_body),
            "alignment_response_payload": self._clone_json(self.last_alignment_response_payload),
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
                lines.append(f"- {neighbor_room_id} ({neighbor_title}) is {direction} of {room_id}.")
        return "\n".join(lines)

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
