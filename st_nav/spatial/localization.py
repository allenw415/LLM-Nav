from __future__ import annotations

import json
import math
import mimetypes
import re
from base64 import b64encode
from difflib import SequenceMatcher
from pathlib import Path
from typing import Callable

from ..common.env import resolve_model_environment
from ..common.model_client import DEFAULT_OPENAI_API_BASE, ModelResponseClient, parse_json_output, resolve_api_kind
from ..common.prompts import (
    build_localization_input,
    build_localization_instructions,
    build_localization_schema,
    build_spatial_alignment_input,
    build_spatial_alignment_instructions,
    build_spatial_alignment_schema,
    build_spatial_context_extraction_input,
    build_spatial_context_extraction_instructions,
    build_spatial_context_extraction_schema,
)
from ..common.types import EntityDetection, Observation
from .grounding import GroundingIndex

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
TOKEN_NORMALIZATION = {
    "greece": "greek",
    "greek": "greek",
    "rome": "roman",
    "roman": "roman",
    "romans": "roman",
    "assyria": "assyria",
    "assyrian": "assyria",
    "sculptures": "sculpture",
    "statues": "statue",
    "reliefs": "relief",
}
STOPWORDS = {
    "a",
    "an",
    "and",
    "ancient",
    "art",
    "artwork",
    "at",
    "bc",
    "by",
    "for",
    "from",
    "gallery",
    "in",
    "museum",
    "of",
    "room",
    "the",
    "to",
    "with",
}


def _clamp_probability(value: float, *, floor: float = 1e-6, ceiling: float = 1.0) -> float:
    return max(floor, min(ceiling, float(value)))


def _normalize_token(token: str) -> str:
    normalized = TOKEN_NORMALIZATION.get(token, token)
    if normalized.endswith("ies") and len(normalized) > 4:
        return normalized[:-3] + "y"
    if normalized.endswith("es") and len(normalized) > 4:
        return normalized[:-2]
    if normalized.endswith("s") and len(normalized) > 3:
        return normalized[:-1]
    return normalized


def _tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for token in TOKEN_PATTERN.findall(text.lower()):
        normalized = _normalize_token(token)
        if normalized and normalized not in STOPWORDS:
            tokens.append(normalized)
    return tokens


def _normalize_text(text: str) -> str:
    return " ".join(_tokenize(text))


class RoomLocalizer:
    def __init__(
        self,
        *,
        room_graph: dict[str, dict],
        grounding_index: GroundingIndex,
        same_floor_only: bool = True,
        self_transition_weight: float = 1.0,
        neighbor_transition_weight: float = 1.0,
        min_entity_likelihood: float = 0.05,
    ):
        self.room_graph = room_graph
        self.grounding_index = grounding_index
        self.same_floor_only = same_floor_only
        self.self_transition_weight = self_transition_weight
        self.neighbor_transition_weight = neighbor_transition_weight
        self.min_entity_likelihood = min_entity_likelihood
        self._room_signatures = {
            room_id: self._build_room_signature(room_id)
            for room_id in sorted(room_graph.keys())
        }

    def localize(
        self,
        *,
        observation: Observation,
        prior_room_belief: dict[str, float] | None,
        fallback_room_id: str | None = None,
    ) -> dict:
        candidate_room_ids = self._candidate_room_ids(observation)
        entity_observation = self._observation_with_inside_entities(observation)
        if not candidate_room_ids:
            return {
                "predicted_room_id": None,
                "confidence": 0.0,
                "room_belief": {},
                "transition_support": {},
                "evidence": [],
            }

        if not entity_observation.entities:
            stable_belief = self._stable_room_belief(
                prior_room_belief or {},
                candidate_room_ids=candidate_room_ids,
                fallback_room_id=fallback_room_id,
            )
            predicted_room_id = max(stable_belief, key=stable_belief.get) if stable_belief else None
            return {
                "predicted_room_id": predicted_room_id,
                "confidence": stable_belief.get(predicted_room_id, 0.0) if predicted_room_id else 0.0,
                "room_belief": stable_belief,
                "transition_support": stable_belief,
                "evidence": [],
            }

        transition_support = self._build_transition_support(
            prior_room_belief or {},
            candidate_room_ids=candidate_room_ids,
            fallback_room_id=fallback_room_id,
        )
        entity_scores_by_room: dict[str, list[tuple[str, float]]] = {}
        log_scores: dict[str, float] = {}

        for room_id in candidate_room_ids:
            transition_probability = float(transition_support.get(room_id, 0.0))
            if transition_probability <= 0.0:
                entity_scores_by_room[room_id] = []
                log_scores[room_id] = float("-inf")
                continue

            log_score = math.log(transition_probability)
            entity_scores = []
            for entity in entity_observation.entities:
                likelihood, match_score = self._entity_likelihood(room_id, entity)
                log_score += math.log(likelihood)
                entity_scores.append((entity.name, match_score))
            entity_scores_by_room[room_id] = sorted(entity_scores, key=lambda item: (-item[1], item[0].lower()))
            log_scores[room_id] = log_score

        posterior = self._normalize_log_scores(log_scores)
        predicted_room_id = max(posterior, key=posterior.get) if posterior else None
        evidence = self._evidence_from_entity_scores(predicted_room_id, entity_scores_by_room)

        return {
            "predicted_room_id": predicted_room_id,
            "confidence": posterior.get(predicted_room_id, 0.0) if predicted_room_id else 0.0,
            "room_belief": posterior,
            "transition_support": transition_support,
            "evidence": evidence,
        }

    @staticmethod
    def _inside_entities(observation: Observation) -> list[EntityDetection]:
        inside_entities = []
        for entity in observation.entities:
            scope = getattr(entity, "location_scope", None)
            if not isinstance(scope, str) or not scope:
                scope = entity.metadata.get("location_scope", "inside")
            if scope == "inside":
                inside_entities.append(entity)
        return inside_entities

    @staticmethod
    def _observation_with_inside_entities(observation: Observation) -> Observation:
        inside_entities = RoomLocalizer._inside_entities(observation)
        if len(inside_entities) == len(observation.entities):
            return observation
        return Observation(
            pano_id=observation.pano_id,
            views=list(observation.views),
            entities=inside_entities,
            heading_estimate=observation.heading_estimate,
            metadata=dict(observation.metadata),
        )

    def _observation_scores_from_entities(
        self,
        observation: Observation,
        candidate_room_ids: list[str],
    ) -> tuple[dict[str, float], dict[str, list[tuple[str, float]]]]:
        observation = self._observation_with_inside_entities(observation)
        if not observation.entities:
            uniform = 1.0 / len(candidate_room_ids) if candidate_room_ids else 0.0
            return (
                {room_id: uniform for room_id in candidate_room_ids},
                {room_id: [] for room_id in candidate_room_ids},
            )

        entity_scores_by_room: dict[str, list[tuple[str, float]]] = {}
        log_scores: dict[str, float] = {}
        for room_id in candidate_room_ids:
            entity_scores = []
            log_score = 0.0
            for entity in observation.entities:
                likelihood, match_score = self._entity_likelihood(room_id, entity)
                log_score += math.log(likelihood)
                entity_scores.append((entity.name, match_score))
            entity_scores_by_room[room_id] = sorted(entity_scores, key=lambda item: (-item[1], item[0].lower()))
            log_scores[room_id] = log_score
        return self._normalize_log_scores(log_scores), entity_scores_by_room

    @staticmethod
    def _evidence_from_entity_scores(
        predicted_room_id: str | None,
        entity_scores_by_room: dict[str, list[tuple[str, float]]],
    ) -> list[str]:
        if not predicted_room_id:
            return []
        evidence = []
        for entity_name, match_score in entity_scores_by_room.get(predicted_room_id, []):
            if match_score < 0.35:
                continue
            evidence.append(entity_name)
            if len(evidence) >= 3:
                break
        return evidence

    def _candidate_room_ids(self, observation: Observation) -> list[str]:
        if not self.same_floor_only:
            return sorted(self.room_graph.keys())
        floor = observation.metadata.get("floor")
        if floor is None:
            return sorted(self.room_graph.keys())
        floor_text = str(floor)
        return [
            room_id
            for room_id in sorted(self.room_graph.keys())
            if str(self.room_graph.get(room_id, {}).get("floor")) == floor_text
        ]

    def _build_transition_support(
        self,
        prior_room_belief: dict[str, float],
        *,
        candidate_room_ids: list[str],
        fallback_room_id: str | None,
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
            uniform = 1.0 / len(candidate_room_ids)
            return {room_id: uniform for room_id in candidate_room_ids}

        support = {room_id: 0.0 for room_id in candidate_room_ids}
        for source_room_id, source_probability in filtered_prior.items():
            targets = [source_room_id]
            for neighbor in self.room_graph.get(source_room_id, {}).get("neighbors", []):
                target_room_id = neighbor.get("target_room_id")
                if target_room_id in candidate_set and target_room_id not in targets:
                    targets.append(target_room_id)

            target_weights = {
                target_room_id: (
                    self.self_transition_weight
                    if target_room_id == source_room_id
                    else self.neighbor_transition_weight
                )
                for target_room_id in targets
            }
            total_weight = sum(weight for weight in target_weights.values() if weight > 0.0)
            if total_weight <= 0.0:
                support[source_room_id] += source_probability
                continue

            for target_room_id, target_weight in target_weights.items():
                if target_weight <= 0.0:
                    continue
                support[target_room_id] += source_probability * (target_weight / total_weight)

        return support

    def _stable_room_belief(
        self,
        prior_room_belief: dict[str, float],
        *,
        candidate_room_ids: list[str],
        fallback_room_id: str | None,
    ) -> dict[str, float]:
        candidate_set = set(candidate_room_ids)
        filtered_prior = {
            room_id: float(probability)
            for room_id, probability in prior_room_belief.items()
            if room_id in candidate_set and isinstance(probability, (int, float)) and probability > 0.0
        }
        if filtered_prior:
            return self._normalize_scores(filtered_prior)
        if fallback_room_id in candidate_set:
            return {room_id: 1.0 if room_id == fallback_room_id else 0.0 for room_id in candidate_room_ids}
        uniform = 1.0 / len(candidate_room_ids)
        return {room_id: uniform for room_id in candidate_room_ids}

    def _entity_likelihood(self, room_id: str, entity: EntityDetection) -> tuple[float, float]:
        room_signature = self._room_signatures.get(room_id, ())
        room_token_sets = [set(_tokenize(signature_text)) for signature_text in room_signature]
        direct_room_id = entity.metadata.get("predicted_room_id")
        if isinstance(direct_room_id, str) and direct_room_id == room_id:
            return 0.99, 1.0

        entity_confidence = _clamp_probability(entity.confidence, floor=0.0)
        match_score = 0.0
        for signature_text, room_tokens in zip(room_signature, room_token_sets):
            match_score = max(match_score, self._phrase_match(entity.name, signature_text, room_tokens))

        effective_confidence = min(1.0, entity_confidence)
        blended_score = (1.0 - effective_confidence) * 0.5 + effective_confidence * match_score
        likelihood = self.min_entity_likelihood + (1.0 - self.min_entity_likelihood) * blended_score
        return _clamp_probability(likelihood), match_score

    def _build_room_signature(self, room_id: str) -> tuple[str, ...]:
        node = self.room_graph.get(room_id, {})
        entry = self.grounding_index.room_entry(room_id) or {}
        values: list[str] = [room_id]
        for key in ("display_name", "title", "category"):
            value = node.get(key)
            if isinstance(value, str) and value:
                values.append(value)
        for key in ("aliases", "anchor_entities"):
            raw_values = node.get(key) if key == "aliases" else entry.get(key)
            if isinstance(raw_values, list):
                for value in raw_values:
                    if isinstance(value, str) and value:
                        values.append(value)
        deduped: list[str] = []
        seen: set[str] = set()
        for value in values:
            normalized = _normalize_text(value)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            deduped.append(value)
        return tuple(deduped)

    @staticmethod
    def _phrase_match(entity_name: str, signature_text: str, room_tokens: set[str]) -> float:
        normalized_entity = _normalize_text(entity_name)
        normalized_signature = _normalize_text(signature_text)
        if not normalized_entity or not normalized_signature:
            return 0.0
        if normalized_entity == normalized_signature:
            return 1.0
        if normalized_entity in normalized_signature or normalized_signature in normalized_entity:
            return 0.95

        entity_tokens = _tokenize(entity_name)
        if not entity_tokens or not room_tokens:
            return 0.0

        matched_count = 0
        unmatched_room_tokens = set(room_tokens)
        for entity_token in entity_tokens:
            matched_token = None
            for room_token in unmatched_room_tokens:
                if entity_token == room_token:
                    matched_token = room_token
                    break
                if entity_token.startswith(room_token) or room_token.startswith(entity_token):
                    matched_token = room_token
                    break
                if SequenceMatcher(None, entity_token, room_token).ratio() >= 0.8:
                    matched_token = room_token
                    break
            if matched_token is not None:
                matched_count += 1
                unmatched_room_tokens.discard(matched_token)

        token_precision = matched_count / len(entity_tokens)
        sequence_score = SequenceMatcher(None, normalized_entity, normalized_signature).ratio()
        return max(token_precision, sequence_score * 0.6)

    @staticmethod
    def _normalize_scores(scores: dict[str, float]) -> dict[str, float]:
        total = sum(value for value in scores.values() if value > 0.0)
        if total <= 0.0:
            return {key: 0.0 for key in scores}
        return {key: value / total for key, value in scores.items()}

    @staticmethod
    def _normalize_log_scores(log_scores: dict[str, float]) -> dict[str, float]:
        if not log_scores:
            return {}
        finite_scores = [log_score for log_score in log_scores.values() if math.isfinite(log_score)]
        if not finite_scores:
            return {room_id: 0.0 for room_id in log_scores}
        max_log_score = max(finite_scores)
        weights = {
            room_id: math.exp(log_score - max_log_score) if math.isfinite(log_score) else 0.0
            for room_id, log_score in log_scores.items()
        }
        return RoomLocalizer._normalize_scores(weights)


class VisualObservationLocalizer(RoomLocalizer):
    def localize(
        self,
        *,
        observation: Observation,
        prior_room_belief: dict[str, float] | None,
        fallback_room_id: str | None = None,
    ) -> dict:
        candidate_room_ids = self._candidate_room_ids(observation)
        entity_observation = self._observation_with_inside_entities(observation)
        if not candidate_room_ids:
            return {
                "predicted_room_id": None,
                "confidence": 0.0,
                "room_belief": {},
                "transition_support": {},
                "observation_distribution": {},
                "observation_likelihood": {},
                "evidence": [],
            }

        visual_localization = observation.metadata.get("visual_localization")
        visual_observation_likelihood = self._visual_observation_likelihood(
            visual_localization,
            candidate_room_ids=candidate_room_ids,
        )
        if not any(value > 0.0 for value in visual_observation_likelihood.values()):
            entity_observation = self._observation_with_inside_entities(observation)
            fallback = super().localize(
                observation=entity_observation,
                prior_room_belief=prior_room_belief,
                fallback_room_id=fallback_room_id,
            )
            entity_distribution, _ = self._observation_scores_from_entities(entity_observation, candidate_room_ids)
            fallback.setdefault("observation_distribution", entity_distribution)
            fallback.setdefault("observation_likelihood", entity_distribution)
            fallback.setdefault("entity_observation_distribution", entity_distribution)
            return fallback

        transition_support = self._build_transition_support(
            prior_room_belief or {},
            candidate_room_ids=candidate_room_ids,
            fallback_room_id=fallback_room_id,
        )
        posterior_scores = {
            room_id: float(transition_support.get(room_id, 0.0))
            * float(visual_observation_likelihood.get(room_id, 0.0))
            for room_id in candidate_room_ids
        }
        posterior = self._normalize_scores(posterior_scores)
        predicted_room_id = max(posterior, key=posterior.get) if posterior else None
        if predicted_room_id and posterior.get(predicted_room_id, 0.0) <= 0.0:
            predicted_room_id = None

        evidence = []
        summary = ""
        if isinstance(visual_localization, dict):
            raw_evidence = visual_localization.get("evidence_entities")
            if isinstance(raw_evidence, list):
                evidence = [value for value in raw_evidence if isinstance(value, str) and value]
            raw_summary = visual_localization.get("summary")
            summary = raw_summary if isinstance(raw_summary, str) else ""

        return {
            "predicted_room_id": predicted_room_id,
            "confidence": posterior.get(predicted_room_id, 0.0) if predicted_room_id else 0.0,
            "room_belief": posterior,
            "transition_support": transition_support,
            "observation_distribution": visual_observation_likelihood,
            "observation_likelihood": visual_observation_likelihood,
            "entity_observation_distribution": visual_observation_likelihood,
            "evidence": evidence[:3],
            "summary": summary,
            "visual_localization": dict(visual_localization) if isinstance(visual_localization, dict) else {},
        }

    @staticmethod
    def _visual_observation_likelihood(
        visual_localization: object,
        *,
        candidate_room_ids: list[str],
    ) -> dict[str, float]:
        scores = {room_id: 0.0 for room_id in candidate_room_ids}
        if not isinstance(visual_localization, dict):
            return scores
        raw_distribution = visual_localization.get("room_distribution")
        if isinstance(raw_distribution, dict):
            for room_id, score in raw_distribution.items():
                if room_id in scores and isinstance(score, (int, float)):
                    scores[room_id] = max(0.0, float(score))
        elif isinstance(raw_distribution, list):
            for record in raw_distribution:
                if not isinstance(record, dict):
                    continue
                room_id = record.get("room_id")
                score = record.get("score")
                if room_id in scores and isinstance(score, (int, float)):
                    scores[room_id] = max(0.0, float(score))
        normalized_scores = RoomLocalizer._normalize_scores(scores)
        if any(value > 0.0 for value in normalized_scores.values()):
            return normalized_scores
        return scores


class LLMRoomLocalizer(RoomLocalizer):
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
        same_floor_only: bool = True,
        self_transition_weight: float = 1.0,
        neighbor_transition_weight: float = 1.0,
        min_entity_likelihood: float = 0.05,
    ):
        super().__init__(
            room_graph=room_graph,
            grounding_index=grounding_index,
            same_floor_only=same_floor_only,
            self_transition_weight=self_transition_weight,
            neighbor_transition_weight=neighbor_transition_weight,
            min_entity_likelihood=min_entity_likelihood,
        )
        settings = resolve_model_environment(
            default_model="gpt-5-mini",
            default_api_base=DEFAULT_OPENAI_API_BASE,
            default_api_kind="responses",
        )
        self.model = model or settings.model_name or "gpt-5-mini"
        self.api_key = api_key or settings.api_key
        self.api_base = (api_base or settings.api_base or DEFAULT_OPENAI_API_BASE).rstrip("/")
        self.api_kind = resolve_api_kind(api_kind or settings.api_kind)
        self.request_timeout = float(request_timeout if request_timeout is not None else (settings.request_timeout or 30.0))
        self.response_client = response_client
        self.last_request_body: dict | None = None
        self.last_response_payload: dict | None = None
        self.model_client = ModelResponseClient(
            provider=settings.provider,
            api_key=self.api_key,
            api_base=self.api_base,
            api_kind=self.api_kind,
            request_timeout=self.request_timeout,
            num_ctx=settings.num_ctx,
            temperature=settings.temperature,
            response_client=self.response_client,
        )

    def localize(
        self,
        *,
        observation: Observation,
        prior_room_belief: dict[str, float] | None,
        fallback_room_id: str | None = None,
    ) -> dict:
        candidate_room_ids = self._candidate_room_ids(observation)
        entity_observation = self._observation_with_inside_entities(observation)
        if not candidate_room_ids:
            return {
                "predicted_room_id": None,
                "confidence": 0.0,
                "room_belief": {},
                "transition_support": {},
                "evidence": [],
            }

        if not entity_observation.entities:
            return super().localize(
                observation=entity_observation,
                prior_room_belief=prior_room_belief,
                fallback_room_id=fallback_room_id,
            )
        if not self.model_client.is_configured():
            raise RuntimeError("Missing model API configuration for LLM-based localization.")

        transition_support = self._build_transition_support(
            prior_room_belief or {},
            candidate_room_ids=candidate_room_ids,
            fallback_room_id=fallback_room_id,
        )
        request_body = self._build_request_body(
            observation=entity_observation,
            candidate_room_ids=candidate_room_ids,
            transition_support=transition_support,
        )
        self.last_request_body = self._clone_json(request_body)
        payload = self._create_response(request_body)
        self.last_response_payload = self._clone_json(payload)
        parsed = self._parse_output_payload(payload)
        observation_likelihood = self._observation_scores_from_llm(parsed, candidate_room_ids)

        posterior_scores = {
            room_id: float(transition_support.get(room_id, 0.0)) * float(observation_likelihood.get(room_id, 0.0))
            for room_id in candidate_room_ids
        }
        posterior = self._normalize_scores(posterior_scores)
        predicted_room_id = max(posterior, key=posterior.get) if posterior else None
        if posterior and posterior.get(predicted_room_id, 0.0) <= 0.0:
            predicted_room_id = None

        raw_evidence = parsed.get("evidence")
        evidence = [value for value in raw_evidence if isinstance(value, str) and value] if isinstance(raw_evidence, list) else []
        raw_summary = parsed.get("summary")
        summary = raw_summary if isinstance(raw_summary, str) else ""
        return {
            "predicted_room_id": predicted_room_id,
            "confidence": posterior.get(predicted_room_id, 0.0) if predicted_room_id else 0.0,
            "room_belief": posterior,
            "transition_support": transition_support,
            "observation_distribution": observation_likelihood,
            "observation_likelihood": observation_likelihood,
            "evidence": evidence[:3],
            "summary": summary,
        }

    def _build_request_body(
        self,
        *,
        observation: Observation,
        candidate_room_ids: list[str],
        transition_support: dict[str, float],
    ) -> dict:
        candidates = []
        for room_id in candidate_room_ids:
            node = self.room_graph.get(room_id, {})
            entry = self.grounding_index.room_entry(room_id) or {}
            candidates.append(
                {
                    "room_id": room_id,
                    "transition_support": float(transition_support.get(room_id, 0.0)),
                    "title": node.get("title"),
                    "category": node.get("category"),
                    "aliases": list(node.get("aliases") or []) + list(entry.get("aliases") or []),
                    "anchor_entities": list(entry.get("anchor_entities") or []),
                }
            )
        observation_entities = [
            {
                "name": entity.name,
                "kind": entity.kind,
                "confidence": float(entity.confidence),
                "source_views": list(entity.metadata.get("source_views", [])) or [entity.source_view],
            }
            for entity in observation.entities
        ]
        return {
            "model": self.model,
            "instructions": build_localization_instructions(),
            "input": build_localization_input(
                observation_entities=observation_entities,
                candidates=candidates,
            ),
            "reasoning": {"effort": "low"},
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "museum_localization",
                    "strict": True,
                    "schema": build_localization_schema(candidate_room_ids),
                }
            },
        }

    def _observation_scores_from_llm(self, parsed: dict, candidate_room_ids: list[str]) -> dict[str, float]:
        scores = {room_id: 0.0 for room_id in candidate_room_ids}
        raw_scores = parsed.get("room_distribution")
        if not isinstance(raw_scores, list):
            raw_scores = parsed.get("room_scores")
        if isinstance(raw_scores, list):
            for record in raw_scores:
                if not isinstance(record, dict):
                    continue
                room_id = record.get("room_id")
                score = record.get("score")
                if room_id in scores and isinstance(score, (int, float)):
                    scores[room_id] = max(0.0, float(score))
        normalized_scores = self._normalize_scores(scores)
        if any(value > 0.0 for value in normalized_scores.values()):
            return normalized_scores
        uniform = 1.0 / len(candidate_room_ids)
        return {room_id: uniform for room_id in candidate_room_ids}

    def _create_response(self, request_body: dict) -> dict:
        return self.model_client.create(request_body)

    def _parse_output_payload(self, payload: dict) -> dict:
        return parse_json_output(payload)

    @staticmethod
    def _clone_json(payload: dict) -> dict:
        return json.loads(json.dumps(payload))


class LLMSpatialAlignmentLocalizer(RoomLocalizer):
    ALIGNMENT_MODES = {"text_from_images", "direct_images"}
    ENTITY_DISTRIBUTION_MODES = {"heuristic", "llm"}

    def __init__(
        self,
        *,
        room_graph: dict[str, dict],
        grounding_index: GroundingIndex,
        alignment_mode: str = "text_from_images",
        model: str | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
        api_kind: str | None = None,
        request_timeout: float | None = None,
        response_client: Callable[[dict], dict] | None = None,
        entity_distribution_mode: str = "heuristic",
        same_floor_only: bool = True,
        self_transition_weight: float = 1.0,
        neighbor_transition_weight: float = 1.0,
        min_entity_likelihood: float = 0.05,
        alignment_apply_gap_threshold: float = 0.10,
    ):
        super().__init__(
            room_graph=room_graph,
            grounding_index=grounding_index,
            same_floor_only=same_floor_only,
            self_transition_weight=self_transition_weight,
            neighbor_transition_weight=neighbor_transition_weight,
            min_entity_likelihood=min_entity_likelihood,
        )
        if alignment_mode not in self.ALIGNMENT_MODES:
            raise ValueError(f"Unsupported alignment_mode: {alignment_mode}")
        if entity_distribution_mode not in self.ENTITY_DISTRIBUTION_MODES:
            raise ValueError(f"Unsupported entity_distribution_mode: {entity_distribution_mode}")
        settings = resolve_model_environment(
            default_model="gpt-5-mini",
            default_api_base=DEFAULT_OPENAI_API_BASE,
            default_api_kind="responses",
        )
        self.alignment_mode = alignment_mode
        self.entity_distribution_mode = entity_distribution_mode
        self.alignment_apply_gap_threshold = float(alignment_apply_gap_threshold)
        self.model = model or settings.model_name or "gpt-5-mini"
        self.api_key = api_key or settings.api_key
        self.api_base = (api_base or settings.api_base or DEFAULT_OPENAI_API_BASE).rstrip("/")
        self.api_kind = resolve_api_kind(api_kind or settings.api_kind)
        self.request_timeout = float(request_timeout if request_timeout is not None else (settings.request_timeout or 30.0))
        self.response_client = response_client
        self.last_extraction_request_body: dict | None = None
        self.last_extraction_response_payload: dict | None = None
        self.last_entity_request_body: dict | None = None
        self.last_entity_response_payload: dict | None = None
        self.last_alignment_request_body: dict | None = None
        self.last_alignment_response_payload: dict | None = None
        self.last_ego_spatial_context: dict | None = None
        self.model_client = ModelResponseClient(
            provider=settings.provider,
            api_key=self.api_key,
            api_base=self.api_base,
            api_kind=self.api_kind,
            request_timeout=self.request_timeout,
            num_ctx=settings.num_ctx,
            temperature=settings.temperature,
            response_client=self.response_client,
        )

    def localize(
        self,
        *,
        observation: Observation,
        prior_room_belief: dict[str, float] | None,
        fallback_room_id: str | None = None,
    ) -> dict:
        candidate_room_ids = self._candidate_room_ids(observation)
        if not candidate_room_ids:
            return {
                "predicted_room_id": None,
                "confidence": 0.0,
                "room_belief": {},
                "transition_support": {},
                "evidence": [],
            }
        if not self.model_client.is_configured():
            raise RuntimeError("Missing model API configuration for LLM spatial alignment localization.")

        ordered_views = self._ordered_views(observation)
        if not ordered_views:
            raise RuntimeError("Spatial alignment localization requires observation.views with image paths.")

        transition_support = self._build_transition_support(
            prior_room_belief or {},
            candidate_room_ids=candidate_room_ids,
            fallback_room_id=fallback_room_id,
        )
        entity_observation_likelihood, entity_scores_by_room = self._compute_entity_observation_likelihood(
            observation,
            candidate_room_ids,
            transition_support=transition_support,
        )
        entity_transition_room_belief = self._entity_transition_room_belief(
            transition_support=transition_support,
            entity_observation_likelihood=entity_observation_likelihood,
        )
        alignment_candidate_room_ids = self._alignment_candidate_room_ids(entity_transition_room_belief)
        candidate_context_text = self._build_candidate_context_text(alignment_candidate_room_ids)
        manifest_path = self._manifest_path_from_observation(observation)
        cached = self._load_alignment_cache(
            manifest_path=manifest_path,
            candidate_room_ids=alignment_candidate_room_ids,
            ordered_views=ordered_views,
        )
        if cached is not None:
            parsed = dict(cached.get("parsed_alignment", {}))
            ego_spatial_context = cached.get("ego_spatial_context")
            if not isinstance(ego_spatial_context, dict):
                ego_spatial_context = None
            self.last_extraction_request_body = None
            self.last_extraction_response_payload = None
            self.last_entity_request_body = None
            self.last_entity_response_payload = None
            self.last_alignment_request_body = None
            self.last_alignment_response_payload = self._clone_json(cached.get("alignment_payload"))
            self.last_ego_spatial_context = self._clone_json(ego_spatial_context)
            alignment_observation_likelihood = self._observation_scores_from_distribution(
                parsed,
                candidate_room_ids,
                fallback_room_ids=alignment_candidate_room_ids,
            )
            observation_likelihood, entity_transition_room_belief, alignment_fusion_applied = (
                self._fuse_observation_likelihoods(
                    transition_support=transition_support,
                    entity_observation_likelihood=entity_observation_likelihood,
                    alignment_observation_likelihood=alignment_observation_likelihood,
                )
            )
            return self._build_localization_result(
                candidate_room_ids=candidate_room_ids,
                transition_support=transition_support,
                observation_likelihood=observation_likelihood,
                entity_observation_likelihood=entity_observation_likelihood,
                alignment_observation_likelihood=alignment_observation_likelihood,
                entity_transition_room_belief=entity_transition_room_belief,
                alignment_fusion_applied=alignment_fusion_applied,
                entity_scores_by_room=entity_scores_by_room,
                parsed=parsed,
                candidate_context_text=candidate_context_text,
                ego_spatial_context=ego_spatial_context,
            )

        ego_spatial_context = None
        if self.alignment_mode == "text_from_images":
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
            alignment_request = self._build_alignment_text_request_body(
                candidate_room_ids=alignment_candidate_room_ids,
                candidate_context_text=candidate_context_text,
                ego_context_text=str(ego_spatial_context.get("text", "")),
                ordered_views=ordered_views,
            )
        else:
            self.last_extraction_request_body = None
            self.last_extraction_response_payload = None
            self.last_ego_spatial_context = None
            alignment_request = self._build_alignment_image_request_body(
                candidate_room_ids=alignment_candidate_room_ids,
                candidate_context_text=candidate_context_text,
                ordered_views=ordered_views,
            )

        self.last_alignment_request_body = self._clone_json(alignment_request)
        alignment_payload = self._create_response(alignment_request)
        self.last_alignment_response_payload = self._clone_json(alignment_payload)
        parsed = self._parse_output_payload(alignment_payload)
        self._write_alignment_cache(
            manifest_path=manifest_path,
            candidate_room_ids=alignment_candidate_room_ids,
            ordered_views=ordered_views,
            ego_spatial_context=ego_spatial_context,
            alignment_payload=alignment_payload,
            parsed_alignment=parsed,
        )
        alignment_observation_likelihood = self._observation_scores_from_distribution(
            parsed,
            candidate_room_ids,
            fallback_room_ids=alignment_candidate_room_ids,
        )
        observation_likelihood, entity_transition_room_belief, alignment_fusion_applied = (
            self._fuse_observation_likelihoods(
                transition_support=transition_support,
                entity_observation_likelihood=entity_observation_likelihood,
                alignment_observation_likelihood=alignment_observation_likelihood,
            )
        )
        return self._build_localization_result(
            candidate_room_ids=candidate_room_ids,
            transition_support=transition_support,
            observation_likelihood=observation_likelihood,
            entity_observation_likelihood=entity_observation_likelihood,
            alignment_observation_likelihood=alignment_observation_likelihood,
            entity_transition_room_belief=entity_transition_room_belief,
            alignment_fusion_applied=alignment_fusion_applied,
            entity_scores_by_room=entity_scores_by_room,
            parsed=parsed,
            candidate_context_text=candidate_context_text,
            ego_spatial_context=ego_spatial_context,
        )

    def _build_localization_result(
        self,
        *,
        candidate_room_ids: list[str],
        transition_support: dict[str, float],
        observation_likelihood: dict[str, float],
        entity_observation_likelihood: dict[str, float],
        alignment_observation_likelihood: dict[str, float],
        entity_transition_room_belief: dict[str, float],
        alignment_fusion_applied: bool,
        entity_scores_by_room: dict[str, list[tuple[str, float]]],
        parsed: dict,
        candidate_context_text: str,
        ego_spatial_context: dict | None,
    ) -> dict:
        posterior_scores = {
            room_id: float(transition_support.get(room_id, 0.0)) * float(observation_likelihood.get(room_id, 0.0))
            for room_id in candidate_room_ids
        }
        posterior = self._normalize_scores(posterior_scores)
        predicted_room_id = max(posterior, key=posterior.get) if posterior else None
        if predicted_room_id and posterior.get(predicted_room_id, 0.0) <= 0.0:
            predicted_room_id = None
        evidence = self._evidence_from_entity_scores(predicted_room_id, entity_scores_by_room)
        raw_summary = parsed.get("summary")
        summary = raw_summary if isinstance(raw_summary, str) else ""
        view_0_direction = parsed.get("view_0_allocentric_direction")
        if not isinstance(view_0_direction, str) or not view_0_direction:
            view_0_direction = None
        spatial_alignment = {
            "mode": self.alignment_mode,
            "view_0_allocentric_direction": view_0_direction,
            "candidate_context_text": candidate_context_text,
            "alignment_predicted_room_id": parsed.get("predicted_room_id"),
            "alignment_confidence": parsed.get("confidence"),
            "alignment_observation_distribution": alignment_observation_likelihood,
        }
        raw_sector_alignment = parsed.get("sector_alignment")
        if isinstance(raw_sector_alignment, list):
            spatial_alignment["sector_alignment"] = [
                record for record in raw_sector_alignment if isinstance(record, dict)
            ]
        raw_alignment_evidence = parsed.get("evidence")
        if isinstance(raw_alignment_evidence, list):
            spatial_alignment["alignment_evidence"] = [
                value for value in raw_alignment_evidence if isinstance(value, str) and value
            ][:3]
        if summary:
            spatial_alignment["alignment_summary"] = summary
        if ego_spatial_context is not None:
            spatial_alignment["ego_context_text"] = str(ego_spatial_context.get("text", ""))
            spatial_alignment["ego_context_views"] = list(ego_spatial_context.get("views", []))

        return {
            "predicted_room_id": predicted_room_id,
            "confidence": posterior.get(predicted_room_id, 0.0) if predicted_room_id else 0.0,
            "room_belief": posterior,
            "transition_support": transition_support,
            "observation_distribution": observation_likelihood,
            "observation_likelihood": observation_likelihood,
            "entity_observation_distribution": entity_observation_likelihood,
            "alignment_observation_distribution": alignment_observation_likelihood,
            "entity_transition_room_belief": entity_transition_room_belief,
            "alignment_fusion_applied": alignment_fusion_applied,
            "entity_distribution_mode": self.entity_distribution_mode,
            "evidence": evidence[:3],
            "summary": summary,
            "spatial_alignment": spatial_alignment,
            "ego_spatial_context": ego_spatial_context,
        }

    def _compute_entity_observation_likelihood(
        self,
        observation: Observation,
        candidate_room_ids: list[str],
        *,
        transition_support: dict[str, float],
    ) -> tuple[dict[str, float], dict[str, list[tuple[str, float]]]]:
        entity_observation = self._observation_with_inside_entities(observation)
        entity_scores_by_room = self._entity_scores_by_room(entity_observation, candidate_room_ids)
        if self.entity_distribution_mode != "llm" or not entity_observation.entities:
            return self._observation_scores_from_entities(entity_observation, candidate_room_ids)[0], entity_scores_by_room

        entity_request = self._build_entity_localization_request_body(
            observation=entity_observation,
            candidate_room_ids=candidate_room_ids,
            transition_support=transition_support,
        )
        self.last_entity_request_body = self._clone_json(entity_request)
        entity_payload = self._create_response(entity_request)
        self.last_entity_response_payload = self._clone_json(entity_payload)
        parsed = self._parse_output_payload(entity_payload)
        return self._observation_scores_from_llm_distribution(parsed, candidate_room_ids), entity_scores_by_room

    def _entity_scores_by_room(
        self,
        observation: Observation,
        candidate_room_ids: list[str],
    ) -> dict[str, list[tuple[str, float]]]:
        observation = self._observation_with_inside_entities(observation)
        if not observation.entities:
            return {room_id: [] for room_id in candidate_room_ids}
        scores: dict[str, list[tuple[str, float]]] = {}
        for room_id in candidate_room_ids:
            entity_scores = []
            for entity in observation.entities:
                _, match_score = self._entity_likelihood(room_id, entity)
                entity_scores.append((entity.name, match_score))
            scores[room_id] = sorted(entity_scores, key=lambda item: (-item[1], item[0].lower()))
        return scores

    def _build_entity_localization_request_body(
        self,
        *,
        observation: Observation,
        candidate_room_ids: list[str],
        transition_support: dict[str, float],
    ) -> dict:
        candidates = []
        for room_id in candidate_room_ids:
            node = self.room_graph.get(room_id, {})
            entry = self.grounding_index.room_entry(room_id) or {}
            candidates.append(
                {
                    "room_id": room_id,
                    "transition_support": float(transition_support.get(room_id, 0.0)),
                    "title": node.get("title"),
                    "category": node.get("category"),
                    "aliases": list(node.get("aliases") or []) + list(entry.get("aliases") or []),
                    "anchor_entities": list(entry.get("anchor_entities") or []),
                }
            )
        observation_entities = [
            {
                "name": entity.name,
                "kind": entity.kind,
                "confidence": float(entity.confidence),
                "source_views": list(entity.metadata.get("source_views", [])) or [entity.source_view],
            }
            for entity in observation.entities
        ]
        return {
            "model": self.model,
            "instructions": build_localization_instructions(),
            "input": build_localization_input(
                observation_entities=observation_entities,
                candidates=candidates,
            ),
            "reasoning": {"effort": "low"},
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "museum_localization",
                    "strict": True,
                    "schema": build_localization_schema(candidate_room_ids),
                }
            },
        }

    def _fuse_observation_likelihoods(
        self,
        *,
        transition_support: dict[str, float],
        entity_observation_likelihood: dict[str, float],
        alignment_observation_likelihood: dict[str, float],
    ) -> tuple[dict[str, float], dict[str, float], bool]:
        entity_transition_room_belief = self._entity_transition_room_belief(
            transition_support=transition_support,
            entity_observation_likelihood=entity_observation_likelihood,
        )
        if not self._should_apply_alignment(entity_transition_room_belief):
            return dict(entity_observation_likelihood), entity_transition_room_belief, False

        room_ids = list(entity_observation_likelihood.keys())
        fused_scores = {
            room_id: float(entity_observation_likelihood.get(room_id, 0.0))
            * float(alignment_observation_likelihood.get(room_id, 0.0))
            for room_id in room_ids
        }
        normalized = self._normalize_scores(fused_scores)
        if any(value > 0.0 for value in normalized.values()):
            return normalized, entity_transition_room_belief, True
        return dict(entity_observation_likelihood), entity_transition_room_belief, False

    @staticmethod
    def _ranked_room_ids(room_belief: dict[str, float]) -> list[str]:
        return [
            room_id
            for room_id, _ in sorted(
                (
                    (room_id, float(probability))
                    for room_id, probability in room_belief.items()
                    if isinstance(probability, (int, float))
                ),
                key=lambda item: (-item[1], item[0]),
            )
        ]

    def _entity_transition_room_belief(
        self,
        *,
        transition_support: dict[str, float],
        entity_observation_likelihood: dict[str, float],
    ) -> dict[str, float]:
        entity_transition_scores = {
            room_id: float(transition_support.get(room_id, 0.0))
            * float(entity_observation_likelihood.get(room_id, 0.0))
            for room_id in entity_observation_likelihood.keys()
        }
        return self._normalize_scores(entity_transition_scores)

    def _alignment_candidate_room_ids(self, entity_transition_room_belief: dict[str, float]) -> list[str]:
        ranked_room_ids = self._ranked_room_ids(entity_transition_room_belief)
        if not ranked_room_ids:
            return []
        return ranked_room_ids[:2]

    def _should_apply_alignment(self, entity_transition_room_belief: dict[str, float]) -> bool:
        ranked_room_ids = self._ranked_room_ids(entity_transition_room_belief)
        if len(ranked_room_ids) < 2:
            return False
        top_1_room_id, top_2_room_id = ranked_room_ids[:2]
        top_1 = float(entity_transition_room_belief.get(top_1_room_id, 0.0))
        top_2 = float(entity_transition_room_belief.get(top_2_room_id, 0.0))
        if top_1 <= 0.0:
            return False
        if self._room_pair_has_high_semantic_overlap(top_1_room_id, top_2_room_id):
            return True
        return (top_1 - top_2) <= self.alignment_apply_gap_threshold

    def _room_pair_has_high_semantic_overlap(self, room_a_id: str, room_b_id: str) -> bool:
        if not room_a_id or not room_b_id:
            return False
        signature_a = set(_tokenize(" ".join(self._build_room_signature(room_a_id))))
        signature_b = set(_tokenize(" ".join(self._build_room_signature(room_b_id))))
        if not signature_a or not signature_b:
            return False
        overlap = len(signature_a & signature_b) / max(1, len(signature_a | signature_b))
        return overlap >= 0.5

    def _load_alignment_cache(
        self,
        *,
        manifest_path: Path | None,
        candidate_room_ids: list[str],
        ordered_views: list[dict],
    ) -> dict | None:
        cache_path = self._alignment_cache_path(manifest_path)
        if cache_path is None or not cache_path.exists():
            return None
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict):
            return None
        if payload.get("cache_version") != 1:
            return None
        if payload.get("alignment_mode") != self.alignment_mode:
            return None
        if payload.get("model") != self.model:
            return None
        if payload.get("api_kind") != self.api_kind:
            return None
        if payload.get("candidate_room_ids") != list(candidate_room_ids):
            return None
        if payload.get("view_ids") != [view["view_id"] for view in ordered_views]:
            return None
        parsed_alignment = payload.get("parsed_alignment")
        if not isinstance(parsed_alignment, dict):
            return None
        return payload

    def _write_alignment_cache(
        self,
        *,
        manifest_path: Path | None,
        candidate_room_ids: list[str],
        ordered_views: list[dict],
        ego_spatial_context: dict | None,
        alignment_payload: dict,
        parsed_alignment: dict,
    ) -> None:
        cache_path = self._alignment_cache_path(manifest_path)
        if cache_path is None:
            return
        payload = {
            "cache_version": 1,
            "alignment_mode": self.alignment_mode,
            "model": self.model,
            "api_kind": self.api_kind,
            "candidate_room_ids": list(candidate_room_ids),
            "view_ids": [view["view_id"] for view in ordered_views],
            "ego_spatial_context": self._clone_json(ego_spatial_context),
            "alignment_payload": self._clone_json(alignment_payload),
            "parsed_alignment": self._clone_json(parsed_alignment),
        }
        cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _alignment_cache_path(self, manifest_path: Path | None) -> Path | None:
        if manifest_path is None:
            return None
        suffix = self.alignment_mode.replace("-", "_")
        return manifest_path.with_name(f"{manifest_path.stem}_spatial_alignment_{suffix}.json")

    @staticmethod
    def _manifest_path_from_observation(observation: Observation) -> Path | None:
        manifest_path = observation.metadata.get("manifest_path")
        if not isinstance(manifest_path, str) or not manifest_path:
            return None
        return Path(manifest_path)

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
        for room_id in candidate_room_ids:
            node = self.room_graph.get(room_id, {})
            for key in ("title", "category"):
                value = node.get(key)
                if isinstance(value, str) and value and value not in labels:
                    labels.append(value)
            for neighbor in node.get("neighbors", []):
                if not isinstance(neighbor, dict):
                    continue
                neighbor_node = self.room_graph.get(str(neighbor.get("target_room_id")), {})
                for key in ("title", "category"):
                    value = neighbor_node.get(key)
                    if isinstance(value, str) and value and value not in labels:
                        labels.append(value)
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

    def _build_alignment_image_request_body(
        self,
        *,
        candidate_room_ids: list[str],
        candidate_context_text: str,
        ordered_views: list[dict],
    ) -> dict:
        content: list[dict] = [
            {
                "type": "input_text",
                "text": build_spatial_alignment_input(
                    candidate_context_text=candidate_context_text,
                    ego_context_text="Use the panorama-sector images below directly instead of a pre-extracted textual context.",
                    view_ids=[view["view_id"] for view in ordered_views],
                    direct_images=True,
                ),
            }
        ]
        for view in ordered_views:
            content.extend(
                [
                    {
                        "type": "input_text",
                        "text": f"Panorama sector {view['view_id']}. The sectors are ordered clockwise, but the global heading is unknown.",
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
            "instructions": build_spatial_alignment_instructions(direct_images=True),
            "input": [{"role": "user", "content": content}],
            "reasoning": {"effort": "low"},
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "spatial_alignment_direct_images",
                    "strict": True,
                    "schema": build_spatial_alignment_schema(
                        candidate_room_ids,
                        view_ids=[view["view_id"] for view in ordered_views],
                        include_sector_alignment=True,
                    ),
                }
            },
        }

    @staticmethod
    def _observation_scores_from_distribution(
        parsed: dict,
        candidate_room_ids: list[str],
        *,
        fallback_room_ids: list[str] | None = None,
    ) -> dict[str, float]:
        scores = {room_id: 0.0 for room_id in candidate_room_ids}
        raw_scores = parsed.get("room_distribution")
        if isinstance(raw_scores, list):
            for record in raw_scores:
                if not isinstance(record, dict):
                    continue
                room_id = record.get("room_id")
                score = record.get("score")
                if room_id in scores and isinstance(score, (int, float)):
                    scores[room_id] = max(0.0, float(score))
        normalized_scores = RoomLocalizer._normalize_scores(scores)
        if any(value > 0.0 for value in normalized_scores.values()):
            return normalized_scores
        if fallback_room_ids:
            fallback_set = {room_id for room_id in fallback_room_ids if room_id in scores}
            if fallback_set:
                uniform = 1.0 / len(fallback_set)
                return {room_id: (uniform if room_id in fallback_set else 0.0) for room_id in candidate_room_ids}
        uniform = 1.0 / len(candidate_room_ids)
        return {room_id: uniform for room_id in candidate_room_ids}

    @staticmethod
    def _observation_scores_from_llm_distribution(parsed: dict, candidate_room_ids: list[str]) -> dict[str, float]:
        scores = {room_id: 0.0 for room_id in candidate_room_ids}
        raw_scores = parsed.get("room_distribution")
        if not isinstance(raw_scores, list):
            raw_scores = parsed.get("room_scores")
        if isinstance(raw_scores, list):
            for record in raw_scores:
                if not isinstance(record, dict):
                    continue
                room_id = record.get("room_id")
                score = record.get("score")
                if room_id in scores and isinstance(score, (int, float)):
                    scores[room_id] = max(0.0, float(score))
        normalized_scores = RoomLocalizer._normalize_scores(scores)
        if any(value > 0.0 for value in normalized_scores.values()):
            return normalized_scores
        uniform = 1.0 / len(candidate_room_ids)
        return {room_id: uniform for room_id in candidate_room_ids}

    def _create_response(self, request_body: dict) -> dict:
        return self.model_client.create(request_body)

    def _parse_output_payload(self, payload: dict) -> dict:
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
