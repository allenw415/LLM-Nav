from __future__ import annotations

import json
import math
import os
import re
import urllib.request
from difflib import SequenceMatcher
from typing import Callable

from ..common.prompts import build_localization_input, build_localization_instructions, build_localization_schema
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
        if not candidate_room_ids:
            return {
                "predicted_room_id": None,
                "confidence": 0.0,
                "room_belief": {},
                "transition_support": {},
                "evidence": [],
            }

        if not observation.entities:
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
            for entity in observation.entities:
                likelihood, match_score = self._entity_likelihood(room_id, entity)
                log_score += math.log(likelihood)
                entity_scores.append((entity.name, match_score))
            entity_scores_by_room[room_id] = sorted(entity_scores, key=lambda item: (-item[1], item[0].lower()))
            log_scores[room_id] = log_score

        posterior = self._normalize_log_scores(log_scores)
        predicted_room_id = max(posterior, key=posterior.get) if posterior else None
        evidence = []
        if predicted_room_id:
            for entity_name, match_score in entity_scores_by_room.get(predicted_room_id, []):
                if match_score < 0.35:
                    continue
                evidence.append(entity_name)
                if len(evidence) >= 3:
                    break

        return {
            "predicted_room_id": predicted_room_id,
            "confidence": posterior.get(predicted_room_id, 0.0) if predicted_room_id else 0.0,
            "room_belief": posterior,
            "transition_support": transition_support,
            "evidence": evidence,
        }

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


class LLMRoomLocalizer(RoomLocalizer):
    def __init__(
        self,
        *,
        room_graph: dict[str, dict],
        grounding_index: GroundingIndex,
        model: str = "gpt-5-mini",
        api_key: str | None = None,
        api_base: str = "https://api.openai.com/v1",
        request_timeout: float = 30.0,
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
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.api_base = api_base.rstrip("/")
        self.request_timeout = request_timeout
        self.response_client = response_client
        self.last_request_body: dict | None = None
        self.last_response_payload: dict | None = None

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

        if not observation.entities:
            return super().localize(
                observation=observation,
                prior_room_belief=prior_room_belief,
                fallback_room_id=fallback_room_id,
            )
        if not self.api_key and self.response_client is None:
            raise RuntimeError("Missing API key for LLM-based localization.")

        transition_support = self._build_transition_support(
            prior_room_belief or {},
            candidate_room_ids=candidate_room_ids,
            fallback_room_id=fallback_room_id,
        )
        request_body = self._build_request_body(
            observation=observation,
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
        if self.response_client is not None:
            return self.response_client(request_body)
        request = urllib.request.Request(
            f"{self.api_base}/responses",
            data=json.dumps(request_body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.request_timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def _parse_output_payload(self, payload: dict) -> dict:
        output_text = payload.get("output_text")
        if not isinstance(output_text, str) or not output_text.strip():
            fragments: list[str] = []
            for item in payload.get("output", []):
                if not isinstance(item, dict) or item.get("type") != "message":
                    continue
                for content in item.get("content", []):
                    if not isinstance(content, dict):
                        continue
                    text = content.get("text")
                    if isinstance(text, str):
                        fragments.append(text)
            output_text = "".join(fragments)
        if not isinstance(output_text, str) or not output_text.strip():
            raise ValueError("Responses API payload did not include output text for localization.")
        return json.loads(output_text)

    @staticmethod
    def _clone_json(payload: dict) -> dict:
        return json.loads(json.dumps(payload))
