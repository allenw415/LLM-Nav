from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable

from ..common.env import resolve_model_environment, resolve_task_num_ctx
from ..common.model_client import (
    DEFAULT_OPENAI_API_BASE,
    ModelResponseClient,
    parse_json_output,
    resolve_api_kind,
)
from ..common.prompts import (
    build_navigation_parse_input,
    build_navigation_parse_instructions,
    build_navigation_parse_schema,
)
from ..common.room_profiles import visual_profile_anchor_entities
from ..common.request_cache import (
    atomic_write_json,
    cache_entry_path,
    canonical_json_sha256,
    exclusive_cache_lock,
    model_transport_identity,
    read_json_object,
    request_fingerprint,
)
from ..common.types import ParsedNavigationEntity, TaskSpec


QUERY_PARSER_CACHE_NAMESPACE = "navigation_query_parser"
QUERY_PARSER_CACHE_VERSION = 1


class LLMInstructionParser:
    """
    Parse only ordered navigation waypoints and goals.

    The navigation source is deliberately excluded from this contract. Runtime
    source resolution derives it from the caller-provided start panorama.
    """

    def __init__(
        self,
        *,
        room_graph: dict[str, dict],
        model: str | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
        api_kind: str | None = None,
        request_timeout: float | None = None,
        num_ctx: int | None = None,
        max_theme_lines: int | None = None,
        response_client: Callable[[dict], dict] | None = None,
        cache_dir: str | Path | None = None,
    ):
        settings = resolve_model_environment(
            default_model="gpt-5-mini",
            default_api_base=DEFAULT_OPENAI_API_BASE,
            default_api_kind="responses",
        )
        self.room_graph = room_graph
        self.model = model or settings.model_name or "gpt-5-mini"
        self.api_key = api_key or settings.api_key
        self.api_base = (api_base or settings.api_base or DEFAULT_OPENAI_API_BASE).rstrip("/")
        self.api_kind = resolve_api_kind(api_kind or settings.api_kind)
        self.request_timeout = float(
            request_timeout
            if request_timeout is not None
            else (settings.request_timeout or 30.0)
        )
        self.num_ctx = resolve_task_num_ctx(
            "parse_instruction",
            explicit_num_ctx=num_ctx,
            fallback_num_ctx=settings.num_ctx,
            default_num_ctx=8192,
        )
        self.response_client = response_client
        self.max_theme_lines = max_theme_lines
        self.cache_dir = Path(cache_dir).resolve() if cache_dir is not None else None
        self.last_request_body: dict | None = None
        self.last_response_payload: dict | None = None
        self.last_cache_hit: bool | None = None
        self.last_cache_path: str | None = None
        self.last_cache_fingerprint: str | None = None
        self.model_client = ModelResponseClient(
            provider=settings.provider,
            api_key=self.api_key,
            api_base=self.api_base,
            api_kind=self.api_kind,
            request_timeout=self.request_timeout,
            num_ctx=self.num_ctx,
            temperature=settings.temperature,
            think=settings.think,
            response_client=self.response_client,
        )

    def parse(self, instruction: str) -> TaskSpec:
        request_body = self._build_request_body(instruction)
        self.last_request_body = self._clone_json(request_body)
        self.last_cache_hit = None
        self.last_cache_path = None
        self.last_cache_fingerprint = None

        if self.cache_dir is None:
            if not self.model_client.is_configured():
                raise RuntimeError(
                    "Missing model API configuration for LLM-based instruction parsing."
                )
            payload = self._create_response(request_body)
            self.last_response_payload = self._clone_json(payload)
            parsed = self._parse_output_payload(payload)
            return self._task_from_llm_parse(instruction, parsed)

        transport = model_transport_identity(
            model=self.model,
            model_client=self.model_client,
        )
        fingerprint = request_fingerprint(
            namespace=QUERY_PARSER_CACHE_NAMESPACE,
            cache_version=QUERY_PARSER_CACHE_VERSION,
            request_body=request_body,
            transport_identity=transport,
        )
        cache_path = cache_entry_path(self.cache_dir, fingerprint)
        self.last_cache_path = str(cache_path)
        self.last_cache_fingerprint = fingerprint

        with exclusive_cache_lock(cache_path):
            cached_payload = self._read_cached_response(
                cache_path,
                fingerprint=fingerprint,
                transport=transport,
            )
            if cached_payload is not None:
                try:
                    parsed = self._parse_output_payload(cached_payload)
                    task = self._task_from_llm_parse(instruction, parsed)
                except (TypeError, ValueError):
                    cached_payload = None
                else:
                    self.last_cache_hit = True
                    self.last_response_payload = self._clone_json(cached_payload)
                    self.model_client.last_call_wall_ms = 0.0
                    self.model_client.last_call_succeeded = True
                    return task

            if not self.model_client.is_configured():
                raise RuntimeError(
                    "Missing model API configuration for LLM-based instruction parsing."
                )
            self.last_cache_hit = False
            payload = self._create_response(request_body)
            parsed = self._parse_output_payload(payload)
            task = self._task_from_llm_parse(instruction, parsed)
            self.last_response_payload = self._clone_json(payload)
            self.last_cache_hit = False
            atomic_write_json(
                cache_path,
                {
                    "cache_schema": QUERY_PARSER_CACHE_NAMESPACE,
                    "cache_version": QUERY_PARSER_CACHE_VERSION,
                    "fingerprint": fingerprint,
                    "transport": transport,
                    "response_sha256": canonical_json_sha256(payload),
                    "response": self._clone_json(payload),
                },
            )
            return task

    @staticmethod
    def _read_cached_response(
        cache_path: Path,
        *,
        fingerprint: str,
        transport: dict[str, object],
    ) -> dict | None:
        entry = read_json_object(cache_path)
        if entry is None:
            return None
        if entry.get("cache_schema") != QUERY_PARSER_CACHE_NAMESPACE:
            return None
        if entry.get("cache_version") != QUERY_PARSER_CACHE_VERSION:
            return None
        if entry.get("fingerprint") != fingerprint:
            return None
        if entry.get("transport") != transport:
            return None
        response = entry.get("response")
        if not isinstance(response, dict):
            return None
        if entry.get("response_sha256") != canonical_json_sha256(response):
            return None
        return response

    def _build_request_body(self, instruction: str) -> dict:
        room_ids = list(self.room_graph)
        return {
            "model": self.model,
            "instructions": build_navigation_parse_instructions(),
            "input": build_navigation_parse_input(
                instruction=instruction,
                room_ids=room_ids,
                theme_lines=self._room_theme_lines(),
            ),
            "reasoning": {"effort": "low"},
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "navigation_query_parse",
                    "strict": True,
                    "schema": build_navigation_parse_schema(room_ids),
                }
            },
        }

    def _create_response(self, request_body: dict) -> dict:
        return self.model_client.create(request_body)

    def _parse_output_payload(self, payload: dict) -> dict:
        return parse_json_output(payload)

    @staticmethod
    def _clone_json(payload: dict) -> dict:
        return json.loads(json.dumps(payload))

    def _task_from_llm_parse(self, instruction: str, parsed: dict) -> TaskSpec:
        if not isinstance(parsed, dict):
            raise ValueError("Parsed navigation query must be a JSON object.")
        goal_entities = self._parse_entities(
            parsed.get("goal_entities"),
            field_name="goal_entities",
        )
        waypoint_entities = self._parse_entities(
            parsed.get("waypoint_entities"),
            field_name="waypoint_entities",
        )
        goal_room_ids = self._ordered_room_ids(goal_entities)
        if not goal_room_ids:
            raise ValueError("Navigation query did not resolve any goal room ids.")

        target_room_id = goal_room_ids[-1]
        waypoint_room_ids = [
            room_id
            for room_id in self._ordered_room_ids(waypoint_entities)
            if room_id != target_room_id
        ]
        return TaskSpec(
            task_type=self._task_type(goal_entities, waypoint_entities),
            raw_instruction=instruction,
            source_room_id=None,
            source_entity=None,
            goal_room_ids=[target_room_id],
            waypoint_room_ids=waypoint_room_ids,
            goal_entities=goal_entities,
            waypoint_entities=waypoint_entities,
        )

    def _room_theme_lines(self) -> str:
        lines: list[str] = []
        for room_id, node in self.room_graph.items():
            title = self._string_value(
                node.get("title") or node.get("display_name") or room_id
            )
            anchors = visual_profile_anchor_entities(node, max_items=4)
            if not anchors:
                category = self._string_value(node.get("category"))
                anchors = [category] if category else []
            details = "; ".join(anchors)
            lines.append(
                f"- {room_id}: {title}. {details}"
                if details
                else f"- {room_id}: {title}"
            )
            if self.max_theme_lines is not None and len(lines) >= self.max_theme_lines:
                break
        return "\n".join(lines)

    def _parse_entities(
        self,
        values: object,
        *,
        field_name: str,
    ) -> list[ParsedNavigationEntity]:
        if not isinstance(values, list):
            raise ValueError(
                f"Parsed navigation query field must be a list: {field_name}"
            )
        entities: list[ParsedNavigationEntity] = []
        for value in values:
            if not isinstance(value, dict):
                raise ValueError(
                    f"Parsed navigation query entity must be an object: {field_name}"
                )
            entity = self._parse_entity(value)
            if entity.predicted_room_id is not None:
                entities.append(entity)
        return entities

    def _parse_entity(self, value: dict[str, Any]) -> ParsedNavigationEntity:
        name = str(value.get("name") or "").strip()
        entity_type = str(value.get("entity_type") or "").strip()
        predicted_room_id = value.get("predicted_room_id")
        confidence = value.get("confidence")

        if not name:
            raise ValueError("Parsed navigation entity is missing a name.")
        if entity_type not in {"gallery", "artwork"}:
            raise ValueError(f"Unsupported parsed entity type: {entity_type}")
        if predicted_room_id is not None and not isinstance(predicted_room_id, str):
            raise ValueError(
                f"Parsed entity has invalid room id: {predicted_room_id!r}"
            )
        if predicted_room_id is not None and predicted_room_id not in self.room_graph:
            raise ValueError(f"Parsed entity used unknown room id: {predicted_room_id}")
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise ValueError(f"Parsed entity has invalid confidence: {confidence!r}")
        numeric_confidence = float(confidence)
        if not math.isfinite(numeric_confidence):
            raise ValueError(
                f"Parsed entity confidence must be finite: {numeric_confidence}"
            )
        if numeric_confidence < 0.0 or numeric_confidence > 1.0:
            raise ValueError(
                f"Parsed entity confidence is out of range: {numeric_confidence}"
            )
        if entity_type == "gallery" and predicted_room_id is None:
            raise ValueError(f"Gallery entity must resolve to a room id: {name}")
        if name in self.room_graph and (
            predicted_room_id != name or numeric_confidence != 1.0
        ):
            raise ValueError(
                f"Explicit room entity must map exactly with confidence 1.0: {name}"
            )

        return ParsedNavigationEntity(
            name=name,
            entity_type=entity_type,
            predicted_room_id=predicted_room_id,
            confidence=numeric_confidence,
        )

    @staticmethod
    def _task_type(
        goal_entities: list[ParsedNavigationEntity],
        waypoint_entities: list[ParsedNavigationEntity],
    ) -> str:
        if not waypoint_entities:
            return (
                "artwork_goal_navigation"
                if goal_entities[-1].entity_type == "artwork"
                else "gallery_goal_navigation"
            )
        entity_types = {
            entity.entity_type for entity in [*waypoint_entities, *goal_entities]
        }
        if entity_types == {"gallery"}:
            return "gallery_instruction_following_navigation"
        if entity_types == {"artwork"}:
            return "artwork_instruction_following_navigation"
        return "artwork_gallery_instruction_following_navigation"

    @staticmethod
    def _ordered_room_ids(
        entities: list[ParsedNavigationEntity],
    ) -> list[str]:
        ordered: list[str] = []
        for entity in entities:
            room_id = entity.predicted_room_id
            if room_id and room_id not in ordered:
                ordered.append(room_id)
        return ordered

    @staticmethod
    def _string_value(value: object) -> str:
        return value.strip() if isinstance(value, str) else ""
