from __future__ import annotations

import json
from typing import Callable

from ..common.env import resolve_model_environment, resolve_task_num_ctx
from ..common.model_client import DEFAULT_OPENAI_API_BASE, ModelResponseClient, parse_json_output, resolve_api_kind
from ..common.prompts import (
    NAVIGATION_TASK_TYPES,
    build_navigation_parse_input,
    build_navigation_parse_instructions,
    build_navigation_parse_schema,
)
from ..common.types import ParsedNavigationEntity, TaskSpec


class LLMInstructionParser:
    """
    LLM-only parser for navigation task types used in the paper:
    gallery-goal, artwork-goal, gallery-instruction-following,
    artwork-instruction-following, and mixed artwork/gallery instruction-following.
    """

    VALID_TASK_TYPES = set(NAVIGATION_TASK_TYPES)

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
        response_client: Callable[[dict], dict] | None = None,
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
        self.request_timeout = float(request_timeout if request_timeout is not None else (settings.request_timeout or 30.0))
        self.num_ctx = resolve_task_num_ctx(
            "parse_instruction",
            explicit_num_ctx=num_ctx,
            fallback_num_ctx=settings.num_ctx,
            default_num_ctx=8192,
        )
        self.response_client = response_client
        self.last_request_body: dict | None = None
        self.last_response_payload: dict | None = None
        self.model_client = ModelResponseClient(
            provider=settings.provider,
            api_key=self.api_key,
            api_base=self.api_base,
            api_kind=self.api_kind,
            request_timeout=self.request_timeout,
            num_ctx=self.num_ctx,
            temperature=settings.temperature,
            response_client=self.response_client,
        )

    def parse(self, instruction: str) -> TaskSpec:
        if not self.model_client.is_configured():
            raise RuntimeError("Missing model API configuration for LLM-based instruction parsing.")

        request_body = self._build_request_body(instruction)
        self.last_request_body = self._clone_json(request_body)
        payload = self._create_response(request_body)
        self.last_response_payload = self._clone_json(payload)
        parsed = self._parse_output_payload(payload)
        task = self._task_from_llm_parse(instruction, parsed)
        if not task.goal_room_ids:
            raise ValueError("LLM parser did not return a valid goal room sequence.")
        return task

    def _build_request_body(self, instruction: str) -> dict:
        room_ids = sorted(self.room_graph)
        theme_lines = self._room_theme_lines()
        return {
            "model": self.model,
            "instructions": build_navigation_parse_instructions(),
            "input": build_navigation_parse_input(
                instruction=instruction,
                room_ids=room_ids,
                theme_lines=theme_lines,
            ),
            "reasoning": {"effort": "low"},
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "museum_navigation_parse",
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
        task_type = parsed.get("task_type")
        if task_type not in self.VALID_TASK_TYPES:
            return TaskSpec(task_type="unknown", raw_instruction=instruction)

        source_room_id = parsed.get("source_room_id")
        source_entity = self._parse_entity(parsed.get("source_entity"))
        goal_entities = self._parse_entities(parsed.get("goal_entities", []))
        waypoint_entities = self._parse_entities(parsed.get("waypoint_entities", []))

        if source_entity and source_entity.predicted_room_id:
            source_room_id = source_entity.predicted_room_id
        if source_room_id not in self.room_graph:
            source_room_id = None

        goal_room_ids = self._ordered_room_ids(goal_entities)
        waypoint_room_ids = self._ordered_room_ids(waypoint_entities)
        if not goal_room_ids:
            return TaskSpec(task_type="unknown", raw_instruction=instruction)

        return TaskSpec(
            task_type=task_type,
            raw_instruction=instruction,
            source_room_id=source_room_id,
            source_entity=source_entity,
            goal_room_ids=goal_room_ids,
            waypoint_room_ids=waypoint_room_ids,
            goal_entities=goal_entities,
            waypoint_entities=waypoint_entities,
        )

    def _room_theme_lines(self) -> str:
        lines = []
        for room_id in sorted(self.room_graph):
            node = self.room_graph[room_id]
            title = node.get("title")
            if isinstance(title, str) and title:
                lines.append(f"- {title}: {room_id}")
        return "\n".join(lines)

    def _parse_entities(self, values: list[dict]) -> list[ParsedNavigationEntity]:
        entities: list[ParsedNavigationEntity] = []
        for value in values:
            entity = self._parse_entity(value)
            if entity is not None:
                entities.append(entity)
        return entities

    def _parse_entity(self, value: dict | None) -> ParsedNavigationEntity | None:
        if not isinstance(value, dict):
            return None
        name = value.get("name")
        entity_type = value.get("entity_type")
        predicted_room_id = value.get("predicted_room_id")
        confidence = value.get("confidence")
        if not isinstance(name, str) or not name:
            return None
        if entity_type not in {"gallery", "artwork"}:
            return None
        if predicted_room_id not in self.room_graph:
            predicted_room_id = None
        if not isinstance(confidence, (int, float)):
            confidence = None
        return ParsedNavigationEntity(
            name=name,
            entity_type=entity_type,
            predicted_room_id=predicted_room_id,
            confidence=None if confidence is None else float(confidence),
        )

    @staticmethod
    def _ordered_room_ids(entities: list[ParsedNavigationEntity]) -> list[str]:
        ordered: list[str] = []
        for entity in entities:
            room_id = entity.predicted_room_id
            if room_id and room_id not in ordered:
                ordered.append(room_id)
        return ordered
