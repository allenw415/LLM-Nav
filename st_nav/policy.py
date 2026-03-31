from __future__ import annotations

import json
import os
import urllib.request
from typing import Callable

from .models import ParsedNavigationEntity, TaskSpec
from .prompts import (
    NAVIGATION_TASK_TYPES,
    build_navigation_parse_input,
    build_navigation_parse_instructions,
    build_navigation_parse_schema,
)


class LLMInstructionParser:
    """
    LLM-only parser for the four task types used in the paper:
    gallery-goal, artwork-goal, gallery-instruction-following, artwork-instruction-following.
    """

    VALID_TASK_TYPES = set(NAVIGATION_TASK_TYPES)

    def __init__(
        self,
        *,
        room_graph: dict[str, dict],
        model: str = "gpt-5-mini",
        api_key: str | None = None,
        api_base: str = "https://api.openai.com/v1",
        request_timeout: float = 30.0,
        response_client: Callable[[dict], dict] | None = None,
    ):
        self.room_graph = room_graph
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.api_base = api_base.rstrip("/")
        self.request_timeout = request_timeout
        self.response_client = response_client
        self.last_request_body: dict | None = None
        self.last_response_payload: dict | None = None

    def parse(self, instruction: str) -> TaskSpec:
        if not self.api_key and self.response_client is None:
            raise RuntimeError("Missing API key for LLM-based instruction parsing.")

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
            raise ValueError("Responses API payload did not include output text.")
        return json.loads(output_text)

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
