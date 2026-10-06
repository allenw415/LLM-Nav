from __future__ import annotations

from base64 import b64encode
import json
import math
import mimetypes
from pathlib import Path
from typing import Callable

from ..common.env import resolve_model_environment
from ..common.model_client import (
    DEFAULT_OPENAI_API_BASE,
    ModelResponseClient,
    parse_json_output,
    resolve_api_kind,
)
from ..common.prompts import (
    build_visual_theme_direction_input,
    build_visual_theme_direction_instructions,
    build_visual_theme_direction_schema,
)
from ..common.request_cache import (
    atomic_write_json,
    cache_entry_path,
    canonical_json_sha256,
    exclusive_cache_lock,
    model_transport_identity,
    read_json_object,
    request_fingerprint,
)
from ..common.room_profiles import compact_visual_profile
from ..common.types import PolicyOutput, ReasoningInput, RenderedView
from .action_policy import GreedyActionPolicy


VISUAL_THEME_DIRECTION_CACHE_NAMESPACE = "visual_theme_direction"
VISUAL_THEME_DIRECTION_CACHE_VERSION = 1


class VisualThemeActionPolicy(GreedyActionPolicy):
    """Choose the graph edge nearest a VLM-selected view without trajectory memory."""

    def __init__(
        self,
        *,
        room_graph: dict[str, dict],
        model: str | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
        api_kind: str | None = None,
        request_timeout: float | None = None,
        response_client: Callable[[dict], dict] | None = None,
        cache_dir: str | Path | None = None,
        schema_validation_retries: int = 1,
        max_output_tokens: int | None = None,
        expected_view_count: int = 8,
        u_turn_threshold_deg: float = 150.0,
        history_return_threshold_deg: float = 30.0,
    ) -> None:
        super().__init__(
            u_turn_threshold_deg=u_turn_threshold_deg,
            history_return_threshold_deg=history_return_threshold_deg,
        )
        if schema_validation_retries < 0:
            raise ValueError("schema_validation_retries must be non-negative.")
        if max_output_tokens is not None and max_output_tokens < 1:
            raise ValueError("max_output_tokens must be positive when set.")
        if expected_view_count < 1:
            raise ValueError("expected_view_count must be positive.")

        settings = resolve_model_environment(
            default_model="gemma-4-31b-it",
            default_api_base=DEFAULT_OPENAI_API_BASE,
            default_api_kind="responses",
        )
        self.room_graph = room_graph
        self.model = model or settings.model_name or "gemma-4-31b-it"
        self.api_key = api_key or settings.api_key
        self.api_base = (api_base or settings.api_base or DEFAULT_OPENAI_API_BASE).rstrip("/")
        self.api_kind = resolve_api_kind(api_kind or settings.api_kind)
        self.request_timeout = float(
            request_timeout
            if request_timeout is not None
            else (settings.request_timeout or 300.0)
        )
        self.model_client = ModelResponseClient(
            provider=settings.provider,
            api_key=self.api_key,
            api_base=self.api_base,
            api_kind=self.api_kind,
            request_timeout=self.request_timeout,
            num_ctx=settings.num_ctx,
            temperature=settings.temperature,
            think=settings.think,
            response_client=response_client,
        )
        self.cache_dir = Path(cache_dir).resolve() if cache_dir is not None else None
        self.schema_validation_retries = int(schema_validation_retries)
        self.max_output_tokens = max_output_tokens
        self.expected_view_count = int(expected_view_count)
        self.last_request_body: dict | None = None
        self.last_response_payload: dict | None = None
        self.last_cache_hit: bool | None = None
        self.last_cache_path: str | None = None
        self.last_cache_fingerprint: str | None = None
        self.last_validation_attempt_count: int = 0

    def reset_episode(self) -> None:
        self.last_request_body = None
        self.last_response_payload = None
        self.last_cache_hit = None
        self.last_cache_path = None
        self.last_cache_fingerprint = None
        self.last_validation_attempt_count = 0

    def choose_next_action(self, reasoning_input: ReasoningInput) -> PolicyOutput:
        if not reasoning_input.candidates:
            return PolicyOutput(action=None, rationale="No candidate actions available.")

        if len(reasoning_input.candidates) == 1:
            self.last_request_body = None
            self.last_response_payload = None
            self.last_cache_hit = None
            self.last_cache_path = None
            self.last_cache_fingerprint = None
            self.last_validation_attempt_count = 0
            only = reasoning_input.candidates[0]
            only.metadata["visual_theme_direction_skipped_reason"] = "single_candidate"
            only.metadata["visual_theme_trajectory_memory_enabled"] = False
            only.metadata["visual_theme_policy_mode"] = "stateless_nearest_heading"
            return PolicyOutput(
                action=only,
                rationale=(
                    "Visual direction call skipped: single candidate; "
                    "trajectory_memory_enabled=false."
                ),
            )

        views = list(reasoning_input.views)
        if len(views) != self.expected_view_count:
            raise ValueError(
                "Visual-theme direction requires exactly "
                f"{self.expected_view_count} views, received {len(views)}."
            )
        if not self.model_client.is_configured():
            raise RuntimeError("Visual-theme direction model is not configured.")

        view_ids = [f"view_{index}" for index in range(len(views))]
        subgoal_theme = self._subgoal_theme(reasoning_input.subgoal_room_id)
        request_body = self._build_request_body(
            views=views,
            view_ids=view_ids,
            subgoal_theme=subgoal_theme,
        )
        parsed = self._load_or_create_validated(
            request_body=request_body,
            view_ids=view_ids,
            current_pano_id=reasoning_input.current_pano_id,
        )
        selected_view_id = str(parsed["selected_view_id"])
        selected_index = view_ids.index(selected_view_id)
        selected_view = views[selected_index]
        confidence = float(parsed["confidence"])
        visual_evidence = str(parsed["visual_evidence"]).strip()

        for candidate in reasoning_input.candidates:
            error = self._angular_distance(selected_view.heading, candidate.absolute_heading)
            candidate.metadata["target_relative_diff_deg"] = error
            candidate.metadata["target_heading_source"] = "visual_theme_selected_view"
            candidate.metadata["visual_theme_selected_view_id"] = selected_view_id
            candidate.metadata["visual_theme_selected_view_heading_deg"] = float(
                selected_view.heading
            )
            candidate.metadata["visual_theme_direction_confidence"] = confidence
            candidate.metadata["visual_theme_direction_evidence"] = visual_evidence
            candidate.metadata["visual_theme_direction_error_deg"] = error
            candidate.metadata["visual_theme_direction_cache_hit"] = self.last_cache_hit
            candidate.metadata["visual_theme_direction_validation_attempt_count"] = (
                self.last_validation_attempt_count
            )
            candidate.metadata["visual_theme_trajectory_memory_enabled"] = False
            candidate.metadata["visual_theme_policy_mode"] = "stateless_nearest_heading"

        output = self._choose_nearest_heading_action(reasoning_input)
        return PolicyOutput(
            action=output.action,
            rationale=(
                f"VLM selected {selected_view_id} with confidence={confidence:.3f}; "
                f"trajectory_memory_enabled=false. {output.rationale}"
            ),
        )

    def _choose_nearest_heading_action(
        self,
        reasoning_input: ReasoningInput,
    ) -> PolicyOutput:
        """Select only by VLM-view heading error; do not inspect movement history."""
        ranked_candidates = sorted(
            reasoning_input.candidates,
            key=lambda item: (
                self._target_direction_error(item),
                self._candidate_index(item),
            ),
        )
        best_action = ranked_candidates[0]
        rationale = (
            f"Selected candidate_index={self._candidate_index(best_action)}, "
            f"target_relative_diff={self._target_direction_error(best_action):.1f}, "
            "policy_mode=stateless_nearest_heading"
        )
        if best_action.reason:
            rationale = f"{rationale} ({best_action.reason})"
        return PolicyOutput(action=best_action, rationale=rationale)

    def _subgoal_theme(self, subgoal_room_id: str | None) -> dict:
        node = self.room_graph.get(subgoal_room_id or "", {})
        if not isinstance(node, dict):
            node = {}
        profile = compact_visual_profile(node)
        return {
            "title": node.get("title"),
            "category": node.get("category"),
            "short_description": profile.get("short_description"),
            "visual_cues": list(profile.get("visual_cues", [])),
        }

    @staticmethod
    def _angular_distance(first: float, second: float) -> float:
        return abs((float(first) - float(second) + 180.0) % 360.0 - 180.0)

    def _build_request_body(
        self,
        *,
        views: list[RenderedView],
        view_ids: list[str],
        subgoal_theme: dict,
    ) -> dict:
        content: list[dict] = [
            {
                "type": "input_text",
                "text": build_visual_theme_direction_input(
                    view_ids=view_ids,
                    subgoal_theme=subgoal_theme,
                ),
            }
        ]
        for view_id, view in zip(view_ids, views):
            content.extend(
                [
                    {"type": "input_text", "text": f"View id: {view_id}."},
                    {
                        "type": "input_image",
                        "image_url": self._image_to_data_url(Path(view.path)),
                        "detail": "high",
                    },
                ]
            )
        request_body = {
            "model": self.model,
            "instructions": build_visual_theme_direction_instructions(),
            "input": [{"role": "user", "content": content}],
            "reasoning": {"effort": "low"},
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "visual_theme_direction",
                    "strict": True,
                    "schema": build_visual_theme_direction_schema(view_ids),
                }
            },
        }
        if self.max_output_tokens is not None:
            request_body["max_output_tokens"] = self.max_output_tokens
        return request_body

    def _load_or_create_validated(
        self,
        *,
        request_body: dict,
        view_ids: list[str],
        current_pano_id: str | None,
    ) -> dict:
        fingerprint = request_fingerprint(
            namespace=VISUAL_THEME_DIRECTION_CACHE_NAMESPACE,
            cache_version=VISUAL_THEME_DIRECTION_CACHE_VERSION,
            request_body=request_body,
            transport_identity=model_transport_identity(
                model=self.model,
                model_client=self.model_client,
            ),
        )
        cache_path = (
            cache_entry_path(
                self.cache_dir,
                fingerprint,
                group=current_pano_id,
            )
            if self.cache_dir is not None
            else None
        )
        self.last_cache_fingerprint = fingerprint
        self.last_cache_path = str(cache_path) if cache_path is not None else None

        if cache_path is None:
            return self._create_validated(request_body=request_body, view_ids=view_ids)

        with exclusive_cache_lock(cache_path):
            cached = read_json_object(cache_path)
            if isinstance(cached, dict) and cached.get("fingerprint") == fingerprint:
                parsed = cached.get("parsed")
                if isinstance(parsed, dict):
                    try:
                        self._validate_output(parsed, view_ids=view_ids)
                    except ValueError:
                        pass
                    else:
                        self.last_cache_hit = True
                        self.last_validation_attempt_count = int(
                            cached.get("validation_attempt_count") or 1
                        )
                        response = cached.get("response")
                        self.last_response_payload = (
                            self._clone_json(response) if isinstance(response, dict) else None
                        )
                        self.last_request_body = self._redact_request_body(request_body)
                        return self._clone_json(parsed)

            parsed = self._create_validated(request_body=request_body, view_ids=view_ids)
            atomic_write_json(
                cache_path,
                {
                    "schema": "visual_theme_direction_cache",
                    "cache_version": VISUAL_THEME_DIRECTION_CACHE_VERSION,
                    "fingerprint": fingerprint,
                    "request_sha256": canonical_json_sha256(request_body),
                    "validation_attempt_count": self.last_validation_attempt_count,
                    "parsed": parsed,
                    "response": self.last_response_payload,
                },
            )
            self.last_cache_hit = False
            return parsed

    def _create_validated(self, *, request_body: dict, view_ids: list[str]) -> dict:
        previous_error: ValueError | None = None
        for attempt in range(1, self.schema_validation_retries + 2):
            current_request = (
                request_body
                if attempt == 1
                else self._repair_request(request_body, view_ids=view_ids, error=previous_error)
            )
            self.last_request_body = self._redact_request_body(current_request)
            payload = self.model_client.create(current_request)
            self.last_response_payload = self._clone_json(payload)
            self.last_validation_attempt_count = attempt
            try:
                parsed = parse_json_output(payload)
                self._validate_output(parsed, view_ids=view_ids)
            except ValueError as error:
                previous_error = error
                if attempt > self.schema_validation_retries:
                    raise
                continue
            self.last_cache_hit = False if self.cache_dir is not None else None
            return parsed
        raise RuntimeError("Visual-theme direction validation ended without a result.")

    @staticmethod
    def _repair_request(request_body: dict, *, view_ids: list[str], error: ValueError | None) -> dict:
        repaired = json.loads(json.dumps(request_body))
        repair = (
            "Schema repair retry: return exactly one JSON object with one selected_view_id, "
            "one numeric confidence from 0 to 1, and one short visual_evidence string. "
            f"selected_view_id must be exactly one of {json.dumps(view_ids)}. "
            "Do not return a list, ranking, repeated choices, markdown, or commentary."
        )
        if error is not None:
            repair += f" Previous validator error: {str(error)[:500]}"
        instructions = repaired.get("instructions")
        repaired["instructions"] = (
            f"{instructions}\n\n{repair}" if isinstance(instructions, str) else repair
        )
        return repaired

    @staticmethod
    def _validate_output(parsed: dict, *, view_ids: list[str]) -> None:
        if set(parsed) != {"selected_view_id", "confidence", "visual_evidence"}:
            raise ValueError("Visual-theme direction output has unexpected fields.")
        selected = parsed.get("selected_view_id")
        confidence = parsed.get("confidence")
        evidence = parsed.get("visual_evidence")
        if selected not in view_ids:
            raise ValueError("Visual-theme direction selected an unknown view id.")
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not math.isfinite(float(confidence))
            or not 0.0 <= float(confidence) <= 1.0
        ):
            raise ValueError("Visual-theme direction confidence is invalid.")
        if not isinstance(evidence, str) or not evidence.strip() or len(evidence) > 240:
            raise ValueError("Visual-theme direction evidence is invalid.")

    @staticmethod
    def _image_to_data_url(path: Path) -> str:
        if not path.is_file():
            raise FileNotFoundError(f"Direction-selection image does not exist: {path}")
        mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
        encoded = b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"

    @staticmethod
    def _redact_request_body(request_body: dict) -> dict:
        redacted = json.loads(json.dumps(request_body))
        for item in redacted.get("input", []):
            if not isinstance(item, dict):
                continue
            for content in item.get("content", []):
                if isinstance(content, dict) and content.get("type") == "input_image":
                    content["image_url"] = "<redacted-image-data>"
        return redacted

    @staticmethod
    def _clone_json(payload: object) -> dict:
        return json.loads(json.dumps(payload))
