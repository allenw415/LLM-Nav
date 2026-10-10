from __future__ import annotations

import base64
import json
import os
import socket
import time
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Callable


DEFAULT_OPENAI_API_BASE = "https://api.openai.com/v1"
DEFAULT_GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_ANTHROPIC_API_BASE = "https://api.anthropic.com/v1"
DEFAULT_ANTHROPIC_MAX_TOKENS = 4096
ANTHROPIC_VERSION = "2023-06-01"
SUPPORTED_API_KINDS = {"responses", "chat_completions", "anthropic_messages"}
TRANSIENT_HTTP_STATUS_CODES = {429, 500, 502, 503, 504}
MAX_HTTP_RETRIES = 5
HTTP_RETRY_BACKOFF_SECONDS = 1.0
MODEL_TELEMETRY_COUNTER_KEYS = (
    "logical_call_count",
    "successful_logical_call_count",
    "failed_logical_call_count",
    "image_logical_call_count",
    "text_logical_call_count",
    "http_attempt_count",
    "successful_http_attempt_count",
    "failed_http_attempt_count",
    "image_http_attempt_count",
    "text_http_attempt_count",
)


def resolve_api_kind(api_kind: str | None = None) -> str:
    value = (api_kind or os.environ.get("SA_NAV_API_KIND") or "responses").strip().lower()
    if value not in SUPPORTED_API_KINDS:
        supported = ", ".join(sorted(SUPPORTED_API_KINDS))
        raise ValueError(f"Unsupported API kind: {value}. Expected one of: {supported}.")
    return value


def extract_output_text(payload: dict) -> str | None:
    output_text = payload.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text

    message = payload.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content

    candidate_text = _extract_gemini_candidate_text(payload)
    if isinstance(candidate_text, str) and candidate_text.strip():
        return candidate_text

    anthropic_text = _extract_anthropic_content_text(payload)
    if isinstance(anthropic_text, str) and anthropic_text.strip():
        return anthropic_text

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
    if fragments:
        return "".join(fragments)

    choices = payload.get("choices")
    if not isinstance(choices, list):
        return None
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message")
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for part in content:
                if not isinstance(part, dict):
                    continue
                text = part.get("text")
                if isinstance(text, str):
                    parts.append(text)
            if parts:
                return "".join(parts)
    return None


def parse_json_output(payload: dict) -> dict:
    output_text = extract_output_text(payload)
    if not isinstance(output_text, str) or not output_text.strip():
        raise ValueError("Model API payload did not include output text.")
    return parse_json_text(output_text)


def parse_json_text(text: str) -> dict:
    stripped = text.strip()
    candidates = [stripped]

    fenced = _extract_fenced_code_block(stripped)
    if fenced and fenced not in candidates:
        candidates.append(fenced)

    bracketed = _extract_first_balanced_json(stripped)
    if bracketed and bracketed not in candidates:
        candidates.append(bracketed)

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    preview = stripped[:400].replace("\n", "\\n")
    raise ValueError(f"Model returned non-JSON output: {preview}")


def _extract_fenced_code_block(text: str) -> str | None:
    marker = "```"
    start = text.find(marker)
    if start < 0:
        return None
    end = text.find(marker, start + len(marker))
    if end < 0:
        return None
    body = text[start + len(marker):end].strip()
    if "\n" in body:
        first_line, rest = body.split("\n", 1)
        if first_line.strip().lower() in {"json", "javascript", "js"}:
            return rest.strip()
    return body.strip()


def _extract_first_balanced_json(text: str) -> str | None:
    openings = {"{": "}", "[": "]"}
    for index, char in enumerate(text):
        if char not in openings:
            continue
        closing = openings[char]
        depth = 0
        in_string = False
        escaped = False
        for end_index in range(index, len(text)):
            current = text[end_index]
            if in_string:
                if escaped:
                    escaped = False
                elif current == "\\":
                    escaped = True
                elif current == '"':
                    in_string = False
                continue
            if current == '"':
                in_string = True
            elif current == char:
                depth += 1
            elif current == closing:
                depth -= 1
                if depth == 0:
                    return text[index : end_index + 1].strip()
    return None


class ModelResponseClient:
    """
    Shared transport wrapper for hosted closed-source APIs and self-hosted
    OpenAI-compatible servers.
    """

    def __init__(
        self,
        *,
        provider: str | None = None,
        api_key: str | None = None,
        api_base: str = DEFAULT_OPENAI_API_BASE,
        api_kind: str | None = None,
        request_timeout: float = 30.0,
        num_ctx: int | None = None,
        temperature: float | None = None,
        think: bool | str | None = None,
        response_client: Callable[[dict], dict] | None = None,
        max_http_retries: int | None = None,
        retry_backoff_seconds: float | None = None,
    ):
        self.provider = (provider or "").strip().lower() or None
        self.api_key = api_key
        self.api_base = api_base.rstrip("/")
        self.api_kind = resolve_api_kind(api_kind)
        self.request_timeout = request_timeout
        self.num_ctx = num_ctx
        self.temperature = temperature
        self.think = think
        self.response_client = response_client
        self.max_http_retries = max(1, int(max_http_retries or _parse_int_env("SA_NAV_MAX_HTTP_RETRIES", MAX_HTTP_RETRIES)))
        self.retry_backoff_seconds = max(
            0.0,
            float(retry_backoff_seconds if retry_backoff_seconds is not None else _parse_float_env(
                "SA_NAV_HTTP_RETRY_BACKOFF_SECONDS",
                HTTP_RETRY_BACKOFF_SECONDS,
            )),
        )
        self._telemetry = {key: 0 for key in MODEL_TELEMETRY_COUNTER_KEYS}
        self.last_call_wall_ms: float | None = None
        self.last_call_succeeded: bool | None = None

    def is_configured(self) -> bool:
        return self.response_client is not None or bool(self.api_key) or self.api_base != DEFAULT_OPENAI_API_BASE

    def telemetry_snapshot(self) -> dict[str, int]:
        """Return cumulative, secret-free request counters for this client."""
        return dict(self._telemetry)

    def create(self, request_body: dict) -> dict:
        call_started = time.perf_counter()
        has_images = self._request_contains_images(request_body)
        self._telemetry["logical_call_count"] += 1
        self._telemetry["image_logical_call_count" if has_images else "text_logical_call_count"] += 1
        succeeded = False
        try:
            if self.response_client is not None:
                result = self._normalize_payload(self.response_client(request_body))
                succeeded = True
                return result

            if self.provider == "anthropic" or self.api_kind == "anthropic_messages":
                if not self.api_key:
                    raise RuntimeError("Missing ANTHROPIC_API_KEY or SA_NAV_API_KEY for Anthropic provider.")
                endpoint = f"{self.api_base}/messages"
                payload = self._responses_to_anthropic_messages_payload(request_body)
            elif self.provider in {"gemini", "gemini_api", "google_gemma_api"}:
                if not self.api_key:
                    raise RuntimeError("Missing GEMINI_API_KEY, GOOGLE_API_KEY, or SA_NAV_API_KEY for Gemini API provider.")
                endpoint = self._gemini_endpoint(request_body)
                payload = self._responses_to_gemini_generate_content_payload(request_body)
            elif self.provider == "ollama":
                endpoint = f"{self._ollama_api_base()}/api/chat"
                payload = self._responses_to_ollama_chat_payload(request_body)
            elif self.api_kind == "responses":
                endpoint = f"{self.api_base}/responses"
                payload = request_body
            else:
                endpoint = f"{self.api_base}/chat/completions"
                payload = self._responses_to_chat_completions_payload(request_body)

            headers = {"Content-Type": "application/json"}
            if self.provider == "anthropic" or self.api_kind == "anthropic_messages":
                headers["x-api-key"] = self.api_key
                headers["anthropic-version"] = ANTHROPIC_VERSION
                workspace_id = os.environ.get("ANTHROPIC_WORKSPACE_ID", "").strip()
                if workspace_id:
                    headers["anthropic-workspace-id"] = workspace_id
            elif self.provider in {"gemini", "gemini_api", "google_gemma_api"}:
                headers["x-goog-api-key"] = self.api_key
            elif self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            request = urllib.request.Request(
                endpoint,
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            for attempt in range(1, self.max_http_retries + 1):
                self._record_http_attempt(has_images=has_images)
                try:
                    with urllib.request.urlopen(request, timeout=self.request_timeout) as response:
                        result = self._normalize_payload(json.loads(response.read().decode("utf-8")))
                    self._telemetry["successful_http_attempt_count"] += 1
                    succeeded = True
                    return result
                except TimeoutError as exc:
                    self._telemetry["failed_http_attempt_count"] += 1
                    if attempt < self.max_http_retries:
                        self._sleep_before_retry(attempt)
                        continue
                    raise TimeoutError(
                        f"Model API timed out after {self.request_timeout:.0f}s while waiting for {self.api_kind} output."
                    ) from exc
                except socket.timeout as exc:
                    self._telemetry["failed_http_attempt_count"] += 1
                    if attempt < self.max_http_retries:
                        self._sleep_before_retry(attempt)
                        continue
                    raise TimeoutError(
                        f"Model API timed out after {self.request_timeout:.0f}s while waiting for {self.api_kind} output."
                    ) from exc
                except urllib.error.HTTPError as exc:
                    self._telemetry["failed_http_attempt_count"] += 1
                    body = _read_http_error_body(exc)
                    if exc.code in TRANSIENT_HTTP_STATUS_CODES and attempt < self.max_http_retries:
                        self._sleep_before_retry(attempt)
                        continue
                    detail = _format_http_error_detail(body)
                    raise RuntimeError(
                        f"Model API request failed with HTTP {exc.code} from {endpoint}: {exc.reason}.{detail}"
                    ) from exc
                except urllib.error.URLError as exc:
                    self._telemetry["failed_http_attempt_count"] += 1
                    if attempt < self.max_http_retries:
                        self._sleep_before_retry(attempt)
                        continue
                    raise RuntimeError(f"Model API request failed from {endpoint}: {exc.reason}") from exc
                except Exception:
                    self._telemetry["failed_http_attempt_count"] += 1
                    raise
        finally:
            self.last_call_wall_ms = (time.perf_counter() - call_started) * 1000.0
            self.last_call_succeeded = succeeded
            self._telemetry[
                "successful_logical_call_count" if succeeded else "failed_logical_call_count"
            ] += 1

    def _record_http_attempt(self, *, has_images: bool) -> None:
        self._telemetry["http_attempt_count"] += 1
        self._telemetry["image_http_attempt_count" if has_images else "text_http_attempt_count"] += 1

    @staticmethod
    def _request_contains_images(value: object) -> bool:
        if isinstance(value, dict):
            block_type = value.get("type")
            if isinstance(block_type, str) and block_type in {"input_image", "image_url"}:
                return True
            return any(ModelResponseClient._request_contains_images(item) for item in value.values())
        if isinstance(value, list):
            return any(ModelResponseClient._request_contains_images(item) for item in value)
        return False

    def _sleep_before_retry(self, attempt: int) -> None:
        time.sleep(self.retry_backoff_seconds * float(attempt))

    @staticmethod
    def _normalize_payload(payload: dict) -> dict:
        normalized = json.loads(json.dumps(payload))
        output_text = extract_output_text(normalized)
        if isinstance(output_text, str) and output_text.strip():
            normalized["output_text"] = output_text
        return normalized

    def _ollama_api_base(self) -> str:
        if self.api_base.endswith("/v1"):
            return self.api_base[:-3]
        return self.api_base

    def _gemini_endpoint(self, request_body: dict) -> str:
        api_base = self.api_base or DEFAULT_GEMINI_API_BASE
        encoded_model = urllib.parse.quote(str(request_body.get("model")), safe="")
        return f"{api_base.rstrip('/')}/models/{encoded_model}:generateContent"

    @staticmethod
    def _iter_input_messages(input_value: object) -> list[dict]:
        if isinstance(input_value, str):
            return [{"role": "user", "content": input_value}]
        if isinstance(input_value, dict):
            return [input_value]
        if not isinstance(input_value, list):
            return []

        messages: list[dict] = []
        for item in input_value:
            if isinstance(item, dict):
                messages.append(item)
            elif isinstance(item, str):
                messages.append({"role": "user", "content": item})
        return messages

    @staticmethod
    def _responses_to_chat_completions_payload(request_body: dict) -> dict:
        messages: list[dict] = []
        instructions = request_body.get("instructions")
        if isinstance(instructions, str) and instructions.strip():
            messages.append({"role": "system", "content": instructions})

        for item in ModelResponseClient._iter_input_messages(request_body.get("input")):
            role = item.get("role")
            if not isinstance(role, str) or not role:
                role = "user"
            messages.append(
                {
                    "role": role,
                    "content": ModelResponseClient._convert_content_blocks(item.get("content")),
                }
            )

        payload = {
            "model": request_body.get("model"),
            "messages": messages,
        }

        response_format = ModelResponseClient._response_format_from_request_body(request_body)
        if response_format is not None:
            payload["response_format"] = response_format

        return payload

    @staticmethod
    def _responses_to_anthropic_messages_payload(
        request_body: dict,
        *,
        default_max_tokens: int = DEFAULT_ANTHROPIC_MAX_TOKENS,
    ) -> dict:
        messages: list[dict] = []
        for item in ModelResponseClient._iter_input_messages(request_body.get("input")):
            role = item.get("role")
            if role not in {"user", "assistant"}:
                role = "user"
            messages.append(
                {
                    "role": role,
                    "content": ModelResponseClient._convert_content_blocks_for_anthropic(
                        item.get("content")
                    ),
                }
            )

        max_output_tokens = request_body.get("max_output_tokens")
        if not isinstance(max_output_tokens, int) or max_output_tokens <= 0:
            max_output_tokens = int(default_max_tokens)

        payload: dict[str, object] = {
            "model": request_body.get("model"),
            "max_tokens": max_output_tokens,
            "messages": messages,
        }
        instructions = request_body.get("instructions")
        if isinstance(instructions, str) and instructions.strip():
            payload["system"] = instructions

        output_config: dict[str, object] = {}
        reasoning = request_body.get("reasoning")
        if isinstance(reasoning, dict):
            effort = reasoning.get("effort")
            if effort in {"low", "medium", "high", "xhigh", "max"}:
                output_config["effort"] = effort

        response_schema = ModelResponseClient._anthropic_response_schema_from_request_body(
            request_body
        )
        if response_schema is not None:
            output_config["format"] = {
                "type": "json_schema",
                "schema": response_schema,
            }
        if output_config:
            payload["output_config"] = output_config
        return payload

    def _responses_to_ollama_chat_payload(self, request_body: dict) -> dict:
        messages: list[dict] = []
        instructions = request_body.get("instructions")
        if isinstance(instructions, str) and instructions.strip():
            messages.append({"role": "system", "content": instructions})

        for item in self._iter_input_messages(request_body.get("input")):
            role = item.get("role")
            if not isinstance(role, str) or not role:
                role = "user"
            converted = self._convert_content_blocks_for_ollama(item.get("content"))
            messages.append(
                {
                    "role": role,
                    "content": converted["content"],
                    **({"images": converted["images"]} if converted["images"] else {}),
                }
            )

        payload: dict[str, object] = {
            "model": request_body.get("model"),
            "messages": messages,
            "stream": False,
        }
        if self.think is not None:
            payload["think"] = self.think

        ollama_format = self._ollama_format_from_request_body(request_body)
        if ollama_format is not None:
            payload["format"] = ollama_format

        options: dict[str, object] = {}
        if self.temperature is not None:
            options["temperature"] = self.temperature
        elif ollama_format is not None:
            options["temperature"] = 0
        if self.num_ctx is not None:
            options["num_ctx"] = self.num_ctx
        elif ollama_format is not None:
            options["num_ctx"] = 4096
        max_output_tokens = request_body.get("max_output_tokens")
        if isinstance(max_output_tokens, int) and max_output_tokens > 0:
            options["num_predict"] = max_output_tokens
        if options:
            payload["options"] = options
        return payload

    def _responses_to_gemini_generate_content_payload(self, request_body: dict) -> dict:
        contents: list[dict] = []
        instructions = request_body.get("instructions")
        if isinstance(instructions, str) and instructions.strip():
            contents.append(
                {
                    "role": "user",
                    "parts": [{"text": instructions}],
                }
            )

        for item in self._iter_input_messages(request_body.get("input")):
            role = item.get("role")
            if role == "assistant":
                mapped_role = "model"
            else:
                mapped_role = "user"
            parts = self._convert_content_blocks_for_gemini(item.get("content"))
            contents.append({"role": mapped_role, "parts": parts})

        payload: dict[str, object] = {"contents": contents}

        generation_config: dict[str, object] = {}
        gemini_schema = self._gemini_response_schema_from_request_body(request_body)
        if gemini_schema is not None:
            generation_config["responseMimeType"] = "application/json"
            generation_config["responseJsonSchema"] = gemini_schema
        if self.temperature is not None:
            generation_config["temperature"] = self.temperature
        elif gemini_schema is not None:
            generation_config["temperature"] = 0
        max_output_tokens = request_body.get("max_output_tokens")
        if isinstance(max_output_tokens, int) and max_output_tokens > 0:
            generation_config["maxOutputTokens"] = max_output_tokens
        if generation_config:
            payload["generationConfig"] = generation_config
        return payload

    @staticmethod
    def _convert_content_blocks(content: object) -> object:
        if isinstance(content, str):
            return content
        if not isinstance(content, list):
            return content

        converted: list[dict] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "input_text":
                text = block.get("text")
                if isinstance(text, str):
                    converted.append({"type": "text", "text": text})
            elif block_type == "input_image":
                image_url = block.get("image_url")
                if isinstance(image_url, str) and image_url:
                    image_payload = {"url": image_url}
                    detail = block.get("detail")
                    if isinstance(detail, str) and detail:
                        image_payload["detail"] = detail
                    converted.append({"type": "image_url", "image_url": image_payload})
        return converted

    @staticmethod
    def _convert_content_blocks_for_anthropic(content: object) -> object:
        if isinstance(content, str):
            return content
        if not isinstance(content, list):
            return content

        converted: list[dict] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "input_text":
                value = block.get("text")
                if isinstance(value, str):
                    converted.append({"type": "text", "text": value})
            elif block_type == "input_image":
                image_url = block.get("image_url")
                if isinstance(image_url, str) and image_url:
                    media_type, data = ModelResponseClient._decode_data_url(image_url)
                    converted.append(
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": data,
                            },
                        }
                    )
        return converted

    @staticmethod
    def _convert_content_blocks_for_ollama(content: object) -> dict[str, object]:
        if isinstance(content, str):
            return {"content": content, "images": []}
        if not isinstance(content, list):
            return {"content": "", "images": []}

        text_parts: list[str] = []
        images: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "input_text":
                text = block.get("text")
                if isinstance(text, str) and text:
                    text_parts.append(text)
            elif block_type == "input_image":
                image_url = block.get("image_url")
                if isinstance(image_url, str) and image_url:
                    images.append(ModelResponseClient._data_url_payload(image_url))
        return {"content": "\n".join(text_parts), "images": images}

    @staticmethod
    def _convert_content_blocks_for_gemini(content: object) -> list[dict]:
        if isinstance(content, str):
            return [{"text": content}]
        if not isinstance(content, list):
            return []

        parts: list[dict] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "input_text":
                text = block.get("text")
                if isinstance(text, str) and text:
                    parts.append({"text": text})
            elif block_type == "input_image":
                image_url = block.get("image_url")
                if isinstance(image_url, str) and image_url:
                    mime_type, data = ModelResponseClient._decode_data_url(image_url)
                    parts.append(
                        {
                            "inline_data": {
                                "mime_type": mime_type,
                                "data": data,
                            }
                        }
                    )
        return parts

    @staticmethod
    def _response_format_from_request_body(request_body: dict) -> dict | None:
        text_config = request_body.get("text")
        if not isinstance(text_config, dict):
            return None
        format_config = text_config.get("format")
        if not isinstance(format_config, dict) or format_config.get("type") != "json_schema":
            return None

        name = format_config.get("name")
        schema = format_config.get("schema")
        strict = format_config.get("strict")
        if not isinstance(name, str) or not isinstance(schema, dict):
            return None

        json_schema: dict[str, object] = {"name": name, "schema": schema}
        if isinstance(strict, bool):
            json_schema["strict"] = strict
        return {"type": "json_schema", "json_schema": json_schema}

    @staticmethod
    def _ollama_format_from_request_body(request_body: dict) -> dict | str | None:
        text_config = request_body.get("text")
        if not isinstance(text_config, dict):
            return None
        format_config = text_config.get("format")
        if not isinstance(format_config, dict):
            return None
        if format_config.get("type") != "json_schema":
            return None
        schema = format_config.get("schema")
        if isinstance(schema, dict):
            return schema
        return "json"

    @staticmethod
    def _gemini_response_schema_from_request_body(request_body: dict) -> dict | None:
        text_config = request_body.get("text")
        if not isinstance(text_config, dict):
            return None
        format_config = text_config.get("format")
        if not isinstance(format_config, dict) or format_config.get("type") != "json_schema":
            return None
        schema = format_config.get("schema")
        if isinstance(schema, dict):
            return schema
        return None

    @staticmethod
    def _anthropic_response_schema_from_request_body(request_body: dict) -> dict | None:
        """Return a Claude structured-output compatible copy of the schema.

        Claude's raw JSON-schema endpoint supports a smaller constraint set than
        the Responses-compatible schema used throughout this repository.  Keep
        the original schema intact for local validation, while simplifying only
        the copy sent to Anthropic.
        """

        schema = ModelResponseClient._gemini_response_schema_from_request_body(
            request_body
        )
        if schema is None:
            return None
        return ModelResponseClient._transform_schema_for_anthropic(schema)

    @staticmethod
    def _transform_schema_for_anthropic(schema: dict) -> dict:
        unsupported_constraints = {
            "minimum",
            "maximum",
            "exclusiveMinimum",
            "exclusiveMaximum",
            "multipleOf",
            "minLength",
            "maxLength",
            "maxItems",
            "uniqueItems",
            "contains",
            "minContains",
            "maxContains",
        }
        supported_string_formats = {
            "date-time",
            "time",
            "date",
            "duration",
            "email",
            "hostname",
            "uri",
            "ipv4",
            "ipv6",
            "uuid",
        }

        def enum_value_matches_type(value: object, type_name: str) -> bool:
            if type_name == "null":
                return value is None
            if type_name == "boolean":
                return isinstance(value, bool)
            if type_name == "integer":
                return isinstance(value, int) and not isinstance(value, bool)
            if type_name == "number":
                return isinstance(value, (int, float)) and not isinstance(value, bool)
            if type_name == "string":
                return isinstance(value, str)
            if type_name == "array":
                return isinstance(value, list)
            if type_name == "object":
                return isinstance(value, dict)
            return False

        def transform(node: object) -> object:
            if isinstance(node, list):
                return [transform(item) for item in node]
            if not isinstance(node, dict):
                return node

            transformed: dict[str, object] = {}
            moved_constraints: list[str] = []
            original_types = node.get("type")

            for key, value in node.items():
                if key in unsupported_constraints:
                    moved_constraints.append(f"{key}: {value}")
                    continue
                if key == "minItems" and value not in {0, 1}:
                    moved_constraints.append(f"{key}: {value}")
                    continue
                if key == "format" and value not in supported_string_formats:
                    moved_constraints.append(f"{key}: {value}")
                    continue
                transformed[key] = transform(value)

            # The Claude API currently rejects a single enum attached to a
            # nullable type array because each enum value is checked against
            # every declared type.  Split it into type-specific anyOf branches.
            if isinstance(original_types, list):
                transformed.pop("type", None)
                original_enum = node.get("enum")
                transformed.pop("enum", None)
                branches: list[dict[str, object]] = []
                for type_name in original_types:
                    if not isinstance(type_name, str):
                        continue
                    branch: dict[str, object] = {"type": type_name}
                    if isinstance(original_enum, list):
                        matching_values = [
                            value
                            for value in original_enum
                            if enum_value_matches_type(value, type_name)
                        ]
                        if not matching_values:
                            continue
                        branch["enum"] = transform(matching_values)
                    branches.append(branch)
                if branches:
                    transformed["anyOf"] = branches

            if original_types == "object" or (
                isinstance(original_types, list) and "object" in original_types
            ):
                transformed["additionalProperties"] = False

            if moved_constraints:
                constraint_note = "{" + ", ".join(moved_constraints) + "}"
                existing_description = transformed.get("description")
                if isinstance(existing_description, str) and existing_description.strip():
                    transformed["description"] = (
                        f"{existing_description.rstrip()} {constraint_note}"
                    )
                else:
                    transformed["description"] = constraint_note

            return transformed

        transformed = transform(schema)
        if not isinstance(transformed, dict):  # pragma: no cover - schema is a dict
            raise TypeError("Anthropic schema transformation did not return an object.")
        return transformed

    @staticmethod
    def _data_url_payload(image_url: str) -> str:
        marker = ";base64,"
        if marker in image_url:
            return image_url.split(marker, 1)[1]
        return image_url

    @staticmethod
    def _decode_data_url(image_url: str) -> tuple[str, str]:
        prefix = "data:"
        marker = ";base64,"
        if image_url.startswith(prefix) and marker in image_url:
            header, data = image_url[len(prefix):].split(marker, 1)
            return ModelResponseClient._sniff_image_media_type(data, fallback=header), data
        return ModelResponseClient._sniff_image_media_type(image_url, fallback="image/png"), image_url

    @staticmethod
    def _sniff_image_media_type(data: str, *, fallback: str) -> str:
        try:
            prefix = data[:64]
            prefix += "=" * (-len(prefix) % 4)
            decoded = base64.b64decode(prefix, validate=False)
        except (ValueError, TypeError):
            return fallback
        if decoded.startswith(b"\xff\xd8\xff"):
            return "image/jpeg"
        if decoded.startswith(b"\x89PNG\r\n\x1a\n"):
            return "image/png"
        if decoded.startswith((b"GIF87a", b"GIF89a")):
            return "image/gif"
        if decoded.startswith(b"RIFF") and decoded[8:12] == b"WEBP":
            return "image/webp"
        return fallback


def _extract_gemini_candidate_text(payload: dict) -> str | None:
    fragments: list[str] = []
    candidates = payload.get("candidates")
    if not isinstance(candidates, list):
        return None
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        content = candidate.get("content")
        if not isinstance(content, dict):
            continue
        for part in content.get("parts", []):
            if not isinstance(part, dict):
                continue
            text = part.get("text")
            if isinstance(text, str):
                fragments.append(text)
    if fragments:
        return "".join(fragments)
    return None


def _extract_anthropic_content_text(payload: dict) -> str | None:
    fragments: list[str] = []
    content = payload.get("content")
    if not isinstance(content, list):
        return None
    for block in content:
        if not isinstance(block, dict) or block.get("type") != "text":
            continue
        value = block.get("text")
        if isinstance(value, str):
            fragments.append(value)
    if fragments:
        return "".join(fragments)
    return None


def _read_http_error_body(exc: urllib.error.HTTPError) -> str | None:
    try:
        payload = exc.read()
    except Exception:
        return None
    if not payload:
        return None
    return payload.decode("utf-8", errors="replace")


def _format_http_error_detail(body: str | None) -> str:
    if not body:
        return ""
    compact = " ".join(body.strip().split())
    if len(compact) > 300:
        compact = compact[:300].rstrip() + "..."
    return f" Response body: {compact}"


def _parse_int_env(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _parse_float_env(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    try:
        return float(value)
    except ValueError:
        return default
