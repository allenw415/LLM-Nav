from __future__ import annotations

import argparse
import sys
from ._common import PROJECT_ROOT, ensure_project_root_on_path, load_normalized_artifacts, render_json

ensure_project_root_on_path()

from st_nav import LLMInstructionParser, load_dotenv, resolve_model_environment

load_dotenv(PROJECT_ROOT / ".env")
MODEL_ENV = resolve_model_environment(
    default_model="gpt-5-mini",
    default_api_base="https://api.openai.com/v1",
    default_api_kind="responses",
)

# Example instructions for the four task types used by the parser:
# 1. gallery_goal_navigation:
#    "Find the way from Room 4 to Room 23."
# 2. artwork_goal_navigation:
#    "Find the way from the Lamassu to the Townley Venus."
# 3. gallery_instruction_following_navigation:
#    "Find the way from Room 4, passing Room 7 and Room 17, to Room 23."
# 4. artwork_instruction_following_navigation:
#    "Find the way from the Bronze Container for Cosmetic Items, passing the Lamassu and the Nereid Monument, to the Townley Venus."
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Parse a navigation instruction with the LLM parser.")
    parser.add_argument("--artifacts-dir", default="dataset/sites/british_museum/normalized")
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--debug-request", action="store_true")
    parser.add_argument("--llm-api-key", default=MODEL_ENV.api_key)
    parser.add_argument("--llm-model", default=MODEL_ENV.model_name)
    parser.add_argument("--llm-api-kind", default=MODEL_ENV.api_kind)
    parser.add_argument("--llm-api-base", default=MODEL_ENV.api_base)
    parser.add_argument("--llm-timeout", type=float, default=MODEL_ENV.request_timeout or 30.0)
    return parser


def _resolve_endpoint(parser: LLMInstructionParser, request_body: dict) -> str:
    client = parser.model_client
    if client.provider in {"gemini", "gemini_api", "google_gemma_api"}:
        return client._gemini_endpoint(request_body)
    if client.provider == "ollama":
        return f"{client._ollama_api_base()}/api/chat"
    if client.api_kind == "responses":
        return f"{client.api_base}/responses"
    return f"{client.api_base}/chat/completions"


def _resolve_transport_payload(parser: LLMInstructionParser, request_body: dict) -> dict:
    client = parser.model_client
    if client.provider in {"gemini", "gemini_api", "google_gemma_api"}:
        return client._responses_to_gemini_generate_content_payload(request_body)
    if client.provider == "ollama":
        return client._responses_to_ollama_chat_payload(request_body)
    if client.api_kind == "responses":
        return request_body
    return client._responses_to_chat_completions_payload(request_body)


def main() -> int:
    args = build_parser().parse_args()
    artifacts = load_normalized_artifacts(args.artifacts_dir, room_graph=True)
    parser = LLMInstructionParser(
        room_graph=artifacts.room_graph or {},
        api_key=args.llm_api_key,
        api_base=args.llm_api_base,
        api_kind=args.llm_api_kind,
        model=args.llm_model,
        request_timeout=args.llm_timeout,
    )
    if args.debug_request:
        request_body = parser._build_request_body(args.instruction)
        debug_payload = {
            "provider": parser.model_client.provider,
            "model": parser.model,
            "api_kind": parser.api_kind,
            "api_base": parser.api_base,
            "endpoint": _resolve_endpoint(parser, request_body),
            "request_timeout": parser.request_timeout,
            "has_api_key": bool(parser.api_key),
            "request_body": request_body,
            "transport_payload": _resolve_transport_payload(parser, request_body),
        }
        print(render_json(debug_payload), file=sys.stderr)
    task = parser.parse(args.instruction)

    print(
        render_json(
            {
                "instruction": task.raw_instruction,
                "task_type": task.task_type,
                "source_room_id": task.source_room_id,
                "source_entity": (
                    {
                        "name": task.source_entity.name,
                        "entity_type": task.source_entity.entity_type,
                        "predicted_room_id": task.source_entity.predicted_room_id,
                        "confidence": task.source_entity.confidence,
                    }
                    if task.source_entity
                    else None
                ),
                "goal_room_ids": task.goal_room_ids,
                "waypoint_room_ids": task.waypoint_room_ids,
                "goal_entities": [
                    {
                        "name": entity.name,
                        "entity_type": entity.entity_type,
                        "predicted_room_id": entity.predicted_room_id,
                        "confidence": entity.confidence,
                    }
                    for entity in task.goal_entities
                ],
                "waypoint_entities": [
                    {
                        "name": entity.name,
                        "entity_type": entity.entity_type,
                        "predicted_room_id": entity.predicted_room_id,
                        "confidence": entity.confidence,
                    }
                    for entity in task.waypoint_entities
                ],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
