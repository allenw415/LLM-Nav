from __future__ import annotations

import argparse
import os
from _common import PROJECT_ROOT, ensure_project_root_on_path, load_normalized_artifacts, render_json

ensure_project_root_on_path()

from st_nav import LLMInstructionParser, load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

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
    parser.add_argument("--llm-api-key", default=os.environ.get("OPENAI_API_KEY"))
    parser.add_argument("--llm-model", default="gpt-5-mini")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not args.llm_api_key:
        raise RuntimeError("Missing OPENAI_API_KEY.")

    artifacts = load_normalized_artifacts(args.artifacts_dir, room_graph=True)
    parser = LLMInstructionParser(
        room_graph=artifacts.room_graph or {},
        api_key=args.llm_api_key,
        model=args.llm_model,
    )
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
