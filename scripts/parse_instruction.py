from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from st_nav import LLMInstructionParser
from st_nav.env import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Parse a navigation instruction with the LLM parser.")
    parser.add_argument("--artifacts-dir", default="dataset/sites/british_museum/normalized")
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--llm-api-key", default=os.environ.get("OPENAI_API_KEY"))
    parser.add_argument("--llm-model", default="gpt-5-mini")
    return parser


def load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict JSON: {path}")
    return payload


def main() -> int:
    args = build_parser().parse_args()
    if not args.llm_api_key:
        raise RuntimeError("Missing OPENAI_API_KEY.")

    artifacts_dir = (PROJECT_ROOT / args.artifacts_dir).resolve()
    room_graph = load_json(artifacts_dir / "room_graph.json")
    parser = LLMInstructionParser(
        room_graph=room_graph,
        api_key=args.llm_api_key,
        model=args.llm_model,
    )
    task = parser.parse(args.instruction)

    print(
        json.dumps(
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
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
