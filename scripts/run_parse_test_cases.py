from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from st_nav import GroundingIndex, InstructionRoutePlanner, LLMInstructionParser, SpatialEngine
from st_nav.env import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the four navigation parse test instructions through the LLM parser."
    )
    parser.add_argument("--artifacts-dir", default="dataset/sites/british_museum/normalized")
    parser.add_argument(
        "--cases-path",
        default="tests/fixtures/navigation_parse_instructions.json",
    )
    parser.add_argument(
        "--llm-api-key",
        default=os.environ.get("OPENAI_API_KEY"),
        help="Defaults to OPENAI_API_KEY.",
    )
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
    pano_graph = load_json(artifacts_dir / "pano_graph.json")
    grounding = load_json(artifacts_dir / "room_grounding.template.json")
    cases = load_json((PROJECT_ROOT / args.cases_path).resolve())

    planner = InstructionRoutePlanner(
        instruction_parser=LLMInstructionParser(
            room_graph=room_graph,
            api_key=args.llm_api_key,
            model=args.llm_model,
        ),
        spatial_engine=SpatialEngine(
            room_graph=room_graph,
            pano_graph=pano_graph,
            grounding_index=GroundingIndex(grounding),
        ),
    )

    results = {}
    for case_name, case in cases.items():
        instruction = case["instruction"]
        plan = planner.plan(instruction)
        results[case_name] = {
            "instruction": instruction,
            "task_type": plan.task.task_type if plan.task else None,
            "source_room_id": plan.source_room_id,
            "target_room_id": plan.target_room_id,
            "waypoint_room_ids": plan.waypoint_room_ids,
            "shortest_path": plan.shortest_path,
        }

    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
