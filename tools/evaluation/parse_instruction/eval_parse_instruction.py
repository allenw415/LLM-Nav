from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from st_nav import load_dotenv, resolve_model_environment, resolve_task_num_ctx
from st_nav.cli._common import PROJECT_ROOT, load_normalized_artifacts, resolve_project_path
from st_nav.decision.instruction_parser import LLMInstructionParser

load_dotenv(PROJECT_ROOT / ".env")


DEFAULT_CASES = [
    "Find the way from Room 4 to Room 23.",
    "Find the way from Room 8 to Room 23.",
    "Find the way from the Lamassu to the Townley Venus.",
    "Find the way from the Nereid Monument to Room 18.",
    "Find the way from Room 4, passing Room 7 and Room 17, to Room 23.",
    "Find the way from Room 8, passing Room 9, to Room 23.",
    "Find the way from the Bronze Container for Cosmetic Items, passing the Lamassu and the Nereid Monument, to the Townley Venus.",
    "Find the way from the Lamassu, passing Room 8, to Room 23.",
    "Find the way from Room 6, passing the Lamassu, to the Townley Venus.",
    "Find the way from the Townley Venus, passing Room 17, to the Lamassu.",
]

MODEL_ENV = resolve_model_environment(
    default_model="gpt-5-mini",
    default_api_base="https://api.openai.com/v1",
    default_api_kind="responses",
)
DEFAULT_LLM_NUM_CTX = resolve_task_num_ctx(
    "parse_instruction",
    fallback_num_ctx=MODEL_ENV.num_ctx,
    default_num_ctx=8192,
)
TOKEN_FIELDS = ("input_tokens", "output_tokens", "total_tokens", "reasoning_tokens")
SUMMARY_TOKEN_KEYS = {
    "input_tokens": ("total_input_tokens", "average_input_tokens"),
    "output_tokens": ("total_output_tokens", "average_output_tokens"),
    "total_tokens": ("total_tokens", "average_total_tokens"),
    "reasoning_tokens": ("total_reasoning_tokens", "average_reasoning_tokens"),
}


def serialize_task(task) -> dict:
    return {
        "task_type": task.task_type,
        "source_room_id": task.source_room_id,
        "source_entity": task.source_entity.name if task.source_entity else None,
        "goal_room_ids": task.goal_room_ids,
        "waypoint_room_ids": task.waypoint_room_ids,
        "goal_entities": [entity.name for entity in task.goal_entities],
        "waypoint_entities": [entity.name for entity in task.waypoint_entities],
    }


def _as_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _clone_json(value: object) -> object:
    return json.loads(json.dumps(value))


def extract_token_usage(payload: object) -> dict[str, object]:
    usage = payload.get("usage") if isinstance(payload, dict) else None
    input_tokens = output_tokens = total_tokens = reasoning_tokens = None
    raw_usage: dict[str, object] | None = None

    if isinstance(usage, dict):
        raw_usage = _clone_json(usage)
        input_tokens = _as_int(usage.get("input_tokens"))
        if input_tokens is None:
            input_tokens = _as_int(usage.get("prompt_tokens"))

        output_tokens = _as_int(usage.get("output_tokens"))
        if output_tokens is None:
            output_tokens = _as_int(usage.get("completion_tokens"))

        total_tokens = _as_int(usage.get("total_tokens"))
        reasoning_tokens = _as_int(usage.get("reasoning_tokens"))
        if reasoning_tokens is None:
            output_details = usage.get("output_tokens_details")
            if isinstance(output_details, dict):
                reasoning_tokens = _as_int(output_details.get("reasoning_tokens"))
        if reasoning_tokens is None:
            completion_details = usage.get("completion_tokens_details")
            if isinstance(completion_details, dict):
                reasoning_tokens = _as_int(completion_details.get("reasoning_tokens"))
    elif isinstance(payload, dict):
        prompt_eval_count = _as_int(payload.get("prompt_eval_count"))
        eval_count = _as_int(payload.get("eval_count"))
        if prompt_eval_count is not None or eval_count is not None:
            input_tokens = prompt_eval_count
            output_tokens = eval_count
            raw_usage = {
                "prompt_eval_count": prompt_eval_count,
                "eval_count": eval_count,
            }

    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "reasoning_tokens": reasoning_tokens,
        "raw_usage": raw_usage,
    }


def run_cases(parser: LLMInstructionParser, *, cases: list[str]) -> list[dict]:
    results: list[dict] = []
    for instruction in cases:
        parser.last_request_body = None
        parser.last_response_payload = None
        start = time.perf_counter()
        try:
            task = parser.parse(instruction)
            elapsed = time.perf_counter() - start
            usage = extract_token_usage(parser.last_response_payload)
            results.append(
                {
                    "instruction": instruction,
                    "elapsed_seconds": round(elapsed, 3),
                    "result": serialize_task(task),
                    **usage,
                }
            )
        except Exception as exc:
            elapsed = time.perf_counter() - start
            usage = extract_token_usage(parser.last_response_payload)
            results.append(
                {
                    "instruction": instruction,
                    "elapsed_seconds": round(elapsed, 3),
                    "error": str(exc),
                    **usage,
                }
            )
    return results


def _sum_token_field(records: list[dict], field: str) -> int | None:
    values = [_as_int(record.get(field)) for record in records]
    filtered = [value for value in values if value is not None]
    if not filtered:
        return None
    return sum(filtered)


def _average_token_field(records: list[dict], field: str) -> float | None:
    values = [_as_int(record.get(field)) for record in records]
    filtered = [value for value in values if value is not None]
    if not filtered:
        return None
    return round(sum(filtered) / len(filtered), 3)


def summarize_results(results: list[dict]) -> dict:
    successes = [record for record in results if isinstance(record.get("result"), dict)]
    errors = [record for record in results if record.get("error")]
    latencies = [
        float(record["elapsed_seconds"])
        for record in results
        if isinstance(record.get("elapsed_seconds"), (int, float))
    ]
    task_type_counts: dict[str, int] = {}
    for record in successes:
        task_type = record["result"].get("task_type")
        if isinstance(task_type, str) and task_type:
            task_type_counts[task_type] = task_type_counts.get(task_type, 0) + 1

    summary = {
        "success_count": len(successes),
        "error_count": len(errors),
        "average_latency_seconds": round(sum(latencies) / len(latencies), 3) if latencies else None,
        "task_type_counts": dict(sorted(task_type_counts.items())),
    }
    for field in TOKEN_FIELDS:
        total_key, average_key = SUMMARY_TOKEN_KEYS[field]
        summary[total_key] = _sum_token_field(results, field)
        summary[average_key] = _average_token_field(results, field)
    return summary


def write_text(path_value: str, text: str) -> Path:
    path = resolve_project_path(path_value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _markdown_list(values: object) -> str:
    if not isinstance(values, list) or not values:
        return "-"
    return ", ".join(f"`{value}`" for value in values)


def render_report(payload: dict) -> str:
    config = payload.get("config", {}) if isinstance(payload.get("config"), dict) else {}
    summary = payload.get("summary", {}) if isinstance(payload.get("summary"), dict) else {}
    results = payload.get("results", []) if isinstance(payload.get("results"), list) else []
    lines = [
        "# `parse_instruction` Evaluation Report",
        "",
        f"Generated at: `{payload.get('generated_at')}`",
        "",
        "## Config",
        "",
        f"- Parser: `{payload.get('parser')}`",
        f"- Active profile: `{config.get('active_profile') or '-'}`",
        f"- Model: `{config.get('model') or '-'}`",
        f"- API base: `{config.get('api_base') or '-'}`",
        f"- API kind: `{config.get('api_kind') or '-'}`",
        f"- Effective num ctx: `{config.get('effective_num_ctx') or '-'}`",
        f"- Artifacts dir: `{config.get('artifacts_dir') or '-'}`",
        "",
        "## Summary",
        "",
        f"- Case count: `{payload.get('case_count')}`",
        f"- Success count: `{summary.get('success_count')}`",
        f"- Error count: `{summary.get('error_count')}`",
        f"- Average latency seconds: `{summary.get('average_latency_seconds')}`",
        f"- Average input tokens: `{summary.get('average_input_tokens')}`",
        f"- Average output tokens: `{summary.get('average_output_tokens')}`",
        f"- Average total tokens: `{summary.get('average_total_tokens')}`",
        f"- Average reasoning tokens: `{summary.get('average_reasoning_tokens')}`",
        f"- Total input tokens: `{summary.get('total_input_tokens')}`",
        f"- Total output tokens: `{summary.get('total_output_tokens')}`",
        f"- Total tokens: `{summary.get('total_tokens')}`",
        f"- Total reasoning tokens: `{summary.get('total_reasoning_tokens')}`",
        "",
        "## Task Type Counts",
        "",
    ]
    task_type_counts = summary.get("task_type_counts")
    if isinstance(task_type_counts, dict) and task_type_counts:
        lines.extend(["| `task_type` | Count |", "|---|---:|"])
        for task_type, count in task_type_counts.items():
            lines.append(f"| `{task_type}` | {count} |")
    else:
        lines.append("No successful task types recorded.")
    lines.extend([
        "",
        "## Cases",
        "",
        "| # | `instruction` | `task_type` | Source | Goals | Waypoints | Latency | Input Tokens | Output Tokens | Total Tokens | Reasoning Tokens | Error |",
        "|---:|---|---|---|---|---|---:|---:|---:|---:|---:|---|",
    ])
    for index, record in enumerate(results, start=1):
        result = record.get("result") if isinstance(record.get("result"), dict) else {}
        error = str(record.get("error") or "").replace("|", "\\|")
        instruction = str(record.get("instruction") or "").replace("|", "\\|")
        lines.append(
            "| "
            f"{index} | "
            f"`{instruction}` | "
            f"`{result.get('task_type') or '-'}` | "
            f"`{result.get('source_room_id') or '-'}` | "
            f"{_markdown_list(result.get('goal_room_ids'))} | "
            f"{_markdown_list(result.get('waypoint_room_ids'))} | "
            f"{record.get('elapsed_seconds')} | "
            f"{record.get('input_tokens') if record.get('input_tokens') is not None else '-'} | "
            f"{record.get('output_tokens') if record.get('output_tokens') is not None else '-'} | "
            f"{record.get('total_tokens') if record.get('total_tokens') is not None else '-'} | "
            f"{record.get('reasoning_tokens') if record.get('reasoning_tokens') is not None else '-'} | "
            f"{error or '-'} |"
        )
    lines.extend([
        "",
        "## Raw Result",
        "",
        "Full JSON output is available from the matching `--output-path` file when provided.",
    ])
    return "\n".join(lines) + "\n"


def main() -> int:
    arg_parser = argparse.ArgumentParser(description="Evaluate the runtime parse_instruction parser on a fixed or user-provided instruction set.")
    arg_parser.add_argument("--artifacts-dir", default="dataset/sites/british_museum/normalized")
    arg_parser.add_argument("--instruction", action="append", dest="instructions")
    arg_parser.add_argument("--llm-model")
    arg_parser.add_argument("--llm-api-key")
    arg_parser.add_argument("--llm-api-base")
    arg_parser.add_argument("--llm-api-kind")
    arg_parser.add_argument("--llm-timeout", type=float)
    arg_parser.add_argument("--llm-num-ctx", type=int, default=DEFAULT_LLM_NUM_CTX)
    arg_parser.add_argument("--output-path", help="Optional path for the full JSON evaluation payload.")
    arg_parser.add_argument("--report-path", help="Optional path for a generated Markdown report.")
    args = arg_parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")
    artifacts = load_normalized_artifacts(args.artifacts_dir, room_graph=True)
    cases = args.instructions or list(DEFAULT_CASES)
    parser = LLMInstructionParser(
        room_graph=artifacts.room_graph or {},
        model=args.llm_model,
        api_key=args.llm_api_key,
        api_base=args.llm_api_base,
        api_kind=args.llm_api_kind,
        request_timeout=args.llm_timeout,
        num_ctx=args.llm_num_ctx,
    )

    results = run_cases(parser, cases=cases)
    payload = {
        "parser": "runtime",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "case_count": len(cases),
        "config": {
            "active_profile": os.environ.get("ST_NAV_ACTIVE_PROFILE") or os.environ.get("ST_NAV_PROFILE"),
            "artifacts_dir": args.artifacts_dir,
            "model": parser.model,
            "api_base": parser.api_base,
            "api_kind": parser.api_kind,
            "request_timeout": parser.request_timeout,
            "effective_num_ctx": parser.num_ctx,
        },
        "summary": summarize_results(results),
        "results": results,
    }
    output_text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output_path:
        write_text(args.output_path, output_text + "\n")
    if args.report_path:
        write_text(args.report_path, render_report(payload))
    print(output_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
