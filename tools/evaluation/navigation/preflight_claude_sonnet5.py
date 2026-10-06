from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from st_nav import (  # noqa: E402
    AnthropicMessageBatchClient,
    ModelResponseClient,
    atomic_write_batch_checkpoint,
    parse_json_output,
    prepare_message_batch_request,
)
from st_nav.common.env import load_dotenv  # noqa: E402


MODEL_NAME = "claude-sonnet-5"
ANTHROPIC_API_BASE = "https://api.anthropic.com/v1"
DEFAULT_CHECKPOINT = PROJECT_ROOT / (
    "outputs/navigation_evaluation/claude_sonnet5_preflight/batch_checkpoint.json"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate Claude Sonnet 5 native Messages or Message Batches access."
    )
    parser.add_argument(
        "--mode",
        choices=["dry-run", "messages", "batch"],
        default="dry-run",
        help="dry-run needs no API key; messages and batch make live Anthropic requests.",
    )
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--poll-interval", type=float, default=30.0)
    parser.add_argument("--max-wait", type=float, default=24 * 60 * 60)
    return parser


def build_request_body() -> dict:
    return {
        "model": MODEL_NAME,
        "instructions": (
            "Return exactly one JSON object matching the supplied schema. "
            "Do not include markdown or commentary."
        ),
        "input": "Return status=ok and identify the model family as Claude Sonnet 5.",
        "reasoning": {"effort": "low"},
        # Sonnet 5 adaptive thinking counts against max_tokens. Keep enough
        # headroom even though the visible JSON response is tiny.
        "max_output_tokens": 1024,
        "text": {
            "format": {
                "type": "json_schema",
                "name": "claude_sonnet5_preflight",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "status": {"type": "string", "enum": ["ok"]},
                        "model_family": {"type": "string"},
                    },
                    "required": ["status", "model_family"],
                    "additionalProperties": False,
                },
            }
        },
    }


def resolve_anthropic_api_key() -> str | None:
    for name in (
        "ST_NAV_PROFILE_CLAUDE_API_KEY",
        "ANTHROPIC_API_KEY",
    ):
        value = os.environ.get(name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def validate_payload(payload: dict) -> dict:
    parsed = parse_json_output(payload)
    if parsed.get("status") != "ok":
        raise RuntimeError(f"Unexpected Claude preflight response: {parsed}")
    return parsed


def run_messages(*, api_key: str) -> dict:
    client = ModelResponseClient(
        provider="anthropic",
        api_key=api_key,
        api_base=ANTHROPIC_API_BASE,
        api_kind="anthropic_messages",
        request_timeout=300.0,
    )
    return validate_payload(client.create(build_request_body()))


def run_batch(
    *,
    api_key: str,
    checkpoint_path: Path,
    poll_interval_seconds: float,
    max_wait_seconds: float,
) -> dict:
    request = prepare_message_batch_request(
        custom_id="claude_sonnet5_preflight",
        request_body=build_request_body(),
    )
    client = AnthropicMessageBatchClient(
        api_key=api_key,
        api_base=ANTHROPIC_API_BASE,
    )
    batch = client.create([request])
    atomic_write_batch_checkpoint(
        checkpoint_path,
        {
            "schema": "anthropic_message_batch_checkpoint",
            "schema_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "model": MODEL_NAME,
            "batch": batch,
            "custom_ids": [request["custom_id"]],
        },
    )
    ended = client.wait_until_ended(
        str(batch["id"]),
        poll_interval_seconds=poll_interval_seconds,
        max_wait_seconds=max_wait_seconds,
    )
    records = list(client.results(str(batch["id"])))
    atomic_write_batch_checkpoint(
        checkpoint_path,
        {
            "schema": "anthropic_message_batch_checkpoint",
            "schema_version": 1,
            "created_at_utc": batch.get("created_at"),
            "ended_at_utc": datetime.now(timezone.utc).isoformat(),
            "model": MODEL_NAME,
            "batch": ended,
            "custom_ids": [request["custom_id"]],
            "result_types": {
                str(record.get("custom_id")): (
                    record.get("result", {}).get("type")
                    if isinstance(record.get("result"), dict)
                    else None
                )
                for record in records
            },
        },
    )
    succeeded = [
        record
        for record in records
        if isinstance(record.get("result"), dict)
        and record["result"].get("type") == "succeeded"
    ]
    if len(succeeded) != 1:
        raise RuntimeError(
            "Claude Sonnet 5 preflight batch did not return exactly one successful result: "
            + json.dumps(records, ensure_ascii=False)[:2000]
        )
    message = succeeded[0]["result"].get("message")
    if not isinstance(message, dict):
        raise RuntimeError("Successful Anthropic batch result did not include a message.")
    return validate_payload(message)


def main() -> int:
    args = build_parser().parse_args()
    load_dotenv(PROJECT_ROOT / ".env")

    native_payload = ModelResponseClient._responses_to_anthropic_messages_payload(
        build_request_body()
    )
    if args.mode == "dry-run":
        print(
            json.dumps(
                {
                    "status": "ready",
                    "mode": "dry-run",
                    "model": native_payload["model"],
                    "max_tokens": native_payload["max_tokens"],
                    "has_system": bool(native_payload.get("system")),
                    "has_structured_output": bool(
                        native_payload.get("output_config", {}).get("format")
                    ),
                    "effort": native_payload.get("output_config", {}).get("effort"),
                    "has_api_key": bool(resolve_anthropic_api_key()),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    api_key = resolve_anthropic_api_key()
    if not api_key:
        raise RuntimeError(
            "Missing ANTHROPIC_API_KEY or ST_NAV_PROFILE_CLAUDE_API_KEY. "
            "Store the key in .env; do not pass it on the command line."
        )

    if args.mode == "messages":
        result = run_messages(api_key=api_key)
    else:
        result = run_batch(
            api_key=api_key,
            checkpoint_path=Path(args.checkpoint).resolve(),
            poll_interval_seconds=args.poll_interval,
            max_wait_seconds=args.max_wait,
        )
    print(
        json.dumps(
            {
                "status": "passed",
                "mode": args.mode,
                "model": MODEL_NAME,
                "response": result,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
