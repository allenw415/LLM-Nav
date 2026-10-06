from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from st_nav import (  # noqa: E402
    AnthropicMessageBatchClient,
    PersistentAnthropicBatchBroker,
    load_dotenv,
)
from st_nav.common.anthropic_batch_gateway import (  # noqa: E402
    start_anthropic_batch_gateway_server,
)
from st_nav.evaluation import summarize_batch_results  # noqa: E402
from tools.evaluation.navigation import (  # noqa: E402
    run_gemma4_12b_with_trajectory_715 as campaign,
)


MODEL_NAME = "claude-sonnet-5"
ANTHROPIC_API_BASE = "https://api.anthropic.com/v1"
LOCAL_GATEWAY_API_KEY = "llm-nav-local-anthropic-batch"
EVALUATOR = PROJECT_ROOT / "tools/evaluation/navigation/evaluate_navigation_batch.py"
DEFAULT_TEST_SET = PROJECT_ROOT / (
    "outputs/navigation_paths/floor0_semantic_targets_topology_constrained/"
    "room_count_2to6_ratio_1p0_2p0_per30_sparse715/test_set.jsonl"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / (
    "outputs/navigation_evaluation/claude_sonnet5_batch_with_trajectory_715"
)
DEFAULT_CACHE_ROOT = PROJECT_ROOT / (
    "outputs/navigation_cache/claude_sonnet5_batch_with_trajectory"
)
DEFAULT_RENDER_ROOT = PROJECT_ROOT / "renders/room_grounding_fov90"
DEFAULT_WORKER_COUNT = 32
WORKER_REQUEST_TIMEOUT_SECONDS = 25 * 60 * 60
MAX_HTTP_RETRIES = 1
VISUAL_MAX_OUTPUT_TOKENS = 3072
VISUAL_SCHEMA_VALIDATION_RETRIES = 1
SPATIAL_SCHEMA_VALIDATION_RETRIES = 1
SPATIAL_CONTEXT_MAX_OUTPUT_TOKENS = 2048
SPATIAL_ALIGNMENT_MAX_OUTPUT_TOKENS = 1024


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return (
        candidate.resolve()
        if candidate.is_absolute()
        else (PROJECT_ROOT / candidate).resolve()
    )


def split_worker_shards(tests: list[dict], worker_count: int) -> list[list[dict]]:
    if worker_count < 1:
        raise ValueError("worker_count must be at least 1.")
    actual_count = min(worker_count, len(tests)) if tests else 0
    return [tests[index::actual_count] for index in range(actual_count)]


def worker_environment(base_environment: dict[str, str]) -> dict[str, str]:
    child = dict(base_environment)
    child.update(
        {
            "ST_NAV_ACTIVE_PROFILE": "claude_batch_worker",
            "ST_NAV_PROFILE": "claude_batch_worker",
            "ST_NAV_MODEL_PROVIDER": "anthropic",
            "ST_NAV_MODEL_NAME": MODEL_NAME,
            "ST_NAV_API_KIND": "anthropic_messages",
            "ST_NAV_TEMPERATURE": "0",
            "ST_NAV_MAX_HTTP_RETRIES": str(MAX_HTTP_RETRIES),
        }
    )
    return child


def build_evaluator_command(
    *,
    tests: list[dict],
    test_set: Path,
    output_dir: Path,
    cache_root: Path,
    render_root: Path,
    gateway_base_url: str,
    max_attempts: int,
    render_missing: bool,
    spatial_context_max_output_tokens: int,
    spatial_alignment_max_output_tokens: int,
) -> list[str]:
    command = [
        sys.executable,
        str(EVALUATOR),
        "--test-set",
        str(test_set),
        "--output-dir",
        str(output_dir),
        "--render-output-dir",
        str(render_root),
        "--render-heading-mode",
        "museum",
        "--llm-model",
        MODEL_NAME,
        "--llm-api-key",
        LOCAL_GATEWAY_API_KEY,
        "--llm-api-kind",
        "anthropic_messages",
        "--llm-api-base",
        gateway_base_url,
        "--llm-timeout",
        str(WORKER_REQUEST_TIMEOUT_SECONDS),
        "--max-http-retries",
        str(MAX_HTTP_RETRIES),
        "--allow-infrastructure-resume",
        "--visual-max-output-tokens",
        str(VISUAL_MAX_OUTPUT_TOKENS),
        "--visual-schema-validation-retries",
        str(VISUAL_SCHEMA_VALIDATION_RETRIES),
        "--spatial-schema-validation-retries",
        str(SPATIAL_SCHEMA_VALIDATION_RETRIES),
        "--spatial-context-max-output-tokens",
        str(spatial_context_max_output_tokens),
        "--spatial-alignment-max-output-tokens",
        str(spatial_alignment_max_output_tokens),
        "--localization-interval",
        "3",
        "--localization-effective-stage",
        "spatial",
        "--start-heading",
        "0",
        "--heading-alignment-mode",
        "initial-spatial",
        "--force-initial-visual-refresh",
        "--direction-policy",
        "geometry-edge-visit",
        "--query-cache-dir",
        str(cache_root / "query_parser"),
        "--visual-cache-dir",
        str(cache_root / "visual_localization"),
        "--retry-errors",
        "--max-attempts-per-case",
        str(max_attempts),
    ]
    if render_missing:
        command.append("--render-missing")
    for test in tests:
        command.extend(("--test-id", str(test["test_id"])))
    return command


def _shard_needs_run(
    tests: list[dict],
    results_path: Path,
    *,
    max_attempts: int,
) -> bool:
    if not results_path.is_file():
        return True
    results = campaign.load_jsonl(results_path)
    by_id = {record.get("test_id"): record for record in results}
    for test in tests:
        record = by_id.get(test["test_id"])
        if record is None:
            return True
        if (
            record.get("stop_reason") == "error"
            and int(record.get("attempt_number") or 1) < max_attempts
        ):
            return True
    return False


def _write_running_status(
    *,
    status_path: Path,
    round_number: int,
    processes: list[tuple[str, subprocess.Popen, object]],
    broker: PersistentAnthropicBatchBroker,
) -> None:
    campaign.atomic_write_json(
        status_path,
        {
            "state": "running",
            "round": round_number,
            "updated_at_utc": _utc_now(),
            "gateway": broker.snapshot(),
            "workers": [
                {
                    "worker": worker_name,
                    "pid": process.pid,
                    "state": "running" if process.poll() is None else "exited",
                    "exit_code": process.poll(),
                }
                for worker_name, process, _ in processes
            ],
        },
    )


def run_worker_round(
    workers: list[tuple[str, list[str], Path]],
    *,
    environment: dict[str, str],
    status_path: Path,
    round_number: int,
    broker: PersistentAnthropicBatchBroker,
) -> bool:
    processes: list[tuple[str, subprocess.Popen, object]] = []
    billing_pause: dict | None = None
    try:
        for worker_name, command, log_path in workers:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_stream = log_path.open("a", encoding="utf-8")
            log_stream.write(
                f"\n[{_utc_now()}] runner round={round_number} "
                + " ".join(shlex.quote(value) for value in command)
                + "\n"
            )
            log_stream.flush()
            process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                env=environment,
                stdout=log_stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            processes.append((worker_name, process, log_stream))

        while any(process.poll() is None for _, process, _ in processes):
            _write_running_status(
                status_path=status_path,
                round_number=round_number,
                processes=processes,
                broker=broker,
            )
            broker_status = broker.snapshot()
            if broker_status.get("billing_paused") is True:
                billing_pause = broker_status.get("billing_pause")
                for _, process, _ in processes:
                    if process.poll() is None:
                        process.terminate()
                break
            failures = [
                (worker_name, process.returncode)
                for worker_name, process, _ in processes
                if process.poll() not in {None, 0}
            ]
            if failures:
                for _, process, _ in processes:
                    if process.poll() is None:
                        process.terminate()
                break
            time.sleep(1.0)
    finally:
        for _, process, _ in processes:
            if process.poll() is None:
                process.wait()
        for _, _, log_stream in processes:
            log_stream.close()

    if billing_pause is not None:
        campaign.atomic_write_json(
            status_path,
            {
                "state": "paused_billing",
                "round": round_number,
                "updated_at_utc": _utc_now(),
                "gateway": broker.snapshot(),
                "billing_pause": billing_pause,
                "resume_command": (
                    "python3 tools/evaluation/navigation/"
                    "run_claude_sonnet5_batch_715.py --confirm-full-run"
                ),
            },
        )
        return True

    failures = [
        {"worker": worker_name, "exit_code": process.returncode}
        for worker_name, process, _ in processes
        if process.returncode != 0
    ]
    if failures:
        campaign.atomic_write_json(
            status_path,
            {
                "state": "worker_failed",
                "round": round_number,
                "updated_at_utc": _utc_now(),
                "gateway": broker.snapshot(),
                "failures": failures,
            },
        )
        raise RuntimeError(f"Claude Batch navigation workers failed: {failures}")
    return False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the 715-case Claude Sonnet 5 navigation experiment through "
            "Anthropic Message Batches with resumable worker shards."
        )
    )
    parser.add_argument("--test-set", default=str(DEFAULT_TEST_SET))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--cache-root", default=str(DEFAULT_CACHE_ROOT))
    parser.add_argument("--render-root", default=str(DEFAULT_RENDER_ROOT))
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKER_COUNT)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--gateway-host", default="127.0.0.1")
    parser.add_argument("--gateway-port", type=int, default=8765)
    parser.add_argument("--batch-flush-seconds", type=float, default=2.0)
    parser.add_argument("--batch-poll-seconds", type=float, default=30.0)
    parser.add_argument("--max-concurrent-batches", type=int, default=4)
    parser.add_argument("--target-batch-mib", type=int, default=200)
    parser.add_argument(
        "--spatial-context-max-output-tokens",
        type=int,
        default=SPATIAL_CONTEXT_MAX_OUTPUT_TOKENS,
    )
    parser.add_argument(
        "--spatial-alignment-max-output-tokens",
        type=int,
        default=SPATIAL_ALIGNMENT_MAX_OUTPUT_TOKENS,
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--test-id", action="append", default=[])
    parser.add_argument(
        "--no-render-missing",
        action="store_true",
        help="Do not request missing Street View images during navigation.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the campaign plan without making API calls.",
    )
    parser.add_argument(
        "--confirm-full-run",
        action="store_true",
        help="Required when launching all 715 paid navigation cases.",
    )
    return parser


def main() -> int:
    load_dotenv(PROJECT_ROOT / ".env")
    args = build_parser().parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be at least 1.")
    if args.max_attempts < 1:
        raise ValueError("--max-attempts must be at least 1.")
    if args.target_batch_mib < 1 or args.target_batch_mib > 256:
        raise ValueError("--target-batch-mib must be between 1 and 256.")
    if args.spatial_context_max_output_tokens < 1:
        raise ValueError("--spatial-context-max-output-tokens must be at least 1.")
    if args.spatial_alignment_max_output_tokens < 1:
        raise ValueError("--spatial-alignment-max-output-tokens must be at least 1.")

    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Missing ANTHROPIC_API_KEY in the project .env.")

    test_set = resolve(args.test_set)
    output_root = resolve(args.output_root)
    cache_root = resolve(args.cache_root)
    render_root = resolve(args.render_root)
    all_tests = campaign.load_jsonl(test_set)
    if len(all_tests) != 715:
        raise RuntimeError(f"Expected exactly 715 tests, found {len(all_tests)}: {test_set}")
    tests = all_tests
    if args.test_id:
        selected_ids = set(args.test_id)
        tests = [test for test in tests if test.get("test_id") in selected_ids]
        missing_ids = selected_ids - {str(test.get("test_id")) for test in tests}
        if missing_ids:
            raise ValueError(f"Unknown --test-id values: {sorted(missing_ids)}")
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be at least 1.")
        tests = tests[: args.limit]
    if not tests:
        raise RuntimeError("No tests selected.")
    if len(tests) == 715 and not args.dry_run and not args.confirm_full_run:
        raise RuntimeError(
            "Launching all 715 paid cases requires --confirm-full-run. "
            "Run --dry-run first."
        )

    shards = split_worker_shards(tests, args.workers)
    gateway_base_url = f"http://{args.gateway_host}:{args.gateway_port}/v1"
    run_config = {
        "schema": "claude_sonnet5_message_batch_navigation_715_run",
        "schema_version": 1,
        "dataset": str(test_set),
        "dataset_sha256": campaign._sha256(test_set),
        "dataset_case_count": len(all_tests),
        "selected_case_count": len(tests),
        "selected_test_ids": [test["test_id"] for test in tests],
        "model": MODEL_NAME,
        "provider": "anthropic",
        "api_kind": "anthropic_messages",
        "api_key_persisted": False,
        "message_batches": {
            "enabled": True,
            "worker_count": len(shards),
            "flush_interval_seconds": args.batch_flush_seconds,
            "poll_interval_seconds": args.batch_poll_seconds,
            "max_concurrent_batches": args.max_concurrent_batches,
            "target_batch_mib": args.target_batch_mib,
            "gateway_base_url": gateway_base_url,
            "successful_response_cache": True,
            "submitted_batch_recovery": True,
        },
        "navigation": {
            "method": "main_spatial_8view_with_trajectory",
            "direction_policy": "geometry-edge-visit",
            "trajectory_memory": "edge_visit_counts",
            "localization_interval": 3,
            "localization_effective_stage": "spatial",
            "heading_alignment_mode": "initial-spatial",
            "force_initial_visual_refresh": True,
            "start_heading": 0,
            "localization_pipeline": [
                "visual_localization",
                "bayesian_filtering",
                "spatial_alignment",
            ],
        },
        "reliability": {
            "max_attempts_per_case": args.max_attempts,
            "visual_schema_validation_retries": VISUAL_SCHEMA_VALIDATION_RETRIES,
            "spatial_schema_validation_retries": SPATIAL_SCHEMA_VALIDATION_RETRIES,
        },
        "cache": {
            "root": str(cache_root),
            "query_understanding": {
                "enabled": True,
                "path": str(cache_root / "query_parser"),
                "shared_across_workers": True,
            },
            "visual_localization": {
                "enabled": True,
                "path": str(cache_root / "visual_localization"),
                "shared_across_workers": True,
                "step_0_force_refresh": True,
            },
            "rendered_images": {
                "enabled": True,
                "path": str(render_root),
                "heading_mode": "museum",
                "view_count": 8,
            },
        },
        "render_root": str(render_root),
        "render_missing": not args.no_render_missing,
        "worker_shards": {
            f"worker{index:02d}": [test["test_id"] for test in shard]
            for index, shard in enumerate(shards)
        },
    }
    output_limit_overrides = {}
    if args.spatial_context_max_output_tokens != SPATIAL_CONTEXT_MAX_OUTPUT_TOKENS:
        output_limit_overrides["spatial_context_max_output_tokens"] = (
            args.spatial_context_max_output_tokens
        )
    if args.spatial_alignment_max_output_tokens != SPATIAL_ALIGNMENT_MAX_OUTPUT_TOKENS:
        output_limit_overrides["spatial_alignment_max_output_tokens"] = (
            args.spatial_alignment_max_output_tokens
        )
    if output_limit_overrides:
        run_config["output_limit_overrides"] = output_limit_overrides
    plan = {
        "status": "ready",
        "mode": "dry_run" if args.dry_run else "live",
        "output_root": str(output_root),
        "selected_case_count": len(tests),
        "worker_count": len(shards),
        "shard_sizes": [len(shard) for shard in shards],
        "gateway_base_url": gateway_base_url,
        "render_missing": not args.no_render_missing,
        "spatial_context_max_output_tokens": args.spatial_context_max_output_tokens,
        "spatial_alignment_max_output_tokens": (
            args.spatial_alignment_max_output_tokens
        ),
    }
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0

    output_root.mkdir(parents=True, exist_ok=True)
    config_path = output_root / "run_config.json"
    if config_path.is_file():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing != run_config:
            raise RuntimeError(f"Existing run config differs: {config_path}")
    else:
        campaign.atomic_write_json(config_path, run_config)

    batch_client = AnthropicMessageBatchClient(
        api_key=api_key,
        api_base=ANTHROPIC_API_BASE,
        request_timeout=300.0,
    )
    broker = PersistentAnthropicBatchBroker(
        client=batch_client,
        state_dir=output_root / "gateway",
        flush_interval_seconds=args.batch_flush_seconds,
        poll_interval_seconds=args.batch_poll_seconds,
        max_wait_seconds=24 * 60 * 60,
        target_batch_bytes=args.target_batch_mib * 1024 * 1024,
        max_concurrent_batches=args.max_concurrent_batches,
    )
    broker.start()
    server = None
    server_thread = None
    status_path = output_root / "status.json"
    try:
        server, server_thread = start_anthropic_batch_gateway_server(
            broker=broker,
            host=args.gateway_host,
            port=args.gateway_port,
            local_api_key=LOCAL_GATEWAY_API_KEY,
        )
        environment = worker_environment(dict(os.environ))
        shard_specs = [
            (f"worker{index:02d}", shard)
            for index, shard in enumerate(shards)
        ]
        for round_number in range(1, args.max_attempts + 1):
            workers = []
            for worker_name, shard in shard_specs:
                shard_dir = output_root / "shards" / worker_name
                if not _shard_needs_run(
                    shard,
                    shard_dir / "results.jsonl",
                    max_attempts=args.max_attempts,
                ):
                    continue
                workers.append(
                    (
                        worker_name,
                        build_evaluator_command(
                            tests=shard,
                            test_set=test_set,
                            output_dir=shard_dir,
                            cache_root=cache_root,
                            render_root=render_root,
                            gateway_base_url=gateway_base_url,
                            max_attempts=args.max_attempts,
                            render_missing=not args.no_render_missing,
                            spatial_context_max_output_tokens=(
                                args.spatial_context_max_output_tokens
                            ),
                            spatial_alignment_max_output_tokens=(
                                args.spatial_alignment_max_output_tokens
                            ),
                        ),
                        output_root / "logs" / f"{worker_name}.log",
                    )
                )
            if not workers:
                break
            billing_paused = run_worker_round(
                workers,
                environment=environment,
                status_path=status_path,
                round_number=round_number,
                broker=broker,
            )
            if billing_paused:
                print(
                    json.dumps(
                        {
                            "state": "paused_billing",
                            "status_path": str(status_path),
                            "resume_after_top_up": (
                                "python3 tools/evaluation/navigation/"
                                "run_claude_sonnet5_batch_715.py --confirm-full-run"
                            ),
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                )
                return 75

        merged = campaign.merge_shard_results(
            tests=tests,
            shard_paths=[
                output_root / "shards" / worker_name / "results.jsonl"
                for worker_name, _ in shard_specs
            ],
        )
        campaign.atomic_write_jsonl(output_root / "results.jsonl", merged)
        summary = summarize_batch_results(merged)
        summary.update(
            {
                "expected_case_count": len(tests),
                "completed_case_count": len(merged),
                "pending_case_count": 0,
                "error_case_count": sum(
                    record.get("stop_reason") == "error" for record in merged
                ),
                "batch_gateway": broker.snapshot(),
            }
        )
        campaign.atomic_write_json(output_root / "summary.json", summary)
        final_state = (
            "completed"
            if summary["error_case_count"] == 0
            else "completed_with_errors"
        )
        campaign.atomic_write_json(
            status_path,
            {
                "state": final_state,
                "updated_at_utc": _utc_now(),
                "completed_case_count": len(merged),
                "error_case_count": summary["error_case_count"],
                "gateway": broker.snapshot(),
            },
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0 if final_state == "completed" else 2
    finally:
        if server is not None:
            server.shutdown()
            server.server_close()
        if server_thread is not None:
            server_thread.join(timeout=5.0)
        broker.close(wait=True)


if __name__ == "__main__":
    raise SystemExit(main())
