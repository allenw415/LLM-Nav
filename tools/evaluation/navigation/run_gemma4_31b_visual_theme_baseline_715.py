from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sa_nav import load_dotenv
from sa_nav.common.model_client import ModelResponseClient
from sa_nav.evaluation import summarize_batch_results
from tools.evaluation.navigation import run_gemma4_12b_with_trajectory_715 as campaign


MODEL_NAME = "gemma-4-31b-it"
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"
API_KIND = "responses"
WORKER_KEY_ENV_NAMES = (
    "GEMMA4_API_KEY_WORKER0",
    "GEMMA4_API_KEY_WORKER1",
)
DEFAULT_TEST_SET = PROJECT_ROOT / (
    "outputs/navigation_paths/floor0_semantic_targets_topology_constrained/"
    "room_count_2to6_ratio_1p0_2p0_per30_sparse715/test_set.jsonl"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / (
    "outputs/navigation_evaluation/"
    "gemma4_31b_visual_theme_8view_no_trajectory_715"
)
DEFAULT_CACHE_ROOT = PROJECT_ROOT / (
    "outputs/navigation_cache/gemma4_31b_visual_theme_baseline"
)
DEFAULT_RENDER_ROOT = PROJECT_ROOT / "renders/room_grounding_fov90"
EVALUATOR = PROJECT_ROOT / "tools/evaluation/navigation/evaluate_navigation_batch.py"
LLM_TIMEOUT_SECONDS = 600
MAX_HTTP_RETRIES = 5
HTTP_RETRY_BACKOFF_SECONDS = 15.0
VISUAL_MAX_OUTPUT_TOKENS = 3072
DIRECTION_MAX_OUTPUT_TOKENS = 512
VISUAL_SCHEMA_VALIDATION_RETRIES = 1
DIRECTION_SCHEMA_VALIDATION_RETRIES = 1


def resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate.resolve() if candidate.is_absolute() else (PROJECT_ROOT / candidate).resolve()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_jsonl(path: Path) -> list[dict]:
    return campaign.load_jsonl(path) if path.is_file() else []


def worker_environment(base_environment: dict[str, str], *, key_slot: int) -> dict[str, str]:
    if key_slot not in {0, 1}:
        raise ValueError("key_slot must be 0 or 1.")
    source_name = WORKER_KEY_ENV_NAMES[key_slot]
    api_key = base_environment.get(source_name)
    if not isinstance(api_key, str) or not api_key.strip():
        raise RuntimeError(f"Missing required API key environment variable: {source_name}")

    child = dict(base_environment)
    for name in WORKER_KEY_ENV_NAMES:
        child.pop(name, None)
    # Pin a dedicated profile so evaluate_navigation_batch.py cannot reload an
    # unrelated profile (for example Ollama) from .env. There are intentionally
    # no profile-scoped GEMMA4_WORKER settings, so resolution falls through to
    # the worker-specific generic settings below.
    child["SA_NAV_ACTIVE_PROFILE"] = "gemma4_worker"
    child["SA_NAV_PROFILE"] = "gemma4_worker"
    child.update(
        {
            "SA_NAV_MODEL_PROVIDER": "gemini",
            "SA_NAV_MODEL_NAME": MODEL_NAME,
            "SA_NAV_API_KEY": api_key.strip(),
            "SA_NAV_API_BASE": GEMINI_API_BASE,
            "SA_NAV_API_KIND": API_KIND,
            "SA_NAV_TEMPERATURE": "0",
            "SA_NAV_MAX_HTTP_RETRIES": str(MAX_HTTP_RETRIES),
            "SA_NAV_HTTP_RETRY_BACKOFF_SECONDS": str(HTTP_RETRY_BACKOFF_SECONDS),
        }
    )
    return child


def validate_distinct_worker_keys(environment: dict[str, str]) -> None:
    keys = []
    for name in WORKER_KEY_ENV_NAMES:
        value = environment.get(name)
        if not isinstance(value, str) or not value.strip():
            raise RuntimeError(f"Missing required API key environment variable: {name}")
        keys.append(value.strip())
    if keys[0] == keys[1]:
        raise RuntimeError("The two worker API key slots must contain different keys.")


def validate_worker_generation_access(
    worker_environments: dict[str, dict[str, str]],
) -> None:
    failures = []
    for worker_name, environment in worker_environments.items():
        client = ModelResponseClient(
            provider="gemini",
            api_key=environment["SA_NAV_API_KEY"],
            api_base=GEMINI_API_BASE,
            api_kind=API_KIND,
            request_timeout=60.0,
            max_http_retries=MAX_HTTP_RETRIES,
            retry_backoff_seconds=HTTP_RETRY_BACKOFF_SECONDS,
        )
        try:
            client.create(
                {
                    "model": MODEL_NAME,
                    "instructions": "Reply with OK.",
                    "input": "OK",
                    "max_output_tokens": 16,
                }
            )
        except Exception as error:
            failures.append(
                {
                    "worker": worker_name,
                    "error_type": type(error).__name__,
                    "message": str(error)[:500],
                }
            )
    if failures:
        raise RuntimeError(
            "Gemma generation preflight failed before starting the campaign: "
            + json.dumps(failures, ensure_ascii=False)
        )


def build_evaluator_command(
    *,
    tests: list[dict],
    test_set: Path,
    output_dir: Path,
    cache_root: Path,
    render_root: Path,
    max_attempts: int,
    render_missing: bool,
    direction_cache_dir: Path | None,
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
        "--llm-api-kind",
        API_KIND,
        "--llm-api-base",
        GEMINI_API_BASE,
        "--llm-timeout",
        str(LLM_TIMEOUT_SECONDS),
        "--max-http-retries",
        str(MAX_HTTP_RETRIES),
        "--visual-max-output-tokens",
        str(VISUAL_MAX_OUTPUT_TOKENS),
        "--visual-schema-validation-retries",
        str(VISUAL_SCHEMA_VALIDATION_RETRIES),
        "--direction-max-output-tokens",
        str(DIRECTION_MAX_OUTPUT_TOKENS),
        "--direction-schema-validation-retries",
        str(DIRECTION_SCHEMA_VALIDATION_RETRIES),
        "--localization-interval",
        "3",
        "--localization-effective-stage",
        "visual",
        "--start-heading",
        "0",
        "--heading-alignment-mode",
        "none",
        "--direction-policy",
        "visual-theme",
        "--query-cache-dir",
        str(cache_root / "query_parser"),
        "--visual-cache-dir",
        str(cache_root / "visual_localization"),
        "--retry-errors",
        "--max-attempts-per-case",
        str(max_attempts),
    ]
    if direction_cache_dir is not None:
        command.extend(
            (
                "--direction-cache-dir",
                str(direction_cache_dir),
            )
        )
    if render_missing:
        command.append("--render-missing")
    for test in tests:
        command.extend(("--test-id", str(test["test_id"])))
    return command


def retryable_error_count(results_path: Path, *, max_attempts: int) -> int:
    return sum(
        1
        for record in _load_jsonl(results_path)
        if record.get("stop_reason") == "error"
        and int(record.get("attempt_number") or 1) < max_attempts
    )


def run_round(
    workers: list[tuple[str, list[str], Path, dict[str, str]]],
    *,
    status_path: Path,
    round_number: int,
) -> None:
    processes: list[tuple[str, subprocess.Popen, object]] = []
    for worker_name, command, log_path, environment in workers:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_stream = log_path.open("a", encoding="utf-8")
        log_stream.write(
            f"\n[{_utc_now()}] runner round={round_number} key_slot={worker_name} "
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

    campaign.atomic_write_json(
        status_path,
        {
            "state": "running",
            "round": round_number,
            "updated_at_utc": _utc_now(),
            "workers": [
                {
                    "worker": worker_name,
                    "key_slot": worker_name,
                    "pid": process.pid,
                }
                for worker_name, process, _ in processes
            ],
        },
    )

    failures = []
    for worker_name, process, log_stream in processes:
        exit_code = process.wait()
        log_stream.close()
        if exit_code != 0:
            failures.append({"worker": worker_name, "exit_code": exit_code})
    if failures:
        campaign.atomic_write_json(
            status_path,
            {
                "state": "worker_failed",
                "round": round_number,
                "updated_at_utc": _utc_now(),
                "failures": failures,
            },
        )
        raise RuntimeError(f"Visual-theme baseline workers failed: {failures}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the formal Gemma 4 31B visual-theme navigation baseline over "
            "715 cases with two API keys and two concurrent shards."
        )
    )
    parser.add_argument("--test-set", default=str(DEFAULT_TEST_SET))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--cache-root", default=str(DEFAULT_CACHE_ROOT))
    parser.add_argument(
        "--direction-cache-dir",
        help=(
            "Override only the direction-selection cache directory while retaining "
            "the shared query and visual-localization caches."
        ),
    )
    parser.add_argument(
        "--require-empty-direction-cache",
        action="store_true",
        help=(
            "For a new output run, require the direction cache to contain no JSON "
            "entries before launching. Resuming the same output run remains allowed."
        ),
    )
    parser.add_argument("--render-root", default=str(DEFAULT_RENDER_ROOT))
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument(
        "--no-direction-cache",
        action="store_true",
        help=(
            "Call the direction-selection VLM for every multi-candidate movement "
            "decision instead of reading or writing the direction request cache."
        ),
    )
    parser.add_argument(
        "--no-render-missing",
        action="store_true",
        help="Require all panorama renders to exist instead of lazily rendering cache misses.",
    )
    return parser


def main() -> int:
    load_dotenv(PROJECT_ROOT / ".env")
    args = build_parser().parse_args()
    if args.max_attempts < 1:
        raise ValueError("--max-attempts must be at least 1.")

    test_set = resolve(args.test_set)
    output_root = resolve(args.output_root)
    cache_root = resolve(args.cache_root)
    render_root = resolve(args.render_root)
    if args.no_direction_cache and args.direction_cache_dir:
        raise ValueError(
            "--no-direction-cache and --direction-cache-dir are mutually exclusive."
        )
    direction_cache_dir = (
        None
        if args.no_direction_cache
        else resolve(args.direction_cache_dir)
        if args.direction_cache_dir
        else cache_root / "direction_selection"
    )
    tests = campaign.load_jsonl(test_set)
    if len(tests) != 715:
        raise RuntimeError(f"Expected exactly 715 tests, found {len(tests)}: {test_set}")
    shard0, shard1 = campaign.split_tests(tests)
    if (len(shard0), len(shard1)) != (358, 357):
        raise AssertionError("Unexpected 715-case shard sizes.")

    base_environment = dict(os.environ)
    validate_distinct_worker_keys(base_environment)
    worker_environments = {
        "worker0": worker_environment(base_environment, key_slot=0),
        "worker1": worker_environment(base_environment, key_slot=1),
    }
    validate_worker_generation_access(worker_environments)
    render_missing = not args.no_render_missing
    direction_cache_enabled = direction_cache_dir is not None
    if render_missing and not base_environment.get("GMAPS_API_KEY"):
        raise RuntimeError(
            "GMAPS_API_KEY is required for the default lazy render-missing policy. "
            "Use --no-render-missing only after verifying the render cache is complete."
        )

    config_path = output_root / "run_config.json"
    if (
        args.require_empty_direction_cache
        and not config_path.is_file()
        and direction_cache_dir is not None
    ):
        existing_direction_entries = list(direction_cache_dir.rglob("*.json"))
        if existing_direction_entries:
            raise RuntimeError(
                "Cold-start direction cache is not empty: "
                f"{direction_cache_dir} ({len(existing_direction_entries)} JSON files)"
            )

    run_config = {
        "schema": "gemma4_31b_visual_theme_no_trajectory_baseline_715_run",
        "schema_version": 4,
        "dataset": str(test_set),
        "dataset_sha256": _sha256(test_set),
        "expected_case_count": 715,
        "model": MODEL_NAME,
        "provider": "gemini",
        "api_kind": API_KIND,
        "two_key_execution": {
            "worker0": {"key_slot": WORKER_KEY_ENV_NAMES[0], "test_count": 358},
            "worker1": {"key_slot": WORKER_KEY_ENV_NAMES[1], "test_count": 357},
            "secrets_persisted": False,
        },
        "navigation": {
            "direction_policy": "visual-theme",
            "trajectory_memory": "disabled",
            "action_selection": "stateless_nearest_selected_view_heading",
            "localization_interval": 3,
            "localization_effective_stage": "visual",
            "heading_alignment_mode": "none",
            "spatial_alignment_calls_expected": 0,
            "views": 8,
            "direction_call_policy": (
                "every_multi_candidate_movement_step_with_request_memoization"
                if direction_cache_enabled
                else "every_multi_candidate_movement_step_fresh_api"
            ),
        },
        "reliability": {
            "max_http_retries": MAX_HTTP_RETRIES,
            "http_retry_backoff_seconds": HTTP_RETRY_BACKOFF_SECONDS,
            "max_attempts_per_case": args.max_attempts,
            "visual_schema_validation_retries": VISUAL_SCHEMA_VALIDATION_RETRIES,
            "direction_schema_validation_retries": DIRECTION_SCHEMA_VALIDATION_RETRIES,
        },
        "cache_root": str(cache_root),
        "request_cache": {
            "query_parser": "enabled",
            "visual_localization": "enabled",
            "direction_selection": {
                "enabled": direction_cache_enabled,
                "root": (
                    str(direction_cache_dir)
                    if direction_cache_dir is not None
                    else None
                ),
                "cold_start_required": bool(args.require_empty_direction_cache),
                "reuse_within_run": direction_cache_enabled,
            },
        },
        "render_root": str(render_root),
        "render_missing": render_missing,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    if config_path.is_file():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing != run_config:
            raise RuntimeError(f"Existing run config differs: {config_path}")
    else:
        campaign.atomic_write_json(config_path, run_config)

    shard_specs = (("worker0", shard0), ("worker1", shard1))
    status_path = output_root / "status.json"
    for round_number in range(1, args.max_attempts + 1):
        workers = []
        for worker_name, shard_tests in shard_specs:
            shard_dir = output_root / "shards" / worker_name
            if round_number > 1 and retryable_error_count(
                shard_dir / "results.jsonl",
                max_attempts=args.max_attempts,
            ) == 0:
                continue
            workers.append(
                (
                    worker_name,
                    build_evaluator_command(
                        tests=shard_tests,
                        test_set=test_set,
                        output_dir=shard_dir,
                        cache_root=cache_root,
                        render_root=render_root,
                        max_attempts=args.max_attempts,
                        render_missing=render_missing,
                        direction_cache_dir=direction_cache_dir,
                    ),
                    output_root / "logs" / f"{worker_name}.log",
                    worker_environments[worker_name],
                )
            )
        if not workers:
            break
        run_round(workers, status_path=status_path, round_number=round_number)

    merged = campaign.merge_shard_results(
        tests=tests,
        shard_paths=[
            output_root / "shards/worker0/results.jsonl",
            output_root / "shards/worker1/results.jsonl",
        ],
    )
    if len(merged) != 715:
        raise AssertionError(f"Merged result count is {len(merged)}, expected 715.")
    campaign.atomic_write_jsonl(output_root / "results.jsonl", merged)
    summary = summarize_batch_results(merged)
    summary.update(
        {
            "expected_case_count": 715,
            "completed_case_count": len(merged),
            "pending_case_count": 0,
            "error_case_count": sum(
                1 for record in merged if record.get("stop_reason") == "error"
            ),
        }
    )
    campaign.atomic_write_json(output_root / "summary.json", summary)
    campaign.atomic_write_json(
        status_path,
        {
            "state": (
                "completed" if summary["error_case_count"] == 0 else "completed_with_errors"
            ),
            "updated_at_utc": _utc_now(),
            "completed_case_count": 715,
            "error_case_count": summary["error_case_count"],
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
