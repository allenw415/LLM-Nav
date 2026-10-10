from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sa_nav import load_dotenv
from sa_nav.evaluation import summarize_batch_results
from tools.evaluation.navigation import run_gemma4_31b_visual_theme_baseline_715 as baseline


DEFAULT_OUTPUT_ROOT = baseline.DEFAULT_OUTPUT_ROOT
DEFAULT_RETRY_ROOT_NAME = "retry_errors"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_original_results(output_root: Path) -> list[dict]:
    records: list[dict] = []
    for worker in ("worker0", "worker1"):
        records.extend(
            baseline.campaign.load_jsonl(
                output_root / "shards" / worker / "results.jsonl"
            )
        )
    ids = [record.get("test_id") for record in records]
    if len(records) != 715 or len(set(ids)) != 715:
        raise RuntimeError(
            "Expected 715 unique original shard results before retrying errors."
        )
    return records


def select_retry_tests(tests: list[dict], original_results: list[dict]) -> list[dict]:
    retry_ids = {
        record["test_id"]
        for record in original_results
        if record.get("stop_reason") == "error"
        and int(record.get("attempt_number") or 1) < 3
    }
    return [test for test in tests if test.get("test_id") in retry_ids]


def merge_retry_results(
    *,
    tests: list[dict],
    original_results: list[dict],
    retry_results: list[dict],
) -> list[dict]:
    original_by_id = {record["test_id"]: record for record in original_results}
    retry_by_id = {record["test_id"]: record for record in retry_results}
    if len(retry_by_id) != len(retry_results):
        raise RuntimeError("Retry results contain duplicate test IDs.")
    unknown = set(retry_by_id) - set(original_by_id)
    if unknown:
        raise RuntimeError(f"Retry results contain unknown IDs: {sorted(unknown)}")
    merged = [
        retry_by_id.get(test["test_id"], original_by_id[test["test_id"]])
        for test in tests
    ]
    if len(merged) != 715 or len({record["test_id"] for record in merged}) != 715:
        raise RuntimeError("Merged retry output does not contain 715 unique cases.")
    return merged


def _seed_retry_shard(
    *,
    shard_dir: Path,
    shard_tests: list[dict],
    original_by_id: dict[str, dict],
) -> None:
    expected_ids = [test["test_id"] for test in shard_tests]
    results_path = shard_dir / "results.jsonl"
    attempts_path = shard_dir / "attempts.jsonl"
    if results_path.is_file():
        existing = baseline.campaign.load_jsonl(results_path)
        if [record.get("test_id") for record in existing] != expected_ids:
            raise RuntimeError(f"Existing retry shard IDs differ: {results_path}")
        return
    seeded = [original_by_id[test_id] for test_id in expected_ids]
    baseline.campaign.atomic_write_jsonl(results_path, seeded)
    baseline.campaign.atomic_write_jsonl(attempts_path, seeded)


def _retryable_count(path: Path) -> int:
    return baseline.retryable_error_count(path, max_attempts=3)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Retry only the infrastructure/schema errors from the Gemma 4 31B "
            "visual-theme 715-case baseline with two balanced API-key shards."
        )
    )
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    return parser


def main() -> int:
    load_dotenv(PROJECT_ROOT / ".env")
    args = build_parser().parse_args()
    output_root = baseline.resolve(args.output_root)
    retry_root = output_root / DEFAULT_RETRY_ROOT_NAME
    retry_root.mkdir(parents=True, exist_ok=True)

    lock_stream = (retry_root / "runner.lock").open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError("A retry runner is already active.") from error

    base_environment = dict(os.environ)
    baseline.validate_distinct_worker_keys(base_environment)
    worker_environments = {
        "worker0": baseline.worker_environment(base_environment, key_slot=0),
        "worker1": baseline.worker_environment(base_environment, key_slot=1),
    }
    baseline.validate_worker_generation_access(worker_environments)

    tests = baseline.campaign.load_jsonl(baseline.DEFAULT_TEST_SET)
    original_results = load_original_results(output_root)
    retry_tests = select_retry_tests(tests, original_results)
    if not retry_tests:
        raise RuntimeError("No retryable error cases were found.")
    retry_shards = (retry_tests[0::2], retry_tests[1::2])
    original_by_id = {record["test_id"]: record for record in original_results}

    retry_config = {
        "schema": "gemma4_31b_visual_theme_baseline_error_retry",
        "schema_version": 1,
        "created_at_utc": _utc_now(),
        "source_output_root": str(output_root),
        "source_error_count": len(retry_tests),
        "worker_test_counts": {
            "worker0": len(retry_shards[0]),
            "worker1": len(retry_shards[1]),
        },
        "max_attempts_per_case": 3,
        "cache_root": str(baseline.DEFAULT_CACHE_ROOT),
        "secrets_persisted": False,
    }
    config_path = retry_root / "run_config.json"
    if config_path.is_file():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        comparable_existing = dict(existing)
        comparable_current = dict(retry_config)
        comparable_existing.pop("created_at_utc", None)
        comparable_current.pop("created_at_utc", None)
        if comparable_existing != comparable_current:
            raise RuntimeError(f"Existing retry config differs: {config_path}")
    else:
        baseline.campaign.atomic_write_json(config_path, retry_config)

    for index, shard_tests in enumerate(retry_shards):
        _seed_retry_shard(
            shard_dir=retry_root / "shards" / f"worker{index}",
            shard_tests=shard_tests,
            original_by_id=original_by_id,
        )

    status_path = output_root / "status.json"
    for round_number in (2, 3):
        workers = []
        for index, shard_tests in enumerate(retry_shards):
            worker_name = f"worker{index}"
            shard_dir = retry_root / "shards" / worker_name
            if _retryable_count(shard_dir / "results.jsonl") == 0:
                continue
            workers.append(
                (
                    worker_name,
                    baseline.build_evaluator_command(
                        tests=shard_tests,
                        test_set=baseline.DEFAULT_TEST_SET,
                        output_dir=shard_dir,
                        cache_root=baseline.DEFAULT_CACHE_ROOT,
                        render_root=baseline.DEFAULT_RENDER_ROOT,
                        max_attempts=3,
                        render_missing=True,
                    ),
                    retry_root / "logs" / f"{worker_name}.log",
                    worker_environments[worker_name],
                )
            )
        if not workers:
            break
        baseline.run_round(
            workers,
            status_path=status_path,
            round_number=round_number,
        )

    retry_results: list[dict] = []
    for worker_name in ("worker0", "worker1"):
        retry_results.extend(
            baseline.campaign.load_jsonl(
                retry_root / "shards" / worker_name / "results.jsonl"
            )
        )
    if len(retry_results) != len(retry_tests):
        raise RuntimeError(
            f"Expected {len(retry_tests)} retry results, found {len(retry_results)}."
        )
    merged = merge_retry_results(
        tests=tests,
        original_results=original_results,
        retry_results=retry_results,
    )
    baseline.campaign.atomic_write_jsonl(output_root / "results.jsonl", merged)
    summary = summarize_batch_results(merged)
    error_count = sum(record.get("stop_reason") == "error" for record in merged)
    summary.update(
        {
            "expected_case_count": 715,
            "completed_case_count": 715,
            "pending_case_count": 0,
            "error_case_count": error_count,
            "retried_case_count": len(retry_results),
        }
    )
    baseline.campaign.atomic_write_json(output_root / "summary.json", summary)
    baseline.campaign.atomic_write_json(
        status_path,
        {
            "state": "completed" if error_count == 0 else "completed_with_errors",
            "updated_at_utc": _utc_now(),
            "completed_case_count": 715,
            "retried_case_count": len(retry_results),
            "error_case_count": error_count,
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
