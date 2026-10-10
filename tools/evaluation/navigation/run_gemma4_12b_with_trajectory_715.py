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
import tempfile
import urllib.request


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sa_nav.evaluation import summarize_batch_results
from sa_nav.evaluation.navigation_batch import load_jsonl


EVALUATOR = PROJECT_ROOT / "tools/evaluation/navigation/evaluate_navigation_batch.py"
DEFAULT_TEST_SET = PROJECT_ROOT / (
    "outputs/navigation_paths/floor0_semantic_targets_topology_constrained/"
    "room_count_2to6_ratio_1p0_2p0_per30_sparse715/test_set.jsonl"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / (
    "outputs/navigation_evaluation/"
    "gemma4_12b_bf16_think_false_ctx16384_f16kv_with_trajectory_715"
)
DEFAULT_CACHE_ROOT = PROJECT_ROOT / (
    "outputs/navigation_cache/gemma4_12b_bf16_think_false_ctx16384_f16kv"
)
DEFAULT_RENDER_ROOT = PROJECT_ROOT / "renders/room_grounding_fov90"
MODEL_NAME = "gemma4:12b-it-bf16"
OLLAMA_API_BASE = "http://127.0.0.1:11434/v1"
LLM_TIMEOUT_SECONDS = 600
MAX_HTTP_RETRIES = 1
VISUAL_SCHEMA_VALIDATION_RETRIES = 1
SPATIAL_SCHEMA_VALIDATION_RETRIES = 1
SPATIAL_CONTEXT_MAX_OUTPUT_TOKENS = 2048
SPATIAL_ALIGNMENT_MAX_OUTPUT_TOKENS = 1024
EXPECTED_SERVICE_ENV = {
    "OLLAMA_CONTEXT_LENGTH": "16384",
    "OLLAMA_FLASH_ATTENTION": "1",
    "OLLAMA_KEEP_ALIVE": "-1",
    "OLLAMA_KV_CACHE_TYPE": "f16",
    "OLLAMA_MAX_LOADED_MODELS": "1",
    "OLLAMA_NUM_PARALLEL": "2",
}


def resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate.resolve() if candidate.is_absolute() else (PROJECT_ROOT / candidate).resolve()


def atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
        temporary_path = Path(stream.name)
    os.replace(temporary_path, path)


def atomic_write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
        temporary_path = Path(stream.name)
    os.replace(temporary_path, path)


def _run_config_resume_compatible(existing: dict, current: dict) -> bool:
    def normalized(value: dict) -> dict:
        result = json.loads(json.dumps(value))
        result.pop("schema_version", None)
        result.pop("runtime_provenance", None)
        result.pop("llm_transport", None)
        result.pop("visual_localization_reliability", None)
        result.pop("spatial_alignment_reliability", None)
        return result

    return normalized(existing) == normalized(current)


def _record_run_config_change(path: Path, *, previous: dict, current: dict) -> None:
    previous_transport = previous.get("llm_transport")
    if not isinstance(previous_transport, dict):
        previous_transport = {
            "request_timeout_seconds": 300,
            "max_http_retries": 5,
        }
    records = load_jsonl(path) if path.is_file() else []
    records.append(
        {
            "schema": "gemma4_12b_run_config_change",
            "schema_version": 1,
            "changed_at_utc": datetime.now(timezone.utc).isoformat(),
            "reason": "approved_reliability_adjustment",
            "previous_transport": previous_transport,
            "current_transport": current["llm_transport"],
            "previous_config": previous,
            "current_config": current,
        }
    )
    atomic_write_jsonl(path, records)


def split_tests(tests: list[dict]) -> tuple[list[dict], list[dict]]:
    left = tests[0::2]
    right = tests[1::2]
    left_ids = {record["test_id"] for record in left}
    right_ids = {record["test_id"] for record in right}
    all_ids = [record["test_id"] for record in tests]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("Dataset contains duplicate test IDs.")
    if left_ids & right_ids:
        raise AssertionError("Shard split overlaps.")
    if left_ids | right_ids != set(all_ids):
        raise AssertionError("Shard split does not cover the dataset.")
    return left, right


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_tree_sha256() -> str:
    roots = [PROJECT_ROOT / "sa_nav", PROJECT_ROOT / "tools/evaluation/navigation"]
    paths = sorted(
        path
        for root in roots
        for path in root.rglob("*.py")
        if path.is_file()
    )
    digest = hashlib.sha256()
    for source_path in paths:
        digest.update(str(source_path.relative_to(PROJECT_ROOT)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(source_path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _run_text(command: list[str]) -> str | None:
    try:
        result = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def _ollama_json(path: str) -> dict | None:
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:11434{path}", timeout=10
        ) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _service_environment() -> dict[str, str]:
    output = _run_text(["systemctl", "show", "ollama", "--property=Environment", "--value"])
    if not output:
        return {}
    try:
        values = shlex.split(output)
    except ValueError:
        values = output.split()
    parsed = {}
    allowed = set(EXPECTED_SERVICE_ENV) | {"OLLAMA_HOST"}
    for value in values:
        key, separator, setting = value.partition("=")
        if separator and key in allowed:
            parsed[key] = setting
    return parsed


def runtime_provenance() -> dict:
    tags = _ollama_json("/api/tags") or {}
    matching_model = None
    for model in tags.get("models", []):
        if not isinstance(model, dict):
            continue
        if model.get("name") == MODEL_NAME or model.get("model") == MODEL_NAME:
            matching_model = {
                "name": model.get("name") or model.get("model"),
                "digest": model.get("digest"),
                "size": model.get("size"),
                "details": model.get("details"),
            }
            break
    git_diff = subprocess.run(
        ["git", "diff", "--binary"],
        cwd=PROJECT_ROOT,
        check=False,
        stdout=subprocess.PIPE,
    ).stdout
    return {
        "ollama_api_version": _ollama_json("/api/version"),
        "ollama_cli_version": _run_text(["ollama", "--version"]),
        "model": matching_model,
        "service_environment": _service_environment(),
        "gpu": _run_text(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ]
        ),
        "git_commit": _run_text(["git", "rev-parse", "HEAD"]),
        "git_worktree_diff_sha256": hashlib.sha256(git_diff).hexdigest(),
        "git_worktree_status": _run_text(["git", "status", "--short"]),
        "git_worktree_dirty": bool(_run_text(["git", "status", "--short"])),
        "source_tree_sha256": _source_tree_sha256(),
    }


def validate_runtime(provenance: dict) -> None:
    model = provenance.get("model")
    if not isinstance(model, dict) or not model.get("digest"):
        raise RuntimeError(f"Ollama model is not installed: {MODEL_NAME}")
    actual = provenance.get("service_environment") or {}
    mismatches = {
        key: {"expected": expected, "actual": actual.get(key)}
        for key, expected in EXPECTED_SERVICE_ENV.items()
        if actual.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(
            "Ollama service environment does not match the formal run: "
            + json.dumps(mismatches, ensure_ascii=False, sort_keys=True)
        )


def build_evaluator_command(
    *,
    tests: list[dict],
    test_set: Path,
    output_dir: Path,
    cache_root: Path,
    render_root: Path,
    max_attempts: int,
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
        "--render-missing",
        "--llm-model",
        MODEL_NAME,
        "--llm-api-kind",
        "chat_completions",
        "--llm-api-base",
        OLLAMA_API_BASE,
        "--llm-timeout",
        str(LLM_TIMEOUT_SECONDS),
        "--max-http-retries",
        str(MAX_HTTP_RETRIES),
        "--allow-infrastructure-resume",
        "--visual-schema-validation-retries",
        str(VISUAL_SCHEMA_VALIDATION_RETRIES),
        "--spatial-schema-validation-retries",
        str(SPATIAL_SCHEMA_VALIDATION_RETRIES),
        "--spatial-context-max-output-tokens",
        str(SPATIAL_CONTEXT_MAX_OUTPUT_TOKENS),
        "--spatial-alignment-max-output-tokens",
        str(SPATIAL_ALIGNMENT_MAX_OUTPUT_TOKENS),
        "--localization-interval",
        "3",
        "--localization-effective-stage",
        "spatial",
        "--start-heading",
        "0",
        "--heading-alignment-mode",
        "initial-spatial",
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
    for test in tests:
        command.extend(("--test-id", str(test["test_id"])))
    return command


def retryable_error_count(results_path: Path, *, max_attempts: int) -> int:
    if not results_path.is_file():
        return 0
    return sum(
        1
        for record in load_jsonl(results_path)
        if record.get("stop_reason") == "error"
        and int(record.get("attempt_number") or 1) < max_attempts
    )


def run_round(
    commands: list[tuple[str, list[str], Path]],
    *,
    status_path: Path,
    round_number: int,
) -> None:
    processes: list[tuple[str, subprocess.Popen, object]] = []
    for shard_name, command, log_path in commands:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_stream = log_path.open("a", encoding="utf-8")
        log_stream.write(f"\n[runner round={round_number}] {' '.join(shlex.quote(v) for v in command)}\n")
        log_stream.flush()
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=log_stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        processes.append((shard_name, process, log_stream))
    atomic_write_json(
        status_path,
        {
            "state": "running",
            "round": round_number,
            "llm_transport": {
                "request_timeout_seconds": LLM_TIMEOUT_SECONDS,
                "max_http_retries": MAX_HTTP_RETRIES,
            },
            "workers": [
                {"shard": shard_name, "pid": process.pid}
                for shard_name, process, _ in processes
            ],
        },
    )
    failures = []
    for shard_name, process, log_stream in processes:
        exit_code = process.wait()
        log_stream.close()
        if exit_code != 0:
            failures.append({"shard": shard_name, "exit_code": exit_code})
    if failures:
        atomic_write_json(
            status_path,
            {"state": "worker_failed", "round": round_number, "failures": failures},
        )
        raise RuntimeError(f"Shard workers failed: {failures}")


def merge_shard_results(
    *,
    tests: list[dict],
    shard_paths: list[Path],
) -> list[dict]:
    dataset_by_id = {record["test_id"]: record for record in tests}
    merged_by_id: dict[str, dict] = {}
    for path in shard_paths:
        if not path.is_file():
            raise RuntimeError(f"Missing shard results: {path}")
        for record in load_jsonl(path):
            test_id = record.get("test_id")
            if test_id not in dataset_by_id:
                raise RuntimeError(f"Unexpected result ID {test_id} in {path}")
            if test_id in merged_by_id:
                raise RuntimeError(f"Duplicate result ID across shards: {test_id}")
            expected = dataset_by_id[test_id]
            if record.get("path_id") != expected.get("path_id"):
                raise RuntimeError(f"Path mismatch for {test_id}")
            merged_by_id[test_id] = record
    missing = [record["test_id"] for record in tests if record["test_id"] not in merged_by_id]
    if missing:
        raise RuntimeError(f"Missing {len(missing)} results; first IDs: {missing[:10]}")
    return [merged_by_id[record["test_id"]] for record in tests]


def _error_category(record: dict) -> str:
    error = record.get("error") or {}
    message = f"{error.get('type', '')} {error.get('message', '')}".lower()
    if "timeout" in message:
        return "timeout"
    if "out of memory" in message or "oom" in message:
        return "oom"
    if "json" in message or "schema" in message or "tokenize" in message:
        return "json_or_schema"
    if "http" in message or "connection" in message:
        return "http_or_connection"
    return "other"


def build_run_report(results: list[dict]) -> dict:
    error_categories: dict[str, int] = {}
    query_hits = query_misses = 0
    visual_hits = visual_misses = 0
    heading_statuses: dict[str, int] = {}
    unlocked_offsets: list[str] = []
    for record in results:
        if record.get("stop_reason") == "error":
            category = _error_category(record)
            error_categories[category] = error_categories.get(category, 0) + 1
        query_cache = record.get("query_cache") or {}
        if query_cache.get("hit") is True:
            query_hits += 1
        elif query_cache.get("hit") is False:
            query_misses += 1
        for event in record.get("localization_events", []):
            if event.get("visual_cache_hit") is True:
                visual_hits += 1
            elif event.get("visual_cache_hit") is False:
                visual_misses += 1
        alignment = record.get("initial_heading_alignment") or {}
        status = str(alignment.get("status") or "missing")
        heading_statuses[status] = heading_statuses.get(status, 0) + 1
        estimated = alignment.get("estimated_offset_deg")
        if estimated is not None and any(
            movement.get("heading_offset_used_deg") != estimated
            for movement in record.get("movements", [])
        ):
            unlocked_offsets.append(str(record.get("test_id")))
    elapsed = [
        float(record["elapsed_seconds"])
        for record in results
        if isinstance(record.get("elapsed_seconds"), (int, float))
    ]
    return {
        "schema": "gemma4_12b_with_trajectory_715_report",
        "schema_version": 1,
        "case_count": len(results),
        "unresolved_error_count": sum(
            record.get("stop_reason") == "error" for record in results
        ),
        "error_categories": error_categories,
        "query_cache": {"hit": query_hits, "miss": query_misses},
        "visual_cache": {"hit": visual_hits, "miss": visual_misses},
        "heading_alignment_statuses": heading_statuses,
        "heading_offset_lock_violation_ids": unlocked_offsets,
        "elapsed_seconds": {
            "total": sum(elapsed),
            "mean": sum(elapsed) / len(elapsed) if elapsed else None,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the formal Gemma4 12B With-Trajectory 715-case experiment in two shards."
    )
    parser.add_argument("--test-set", default=str(DEFAULT_TEST_SET))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--cache-root", default=str(DEFAULT_CACHE_ROOT))
    parser.add_argument("--render-root", default=str(DEFAULT_RENDER_ROOT))
    parser.add_argument("--expected-case-count", type=int, default=715)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument(
        "--resume-infrastructure-change",
        action="store_true",
        help="Resume existing shards after an approved timeout/retry-only adjustment.",
    )
    parser.add_argument(
        "--skip-runtime-validation",
        action="store_true",
        help="Intended only for unit tests; formal runs must validate Ollama settings.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    test_set = resolve(args.test_set)
    output_root = resolve(args.output_root)
    cache_root = resolve(args.cache_root)
    render_root = resolve(args.render_root)
    tests = load_jsonl(test_set)
    if len(tests) != args.expected_case_count:
        raise RuntimeError(
            f"Expected {args.expected_case_count} cases, found {len(tests)}: {test_set}"
        )
    if args.max_attempts < 1:
        raise ValueError("--max-attempts must be at least 1.")
    shard0, shard1 = split_tests(tests)
    provenance = runtime_provenance()
    if not args.skip_runtime_validation:
        validate_runtime(provenance)

    run_config = {
        "schema": "gemma4_12b_with_trajectory_715_run",
        "schema_version": 4,
        "dataset": str(test_set),
        "dataset_sha256": _sha256(test_set),
        "expected_case_count": len(tests),
        "shards": {"worker0": len(shard0), "worker1": len(shard1)},
        "model": MODEL_NAME,
        "think": False,
        "temperature": 0,
        "num_ctx": 16384,
        "llm_transport": {
            "request_timeout_seconds": LLM_TIMEOUT_SECONDS,
            "max_http_retries": MAX_HTTP_RETRIES,
            "case_level_max_attempts": args.max_attempts,
        },
        "visual_localization_reliability": {
            "schema_validation_retries": VISUAL_SCHEMA_VALIDATION_RETRIES,
            "failure_diagnostics": "separate_from_validated_cache",
            "legacy_validated_cache_policy": "validate_and_migrate",
        },
        "spatial_alignment_reliability": {
            "schema_validation_retries": SPATIAL_SCHEMA_VALIDATION_RETRIES,
            "context_max_output_tokens": SPATIAL_CONTEXT_MAX_OUTPUT_TOKENS,
            "alignment_max_output_tokens": SPATIAL_ALIGNMENT_MAX_OUTPUT_TOKENS,
            "timeout_retry_policy": "case_level_only",
        },
        "ollama": {
            "kv_cache_type": "f16",
            "num_parallel": 2,
            "flash_attention": True,
            "keep_alive": -1,
        },
        "navigation": {
            "direction_policy": "geometry-edge-visit",
            "localization_interval": 3,
            "localization_effective_stage": "spatial",
            "heading_alignment_mode": "initial-spatial",
            "offset_write_policy": "step_0_once_then_locked",
            "initial_failure_policy": "stop_without_fixed_offset_fallback",
        },
        "render": {"heading_mode": "museum", "views": 8, "width": 512, "height": 512, "fov": 90},
        "cache_root": str(cache_root),
        "output_root": str(output_root),
        "max_attempts_per_case": args.max_attempts,
        "runtime_provenance": provenance,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    config_path = output_root / "run_config.json"
    if config_path.is_file():
        existing_config = json.loads(config_path.read_text(encoding="utf-8"))
        if existing_config != run_config:
            if (
                args.resume_infrastructure_change
                and _run_config_resume_compatible(existing_config, run_config)
            ):
                _record_run_config_change(
                    output_root / "run_config_history.jsonl",
                    previous=existing_config,
                    current=run_config,
                )
                atomic_write_json(config_path, run_config)
            else:
                raise RuntimeError(f"Existing run config differs: {config_path}")
    else:
        atomic_write_json(config_path, run_config)

    shard_specs = [("worker0", shard0), ("worker1", shard1)]
    status_path = output_root / "status.json"
    for round_number in range(1, args.max_attempts + 1):
        commands = []
        for shard_name, shard_tests in shard_specs:
            shard_dir = output_root / "shards" / shard_name
            if round_number > 1 and retryable_error_count(
                shard_dir / "results.jsonl", max_attempts=args.max_attempts
            ) == 0:
                continue
            commands.append(
                (
                    shard_name,
                    build_evaluator_command(
                        tests=shard_tests,
                        test_set=test_set,
                        output_dir=shard_dir,
                        cache_root=cache_root,
                        render_root=render_root,
                        max_attempts=args.max_attempts,
                    ),
                    output_root / "logs" / f"{shard_name}.log",
                )
            )
        if not commands:
            break
        run_round(commands, status_path=status_path, round_number=round_number)

    merged = merge_shard_results(
        tests=tests,
        shard_paths=[
            output_root / "shards" / "worker0" / "results.jsonl",
            output_root / "shards" / "worker1" / "results.jsonl",
        ],
    )
    atomic_write_jsonl(output_root / "results.jsonl", merged)
    summary = summarize_batch_results(merged)
    summary.update(
        {
            "expected_case_count": len(tests),
            "completed_case_count": len(merged),
            "pending_case_count": 0,
        }
    )
    atomic_write_json(output_root / "summary.json", summary)
    report = build_run_report(merged)
    atomic_write_json(output_root / "run_report.json", report)
    final_state = "complete" if report["unresolved_error_count"] == 0 else "unresolved_errors"
    atomic_write_json(
        status_path,
        {"state": final_state, "report": str(output_root / "run_report.json")},
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if final_state == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
