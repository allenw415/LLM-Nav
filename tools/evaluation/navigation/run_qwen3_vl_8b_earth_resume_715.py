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
import urllib.request


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sa_nav.evaluation import summarize_batch_results
from sa_nav.evaluation.navigation_batch import load_jsonl
from tools.evaluation.navigation import run_gemma4_12b_with_trajectory_715 as campaign


EVALUATOR = PROJECT_ROOT / "tools/evaluation/navigation/evaluate_navigation_batch.py"
DEFAULT_TEST_SET = PROJECT_ROOT / (
    "outputs/navigation_paths/floor0_semantic_targets_topology_constrained/"
    "room_count_2to6_ratio_1p0_2p0_per30_sparse715/test_set.jsonl"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / (
    "outputs/navigation_evaluation/"
    "qwen3_vl_8b_instruct_bf16_think_false_ctx16384_f16kv_with_trajectory_715"
)
DEFAULT_CACHE_ROOT = PROJECT_ROOT / (
    "outputs/navigation_cache/qwen3_vl_8b_instruct_bf16_think_false_ctx16384_f16kv"
)
DEFAULT_RENDER_ROOT = PROJECT_ROOT / "renders/room_grounding_fov90"
MODEL_NAME = "qwen3-vl:8b-instruct-bf16"
MODEL_DIGEST = "55aa31bad4e2ded596d2d491708e0c07b72b15c68606b523ce2d86b6231b4e9d"
OLLAMA_API_BASE = "http://127.0.0.1:11434/v1"
LLM_TIMEOUT_SECONDS = 180
MAX_HTTP_RETRIES = 1
VISUAL_MAX_OUTPUT_TOKENS = 3072
VISUAL_SCHEMA_VALIDATION_RETRIES = 1
SPATIAL_SCHEMA_VALIDATION_RETRIES = 1
SPATIAL_CONTEXT_MAX_OUTPUT_TOKENS = 2048
SPATIAL_ALIGNMENT_MAX_OUTPUT_TOKENS = 768
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


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
        with urllib.request.urlopen(f"http://127.0.0.1:11434{path}", timeout=10) as response:
            value = json.loads(response.read().decode("utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _service_environment() -> dict[str, str]:
    output = _run_text(["systemctl", "show", "ollama", "--property=Environment", "--value"])
    if not output:
        return {}
    try:
        values = shlex.split(output)
    except ValueError:
        values = output.split()
    allowed = set(EXPECTED_SERVICE_ENV) | {"OLLAMA_HOST"}
    parsed: dict[str, str] = {}
    for value in values:
        key, separator, setting = value.partition("=")
        if separator and key in allowed:
            parsed[key] = setting
    return parsed


def runtime_provenance() -> dict:
    tags = _ollama_json("/api/tags") or {}
    matching_model = None
    for model in tags.get("models", []):
        if isinstance(model, dict) and (model.get("name") == MODEL_NAME or model.get("model") == MODEL_NAME):
            matching_model = {
                "name": model.get("name") or model.get("model"),
                "digest": model.get("digest"),
                "size": model.get("size"),
                "details": model.get("details"),
            }
            break
    source_paths = [
        PROJECT_ROOT / "sa_nav/perception/detector.py",
        PROJECT_ROOT / "sa_nav/spatial/localization.py",
        EVALUATOR,
        Path(__file__).resolve(),
    ]
    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": _run_text(["hostname"]),
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
        "git_worktree_status": _run_text(["git", "status", "--short"]),
        "source_sha256": {
            str(path.relative_to(PROJECT_ROOT)): sha256_file(path) for path in source_paths
        },
    }


def validate_runtime(provenance: dict) -> None:
    model = provenance.get("model")
    if not isinstance(model, dict) or model.get("digest") != MODEL_DIGEST:
        raise RuntimeError(
            f"Expected {MODEL_NAME} digest {MODEL_DIGEST}, found {model!r}."
        )
    actual = provenance.get("service_environment") or {}
    mismatches = {
        key: {"expected": expected, "actual": actual.get(key)}
        for key, expected in EXPECTED_SERVICE_ENV.items()
        if actual.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(
            "Ollama service environment does not match the two-worker BF16 run: "
            + json.dumps(mismatches, ensure_ascii=False, sort_keys=True)
        )


def validate_snapshot(*, tests: list[dict], snapshot_dir: Path) -> tuple[list[dict], list[dict]]:
    results_path = snapshot_dir / "results.jsonl"
    attempts_path = snapshot_dir / "attempts.jsonl"
    if not results_path.is_file() or not attempts_path.is_file():
        raise RuntimeError(f"Snapshot is missing results or attempts: {snapshot_dir}")
    results = load_jsonl(results_path)
    attempts = load_jsonl(attempts_path)
    test_by_id = {test["test_id"]: test for test in tests}
    ids = [str(record.get("test_id")) for record in results]
    expected_prefix = [str(test["test_id"]) for test in tests[: len(results)]]
    if ids != expected_prefix:
        raise RuntimeError("Neptune snapshot results are not the canonical dataset prefix.")
    if len(ids) != len(set(ids)):
        raise RuntimeError("Neptune snapshot contains duplicate result IDs.")
    for record in results:
        test_id = str(record["test_id"])
        if record.get("path_id") != test_by_id[test_id].get("path_id"):
            raise RuntimeError(f"Snapshot path mismatch for {test_id}.")
    unknown_attempts = sorted(
        {
            str(record.get("test_id"))
            for record in attempts
            if record.get("test_id") not in test_by_id
        }
    )
    if unknown_attempts:
        raise RuntimeError(f"Snapshot attempts contain unknown IDs: {unknown_attempts[:10]}")
    return results, attempts


def prepare_shards(
    *,
    tests: list[dict],
    snapshot_dir: Path,
    output_root: Path,
) -> dict:
    snapshot_results, snapshot_attempts = validate_snapshot(tests=tests, snapshot_dir=snapshot_dir)
    shards = campaign.split_tests(tests)
    shard_by_id = {
        test["test_id"]: shard_index
        for shard_index, shard_tests in enumerate(shards)
        for test in shard_tests
    }
    seeded_results = [[], []]
    seeded_attempts = [[], []]
    for record in snapshot_results:
        seeded_results[shard_by_id[record["test_id"]]].append(record)
    for record in snapshot_attempts:
        test_id = record.get("test_id")
        if test_id in shard_by_id:
            seeded_attempts[shard_by_id[test_id]].append(record)

    for shard_index, shard_tests in enumerate(shards):
        shard_dir = output_root / "shards" / f"worker{shard_index}"
        shard_dir.mkdir(parents=True, exist_ok=True)
        results_path = shard_dir / "results.jsonl"
        attempts_path = shard_dir / "attempts.jsonl"
        if not results_path.exists():
            campaign.atomic_write_jsonl(results_path, seeded_results[shard_index])
        else:
            existing_by_id = {
                record["test_id"]: record for record in load_jsonl(results_path)
            }
            selected_ids = {test["test_id"] for test in shard_tests}
            unexpected = set(existing_by_id) - selected_ids
            missing_seeded = {
                record["test_id"] for record in seeded_results[shard_index]
            } - set(existing_by_id)
            if unexpected or missing_seeded:
                raise RuntimeError(
                    f"Existing worker{shard_index} state is incompatible: "
                    f"unexpected={sorted(unexpected)[:10]}, missing_seeded={sorted(missing_seeded)[:10]}"
                )
        if not attempts_path.exists():
            campaign.atomic_write_jsonl(attempts_path, seeded_attempts[shard_index])

    snapshot_files = {}
    for name in ["results.jsonl", "attempts.jsonl", "config.json", "summary.json", "run.log"]:
        path = snapshot_dir / name
        if path.is_file():
            snapshot_files[name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    manifest = {
        "schema": "qwen3_vl_8b_neptune_to_earth_migration",
        "schema_version": 1,
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_host": "neptune",
        "destination_host": "Earth",
        "snapshot_dir": str(snapshot_dir),
        "snapshot_files": snapshot_files,
        "snapshot_result_count": len(snapshot_results),
        "snapshot_attempt_count": len(snapshot_attempts),
        "snapshot_test_ids": [record["test_id"] for record in snapshot_results],
        "snapshot_error_ids": [
            record["test_id"]
            for record in snapshot_results
            if record.get("stop_reason") == "error"
        ],
        "shards": {
            "worker0": {"test_count": len(shards[0]), "seeded_result_count": len(seeded_results[0])},
            "worker1": {"test_count": len(shards[1]), "seeded_result_count": len(seeded_results[1])},
        },
        "model": {"name": MODEL_NAME, "digest": MODEL_DIGEST, "weights": "BF16"},
    }
    manifest_path = output_root / "migration_manifest.json"
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("snapshot_files") != manifest["snapshot_files"]:
            raise RuntimeError("Existing migration manifest points to a different snapshot.")
        manifest = previous
    else:
        campaign.atomic_write_json(manifest_path, manifest)
    return manifest


def build_evaluator_command(
    *,
    tests: list[dict],
    test_set: Path,
    output_dir: Path,
    cache_root: Path,
    render_root: Path,
    max_attempts: int,
    render_missing: bool = False,
    retry_stop_reasons: tuple[str, ...] = (),
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
        "ollama",
        "--llm-api-kind",
        "chat_completions",
        "--llm-api-base",
        OLLAMA_API_BASE,
        "--llm-timeout",
        str(LLM_TIMEOUT_SECONDS),
        "--max-http-retries",
        str(MAX_HTTP_RETRIES),
        "--visual-max-output-tokens",
        str(VISUAL_MAX_OUTPUT_TOKENS),
        "--visual-schema-validation-retries",
        str(VISUAL_SCHEMA_VALIDATION_RETRIES),
        "--spatial-schema-validation-retries",
        str(SPATIAL_SCHEMA_VALIDATION_RETRIES),
        "--spatial-context-max-output-tokens",
        str(SPATIAL_CONTEXT_MAX_OUTPUT_TOKENS),
        "--spatial-alignment-max-output-tokens",
        str(SPATIAL_ALIGNMENT_MAX_OUTPUT_TOKENS),
        "--allow-infrastructure-resume",
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
    if render_missing:
        command.append("--render-missing")
    for stop_reason in retry_stop_reasons:
        command.extend(("--retry-stop-reason", stop_reason))
    for test in tests:
        command.extend(("--test-id", str(test["test_id"])))
    return command


def pending_case_count(
    *, shard_tests: list[dict], results_path: Path, max_attempts: int
) -> int:
    existing = {
        record["test_id"]: record
        for record in load_jsonl(results_path)
    } if results_path.is_file() else {}
    pending = 0
    for test in shard_tests:
        record = existing.get(test["test_id"])
        if record is None:
            pending += 1
        elif record.get("stop_reason") == "error" and int(record.get("attempt_number") or 1) < max_attempts:
            pending += 1
    return pending


def run_round(
    commands: list[tuple[str, list[str], Path]],
    *, status_path: Path,
    round_number: int,
) -> None:
    processes: list[tuple[str, subprocess.Popen, object]] = []
    for shard_name, command, log_path in commands:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_stream = log_path.open("a", encoding="utf-8")
        log_stream.write(
            f"\n[earth qwen runner round={round_number}] "
            + " ".join(shlex.quote(value) for value in command)
            + "\n"
        )
        log_stream.flush()
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=log_stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        processes.append((shard_name, process, log_stream))
    campaign.atomic_write_json(
        status_path,
        {
            "state": "running",
            "round": round_number,
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
        campaign.atomic_write_json(
            status_path,
            {"state": "worker_failed", "round": round_number, "failures": failures},
        )
        raise RuntimeError(f"Qwen shard workers failed: {failures}")


def build_result_provenance(
    *, results: list[dict], snapshot_results: list[dict]
) -> list[dict]:
    snapshot_by_id = {record["test_id"]: record for record in snapshot_results}
    provenance = []
    for record in results:
        original = snapshot_by_id.get(record["test_id"])
        source_host = "neptune" if original == record else "Earth"
        provenance.append(
            {
                "test_id": record["test_id"],
                "final_result_host": source_host,
                "attempt_number": int(record.get("attempt_number") or 1),
                "present_in_neptune_snapshot": original is not None,
                "replaced_after_migration": original is not None and original != record,
            }
        )
    return provenance


def build_run_report(results: list[dict], provenance: list[dict]) -> dict:
    report = campaign.build_run_report(results)
    report["schema"] = "qwen3_vl_8b_with_trajectory_715_report"
    report["model"] = {"name": MODEL_NAME, "digest": MODEL_DIGEST, "weights": "BF16"}
    report["final_result_hosts"] = {
        host: sum(item["final_result_host"] == host for item in provenance)
        for host in ["neptune", "Earth"]
    }
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Resume the Qwen3-VL 8B BF16 715-case experiment on Earth with two workers."
    )
    parser.add_argument("--snapshot-dir", required=True)
    parser.add_argument("--test-set", default=str(DEFAULT_TEST_SET))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--cache-root", default=str(DEFAULT_CACHE_ROOT))
    parser.add_argument("--render-root", default=str(DEFAULT_RENDER_ROOT))
    parser.add_argument("--expected-case-count", type=int, default=715)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--skip-runtime-validation", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    test_set = resolve(args.test_set)
    output_root = resolve(args.output_root)
    cache_root = resolve(args.cache_root)
    render_root = resolve(args.render_root)
    snapshot_dir = resolve(args.snapshot_dir)
    tests = load_jsonl(test_set)
    if len(tests) != args.expected_case_count:
        raise RuntimeError(
            f"Expected {args.expected_case_count} cases, found {len(tests)}: {test_set}"
        )
    if args.max_attempts < 1:
        raise ValueError("--max-attempts must be at least 1.")

    provenance = runtime_provenance()
    if not args.skip_runtime_validation:
        validate_runtime(provenance)
    output_root.mkdir(parents=True, exist_ok=True)
    migration = prepare_shards(
        tests=tests,
        snapshot_dir=snapshot_dir,
        output_root=output_root,
    )
    run_config = {
        "schema": "qwen3_vl_8b_earth_resume_715_run",
        "schema_version": 1,
        "dataset": str(test_set),
        "dataset_sha256": sha256_file(test_set),
        "expected_case_count": len(tests),
        "shards": {"worker0": 358, "worker1": 357},
        "model": {"name": MODEL_NAME, "digest": MODEL_DIGEST, "weights": "BF16"},
        "think": False,
        "temperature": 0,
        "num_ctx": 16384,
        "llm_timeout_seconds": LLM_TIMEOUT_SECONDS,
        "max_http_retries": MAX_HTTP_RETRIES,
        "max_attempts_per_case": args.max_attempts,
        "visual_max_output_tokens": VISUAL_MAX_OUTPUT_TOKENS,
        "spatial_context_max_output_tokens": SPATIAL_CONTEXT_MAX_OUTPUT_TOKENS,
        "spatial_alignment_max_output_tokens": SPATIAL_ALIGNMENT_MAX_OUTPUT_TOKENS,
        "cache_root": str(cache_root),
        "render_root": str(render_root),
        "snapshot": migration["snapshot_files"],
        "runtime_provenance": provenance,
    }
    config_path = output_root / "run_config.json"
    if config_path.is_file():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        stable_keys = set(run_config) - {"runtime_provenance"}
        if any(existing.get(key) != run_config.get(key) for key in stable_keys):
            raise RuntimeError(f"Existing Qwen run config differs: {config_path}")
    else:
        campaign.atomic_write_json(config_path, run_config)

    shards = campaign.split_tests(tests)
    status_path = output_root / "status.json"
    for round_number in range(1, args.max_attempts + 1):
        commands = []
        for shard_index, shard_tests in enumerate(shards):
            shard_name = f"worker{shard_index}"
            shard_dir = output_root / "shards" / shard_name
            if pending_case_count(
                shard_tests=shard_tests,
                results_path=shard_dir / "results.jsonl",
                max_attempts=args.max_attempts,
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

    merged = campaign.merge_shard_results(
        tests=tests,
        shard_paths=[
            output_root / "shards/worker0/results.jsonl",
            output_root / "shards/worker1/results.jsonl",
        ],
    )
    campaign.atomic_write_jsonl(output_root / "results.jsonl", merged)
    summary = summarize_batch_results(merged)
    summary.update(
        {
            "expected_case_count": len(tests),
            "completed_case_count": len(merged),
            "pending_case_count": 0,
        }
    )
    campaign.atomic_write_json(output_root / "summary.json", summary)
    snapshot_results = load_jsonl(snapshot_dir / "results.jsonl")
    result_provenance = build_result_provenance(
        results=merged,
        snapshot_results=snapshot_results,
    )
    campaign.atomic_write_jsonl(output_root / "result_provenance.jsonl", result_provenance)
    report = build_run_report(merged, result_provenance)
    campaign.atomic_write_json(output_root / "run_report.json", report)
    final_state = "complete" if report["unresolved_error_count"] == 0 else "unresolved_errors"
    campaign.atomic_write_json(
        status_path,
        {"state": final_state, "report": str(output_root / "run_report.json")},
    )
    migration["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    migration["final_result_hosts"] = report["final_result_hosts"]
    campaign.atomic_write_json(output_root / "migration_manifest.json", migration)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if final_state == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
