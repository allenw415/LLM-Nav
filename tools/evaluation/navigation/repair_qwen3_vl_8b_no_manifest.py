from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sa_nav import load_dotenv
from sa_nav.evaluation import summarize_batch_results
from sa_nav.evaluation.navigation_batch import load_jsonl
from tools.evaluation.navigation import run_gemma4_12b_with_trajectory_715 as campaign
from tools.evaluation.navigation import run_qwen3_vl_8b_earth_resume_715 as qwen


STOP_REASON = "stop_no_manifest"
RETRYABLE_STOP_REASONS = {STOP_REASON, "error"}
DEFAULT_REPAIR_ROOT = qwen.DEFAULT_OUTPUT_ROOT / "repairs" / STOP_REASON


def _record_weight(record: dict) -> float:
    elapsed = record.get("elapsed_seconds")
    if isinstance(elapsed, (int, float)) and elapsed > 0:
        return float(elapsed)
    return float(record.get("movement_steps") or 1)


def _missing_pano_key(record: dict) -> str:
    missing = record.get("render", {}).get("cache_miss_pano_ids", [])
    if isinstance(missing, list) and missing:
        return str(missing[-1])
    return f"case:{record['test_id']}"


def partition_repairs(records: list[dict]) -> list[list[str]]:
    """Balance prior runtime while keeping cases for one missing pano together."""
    groups: dict[str, list[dict]] = {}
    for record in records:
        groups.setdefault(_missing_pano_key(record), []).append(record)
    weighted_groups = sorted(
        groups.values(),
        key=lambda group: (
            -sum(_record_weight(record) for record in group),
            min(str(record["test_id"]) for record in group),
        ),
    )
    partitions: list[list[str]] = [[], []]
    weights = [0.0, 0.0]
    for group in weighted_groups:
        target = min(range(2), key=lambda index: (weights[index], index))
        partitions[target].extend(str(record["test_id"]) for record in group)
        weights[target] += sum(_record_weight(record) for record in group)
    for partition in partitions:
        partition.sort()
    return partitions


def _attempt_key(record: dict) -> tuple[str, int]:
    return str(record.get("test_id")), int(record.get("attempt_number") or 1)


def _retry_pending(records: list[dict], *, max_attempts: int) -> list[str]:
    return [
        str(record["test_id"])
        for record in records
        if record.get("stop_reason") in RETRYABLE_STOP_REASONS
        and int(record.get("attempt_number") or 1) < max_attempts
    ]


def _seed_repair_workers(
    *,
    tests_by_id: dict[str, dict],
    before_by_id: dict[str, dict],
    partitions: list[list[str]],
    output_root: Path,
    repair_root: Path,
) -> None:
    attempts: list[dict] = []
    for shard_index in range(2):
        attempts_path = output_root / "shards" / f"worker{shard_index}" / "attempts.jsonl"
        if attempts_path.is_file():
            attempts.extend(load_jsonl(attempts_path))
    selected_ids = set(before_by_id)
    selected_attempts = [
        record for record in attempts if str(record.get("test_id")) in selected_ids
    ]

    for worker_index, test_ids in enumerate(partitions):
        worker_dir = repair_root / f"worker{worker_index}"
        worker_dir.mkdir(parents=True, exist_ok=True)
        results_path = worker_dir / "results.jsonl"
        attempts_path = worker_dir / "attempts.jsonl"
        expected_results = [before_by_id[test_id] for test_id in test_ids]
        if results_path.is_file():
            existing_ids = {str(record["test_id"]) for record in load_jsonl(results_path)}
            if existing_ids != set(test_ids):
                raise RuntimeError(
                    f"Repair worker{worker_index} IDs differ from the saved partition."
                )
        else:
            campaign.atomic_write_jsonl(results_path, expected_results)
        if not attempts_path.is_file():
            campaign.atomic_write_jsonl(
                attempts_path,
                [
                    record
                    for record in selected_attempts
                    if str(record.get("test_id")) in set(test_ids)
                ],
            )
        unknown = set(test_ids) - set(tests_by_id)
        if unknown:
            raise RuntimeError(f"Repair partition contains unknown tests: {sorted(unknown)}")


def _merge_repaired_attempts(main_path: Path, repair_records: list[dict]) -> None:
    main_records = load_jsonl(main_path) if main_path.is_file() else []
    by_key = {_attempt_key(record): record for record in main_records}
    for record in repair_records:
        key = _attempt_key(record)
        existing = by_key.get(key)
        if existing is not None:
            if existing != record:
                raise RuntimeError(f"Conflicting attempt record for {key[0]} attempt {key[1]}.")
            continue
        by_key[key] = record
        main_records.append(record)
    campaign.atomic_write_jsonl(main_path, main_records)


def _merge_into_campaign(
    *,
    tests: list[dict],
    repaired_by_id: dict[str, dict],
    output_root: Path,
    repair_root: Path,
    before_sha256: str,
) -> dict:
    shard_by_id = {
        str(test["test_id"]): shard_index
        for shard_index, shard_tests in enumerate(campaign.split_tests(tests))
        for test in shard_tests
    }
    repair_attempts: list[dict] = []
    for worker_index in range(2):
        repair_attempts.extend(load_jsonl(repair_root / f"worker{worker_index}" / "attempts.jsonl"))

    for shard_index in range(2):
        shard_dir = output_root / "shards" / f"worker{shard_index}"
        results_path = shard_dir / "results.jsonl"
        shard_results = load_jsonl(results_path)
        result_index = {
            str(record["test_id"]): index for index, record in enumerate(shard_results)
        }
        shard_repair_ids = [
            test_id for test_id in repaired_by_id if shard_by_id[test_id] == shard_index
        ]
        for test_id in shard_repair_ids:
            shard_results[result_index[test_id]] = repaired_by_id[test_id]
        campaign.atomic_write_jsonl(results_path, shard_results)
        _merge_repaired_attempts(
            shard_dir / "attempts.jsonl",
            [
                record
                for record in repair_attempts
                if str(record.get("test_id")) in set(shard_repair_ids)
            ],
        )

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

    migration_path = output_root / "migration_manifest.json"
    migration = json.loads(migration_path.read_text(encoding="utf-8"))
    snapshot_dir = Path(str(migration["snapshot_dir"]))
    snapshot_results = load_jsonl(snapshot_dir / "results.jsonl")
    provenance = qwen.build_result_provenance(
        results=merged,
        snapshot_results=snapshot_results,
    )
    campaign.atomic_write_jsonl(output_root / "result_provenance.jsonl", provenance)
    report = qwen.build_run_report(merged, provenance)
    repaired_records = [repaired_by_id[test_id] for test_id in sorted(repaired_by_id)]
    report["stop_no_manifest_repair"] = {
        "case_count": len(repaired_records),
        "attempted_test_ids": sorted(repaired_by_id),
        "final_stop_reasons": dict(Counter(record.get("stop_reason") for record in repaired_records)),
        "new_render_pano_ids": sorted(
            {
                str(pano_id)
                for record in repaired_records
                for pano_id in record.get("render", {}).get("new_render_pano_ids", [])
            }
        ),
        "new_image_requests": sum(
            int(record.get("render", {}).get("new_image_requests") or 0)
            for record in repaired_records
        ),
        "before_results_sha256": before_sha256,
        "after_results_sha256": qwen.sha256_file(output_root / "results.jsonl"),
    }
    campaign.atomic_write_json(output_root / "run_report.json", report)
    campaign.atomic_write_json(
        output_root / "status.json",
        {"state": "complete", "report": str(output_root / "run_report.json")},
    )
    migration["final_result_hosts"] = report["final_result_hosts"]
    migration["last_repair_completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    campaign.atomic_write_json(migration_path, migration)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render missing panoramas and repair Qwen stop_no_manifest results."
    )
    parser.add_argument("--test-set", default=str(qwen.DEFAULT_TEST_SET))
    parser.add_argument("--output-root", default=str(qwen.DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--cache-root", default=str(qwen.DEFAULT_CACHE_ROOT))
    parser.add_argument("--render-root", default=str(qwen.DEFAULT_RENDER_ROOT))
    parser.add_argument("--repair-root", default=str(DEFAULT_REPAIR_ROOT))
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--skip-runtime-validation", action="store_true")
    return parser


def main() -> int:
    load_dotenv(PROJECT_ROOT / ".env")
    args = build_parser().parse_args()
    if not os.environ.get("GMAPS_API_KEY"):
        raise RuntimeError("GMAPS_API_KEY is required for --render-missing repair.")
    test_set = qwen.resolve(args.test_set)
    output_root = qwen.resolve(args.output_root)
    cache_root = qwen.resolve(args.cache_root)
    render_root = qwen.resolve(args.render_root)
    repair_root = qwen.resolve(args.repair_root)
    tests = load_jsonl(test_set)
    tests_by_id = {str(test["test_id"]): test for test in tests}
    if len(tests) != 715 or len(tests_by_id) != 715:
        raise RuntimeError("Qwen repair requires the canonical 715-case test set.")
    if args.max_attempts < 2:
        raise ValueError("--max-attempts must permit at least one repair attempt.")

    runtime = qwen.runtime_provenance()
    if not args.skip_runtime_validation:
        qwen.validate_runtime(runtime)
    results_path = output_root / "results.jsonl"
    results = load_jsonl(results_path)
    if len(results) != 715 or len({record["test_id"] for record in results}) != 715:
        raise RuntimeError("Existing Qwen results are not a complete unique 715-case run.")
    before_records = [record for record in results if record.get("stop_reason") == STOP_REASON]
    if not before_records:
        print("No stop_no_manifest results remain; nothing to repair.")
        return 0

    before_by_id = {str(record["test_id"]): record for record in before_records}
    partitions = partition_repairs(before_records)
    repair_root.mkdir(parents=True, exist_ok=True)
    before_path = repair_root / "before_results.jsonl"
    if before_path.is_file():
        saved = load_jsonl(before_path)
        if saved != before_records:
            raise RuntimeError("Saved repair baseline differs from current stop_no_manifest records.")
    else:
        campaign.atomic_write_jsonl(before_path, before_records)
    before_sha256 = qwen.sha256_file(results_path)
    repair_config = {
        "schema": "qwen3_vl_8b_stop_no_manifest_repair",
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "render_missing": True,
        "retry_stop_reasons": sorted(RETRYABLE_STOP_REASONS),
        "max_attempts_per_case": args.max_attempts,
        "test_ids": sorted(before_by_id),
        "partitions": {f"worker{index}": test_ids for index, test_ids in enumerate(partitions)},
        "before_results_sha256": before_sha256,
        "runtime_provenance": runtime,
    }
    repair_config_path = repair_root / "repair_config.json"
    if repair_config_path.is_file():
        existing = json.loads(repair_config_path.read_text(encoding="utf-8"))
        stable_keys = set(repair_config) - {"created_at_utc", "runtime_provenance"}
        if any(existing.get(key) != repair_config.get(key) for key in stable_keys):
            raise RuntimeError("Existing repair config differs from this repair run.")
    else:
        campaign.atomic_write_json(repair_config_path, repair_config)

    _seed_repair_workers(
        tests_by_id=tests_by_id,
        before_by_id=before_by_id,
        partitions=partitions,
        output_root=output_root,
        repair_root=repair_root,
    )
    status_path = repair_root / "status.json"
    for round_number in range(1, args.max_attempts + 1):
        commands = []
        for worker_index, test_ids in enumerate(partitions):
            worker_dir = repair_root / f"worker{worker_index}"
            worker_results = load_jsonl(worker_dir / "results.jsonl")
            if not _retry_pending(worker_results, max_attempts=args.max_attempts):
                continue
            selected_tests = [tests_by_id[test_id] for test_id in test_ids]
            commands.append(
                (
                    f"worker{worker_index}",
                    qwen.build_evaluator_command(
                        tests=selected_tests,
                        test_set=test_set,
                        output_dir=worker_dir,
                        cache_root=cache_root,
                        render_root=render_root,
                        max_attempts=args.max_attempts,
                        render_missing=True,
                        retry_stop_reasons=(STOP_REASON,),
                    ),
                    repair_root / "logs" / f"worker{worker_index}.log",
                )
            )
        if not commands:
            break
        qwen.run_round(commands, status_path=status_path, round_number=round_number)

    repaired_records: list[dict] = []
    for worker_index in range(2):
        repaired_records.extend(load_jsonl(repair_root / f"worker{worker_index}" / "results.jsonl"))
    repaired_by_id = {str(record["test_id"]): record for record in repaired_records}
    unresolved = [
        test_id
        for test_id, record in repaired_by_id.items()
        if record.get("stop_reason") in RETRYABLE_STOP_REASONS
    ]
    if unresolved:
        campaign.atomic_write_json(
            status_path,
            {"state": "unresolved", "test_ids": sorted(unresolved)},
        )
        raise RuntimeError(f"Repair exhausted attempts for: {sorted(unresolved)}")

    report = _merge_into_campaign(
        tests=tests,
        repaired_by_id=repaired_by_id,
        output_root=output_root,
        repair_root=repair_root,
        before_sha256=before_sha256,
    )
    final_manifest = dict(repair_config)
    final_manifest.update(
        {
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "state": "complete",
            "final_stop_reasons": report["stop_no_manifest_repair"]["final_stop_reasons"],
            "new_render_pano_ids": report["stop_no_manifest_repair"]["new_render_pano_ids"],
            "new_image_requests": report["stop_no_manifest_repair"]["new_image_requests"],
            "after_results_sha256": report["stop_no_manifest_repair"]["after_results_sha256"],
        }
    )
    campaign.atomic_write_json(repair_root / "repair_manifest.json", final_manifest)
    campaign.atomic_write_json(
        status_path,
        {"state": "complete", "manifest": str(repair_root / "repair_manifest.json")},
    )
    print(json.dumps(report["stop_no_manifest_repair"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
