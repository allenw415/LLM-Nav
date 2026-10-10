from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.evaluation.navigation import retry_incremental_api_errors as retry_helpers
from tools.evaluation.navigation import run_cardinal4_initial_offset_incremental as campaign


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Retry every API/runtime error in the completed cardinal4 stages, "
            "atomically replace only successful retries, and rebuild all reports."
        )
    )
    parser.add_argument("--run-id", help="Retry directory name; defaults to a UTC timestamp.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--llm-timeout", type=float, default=600.0)
    parser.add_argument("--max-http-retries", type=int, default=5)
    return parser


def expected_stage_ids(dataset: list[dict], stage: dict) -> set[str]:
    return {
        str(record["test_id"])
        for record in dataset
        if stage["minimum"] <= campaign.test_number(record.get("test_id")) <= stage["maximum"]
    }


def retry_command(
    *,
    test_ids: list[str],
    output_dir: Path,
    direction_policy: str,
    llm_timeout: float,
) -> list[str]:
    command = campaign.evaluation_command(
        test_ids=test_ids,
        output_dir=output_dir,
        direction_policy=direction_policy,
    )
    timeout_index = command.index("--llm-timeout") + 1
    command[timeout_index] = f"{llm_timeout:g}"
    return command


def merge_retry_records(
    *,
    originals: list[dict],
    requested_ids: list[str],
    retry_by_id: dict[str, dict],
) -> tuple[list[dict], dict]:
    requested_set = set(requested_ids)
    replaced_ids: list[str] = []
    retry_error_ids: list[str] = []
    missing_retry_ids: list[str] = []
    merged: list[dict] = []

    for original_record in originals:
        record = original_record
        test_id = str(record["test_id"])
        if test_id in requested_set:
            if not retry_helpers.is_api_error(record):
                raise RuntimeError(f"Requested ID is no longer an API error: {test_id}")
            replacement = retry_by_id.get(test_id)
            if replacement is None:
                missing_retry_ids.append(test_id)
            elif replacement.get("error") is None:
                record = replacement
                replaced_ids.append(test_id)
            else:
                retry_error_ids.append(test_id)
        merged.append(record)

    merged.sort(key=lambda record: campaign.test_number(record["test_id"]))
    remaining_error_ids = [
        str(record["test_id"])
        for record in merged
        if retry_helpers.is_api_error(record)
    ]
    return merged, {
        "requested_retry_count": len(requested_ids),
        "replaced_count": len(replaced_ids),
        "replaced_ids": replaced_ids,
        "retry_error_count": len(retry_error_ids),
        "retry_error_ids": retry_error_ids,
        "missing_retry_count": len(missing_retry_ids),
        "missing_retry_ids": missing_retry_ids,
        "remaining_api_error_count": len(remaining_error_ids),
        "remaining_api_error_ids": remaining_error_ids,
        "successful_original_records_overwritten": 0,
    }


def main() -> int:
    args = build_parser().parse_args()
    if args.poll_seconds <= 0:
        raise ValueError("--poll-seconds must be positive")
    if args.llm_timeout <= 0:
        raise ValueError("--llm-timeout must be positive")
    if args.max_http_retries <= 0:
        raise ValueError("--max-http-retries must be positive")

    os.chdir(PROJECT_ROOT)
    campaign.load_environment_file(PROJECT_ROOT / ".env")
    keys = campaign.validate_api_keys()
    dataset = campaign.load_jsonl(campaign.DATASET)
    if len(dataset) != 715:
        raise RuntimeError(f"Expected 715 dataset records, found {len(dataset)}")

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_id):
        raise ValueError("--run-id may contain only letters, digits, dot, underscore, and dash")
    output_root = campaign.EXPERIMENT_ROOT / "retries/all_stage_api_errors" / run_id
    if output_root.exists():
        raise RuntimeError(f"Retry output already exists: {output_root}")

    source_records: dict[tuple[str, str], list[dict]] = {}
    source_paths: dict[tuple[str, str], Path] = {}
    source_hashes: dict[tuple[str, str], str] = {}
    stage_error_ids: dict[tuple[str, str], list[str]] = {}
    condition_error_ids: dict[str, list[str]] = {
        condition["name"]: [] for condition in campaign.CONDITIONS
    }

    for stage in campaign.STAGES:
        required_ids = expected_stage_ids(dataset, stage)
        for condition in campaign.CONDITIONS:
            condition_name = condition["name"]
            source_path = (
                campaign.EXPERIMENT_ROOT
                / "stages"
                / stage["name"]
                / condition_name
                / "results.jsonl"
            )
            records = campaign.load_jsonl(source_path)
            record_ids = [str(record.get("test_id")) for record in records]
            if len(record_ids) != len(set(record_ids)):
                raise RuntimeError(f"Duplicate test IDs in {source_path}")
            if set(record_ids) != required_ids:
                raise RuntimeError(
                    f"Coverage mismatch in {source_path}: "
                    f"expected={len(required_ids)} actual={len(set(record_ids))}"
                )
            non_api_errors = [
                str(record["test_id"])
                for record in records
                if record.get("error") is not None and not retry_helpers.is_api_error(record)
            ]
            if non_api_errors:
                raise RuntimeError(
                    f"Found non-API errors in {source_path}; refusing broad retry: "
                    f"{' '.join(non_api_errors)}"
                )
            ids = sorted(
                [
                    str(record["test_id"])
                    for record in records
                    if retry_helpers.is_api_error(record)
                ],
                key=campaign.test_number,
            )
            key = (stage["name"], condition_name)
            source_records[key] = records
            source_paths[key] = source_path
            source_hashes[key] = sha256(source_path)
            stage_error_ids[key] = ids
            condition_error_ids[condition_name].extend(ids)

    preview = {
        "created_at": utc_now(),
        "experiment_root": str(campaign.EXPERIMENT_ROOT),
        "render_root": str(campaign.RENDER_ROOT),
        "render_heading_mode": "cardinal4",
        "llm_timeout_seconds": args.llm_timeout,
        "max_http_retries": args.max_http_retries,
        "condition_execution": "concurrent",
        "conditions": {},
    }
    for condition in campaign.CONDITIONS:
        condition_name = condition["name"]
        ids = condition_error_ids[condition_name]
        signatures = Counter()
        for stage in campaign.STAGES:
            for record in source_records[(stage["name"], condition_name)]:
                if retry_helpers.is_api_error(record):
                    signatures[retry_helpers.api_error_signature(record)] += 1
        preview["conditions"][condition_name] = {
            "api_key_source": condition["api_key_source"],
            "api_error_count": len(ids),
            "api_error_ids": ids,
            "api_error_signatures": dict(signatures),
        }

    if args.dry_run:
        print(json.dumps(preview, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    output_root.mkdir(parents=True, exist_ok=False)
    audit_root = output_root / "audit"
    campaign.atomic_write_json(audit_root / "initial.json", preview)
    for key, source_path in source_paths.items():
        stage_name, condition_name = key
        snapshot_path = audit_root / "originals" / stage_name / condition_name / "results.jsonl"
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, snapshot_path)

    jobs: list[tuple[str, subprocess.Popen, object]] = []
    process_exit_codes: dict[str, int] = {}
    for condition in campaign.CONDITIONS:
        condition_name = condition["name"]
        ids = condition_error_ids[condition_name]
        retry_dir = output_root / condition_name
        retry_dir.mkdir(parents=True, exist_ok=True)
        log_path = retry_dir / "run.log"
        log_stream = log_path.open("a", encoding="utf-8")
        log_stream.write(
            f"[{utc_now()}] retry_count={len(ids)} "
            f"key_source={condition['api_key_source']} "
            f"llm_timeout={args.llm_timeout:g} max_http_retries={args.max_http_retries}\n"
        )
        log_stream.flush()
        if not ids:
            log_stream.close()
            process_exit_codes[condition_name] = 0
            continue
        environment = os.environ.copy()
        environment["SA_NAV_PROFILE_GEMINI_API_KEY"] = keys[condition_name]
        environment["SA_NAV_MAX_HTTP_RETRIES"] = str(args.max_http_retries)
        process = subprocess.Popen(
            retry_command(
                test_ids=ids,
                output_dir=retry_dir,
                direction_policy=condition["direction_policy"],
                llm_timeout=args.llm_timeout,
            ),
            cwd=PROJECT_ROOT,
            stdout=log_stream,
            stderr=subprocess.STDOUT,
            env=environment,
            start_new_session=True,
        )
        jobs.append((condition_name, process, log_stream))
        print(
            f"[{utc_now()}] {condition_name} retry_count={len(ids)} pid={process.pid} "
            f"key_source={condition['api_key_source']} output={retry_dir}",
            flush=True,
        )

    while jobs:
        active: list[tuple[str, subprocess.Popen, object]] = []
        for condition_name, process, log_stream in jobs:
            return_code = process.poll()
            if return_code is None:
                active.append((condition_name, process, log_stream))
                continue
            log_stream.close()
            process_exit_codes[condition_name] = return_code
            print(
                f"[{utc_now()}] {condition_name} retry exited {return_code}",
                flush=True,
            )
        jobs = active
        if jobs:
            time.sleep(args.poll_seconds)

    retry_by_condition: dict[str, dict[str, dict]] = {}
    for condition in campaign.CONDITIONS:
        condition_name = condition["name"]
        retry_records = campaign.load_jsonl(output_root / condition_name / "results.jsonl")
        retry_by_id = {str(record["test_id"]): record for record in retry_records}
        if len(retry_by_id) != len(retry_records):
            raise RuntimeError(f"Duplicate retry records for {condition_name}")
        unexpected = set(retry_by_id) - set(condition_error_ids[condition_name])
        if unexpected:
            raise RuntimeError(
                f"Retry output for {condition_name} contains unrequested IDs: {sorted(unexpected)}"
            )
        retry_by_condition[condition_name] = retry_by_id

    for key, source_path in source_paths.items():
        if sha256(source_path) != source_hashes[key]:
            raise RuntimeError(f"Source changed during retry; refusing merge: {source_path}")

    merged_records: dict[tuple[str, str], list[dict]] = {}
    merge_audit: dict[str, dict] = {}
    for stage in campaign.STAGES:
        stage_name = stage["name"]
        merge_audit[stage_name] = {}
        for condition in campaign.CONDITIONS:
            condition_name = condition["name"]
            key = (stage_name, condition_name)
            merged, audit = merge_retry_records(
                originals=source_records[key],
                requested_ids=stage_error_ids[key],
                retry_by_id=retry_by_condition[condition_name],
            )
            audit["source_path"] = str(source_paths[key])
            audit["source_sha256_before"] = source_hashes[key]
            merged_records[key] = merged
            merge_audit[stage_name][condition_name] = audit

    for key, merged in merged_records.items():
        stage_name, condition_name = key
        source_path = source_paths[key]
        campaign.atomic_write_jsonl(source_path, merged)
        campaign.atomic_write_jsonl(
            audit_root / "merged" / stage_name / condition_name / "results.jsonl",
            merged,
        )
        merge_audit[stage_name][condition_name]["source_sha256_after"] = sha256(source_path)

    report_refresh_error: str | None = None
    try:
        for stage_index in range(len(campaign.STAGES)):
            campaign.build_milestone_reports(dataset, stage_index)
        campaign.build_view_count_report()
    except Exception as exc:
        report_refresh_error = f"{type(exc).__name__}: {exc}"
        print(f"report refresh failed: {report_refresh_error}", flush=True)

    remaining_errors = sum(
        audit["remaining_api_error_count"]
        for stage_audit in merge_audit.values()
        for audit in stage_audit.values()
    )
    status = {
        "completed_at": utc_now(),
        "status": "complete" if remaining_errors == 0 and report_refresh_error is None else "incomplete",
        "process_exit_codes": process_exit_codes,
        "remaining_api_error_count": remaining_errors,
        "report_refresh_error": report_refresh_error,
        "stages": merge_audit,
    }
    campaign.atomic_write_json(audit_root / "merge.json", status)
    campaign.atomic_write_json(output_root / "status.json", status)
    print(
        f"[{utc_now()}] cardinal4 retry {status['status']}; "
        f"remaining_api_errors={remaining_errors}",
        flush=True,
    )
    return 0 if status["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
