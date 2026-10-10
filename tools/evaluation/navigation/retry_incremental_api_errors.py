from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET = PROJECT_ROOT / (
    "outputs/navigation_paths/floor0_semantic_targets_topology_constrained/"
    "room_count_2to6_ratio_1p0_2p0_per30_sparse715/test_set.jsonl"
)
ROOT = PROJECT_ROOT / (
    "outputs/navigation_evaluation/floor0_semantic_targets_topology_constrained/"
    "room_count_2to6_ratio_1p0_2p0_per30_sparse715/initial_offset_incremental_10_20_30"
)
EVALUATOR = PROJECT_ROOT / "tools/evaluation/navigation/evaluate_navigation_batch.py"
PAIRED_REPORTER = PROJECT_ROOT / "tools/evaluation/navigation/report_initial_offset_experiment.py"
STAGES = {"per10": (1, 249), "per20": (250, 485), "per30": (486, 715)}
MILESTONE_MAXIMUMS = {"per10": 249, "per20": 485, "per30": 715}
CONDITIONS = (
    ("without_trajectory", "geometry", "active_profile"),
    ("with_trajectory", "geometry-edge-visit", "SA_NAV_WITH_TRAJECTORY_API_KEY"),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Retry only API-error records from an incremental navigation stage or cumulative milestone, "
            "then atomically replace only successful retries while preserving an audit snapshot."
        )
    )
    parser.add_argument("--stage", choices=sorted(STAGES), default="per10")
    parser.add_argument("--scope", choices=("stage", "milestone"), default="stage")
    parser.add_argument("--run-id", help="Audit/retry directory name; defaults to a UTC timestamp.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--llm-timeout", type=float, default=300.0)
    parser.add_argument("--max-http-retries", type=int, default=5)
    parser.add_argument(
        "--serial-conditions",
        action="store_true",
        help="Run Without trajectory to completion before starting With trajectory.",
    )
    parser.add_argument("--skip-report-refresh", action="store_true")
    return parser


def load_env(path: Path) -> None:
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if value and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if key:
            os.environ[key] = value


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def atomic_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_number(test_id: object) -> int:
    if not isinstance(test_id, str) or not re.fullmatch(r"TEST\d+", test_id):
        raise ValueError(f"Invalid test_id: {test_id!r}")
    return int(test_id[4:])


def target_root(stage: str, scope: str) -> Path:
    return ROOT / ("stages" if scope == "stage" else "milestones") / stage


def expected_ids(dataset: list[dict], stage: str, scope: str) -> set[str]:
    minimum, maximum = STAGES[stage] if scope == "stage" else (1, MILESTONE_MAXIMUMS[stage])
    return {
        str(record["test_id"])
        for record in dataset
        if minimum <= test_number(record.get("test_id")) <= maximum
    }


def api_error_signature(record: dict) -> str:
    error = record.get("error")
    if not isinstance(error, dict):
        return "not_api_error"
    error_type = str(error.get("type") or "unknown")
    message = str(error.get("message") or "")
    http_match = re.search(r"HTTP\s+(\d{3})", message, re.IGNORECASE)
    if http_match:
        return f"{error_type}:HTTP{http_match.group(1)}"
    if "timed out" in message.lower() or error_type == "TimeoutError":
        return f"{error_type}:timeout"
    if error_type in {"RemoteDisconnected", "ConnectionError", "URLError"}:
        return f"{error_type}:connection"
    return f"{error_type}:other"


def is_api_error(record: dict) -> bool:
    error = record.get("error")
    if not isinstance(error, dict):
        return False
    error_type = str(error.get("type") or "")
    message = str(error.get("message") or "").lower()
    return (
        "model api" in message
        or bool(re.search(r"http\s+[45]\d{2}", message))
        or "timed out" in message
        or error_type in {"TimeoutError", "RemoteDisconnected", "ConnectionError", "URLError"}
    )


def validate_source_records(path: Path, records: list[dict], required_ids: set[str]) -> None:
    ids = [record.get("test_id") for record in records]
    if len(ids) != len(set(ids)):
        raise RuntimeError(f"Duplicate test IDs in {path}")
    if set(ids) != required_ids:
        raise RuntimeError(f"Coverage mismatch in {path}: expected={len(required_ids)} actual={len(set(ids))}")
    non_api_errors = [
        str(record["test_id"])
        for record in records
        if record.get("error") is not None and not is_api_error(record)
    ]
    if non_api_errors:
        raise RuntimeError(f"Found non-API errors in {path}; refusing broad retry: {' '.join(non_api_errors)}")


def error_ids(records: list[dict]) -> list[str]:
    return sorted([str(record["test_id"]) for record in records if is_api_error(record)], key=test_number)


def command(
    test_ids: list[str],
    output_dir: Path,
    direction_policy: str,
    llm_timeout: float,
) -> list[str]:
    args = [
        sys.executable,
        str(EVALUATOR),
        "--test-set", str(DATASET),
        "--output-dir", str(output_dir),
        "--artifacts-dir", str(PROJECT_ROOT / "dataset/sites/british_museum/normalized"),
        "--render-output-dir", str(PROJECT_ROOT / "renders/room_grounding_fov90"),
        "--render-missing",
        "--direction-policy", direction_policy,
        "--heading-alignment-mode", "initial-spatial",
        "--force-initial-visual-refresh",
        "--localization-interval", "3",
        "--start-heading", "0",
        "--seed", "42",
        "--llm-timeout", str(llm_timeout),
    ]
    for test_id in test_ids:
        args.extend(("--test-id", test_id))
    return args


def active_profile_key() -> tuple[str, str]:
    profile = os.environ.get("SA_NAV_ACTIVE_PROFILE", "gemini").strip().upper()
    variable = f"SA_NAV_PROFILE_{profile}_API_KEY"
    value = os.environ.get(variable)
    if not value:
        raise RuntimeError(f"Missing active-profile API key variable: {variable}")
    return variable, value


def condition_environment(key_source: str) -> tuple[dict[str, str], str]:
    environment = os.environ.copy()
    if key_source == "active_profile":
        variable, _ = active_profile_key()
        return environment, variable
    secondary = os.environ.get(key_source)
    if not secondary:
        raise RuntimeError(f"Missing {key_source} in .env")
    environment["SA_NAV_PROFILE_GEMINI_API_KEY"] = secondary
    return environment, key_source


def retry_root(stage: str, scope: str, run_id: str) -> Path:
    count_label = MILESTONE_MAXIMUMS[stage] if scope == "milestone" else STAGES[stage][1] - STAGES[stage][0] + 1
    return ROOT / "retries" / f"{scope}_{stage}_{count_label}_api_errors" / run_id


def replace_successful_records(
    *,
    source_path: Path,
    source_sha256: str,
    requested_ids: list[str],
    retry_dir: Path,
    merged_copy_path: Path,
) -> dict:
    if sha256(source_path) != source_sha256:
        raise RuntimeError(f"Source changed during retry; refusing merge: {source_path}")
    originals = read_jsonl(source_path)
    requested_set = set(requested_ids)
    retry_records = read_jsonl(retry_dir / "results.jsonl")
    retry_by_id = {str(record["test_id"]): record for record in retry_records}
    if len(retry_by_id) != len(retry_records):
        raise RuntimeError(f"Duplicate retry records in {retry_dir / 'results.jsonl'}")
    unexpected = set(retry_by_id) - requested_set
    if unexpected:
        raise RuntimeError(f"Retry output contains unrequested IDs: {sorted(unexpected)}")

    replaced_ids: list[str] = []
    retry_error_ids: list[str] = []
    missing_retry_ids: list[str] = []
    merged: list[dict] = []
    for original in originals:
        test_id = str(original["test_id"])
        replacement = retry_by_id.get(test_id)
        if test_id in requested_set:
            if not is_api_error(original):
                raise RuntimeError(f"Requested ID is no longer an API error: {test_id}")
            if replacement is None:
                missing_retry_ids.append(test_id)
            elif replacement.get("error") is None:
                original = replacement
                replaced_ids.append(test_id)
            else:
                retry_error_ids.append(test_id)
        merged.append(original)

    merged.sort(key=lambda record: test_number(record["test_id"]))
    remaining_api_error_ids = [str(record["test_id"]) for record in merged if is_api_error(record)]
    atomic_jsonl(source_path, merged)
    atomic_jsonl(merged_copy_path, merged)
    return {
        "source_path": str(source_path),
        "source_sha256_before": source_sha256,
        "source_sha256_after": sha256(source_path),
        "source_record_count": len(merged),
        "requested_retry_count": len(requested_ids),
        "retry_record_count": len(retry_records),
        "replaced_count": len(replaced_ids),
        "replaced_ids": replaced_ids,
        "retry_error_count": len(retry_error_ids),
        "retry_error_ids": retry_error_ids,
        "missing_retry_count": len(missing_retry_ids),
        "missing_retry_ids": missing_retry_ids,
        "remaining_api_error_count": len(remaining_api_error_ids),
        "remaining_api_error_ids": remaining_api_error_ids,
        "successful_original_records_overwritten": 0,
    }


def refresh_reports(scope_root: Path, log_path: Path) -> None:
    from tools.evaluation.navigation.report_localization_stages import build_report

    for condition, _, _ in CONDITIONS:
        results_path = scope_root / condition / "results.jsonl"
        build_report(results_path=results_path, output_dir=scope_root / condition / "localization_report")
    with log_path.open("a", encoding="utf-8") as log:
        subprocess.run(
            [
                sys.executable,
                str(PAIRED_REPORTER),
                "--without-trajectory", str(scope_root / "without_trajectory"),
                "--with-trajectory", str(scope_root / "with_trajectory"),
                "--output-dir", str(scope_root / "paired_report"),
            ],
            cwd=PROJECT_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )


def main() -> int:
    args = build_parser().parse_args()
    if args.poll_seconds <= 0:
        raise ValueError("--poll-seconds must be positive")
    if args.llm_timeout <= 0:
        raise ValueError("--llm-timeout must be positive")
    if args.max_http_retries <= 0:
        raise ValueError("--max-http-retries must be positive")
    os.chdir(PROJECT_ROOT)
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    load_env(PROJECT_ROOT / ".env")
    _, active_key_value = active_profile_key()
    secondary_key_value = os.environ.get("SA_NAV_WITH_TRAJECTORY_API_KEY")
    if not secondary_key_value:
        raise RuntimeError("Missing SA_NAV_WITH_TRAJECTORY_API_KEY in .env")
    if active_key_value == secondary_key_value:
        raise RuntimeError("Without/With trajectory API keys must be distinct for this retry")

    dataset = read_jsonl(DATASET)
    if len(dataset) != 715:
        raise RuntimeError(f"Expected 715 dataset records, found {len(dataset)}")
    required_ids = expected_ids(dataset, args.stage, args.scope)
    scope_root = target_root(args.stage, args.scope)
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_id):
        raise ValueError("--run-id may contain only letters, digits, dot, underscore, and dash")
    output_root = retry_root(args.stage, args.scope, run_id)
    if output_root.exists():
        raise RuntimeError(f"Retry output already exists: {output_root}")

    sources: dict[str, Path] = {}
    initial_records: dict[str, list[dict]] = {}
    initial_error_ids: dict[str, list[str]] = {}
    source_hashes: dict[str, str] = {}
    key_sources: dict[str, str] = {}
    for condition, _, key_source in CONDITIONS:
        source_path = scope_root / condition / "results.jsonl"
        records = read_jsonl(source_path)
        validate_source_records(source_path, records, required_ids)
        sources[condition] = source_path
        initial_records[condition] = records
        initial_error_ids[condition] = error_ids(records)
        source_hashes[condition] = sha256(source_path)
        _, variable = condition_environment(key_source)
        key_sources[condition] = variable

    preview = {
        "stage": args.stage,
        "scope": args.scope,
        "expected_record_count_per_condition": len(required_ids),
        "output_root": str(output_root),
        "llm_timeout_seconds": args.llm_timeout,
        "max_http_retries": args.max_http_retries,
        "condition_execution": "serial" if args.serial_conditions else "concurrent",
        "conditions": {
            condition: {
                "source_path": str(sources[condition]),
                "source_sha256": source_hashes[condition],
                "api_key_source": key_sources[condition],
                "api_error_count": len(initial_error_ids[condition]),
                "api_error_signatures": dict(
                    Counter(
                        api_error_signature(record)
                        for record in initial_records[condition]
                        if is_api_error(record)
                    )
                ),
                "api_error_ids": initial_error_ids[condition],
            }
            for condition, _, _ in CONDITIONS
        },
        "keys_are_distinct": True,
    }
    if args.dry_run:
        print(json.dumps(preview, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    output_root.mkdir(parents=True, exist_ok=False)
    audit_root = output_root / "audit"
    for condition, _, _ in CONDITIONS:
        snapshot_path = audit_root / "originals" / condition / "results.jsonl"
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(sources[condition], snapshot_path)
    atomic_json(audit_root / "initial.json", {"created_at": utc_now(), **preview})

    jobs: list[tuple[str, subprocess.Popen, object, Path]] = []
    process_exit_codes: dict[str, int] = {}
    for condition, direction_policy, key_source in CONDITIONS:
        ids = initial_error_ids[condition]
        condition_retry_dir = output_root / condition
        condition_retry_dir.mkdir(parents=True, exist_ok=True)
        log = (condition_retry_dir / "run.log").open("a", encoding="utf-8")
        log.write(
            f"[{utc_now()}] retry_count={len(ids)} key_source={key_sources[condition]} "
            f"llm_timeout={args.llm_timeout:g} max_http_retries={args.max_http_retries}\n"
        )
        log.flush()
        if not ids:
            log.close()
            process_exit_codes[condition] = 0
            print(f"{condition}: no API-error cases", flush=True)
            continue
        environment, _ = condition_environment(key_source)
        environment["SA_NAV_MAX_HTTP_RETRIES"] = str(args.max_http_retries)
        process = subprocess.Popen(
            command(ids, condition_retry_dir, direction_policy, args.llm_timeout),
            cwd=PROJECT_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=environment,
            start_new_session=True,
        )
        print(
            f"[{utc_now()}] {condition} retry_count={len(ids)} pid={process.pid} "
            f"key_source={key_sources[condition]} output={condition_retry_dir}",
            flush=True,
        )
        if args.serial_conditions:
            return_code = process.wait()
            log.close()
            process_exit_codes[condition] = return_code
            print(f"[{utc_now()}] {condition} retry exited {return_code}", flush=True)
        else:
            jobs.append((condition, process, log, condition_retry_dir))

    while jobs:
        active = []
        for condition, process, log, condition_retry_dir in jobs:
            return_code = process.poll()
            if return_code is None:
                active.append((condition, process, log, condition_retry_dir))
                continue
            log.close()
            process_exit_codes[condition] = return_code
            print(f"[{utc_now()}] {condition} retry exited {return_code}", flush=True)
        jobs = active
        if jobs:
            time.sleep(args.poll_seconds)

    merge_audit: dict[str, dict] = {}
    for condition, _, _ in CONDITIONS:
        merge_audit[condition] = replace_successful_records(
            source_path=sources[condition],
            source_sha256=source_hashes[condition],
            requested_ids=initial_error_ids[condition],
            retry_dir=output_root / condition,
            merged_copy_path=audit_root / "merged" / condition / "results.jsonl",
        )
        merge_audit[condition]["retry_process_exit_code"] = process_exit_codes.get(condition)
        print(
            f"{condition}: replaced={merge_audit[condition]['replaced_count']} "
            f"remaining_api_errors={merge_audit[condition]['remaining_api_error_count']}",
            flush=True,
        )

    report_refresh_error: str | None = None
    if not args.skip_report_refresh:
        try:
            refresh_reports(scope_root, output_root / "report_refresh.log")
        except Exception as exc:
            report_refresh_error = f"{type(exc).__name__}: {exc}"
            print(f"report refresh failed: {report_refresh_error}", flush=True)
    atomic_json(
        audit_root / "merge.json",
        {
            "completed_at": utc_now(),
            "stage": args.stage,
            "scope": args.scope,
            "conditions": merge_audit,
            "report_refresh_error": report_refresh_error,
        },
    )
    return 1 if any(code != 0 for code in process_exit_codes.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
