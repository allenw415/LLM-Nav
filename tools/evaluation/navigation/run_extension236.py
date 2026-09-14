from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile


PROJECT_ROOT = Path(__file__).resolve().parents[3]
EVALUATOR = PROJECT_ROOT / "tools/evaluation/navigation/evaluate_navigation_batch.py"
REPORTER = PROJECT_ROOT / "tools/evaluation/navigation/report_navigation_grid.py"
DATASET_DIR = (
    PROJECT_ROOT
    / "outputs/navigation_paths/floor0_semantic_targets_topology_constrained/"
    "room_count_2to6_ratio_1p0_2p0_per20_sparse485"
)
EVALUATION_DIR = (
    PROJECT_ROOT
    / "outputs/navigation_evaluation/floor0_semantic_targets_topology_constrained/"
    "room_count_2to6_ratio_1p0_2p0_per20_sparse485"
)
LEGACY_DIR = (
    PROJECT_ROOT
    / "outputs/navigation_evaluation/floor0_semantic_targets_topology_constrained/"
    "room_count_2to6_ratio_1p0_2p0_per10_sparse249/llm_nav_combined249"
)
DEFAULT_DATASET = DATASET_DIR / "test_set.jsonl"
DEFAULT_TEST_SET = DATASET_DIR / "extension_test_set.jsonl"
DEFAULT_LEGACY_RESULTS = LEGACY_DIR / "results.jsonl"
DEFAULT_LEGACY_SUMMARY = LEGACY_DIR / "summary.json"
DEFAULT_OUTPUT_DIR = EVALUATION_DIR / "llm_nav_extension236"
DEFAULT_COMBINED_OUTPUT_DIR = EVALUATION_DIR / "llm_nav_combined485"
DEFAULT_RENDER_DIR = PROJECT_ROOT / "renders/room_grounding_fov90"
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"
START_PROGRESS_PATTERN = re.compile(
    r"^\[start (?P<position>\d+)/(?P<total>\d+)\] (?P<test_id>\S+)$"
)
COMPLETED_PROGRESS_PATTERN = re.compile(
    r"^\[(?P<position>\d+)/(?P<total>\d+)\] (?P<test_id>\S+) "
    r"success=(?P<success>True|False)\b"
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload: object) -> None:
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
        temporary_path = Path(stream.name)
    os.replace(temporary_path, path)


def _load_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if raw_line.strip():
            payload = json.loads(raw_line)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object in {path}.")
            records.append(payload)
    return records


def _load_json_object(path: Path) -> dict:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}.")
    return payload


def parse_evaluator_progress_line(line: str) -> dict | None:
    stripped = line.strip()
    match = START_PROGRESS_PATTERN.fullmatch(stripped)
    if match is not None:
        return {
            "event": "started",
            "position": int(match.group("position")),
            "total": int(match.group("total")),
            "test_id": match.group("test_id"),
        }
    match = COMPLETED_PROGRESS_PATTERN.match(stripped)
    if match is not None:
        return {
            "event": "completed",
            "position": int(match.group("position")),
            "total": int(match.group("total")),
            "test_id": match.group("test_id"),
            "success": match.group("success") == "True",
        }
    return None


def validate_complete_result_ids(
    expected_test_ids: tuple[str, ...],
    records: list[dict],
    *,
    label: str,
) -> None:
    actual_test_ids = tuple(str(record.get("test_id")) for record in records)
    if actual_test_ids == expected_test_ids:
        return
    expected = set(expected_test_ids)
    actual = set(actual_test_ids)
    raise ValueError(
        f"{label} test IDs differ: missing={sorted(expected - actual)}, "
        f"extra={sorted(actual - expected)}, duplicate_count="
        f"{len(actual_test_ids) - len(actual)}."
    )


def build_initial_status(
    *,
    test_set: Path,
    output_dir: Path,
    expected_case_count: int,
    existing_results: list[dict],
    previous_status: dict,
) -> dict:
    now = _utc_now()
    attempt_count = int(previous_status.get("attempt_count", 0)) + 1
    return {
        "schema": "llm_nav_extension_run_status",
        "schema_version": 2,
        "started_at": previous_status.get("started_at", now),
        "attempt_started_at": now,
        "attempt_count": attempt_count,
        "process_restart_count": max(attempt_count - 1, 0),
        "updated_at": now,
        "phase": "starting",
        "status": "running",
        "test_set": str(test_set),
        "output_dir": str(output_dir),
        "runner_pid": os.getpid(),
        "child_pid": None,
        "expected_case_count": expected_case_count,
        "completed_case_count": len(existing_results),
        "last_completed_test_id": (
            existing_results[-1].get("test_id") if existing_results else None
        ),
        "last_completed_success": (
            bool(existing_results[-1].get("success")) if existing_results else None
        ),
        "last_progress_at": previous_status.get("last_progress_at"),
        "current_test_id": None,
        "current_test_started_at": None,
        "retry_test_ids": [],
        "completed_retry_count": 0,
    }


def _update_status(path: Path, status: dict, **updates: object) -> None:
    status.update(updates)
    status["updated_at"] = _utc_now()
    _atomic_write_json(path, status)


def _run_logged(
    command: list[str],
    *,
    log_path: Path,
    status_path: Path,
    status: dict,
    progress_results_path: Path | None = None,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_stream:
        log_stream.write(f"\n[{_utc_now()}] command: {' '.join(command)}\n")
        log_stream.flush()
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        _update_status(status_path, status, child_pid=process.pid)
        assert process.stdout is not None
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log_stream.write(line)
                log_stream.flush()
                progress = parse_evaluator_progress_line(line)
                if progress is None:
                    continue
                if progress["event"] == "started":
                    _update_status(
                        status_path,
                        status,
                        current_test_id=progress["test_id"],
                        current_test_started_at=_utc_now(),
                    )
                    continue
                updates: dict[str, object] = {
                    "last_completed_test_id": progress["test_id"],
                    "last_completed_success": progress["success"],
                    "last_progress_at": _utc_now(),
                    "current_test_id": None,
                    "current_test_started_at": None,
                }
                if progress_results_path is not None and progress_results_path.is_file():
                    updates["completed_case_count"] = len(
                        _load_jsonl(progress_results_path)
                    )
                _update_status(status_path, status, **updates)
        finally:
            return_code = process.wait()
            _update_status(status_path, status, child_pid=None)
        if return_code != 0:
            raise RuntimeError(
                f"Command exited with status {return_code}: {' '.join(command)}"
            )


def _evaluator_command(
    *,
    test_set: Path,
    output_dir: Path,
    test_id: str | None = None,
) -> list[str]:
    command = [
        sys.executable,
        "-u",
        str(EVALUATOR),
        "--test-set",
        str(test_set),
        "--output-dir",
        str(output_dir),
        "--render-output-dir",
        str(DEFAULT_RENDER_DIR),
        "--floor",
        "0",
        "--seed",
        "42",
        "--localization-interval",
        "3",
        "--llm-model",
        "gemma-4-31b-it",
        "--llm-api-kind",
        "responses",
        "--llm-api-base",
        GEMINI_API_BASE,
        "--llm-timeout",
        "300",
        "--render-missing",
    ]
    if test_id is not None:
        command.extend(["--test-id", test_id])
    return command


def _reporter_command(
    *,
    dataset: Path,
    extension_test_set: Path,
    legacy_results: Path,
    legacy_summary: Path,
    extension_results: Path,
    extension_output_dir: Path,
    combined_output_dir: Path,
    replacement_paths: list[Path],
) -> list[str]:
    command = [
        sys.executable,
        "-u",
        str(REPORTER),
        "--dataset",
        str(dataset),
        "--extension-test-set",
        str(extension_test_set),
        "--legacy-results",
        str(legacy_results),
        "--legacy-summary",
        str(legacy_summary),
        "--extension-results",
        str(extension_results),
        "--extension-output-dir",
        str(extension_output_dir),
        "--combined-output-dir",
        str(combined_output_dir),
    ]
    for replacement_path in replacement_paths:
        command.extend(["--replacement-results", str(replacement_path)])
    return command


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the additive TEST250-TEST485 extension, retry execution errors once, "
            "then build Extension 236 and Combined 485 reports."
        )
    )
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--test-set", default=str(DEFAULT_TEST_SET))
    parser.add_argument("--legacy-results", default=str(DEFAULT_LEGACY_RESULTS))
    parser.add_argument("--legacy-summary", default=str(DEFAULT_LEGACY_SUMMARY))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--combined-output-dir",
        default=str(DEFAULT_COMBINED_OUTPUT_DIR),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    dataset = Path(args.dataset).resolve()
    test_set = Path(args.test_set).resolve()
    legacy_results = Path(args.legacy_results).resolve()
    legacy_summary = Path(args.legacy_summary).resolve()
    output_dir = Path(args.output_dir).resolve()
    combined_output_dir = Path(args.combined_output_dir).resolve()
    status_path = output_dir / "run_status.json"
    log_path = output_dir / "run.log"
    results_path = output_dir / "results.jsonl"

    tests = _load_jsonl(test_set)
    expected_test_ids = tuple(str(test.get("test_id")) for test in tests)
    if not expected_test_ids or len(expected_test_ids) != len(set(expected_test_ids)):
        raise ValueError("Extension test set must contain unique non-empty test IDs.")
    existing_results = _load_jsonl(results_path) if results_path.is_file() else []
    previous_status = _load_json_object(status_path)
    status = build_initial_status(
        test_set=test_set,
        output_dir=output_dir,
        expected_case_count=len(expected_test_ids),
        existing_results=existing_results,
        previous_status=previous_status,
    )
    _atomic_write_json(status_path, status)

    try:
        _update_status(status_path, status, phase="extension_batch")
        _run_logged(
            _evaluator_command(test_set=test_set, output_dir=output_dir),
            log_path=log_path,
            status_path=status_path,
            status=status,
            progress_results_path=results_path,
        )

        records = _load_jsonl(results_path)
        validate_complete_result_ids(
            expected_test_ids,
            records,
            label="extension results",
        )
        error_records = [record for record in records if record.get("error")]
        retry_paths: list[Path] = []
        _update_status(
            status_path,
            status,
            phase="error_reruns",
            completed_case_count=len(records),
            current_test_id=None,
            current_test_started_at=None,
            retry_test_ids=[record["test_id"] for record in error_records],
        )
        for retry_index, record in enumerate(error_records, start=1):
            test_id = str(record["test_id"])
            retry_output_dir = output_dir / "reruns" / "api_errors" / test_id
            _update_status(
                status_path,
                status,
                current_test_id=test_id,
                current_test_started_at=_utc_now(),
            )
            _run_logged(
                _evaluator_command(
                    test_set=test_set,
                    output_dir=retry_output_dir,
                    test_id=test_id,
                ),
                log_path=log_path,
                status_path=status_path,
                status=status,
            )
            retry_paths.append(retry_output_dir / "results.jsonl")
            _update_status(
                status_path,
                status,
                completed_retry_count=retry_index,
                current_test_id=None,
                current_test_started_at=None,
            )

        _update_status(status_path, status, phase="reporting")
        _run_logged(
            _reporter_command(
                dataset=dataset,
                extension_test_set=test_set,
                legacy_results=legacy_results,
                legacy_summary=legacy_summary,
                extension_results=results_path,
                extension_output_dir=output_dir,
                combined_output_dir=combined_output_dir,
                replacement_paths=retry_paths,
            ),
            log_path=log_path,
            status_path=status_path,
            status=status,
        )

        final_records = _load_jsonl(output_dir / "final_results.jsonl")
        validate_complete_result_ids(
            expected_test_ids,
            final_records,
            label="final extension results",
        )
        _update_status(
            status_path,
            status,
            phase="complete",
            status="complete",
            completed_case_count=len(final_records),
            completed_at=_utc_now(),
            current_test_id=None,
            current_test_started_at=None,
            final_error_test_ids=[
                record["test_id"] for record in final_records if record.get("error")
            ],
            meeting_summary=str(output_dir / "meeting_summary.md"),
            combined_meeting_summary=str(combined_output_dir / "meeting_summary.md"),
        )
        return 0
    except BaseException as error:
        _update_status(
            status_path,
            status,
            status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            current_test_id=None,
            current_test_started_at=None,
            error={"type": type(error).__name__, "message": str(error)},
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
