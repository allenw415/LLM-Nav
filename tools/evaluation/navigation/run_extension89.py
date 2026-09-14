from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


PROJECT_ROOT = Path(__file__).resolve().parents[3]
EVALUATOR = PROJECT_ROOT / "tools/evaluation/navigation/evaluate_navigation_batch.py"
REPORTER = PROJECT_ROOT / "tools/evaluation/navigation/report_navigation_grid.py"
DEFAULT_TEST_SET = (
    PROJECT_ROOT
    / "outputs/navigation_paths/floor0_semantic_targets_topology_constrained/"
    "room_count_2to6_ratio_1p0_2p0_per10_sparse249/extension_test_set.jsonl"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs/navigation_evaluation/floor0_semantic_targets_topology_constrained/"
    "room_count_2to6_ratio_1p0_2p0_per10_sparse249/llm_nav_extension89"
)
DEFAULT_RENDER_DIR = PROJECT_ROOT / "renders/room_grounding_fov90"
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"


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


def _run_logged(command: list[str], *, log_path: Path) -> None:
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
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_stream.write(line)
            log_stream.flush()
        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(f"Command exited with status {return_code}: {' '.join(command)}")


def _evaluator_command(*, test_set: Path, output_dir: Path, test_id: str | None = None) -> list[str]:
    command = [
        sys.executable,
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run TEST161-TEST249, retry final errors once, then build extension and combined reports."
    )
    parser.add_argument("--test-set", default=str(DEFAULT_TEST_SET))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    test_set = Path(args.test_set).resolve()
    output_dir = Path(args.output_dir).resolve()
    status_path = output_dir / "run_status.json"
    log_path = output_dir / "run.log"
    status = {
        "schema": "llm_nav_extension89_run_status",
        "schema_version": 1,
        "started_at": _utc_now(),
        "updated_at": _utc_now(),
        "phase": "starting",
        "status": "running",
        "test_set": str(test_set),
        "output_dir": str(output_dir),
        "completed_case_count": 0,
        "retry_test_ids": [],
    }
    _atomic_write_json(status_path, status)

    try:
        status.update({"phase": "extension_batch", "updated_at": _utc_now()})
        _atomic_write_json(status_path, status)
        _run_logged(
            _evaluator_command(test_set=test_set, output_dir=output_dir),
            log_path=log_path,
        )

        results_path = output_dir / "results.jsonl"
        records = _load_jsonl(results_path)
        error_records = [record for record in records if record.get("error")]
        retry_paths: list[Path] = []
        status.update(
            {
                "phase": "error_reruns",
                "updated_at": _utc_now(),
                "completed_case_count": len(records),
                "retry_test_ids": [record["test_id"] for record in error_records],
            }
        )
        _atomic_write_json(status_path, status)
        for record in error_records:
            test_id = record["test_id"]
            retry_output_dir = output_dir / "reruns" / "api_errors" / test_id
            _run_logged(
                _evaluator_command(
                    test_set=test_set,
                    output_dir=retry_output_dir,
                    test_id=test_id,
                ),
                log_path=log_path,
            )
            retry_paths.append(retry_output_dir / "results.jsonl")

        status.update({"phase": "reporting", "updated_at": _utc_now()})
        _atomic_write_json(status_path, status)
        report_command = [
            sys.executable,
            str(REPORTER),
            "--extension-results",
            str(results_path),
            "--extension-output-dir",
            str(output_dir),
        ]
        for retry_path in retry_paths:
            report_command.extend(["--replacement-results", str(retry_path)])
        _run_logged(report_command, log_path=log_path)

        final_records = _load_jsonl(output_dir / "final_results.jsonl")
        status.update(
            {
                "phase": "complete",
                "status": "complete",
                "updated_at": _utc_now(),
                "completed_at": _utc_now(),
                "completed_case_count": len(final_records),
                "final_error_test_ids": [
                    record["test_id"] for record in final_records if record.get("error")
                ],
                "meeting_summary": str(output_dir / "meeting_summary.md"),
            }
        )
        _atomic_write_json(status_path, status)
        return 0
    except BaseException as error:
        status.update(
            {
                "status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                "updated_at": _utc_now(),
                "error": {"type": type(error).__name__, "message": str(error)},
            }
        )
        _atomic_write_json(status_path, status)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
