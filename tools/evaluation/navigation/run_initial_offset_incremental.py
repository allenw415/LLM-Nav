from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET = PROJECT_ROOT / (
    "outputs/navigation_paths/floor0_semantic_targets_topology_constrained/"
    "room_count_2to6_ratio_1p0_2p0_per30_sparse715/test_set.jsonl"
)
OUTPUT_ROOT = PROJECT_ROOT / (
    "outputs/navigation_evaluation/floor0_semantic_targets_topology_constrained/"
    "room_count_2to6_ratio_1p0_2p0_per30_sparse715"
)
EXPERIMENT_ROOT = OUTPUT_ROOT / "initial_offset_incremental_10_20_30"
LEGACY_WITHOUT_RESULTS = (
    OUTPUT_ROOT / "geometry_initial_offset_timing715/results.jsonl"
)
EVALUATOR = PROJECT_ROOT / "tools/evaluation/navigation/evaluate_navigation_batch.py"
PAIRED_REPORTER = (
    PROJECT_ROOT / "tools/evaluation/navigation/report_initial_offset_experiment.py"
)

STAGES = (
    {"name": "per10", "minimum": 1, "maximum": 249},
    {"name": "per20", "minimum": 250, "maximum": 485},
    {"name": "per30", "minimum": 486, "maximum": 715},
)
CONDITIONS = (
    {
        "name": "without_trajectory",
        "direction_policy": "geometry",
        "api_key_source": "active_profile",
    },
    {
        "name": "with_trajectory",
        "direction_policy": "geometry-edge-visit",
        "api_key_source": "ST_NAV_WITH_TRAJECTORY_API_KEY",
    },
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_environment_file(path: Path) -> None:
    initial_keys = set(os.environ)
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if value and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if key and key not in initial_keys:
            os.environ[key] = value


def load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_number(test_id: object) -> int:
    if not isinstance(test_id, str) or not test_id.startswith("TEST"):
        raise ValueError(f"Invalid test_id: {test_id!r}")
    return int(test_id[4:])


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
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        temporary_path = Path(stream.name)
    os.replace(temporary_path, path)


def stage_test_ids(dataset: list[dict], stage: dict) -> list[str]:
    return [
        record["test_id"]
        for record in dataset
        if stage["minimum"] <= test_number(record.get("test_id")) <= stage["maximum"]
    ]


def condition_dir(stage_name: str, condition_name: str) -> Path:
    return EXPERIMENT_ROOT / "stages" / stage_name / condition_name


def bootstrap_preserved_results() -> None:
    destination = condition_dir("per10", "without_trajectory") / "results.jsonl"
    existing = load_jsonl(destination)
    existing_by_id = {record.get("test_id"): record for record in existing}
    for record in load_jsonl(LEGACY_WITHOUT_RESULTS):
        number = test_number(record.get("test_id"))
        if 1 <= number <= 249:
            existing_by_id.setdefault(record["test_id"], record)
    if existing_by_id:
        atomic_write_jsonl(
            destination,
            sorted(existing_by_id.values(), key=lambda record: test_number(record["test_id"])),
        )


def evaluation_command(
    *,
    test_ids: list[str],
    output_dir: Path,
    direction_policy: str,
) -> list[str]:
    command = [
        sys.executable,
        str(EVALUATOR),
        "--test-set",
        str(DATASET),
        "--output-dir",
        str(output_dir),
        "--artifacts-dir",
        str(PROJECT_ROOT / "dataset/sites/british_museum/normalized"),
        "--render-output-dir",
        str(PROJECT_ROOT / "renders/room_grounding_fov90"),
        "--render-missing",
        "--direction-policy",
        direction_policy,
        "--heading-alignment-mode",
        "initial-spatial",
        "--force-initial-visual-refresh",
        "--localization-interval",
        "3",
        "--start-heading",
        "0",
        "--seed",
        "42",
        "--llm-timeout",
        "300",
    ]
    for test_id in test_ids:
        command.extend(("--test-id", test_id))
    return command


def run_stage(dataset: list[dict], stage: dict) -> None:
    stage_name = stage["name"]
    test_ids = stage_test_ids(dataset, stage)
    print(
        f"[{utc_now()}] start {stage_name}: {len(test_ids)} cases per condition",
        flush=True,
    )
    processes: list[tuple[str, subprocess.Popen, object]] = []
    for condition in CONDITIONS:
        output_dir = condition_dir(stage_name, condition["name"])
        output_dir.mkdir(parents=True, exist_ok=True)
        log_path = output_dir / "run.log"
        log_stream = log_path.open("a", encoding="utf-8")
        log_stream.write(f"\n[{utc_now()}] launcher start {stage_name}\n")
        log_stream.flush()
        process_environment = os.environ.copy()
        api_key_source = condition["api_key_source"]
        if api_key_source != "active_profile":
            condition_api_key = os.environ.get(api_key_source)
            if not condition_api_key:
                raise RuntimeError(f"Missing required .env variable: {api_key_source}")
            process_environment["ST_NAV_PROFILE_GEMINI_API_KEY"] = condition_api_key
        process = subprocess.Popen(
            evaluation_command(
                test_ids=test_ids,
                output_dir=output_dir,
                direction_policy=condition["direction_policy"],
            ),
            cwd=PROJECT_ROOT,
            stdout=log_stream,
            stderr=subprocess.STDOUT,
            env=process_environment,
            start_new_session=True,
        )
        processes.append((condition["name"], process, log_stream))
        print(
            f"[{utc_now()}] {stage_name} {condition['name']} pid={process.pid} "
            f"key_source={api_key_source} log={log_path}",
            flush=True,
        )

    failure: tuple[str, int] | None = None
    while processes:
        active: list[tuple[str, subprocess.Popen, object]] = []
        for condition_name, process, log_stream in processes:
            return_code = process.poll()
            if return_code is None:
                active.append((condition_name, process, log_stream))
                continue
            log_stream.close()
            print(
                f"[{utc_now()}] {stage_name} {condition_name} exited {return_code}",
                flush=True,
            )
            if return_code != 0 and failure is None:
                failure = (condition_name, return_code)
        processes = active
        if failure is not None:
            for _, process, _ in processes:
                process.terminate()
            for _, process, log_stream in processes:
                process.wait(timeout=30)
                log_stream.close()
            condition_name, return_code = failure
            raise RuntimeError(
                f"{stage_name} {condition_name} failed with exit code {return_code}"
            )
        if processes:
            time.sleep(5)

    expected_ids = set(test_ids)
    for condition in CONDITIONS:
        results = load_jsonl(
            condition_dir(stage_name, condition["name"]) / "results.jsonl"
        )
        actual_ids = {record.get("test_id") for record in results}
        if actual_ids != expected_ids:
            raise RuntimeError(
                f"{stage_name} {condition['name']} result coverage mismatch: "
                f"expected={len(expected_ids)} actual={len(actual_ids)}"
            )


def build_milestone_reports(dataset: list[dict], stage_index: int) -> None:
    from tools.evaluation.navigation.report_localization_stages import build_report

    stage = STAGES[stage_index]
    milestone_root = EXPERIMENT_ROOT / "milestones" / stage["name"]
    cumulative_ids = {
        record["test_id"]
        for record in dataset
        if test_number(record.get("test_id")) <= stage["maximum"]
    }
    cumulative_dirs: dict[str, Path] = {}
    for condition in CONDITIONS:
        records_by_id: dict[str, dict] = {}
        for included_stage in STAGES[: stage_index + 1]:
            for record in load_jsonl(
                condition_dir(included_stage["name"], condition["name"])
                / "results.jsonl"
            ):
                records_by_id[record["test_id"]] = record
        if set(records_by_id) != cumulative_ids:
            raise RuntimeError(
                f"Cumulative {stage['name']} {condition['name']} coverage mismatch"
            )
        cumulative_dir = milestone_root / condition["name"]
        cumulative_dirs[condition["name"]] = cumulative_dir
        results_path = cumulative_dir / "results.jsonl"
        atomic_write_jsonl(
            results_path,
            sorted(
                records_by_id.values(),
                key=lambda record: test_number(record["test_id"]),
            ),
        )
        build_report(
            results_path=results_path,
            output_dir=cumulative_dir / "localization_report",
        )

    paired_dir = milestone_root / "paired_report"
    paired_dir.mkdir(parents=True, exist_ok=True)
    with (paired_dir / "run.log").open("a", encoding="utf-8") as log_stream:
        subprocess.run(
            [
                sys.executable,
                str(PAIRED_REPORTER),
                "--without-trajectory",
                str(cumulative_dirs["without_trajectory"]),
                "--with-trajectory",
                str(cumulative_dirs["with_trajectory"]),
                "--output-dir",
                str(paired_dir),
            ],
            cwd=PROJECT_ROOT,
            stdout=log_stream,
            stderr=subprocess.STDOUT,
            check=True,
        )
    print(
        f"[{utc_now()}] completed {stage['name']} milestone: "
        f"{len(cumulative_ids)} paired cases",
        flush=True,
    )


def main() -> int:
    os.chdir(PROJECT_ROOT)
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    environment_path = PROJECT_ROOT / ".env"
    if not environment_path.is_file():
        raise RuntimeError(f"Missing environment file: {environment_path}")
    load_environment_file(environment_path)
    dataset = load_jsonl(DATASET)
    if len(dataset) != 715:
        raise RuntimeError(f"Expected 715 dataset records, found {len(dataset)}")
    bootstrap_preserved_results()
    for stage_index, stage in enumerate(STAGES):
        run_stage(dataset, stage)
        build_milestone_reports(dataset, stage_index)
    print(f"[{utc_now()}] all incremental stages complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
