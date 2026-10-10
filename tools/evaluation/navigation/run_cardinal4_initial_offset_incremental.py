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
EXPERIMENT_ROOT = OUTPUT_ROOT / "initial_offset_cardinal4_incremental_10_20_30"
SMOKE_ROOT = OUTPUT_ROOT / "initial_offset_cardinal4_smoke20"
RENDER_ROOT = PROJECT_ROOT / "renders/room_grounding_cardinal4_fov90"
BASELINE_ROOT = OUTPUT_ROOT / "initial_offset_incremental_10_20_30/milestones/per30"
EVALUATOR = PROJECT_ROOT / "tools/evaluation/navigation/evaluate_navigation_batch.py"
PAIRED_REPORTER = PROJECT_ROOT / "tools/evaluation/navigation/report_initial_offset_experiment.py"
VIEW_REPORTER = PROJECT_ROOT / "tools/evaluation/navigation/report_view_count_comparison.py"

STAGES = (
    {"name": "per10", "minimum": 1, "maximum": 249},
    {"name": "per20", "minimum": 250, "maximum": 485},
    {"name": "per30", "minimum": 486, "maximum": 715},
)
CONDITIONS = (
    {
        "name": "without_trajectory",
        "direction_policy": "geometry",
        "api_key_source": "SA_NAV_PROFILE_GEMINI_API_KEY",
    },
    {
        "name": "with_trajectory",
        "direction_policy": "geometry-edge-visit",
        "api_key_source": "SA_NAV_WITH_TRAJECTORY_API_KEY",
    },
)
SMOKE_CASE_COUNT = 20
EXPECTED_CAPTURE_HEADINGS = [0.0, 90.0, 180.0, 270.0]
EXPECTED_CAPTURE_LABELS = ["view_0", "view_1", "view_2", "view_3"]


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
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        temporary_path = Path(stream.name)
    os.replace(temporary_path, path)


def test_number(test_id: object) -> int:
    if not isinstance(test_id, str) or not test_id.startswith("TEST"):
        raise ValueError(f"Invalid test_id: {test_id!r}")
    return int(test_id[4:])


def stage_test_ids(dataset: list[dict], stage: dict) -> list[str]:
    return [
        record["test_id"]
        for record in dataset
        if stage["minimum"] <= test_number(record.get("test_id")) <= stage["maximum"]
    ]


def validate_api_keys() -> dict[str, str]:
    if os.environ.get("SA_NAV_ACTIVE_PROFILE", "").strip().lower() != "gemini":
        raise RuntimeError("The cardinal4 campaign requires SA_NAV_ACTIVE_PROFILE=gemini.")
    keys: dict[str, str] = {}
    for condition in CONDITIONS:
        source = condition["api_key_source"]
        value = os.environ.get(source)
        if not value:
            raise RuntimeError(f"Missing required .env variable: {source}")
        keys[condition["name"]] = value
    if len(set(keys.values())) != len(keys):
        raise RuntimeError("The two cardinal4 conditions must use distinct API keys.")
    if not os.environ.get("GMAPS_API_KEY"):
        raise RuntimeError("Missing required .env variable: GMAPS_API_KEY")
    atomic_write_json(
        EXPERIMENT_ROOT / "api_key_audit.json",
        {
            "checked_at": utc_now(),
            "key_sources": {
                condition["name"]: condition["api_key_source"]
                for condition in CONDITIONS
            },
            "keys_present": True,
            "keys_distinct": True,
            "note": "Raw API keys and derived identifiers are never logged.",
        },
    )
    return keys


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
        str(RENDER_ROOT),
        "--render-heading-mode",
        "cardinal4",
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


def _run_condition(
    *,
    label: str,
    condition: dict,
    test_ids: list[str],
    output_dir: Path,
    api_key: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "run.log"
    process_environment = os.environ.copy()
    process_environment["SA_NAV_PROFILE_GEMINI_API_KEY"] = api_key
    with log_path.open("a", encoding="utf-8") as log_stream:
        log_stream.write(f"\n[{utc_now()}] launcher start {label}\n")
        log_stream.flush()
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
        print(
            f"[{utc_now()}] {label} {condition['name']} pid={process.pid} "
            f"key_source={condition['api_key_source']} log={log_path}",
            flush=True,
        )
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(
            f"{label} {condition['name']} failed with exit code {return_code}"
        )


def run_condition_group(
    *,
    label: str,
    test_ids: list[str],
    root: Path,
    keys: dict[str, str],
    parallel: bool,
) -> None:
    if not parallel:
        for condition in CONDITIONS:
            _run_condition(
                label=label,
                condition=condition,
                test_ids=test_ids,
                output_dir=root / condition["name"],
                api_key=keys[condition["name"]],
            )
        return

    processes: list[tuple[dict, subprocess.Popen, object, Path]] = []
    for condition in CONDITIONS:
        output_dir = root / condition["name"]
        output_dir.mkdir(parents=True, exist_ok=True)
        log_path = output_dir / "run.log"
        log_stream = log_path.open("a", encoding="utf-8")
        log_stream.write(f"\n[{utc_now()}] launcher start {label}\n")
        log_stream.flush()
        process_environment = os.environ.copy()
        process_environment["SA_NAV_PROFILE_GEMINI_API_KEY"] = keys[condition["name"]]
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
        processes.append((condition, process, log_stream, log_path))
        print(
            f"[{utc_now()}] {label} {condition['name']} pid={process.pid} "
            f"key_source={condition['api_key_source']} log={log_path}",
            flush=True,
        )

    failure: tuple[str, int] | None = None
    while processes:
        active: list[tuple[dict, subprocess.Popen, object, Path]] = []
        for condition, process, log_stream, log_path in processes:
            return_code = process.poll()
            if return_code is None:
                active.append((condition, process, log_stream, log_path))
                continue
            log_stream.close()
            print(
                f"[{utc_now()}] {label} {condition['name']} exited {return_code}",
                flush=True,
            )
            if return_code != 0 and failure is None:
                failure = (condition["name"], return_code)
        processes = active
        if failure is not None:
            for _, process, _, _ in processes:
                process.terminate()
            for _, process, log_stream, _ in processes:
                process.wait(timeout=30)
                log_stream.close()
            condition_name, return_code = failure
            raise RuntimeError(
                f"{label} {condition_name} failed with exit code {return_code}"
            )
        if processes:
            time.sleep(5)


def validate_results(
    *,
    root: Path,
    expected_ids: set[str],
    require_no_errors: bool,
    require_alignment_contract: bool = False,
) -> None:
    for condition in CONDITIONS:
        condition_root = root / condition["name"]
        results = load_jsonl(condition_root / "results.jsonl")
        actual_ids = {record.get("test_id") for record in results}
        if actual_ids != expected_ids:
            raise RuntimeError(
                f"{condition['name']} coverage mismatch: "
                f"expected={len(expected_ids)} actual={len(actual_ids)}"
            )
        errors = [record for record in results if record.get("error") is not None]
        if require_no_errors and errors:
            raise RuntimeError(
                f"{condition['name']} smoke has {len(errors)} API/runtime errors."
            )
        config = json.loads((condition_root / "config.json").read_text(encoding="utf-8"))
        if config.get("render", {}).get("heading_mode") != "cardinal4":
            raise RuntimeError(f"{condition['name']} did not use cardinal4 rendering.")
        if not require_alignment_contract:
            continue
        for record in results:
            if record.get("error") is not None:
                continue
            alignment = record.get("initial_heading_alignment")
            if not isinstance(alignment, dict):
                raise RuntimeError(f"{record.get('test_id')} has no initial alignment record.")
            if alignment.get("view_0_geocentric_heading_deg") != 0.0:
                raise RuntimeError(f"{record.get('test_id')} did not align from view_0=0 degrees.")
            if alignment.get("ground_truth_offset_deg") != 30.0:
                raise RuntimeError(f"{record.get('test_id')} ground-truth offset is not +30 degrees.")


def validate_start_manifests(dataset_by_id: dict[str, dict], test_ids: list[str]) -> None:
    for test_id in test_ids:
        pano_id = dataset_by_id[test_id]["start_pano_id"]
        manifest_path = RENDER_ROOT / pano_id / f"{pano_id}_manifest.json"
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        captures = payload.get("captures", [])
        if payload.get("heading_mode") != "cardinal4":
            raise RuntimeError(f"Unexpected heading mode in {manifest_path}")
        if [capture.get("label") for capture in captures] != EXPECTED_CAPTURE_LABELS:
            raise RuntimeError(f"Unexpected capture labels in {manifest_path}")
        if [capture.get("heading") for capture in captures] != EXPECTED_CAPTURE_HEADINGS:
            raise RuntimeError(f"Unexpected capture headings in {manifest_path}")


def bootstrap_smoke_results() -> None:
    audit = {"created_at": utc_now(), "conditions": {}}
    for condition in CONDITIONS:
        source = SMOKE_ROOT / condition["name"] / "results.jsonl"
        destination = (
            EXPERIMENT_ROOT
            / "stages/per10"
            / condition["name"]
            / "results.jsonl"
        )
        source_records = load_jsonl(source)
        existing_records = load_jsonl(destination)
        records_by_id = {record["test_id"]: record for record in existing_records}
        added_ids = []
        for record in source_records:
            if record.get("error") is None and record["test_id"] not in records_by_id:
                records_by_id[record["test_id"]] = record
                added_ids.append(record["test_id"])
        atomic_write_jsonl(
            destination,
            sorted(records_by_id.values(), key=lambda record: test_number(record["test_id"])),
        )
        audit["conditions"][condition["name"]] = {
            "source": str(source),
            "destination": str(destination),
            "added_test_ids": added_ids,
            "preserved_existing_count": len(existing_records),
        }
    atomic_write_json(EXPERIMENT_ROOT / "smoke_bootstrap_audit.json", audit)


def condition_dir(stage_name: str, condition_name: str) -> Path:
    return EXPERIMENT_ROOT / "stages" / stage_name / condition_name


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
            sorted(records_by_id.values(), key=lambda record: test_number(record["test_id"])),
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


def build_view_count_report() -> None:
    destination = EXPERIMENT_ROOT / "milestones/per30/view_count_comparison"
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / "run.log").open("a", encoding="utf-8") as log_stream:
        subprocess.run(
            [
                sys.executable,
                str(VIEW_REPORTER),
                "--eight-view-root",
                str(BASELINE_ROOT),
                "--four-view-root",
                str(EXPERIMENT_ROOT / "milestones/per30"),
                "--output-dir",
                str(destination),
            ],
            cwd=PROJECT_ROOT,
            stdout=log_stream,
            stderr=subprocess.STDOUT,
            check=True,
        )


def main() -> int:
    os.chdir(PROJECT_ROOT)
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    environment_path = PROJECT_ROOT / ".env"
    if not environment_path.is_file():
        raise RuntimeError(f"Missing environment file: {environment_path}")
    load_environment_file(environment_path)
    keys = validate_api_keys()
    dataset = load_jsonl(DATASET)
    if len(dataset) != 715:
        raise RuntimeError(f"Expected 715 dataset records, found {len(dataset)}")
    dataset_by_id = {record["test_id"]: record for record in dataset}

    smoke_ids = [record["test_id"] for record in dataset[:SMOKE_CASE_COUNT]]
    print(f"[{utc_now()}] starting cardinal4 smoke: {len(smoke_ids)} cases per condition", flush=True)
    run_condition_group(
        label="smoke20",
        test_ids=smoke_ids,
        root=SMOKE_ROOT,
        keys=keys,
        parallel=False,
    )
    validate_results(
        root=SMOKE_ROOT,
        expected_ids=set(smoke_ids),
        require_no_errors=True,
        require_alignment_contract=True,
    )
    validate_start_manifests(dataset_by_id, smoke_ids)
    atomic_write_json(
        SMOKE_ROOT / "smoke_gate.json",
        {
            "passed": True,
            "checked_at": utc_now(),
            "case_count_per_condition": SMOKE_CASE_COUNT,
            "render_heading_mode": "cardinal4",
            "capture_headings_deg": EXPECTED_CAPTURE_HEADINGS,
            "ground_truth_offset_deg": 30.0,
        },
    )
    print(f"[{utc_now()}] smoke gate passed; starting full incremental run", flush=True)
    bootstrap_smoke_results()

    for stage_index, stage in enumerate(STAGES):
        test_ids = stage_test_ids(dataset, stage)
        run_condition_group(
            label=stage["name"],
            test_ids=test_ids,
            root=EXPERIMENT_ROOT / "stages" / stage["name"],
            keys=keys,
            parallel=True,
        )
        validate_results(
            root=EXPERIMENT_ROOT / "stages" / stage["name"],
            expected_ids=set(test_ids),
            require_no_errors=False,
        )
        build_milestone_reports(dataset, stage_index)
        print(
            f"[{utc_now()}] completed {stage['name']} milestone: {stage['maximum']} cases per condition",
            flush=True,
        )

    build_view_count_report()
    print(f"[{utc_now()}] cardinal4 campaign complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
