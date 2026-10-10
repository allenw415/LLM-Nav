from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
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
EXPERIMENT_ROOT = (
    OUTPUT_ROOT / "localization_stage_ablation_8view_trajectory_initial_offset715"
)
SPATIAL_BASELINE = (
    OUTPUT_ROOT
    / "initial_offset_incremental_10_20_30/milestones/per30/with_trajectory"
)
EVALUATOR = PROJECT_ROOT / "tools/evaluation/navigation/evaluate_navigation_batch.py"
REPORTER = (
    PROJECT_ROOT
    / "tools/evaluation/navigation/report_localization_stage_ablation.py"
)
RENDER_ROOT = PROJECT_ROOT / "renders/room_grounding_fov90"
EXPECTED_CASE_COUNT = 715
CONDITIONS = (
    {
        "name": "visual",
        "effective_stage": "visual",
        "api_key_source": "SA_NAV_PROFILE_GEMINI_API_KEY",
    },
    {
        "name": "bayesian",
        "effective_stage": "bayesian",
        "api_key_source": "SA_NAV_WITH_TRAJECTORY_API_KEY",
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


def validate_environment() -> dict[str, str]:
    if os.environ.get("SA_NAV_ACTIVE_PROFILE", "").strip().lower() != "gemini":
        raise RuntimeError("This campaign requires SA_NAV_ACTIVE_PROFILE=gemini.")
    keys: dict[str, str] = {}
    for condition in CONDITIONS:
        source = condition["api_key_source"]
        value = os.environ.get(source)
        if not value:
            raise RuntimeError(f"Missing required .env variable: {source}")
        keys[condition["name"]] = value
    if len(set(keys.values())) != len(keys):
        raise RuntimeError("The two ablation conditions must use distinct API keys.")
    if not os.environ.get("GMAPS_API_KEY"):
        raise RuntimeError("Missing required .env variable: GMAPS_API_KEY")
    return keys


def evaluation_command(*, output_dir: Path, effective_stage: str) -> list[str]:
    return [
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
        "museum",
        "--render-missing",
        "--direction-policy",
        "geometry-edge-visit",
        "--localization-effective-stage",
        effective_stage,
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


def write_campaign_audit(keys: dict[str, str]) -> None:
    EXPERIMENT_ROOT.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "localization_stage_ablation_campaign",
        "schema_version": 1,
        "created_at": utc_now(),
        "expected_case_count": EXPECTED_CASE_COUNT,
        "conditions": [
            {
                "name": condition["name"],
                "localization_effective_stage": condition["effective_stage"],
                "api_key_source": condition["api_key_source"],
            }
            for condition in CONDITIONS
        ],
        "shared_settings": {
            "estimated_heading_offset": True,
            "step_0_offset_estimation_stage": "spatial",
            "offset_write_policy": "step_0_once_then_locked",
            "observation": "8-view",
            "render_heading_mode": "museum",
            "trajectory_memory": True,
            "direction_policy": "geometry-edge-visit",
            "localization_interval": 3,
            "start_heading_deg": 0.0,
            "visual_cache_enabled": True,
        },
        "spatial_baseline": str(SPATIAL_BASELINE),
        "keys_present": bool(keys),
        "keys_distinct": True,
        "note": "Raw API keys and derived identifiers are never logged.",
    }
    (EXPERIMENT_ROOT / "campaign_config.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def validate_dataset_and_baseline() -> None:
    dataset = load_jsonl(DATASET)
    if len(dataset) != EXPECTED_CASE_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_CASE_COUNT} dataset records, found {len(dataset)}"
        )
    baseline = load_jsonl(SPATIAL_BASELINE / "results.jsonl")
    if len(baseline) != EXPECTED_CASE_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_CASE_COUNT} spatial baseline results, found {len(baseline)}"
        )


def validate_condition_output(condition: dict) -> None:
    condition_root = EXPERIMENT_ROOT / condition["name"]
    config = json.loads((condition_root / "config.json").read_text(encoding="utf-8"))
    expected = {
        "direction_policy": "geometry-edge-visit",
        "localization_interval": 3,
        "localization_effective_stage": condition["effective_stage"],
        "start_heading": 0.0,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise RuntimeError(
                f"{condition['name']} config mismatch for {key}: {config.get(key)!r}"
            )
    if config.get("render", {}).get("heading_mode") != "museum":
        raise RuntimeError(f"{condition['name']} did not use 8-view museum rendering.")
    if config.get("heading_alignment", {}).get("mode") != "initial-spatial":
        raise RuntimeError(f"{condition['name']} did not estimate the initial offset.")
    if config.get("heading_alignment", {}).get("force_initial_visual_refresh") is not True:
        raise RuntimeError(f"{condition['name']} did not refresh step-0 visual evidence.")

    results = load_jsonl(condition_root / "results.jsonl")
    if len(results) != EXPECTED_CASE_COUNT:
        raise RuntimeError(
            f"{condition['name']} coverage mismatch: {len(results)}/{EXPECTED_CASE_COUNT}"
        )
    if len({record.get("test_id") for record in results}) != EXPECTED_CASE_COUNT:
        raise RuntimeError(f"{condition['name']} has duplicate or invalid test IDs.")

    runtime_event_count = 0
    for record in results:
        events = record.get("localization_events")
        if not isinstance(events, list):
            continue
        for event in events:
            if event.get("effective_localization_stage") != condition["effective_stage"]:
                raise RuntimeError(
                    f"{record.get('test_id')} logged the wrong effective stage."
                )
            if event.get("step_index") == 0:
                continue
            runtime_event_count += 1
            if event.get("alignment_skipped_reason") != "disabled_by_effective_stage":
                raise RuntimeError(
                    f"{record.get('test_id')} unexpectedly ran runtime spatial alignment."
                )
    if runtime_event_count == 0:
        raise RuntimeError(f"{condition['name']} has no post-step-0 localization events.")


def run_conditions(keys: dict[str, str]) -> None:
    processes: list[tuple[dict, subprocess.Popen, object, Path]] = []
    for condition in CONDITIONS:
        condition_root = EXPERIMENT_ROOT / condition["name"]
        condition_root.mkdir(parents=True, exist_ok=True)
        log_path = condition_root / "run.log"
        log_stream = log_path.open("a", encoding="utf-8")
        log_stream.write(f"\n[{utc_now()}] campaign launcher start\n")
        log_stream.flush()
        environment = os.environ.copy()
        environment["SA_NAV_PROFILE_GEMINI_API_KEY"] = keys[condition["name"]]
        process = subprocess.Popen(
            evaluation_command(
                output_dir=condition_root,
                effective_stage=condition["effective_stage"],
            ),
            cwd=PROJECT_ROOT,
            stdout=log_stream,
            stderr=subprocess.STDOUT,
            env=environment,
            start_new_session=True,
        )
        processes.append((condition, process, log_stream, log_path))
        print(
            f"[{utc_now()}] {condition['name']} pid={process.pid} "
            f"key_source={condition['api_key_source']} log={log_path}",
            flush=True,
        )

    failures = []
    while processes:
        active = []
        for condition, process, log_stream, log_path in processes:
            return_code = process.poll()
            if return_code is None:
                active.append((condition, process, log_stream, log_path))
                continue
            log_stream.close()
            print(
                f"[{utc_now()}] {condition['name']} exited {return_code}",
                flush=True,
            )
            if return_code != 0:
                failures.append((condition["name"], return_code, log_path))
        processes = active
        if processes:
            time.sleep(5)
    if failures:
        raise RuntimeError(f"Ablation evaluator failures: {failures}")


def build_final_report() -> None:
    comparison_root = EXPERIMENT_ROOT / "comparison"
    comparison_root.mkdir(parents=True, exist_ok=True)
    with (comparison_root / "run.log").open("a", encoding="utf-8") as log_stream:
        subprocess.run(
            [
                sys.executable,
                str(REPORTER),
                "--visual",
                str(EXPERIMENT_ROOT / "visual"),
                "--bayesian",
                str(EXPERIMENT_ROOT / "bayesian"),
                "--spatial",
                str(SPATIAL_BASELINE),
                "--output-dir",
                str(comparison_root),
                "--expected-case-count",
                str(EXPECTED_CASE_COUNT),
            ],
            cwd=PROJECT_ROOT,
            stdout=log_stream,
            stderr=subprocess.STDOUT,
            check=True,
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the 8-view trajectory-memory localization-stage ablation. "
            "The command is resumable through each condition's results.jsonl."
        )
    )
    parser.parse_args()
    os.chdir(PROJECT_ROOT)
    environment_path = PROJECT_ROOT / ".env"
    if not environment_path.is_file():
        raise RuntimeError(f"Missing environment file: {environment_path}")
    load_environment_file(environment_path)
    validate_dataset_and_baseline()
    keys = validate_environment()
    write_campaign_audit(keys)
    run_conditions(keys)
    for condition in CONDITIONS:
        validate_condition_output(condition)
    build_final_report()
    print(f"[{utc_now()}] localization-stage ablation complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
