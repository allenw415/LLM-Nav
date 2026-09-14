from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_EXPERIMENT_ROOT = PROJECT_ROOT / (
    "outputs/navigation_evaluation/floor0_semantic_targets_topology_constrained/"
    "room_count_2to6_ratio_1p0_2p0_per30_sparse715/"
    "initial_offset_incremental_10_20_30"
)
STAGES = (("per10", 249), ("per20", 236), ("per30", 230))
CONDITIONS = ("without_trajectory", "with_trajectory")
START_PATTERN = re.compile(r"\[start \d+/\d+\] (TEST\d+)")


def load_results(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    records: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def current_test_id(log_path: Path) -> str:
    if not log_path.is_file():
        return "waiting"
    matches = START_PATTERN.findall(log_path.read_text(encoding="utf-8", errors="replace"))
    return matches[-1] if matches else "waiting"


def select_stage(root: Path) -> tuple[str, int]:
    for stage_name, expected in STAGES:
        counts = [
            len(load_results(root / "stages" / stage_name / condition / "results.jsonl"))
            for condition in CONDITIONS
        ]
        if min(counts) < expected:
            return stage_name, expected
    return STAGES[-1]


def snapshot(root: Path) -> tuple[str, bool]:
    stage_name, expected = select_stage(root)
    condition_records: dict[str, list[dict]] = {}
    parts = []
    for condition in CONDITIONS:
        condition_root = root / "stages" / stage_name / condition
        records = load_results(condition_root / "results.jsonl")
        condition_records[condition] = records
        error_count = sum(record.get("error") is not None for record in records)
        valid_count = len(records) - error_count
        success_count = sum(
            record.get("error") is None and record.get("success") is True
            for record in records
        )
        label = "Without" if condition == "without_trajectory" else "With"
        parts.append(
            f"{label} {len(records)}/{expected} "
            f"(valid={valid_count}, api_error={error_count}, success={success_count}, "
            f"current={current_test_id(condition_root / 'run.log')})"
        )

    valid_ids = {
        condition: {
            record.get("test_id")
            for record in records
            if record.get("error") is None
        }
        for condition, records in condition_records.items()
    }
    paired_valid = len(
        valid_ids["without_trajectory"] & valid_ids["with_trajectory"]
    )
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    line = (
        f"[{timestamp}] stage={stage_name} | "
        + " | ".join(parts)
        + f" | valid_paired={paired_valid}/{expected}"
    )
    completed = all(
        len(condition_records[condition]) >= expected for condition in CONDITIONS
    ) and stage_name == STAGES[-1][0]
    return line, completed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", default=str(DEFAULT_EXPERIMENT_ROOT))
    parser.add_argument("--output", default=None)
    parser.add_argument("--interval", type=float, default=30.0)
    args = parser.parse_args()
    root = Path(args.experiment_root).resolve()
    output = Path(args.output).resolve() if args.output else root / "progress.log"
    output.parent.mkdir(parents=True, exist_ok=True)
    while True:
        line, completed = snapshot(root)
        with output.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")
        if completed:
            break
        time.sleep(max(args.interval, 5.0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
