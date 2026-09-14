from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from st_nav.evaluation.navigation_batch import load_jsonl
from tools.evaluation.navigation.report_navigation_grid import (
    build_experiment_summary,
    build_meeting_markdown,
    standardize_spl_metrics,
)


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
DEFAULT_DATASET = DATASET_DIR / "test_set.jsonl"
DEFAULT_SOURCE_RESULTS = EVALUATION_DIR / "llm_nav_combined485/results.jsonl"
DEFAULT_RERUN_OUTPUT = EVALUATION_DIR / "room4_room9_edge_rerun34"
DEFAULT_CORRECTED_OUTPUT = EVALUATION_DIR / "llm_nav_combined485_room4_room9_corrected"
DEFAULT_RENDER_DIR = PROJECT_ROOT / "renders/room_grounding_fov90"
DEFAULT_ARTIFACTS_DIR = PROJECT_ROOT / "dataset/sites/british_museum/normalized"
DEFAULT_ROOM_GRAPH = DEFAULT_ARTIFACTS_DIR / "room_graph.json"
DEFAULT_VISUAL_ROOM_GRAPH = DEFAULT_ARTIFACTS_DIR / "room_graph_with_visual_profiles.json"
DEFAULT_PANO_GRAPH = DEFAULT_ARTIFACTS_DIR / "pano_graph.json"
EVALUATOR = PROJECT_ROOT / "tools/evaluation/navigation/evaluate_navigation_batch.py"
DISTANCE_REPORTER = PROJECT_ROOT / "tools/evaluation/navigation/report_navigation_by_shortest_distance.py"
LOCALIZATION_REPORTER = (
    PROJECT_ROOT
    / "tools/evaluation/navigation/report_navigation_final_error_and_runtime_localization.py"
)

EXPECTED_REFERENCE_EDGE_COUNT = 26
EXPECTED_RUNTIME_EDGE_COUNT = 33
EXPECTED_AFFECTED_COUNT = 34
DEFAULT_ROOM_A = "Room 4"
DEFAULT_ROOM_B = "Room 9"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def _atomic_write_jsonl(path: Path, records: Iterable[dict]) -> None:
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
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        temporary_path = Path(stream.name)
    os.replace(temporary_path, path)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        stream.write(text)
        temporary_path = Path(stream.name)
    os.replace(temporary_path, path)


def _index(records: list[dict], *, label: str) -> dict[str, dict]:
    indexed: dict[str, dict] = {}
    for record in records:
        test_id = record.get("test_id")
        if not isinstance(test_id, str) or not test_id:
            raise ValueError(f"{label} contains a record without test_id.")
        if test_id in indexed:
            raise ValueError(f"{label} contains duplicate test_id {test_id}.")
        indexed[test_id] = record
    return indexed


def _is_room_edge(source: object, target: object, *, room_a: str, room_b: str) -> bool:
    return (source == room_a and target == room_b) or (
        source == room_b and target == room_a
    )


def reference_uses_room_edge(test: dict, *, room_a: str, room_b: str) -> bool:
    rooms = test.get("shortest_path_rooms")
    if not isinstance(rooms, list):
        return False
    return any(
        _is_room_edge(source, target, room_a=room_a, room_b=room_b)
        for source, target in zip(rooms, rooms[1:])
    )


def runtime_uses_room_edge(result: dict, *, room_a: str, room_b: str) -> bool:
    movements = result.get("movements")
    if not isinstance(movements, list):
        return False
    return any(
        isinstance(movement, dict)
        and _is_room_edge(
            movement.get("estimated_room_id"),
            movement.get("subgoal_room_id"),
            room_a=room_a,
            room_b=room_b,
        )
        for movement in movements
    )


def select_affected_cases(
    dataset_records: list[dict],
    result_records: list[dict],
    *,
    room_a: str = DEFAULT_ROOM_A,
    room_b: str = DEFAULT_ROOM_B,
) -> dict:
    dataset = _index(dataset_records, label="dataset")
    results = _index(result_records, label="source results")
    if tuple(dataset) != tuple(results):
        missing = sorted(set(dataset) - set(results))
        extra = sorted(set(results) - set(dataset))
        raise ValueError(
            f"Dataset and source result order/IDs differ: missing={missing}, extra={extra}."
        )
    reference_ids = {
        test_id
        for test_id, test in dataset.items()
        if reference_uses_room_edge(test, room_a=room_a, room_b=room_b)
    }
    runtime_ids = {
        test_id
        for test_id, result in results.items()
        if runtime_uses_room_edge(result, room_a=room_a, room_b=room_b)
    }
    selected_ids = [
        test_id for test_id in dataset if test_id in reference_ids | runtime_ids
    ]
    cases = []
    for test_id in selected_ids:
        test = dataset[test_id]
        old = results[test_id]
        cases.append(
            {
                "test_id": test_id,
                "path_id": test.get("path_id"),
                "start_room_id": test.get("start_room_id"),
                "target_room_id": test.get("target_room_id"),
                "shortest_path_rooms": test.get("shortest_path_rooms"),
                "reference_edge": test_id in reference_ids,
                "runtime_edge": test_id in runtime_ids,
                "old_success": bool(old.get("success")),
                "old_spl": old.get("spl"),
                "old_movement_steps": old.get("movement_steps"),
                "old_elapsed_seconds": old.get("elapsed_seconds"),
            }
        )
    return {
        "room_a": room_a,
        "room_b": room_b,
        "reference_edge_count": len(reference_ids),
        "runtime_edge_count": len(runtime_ids),
        "affected_count": len(selected_ids),
        "reference_only_test_ids": sorted(reference_ids - runtime_ids),
        "runtime_only_test_ids": sorted(runtime_ids - reference_ids),
        "both_test_ids": sorted(reference_ids & runtime_ids),
        "affected_test_ids": selected_ids,
        "cases": cases,
    }


def validate_expected_selection(selection: dict) -> None:
    expected = {
        "reference_edge_count": EXPECTED_REFERENCE_EDGE_COUNT,
        "runtime_edge_count": EXPECTED_RUNTIME_EDGE_COUNT,
        "affected_count": EXPECTED_AFFECTED_COUNT,
    }
    actual = {key: selection.get(key) for key in expected}
    if actual != expected:
        raise ValueError(f"Affected-case selection drifted: expected={expected}, actual={actual}.")


def _neighbor(room_graph: dict, source_room: str, target_room: str) -> dict:
    room = room_graph.get(source_room)
    if not isinstance(room, dict):
        raise ValueError(f"Room graph is missing {source_room}.")
    for neighbor in room.get("neighbors", []):
        if isinstance(neighbor, dict) and neighbor.get("target_room_id") == target_room:
            return neighbor
    raise ValueError(f"Room graph is missing edge {source_room} -> {target_room}.")


def validate_corrected_room_edge(room_graph: dict) -> None:
    forward = _neighbor(room_graph, DEFAULT_ROOM_A, DEFAULT_ROOM_B)
    reverse = _neighbor(room_graph, DEFAULT_ROOM_B, DEFAULT_ROOM_A)
    expected = (
        (forward, "west", 270.0, f"{DEFAULT_ROOM_A} -> {DEFAULT_ROOM_B}"),
        (reverse, "east", 90.0, f"{DEFAULT_ROOM_B} -> {DEFAULT_ROOM_A}"),
    )
    for edge, direction, heading, label in expected:
        if edge.get("allocentric_direction") != direction or float(
            edge.get("allocentric_heading_deg", -1.0)
        ) != heading:
            raise ValueError(
                f"Incorrect corrected edge {label}: expected {direction}/{heading}, got {edge}."
            )


def merge_authorized_replacements(
    dataset_records: list[dict],
    source_records: list[dict],
    replacement_records: list[dict],
    *,
    authorized_test_ids: list[str],
) -> tuple[list[dict], list[dict]]:
    dataset = _index(dataset_records, label="dataset")
    source = _index(source_records, label="source results")
    replacements = _index(replacement_records, label="replacement results")
    authorized = set(authorized_test_ids)
    if set(replacements) != authorized:
        raise ValueError(
            "Replacement test IDs differ from the authorized set: "
            f"missing={sorted(authorized - set(replacements))}, "
            f"extra={sorted(set(replacements) - authorized)}."
        )
    if tuple(dataset) != tuple(source):
        raise ValueError("Dataset and source results must have identical ordered test IDs.")
    audit: list[dict] = []
    for test_id in authorized_test_ids:
        if test_id not in dataset or test_id not in source:
            raise ValueError(f"Authorized replacement has unknown test ID {test_id}.")
        old = source[test_id]
        new = replacements[test_id]
        expected_path_id = dataset[test_id].get("path_id")
        if old.get("path_id") != expected_path_id or new.get("path_id") != expected_path_id:
            raise ValueError(f"Replacement path_id mismatch for {test_id}.")
        if new.get("error"):
            raise ValueError(f"Replacement still contains an error for {test_id}.")
        audit.append(
            {
                "test_id": test_id,
                "path_id": expected_path_id,
                "old": {
                    "success": bool(old.get("success")),
                    "spl": old.get("spl"),
                    "movement_steps": old.get("movement_steps"),
                    "elapsed_seconds": old.get("elapsed_seconds"),
                    "stop_reason": old.get("stop_reason"),
                },
                "new": {
                    "success": bool(new.get("success")),
                    "spl": new.get("spl"),
                    "movement_steps": new.get("movement_steps"),
                    "elapsed_seconds": new.get("elapsed_seconds"),
                    "stop_reason": new.get("stop_reason"),
                },
            }
        )
    merged = [replacements.get(test_id, record) for test_id, record in source.items()]
    for record in merged:
        test_id = record["test_id"]
        if test_id not in authorized and record != source[test_id]:
            raise AssertionError(f"Unrelated source result changed for {test_id}.")
    return merged, audit


def _git_snapshot() -> dict:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--short"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {
        "head": head,
        "dirty": bool(status.strip()),
        "status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
    }


def _run(command: list[str]) -> None:
    print(f"[command] {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def _evaluator_command(
    *,
    dataset_path: Path,
    output_dir: Path,
    test_ids: list[str],
) -> list[str]:
    command = [
        sys.executable,
        "-u",
        str(EVALUATOR),
        "--test-set",
        str(dataset_path),
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
        "--direction-policy",
        "geometry",
        "--llm-model",
        "gemma-4-31b-it",
        "--llm-api-kind",
        "responses",
        "--llm-api-base",
        "https://generativelanguage.googleapis.com/v1beta",
        "--llm-timeout",
        "300",
        "--render-missing",
    ]
    for test_id in test_ids:
        command.extend(["--test-id", test_id])
    return command


def _load_effective_reruns(
    *,
    dataset_path: Path,
    rerun_output_dir: Path,
    selected_ids: list[str],
) -> tuple[list[dict], list[dict]]:
    main_records = load_jsonl(rerun_output_dir / "results.jsonl")
    main = _index(main_records, label="main rerun results")
    if set(main) != set(selected_ids):
        raise RuntimeError(
            f"Main rerun is incomplete: completed={len(main)}, expected={len(selected_ids)}."
        )
    retry_audit: list[dict] = []
    for test_id in selected_ids:
        record = main[test_id]
        if not record.get("error"):
            continue
        retry_dir = rerun_output_dir / "reruns" / "errors" / test_id
        retry_results = retry_dir / "results.jsonl"
        if not retry_results.is_file():
            _run(
                _evaluator_command(
                    dataset_path=dataset_path,
                    output_dir=retry_dir,
                    test_ids=[test_id],
                )
            )
        retry_records = load_jsonl(retry_results)
        retry = _index(retry_records, label=f"retry results for {test_id}")
        if set(retry) != {test_id}:
            raise RuntimeError(f"Retry output for {test_id} is incomplete.")
        replacement = retry[test_id]
        retry_audit.append(
            {
                "test_id": test_id,
                "original_error": record.get("error"),
                "retry_error": replacement.get("error"),
                "accepted": not bool(replacement.get("error")),
            }
        )
        if replacement.get("error"):
            raise RuntimeError(f"Retry still failed for {test_id}: {replacement['error']}")
        main[test_id] = replacement
    return [main[test_id] for test_id in selected_ids], retry_audit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rerun and replace navigation cases affected by Room 4 <-> Room 9."
    )
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--source-results", default=str(DEFAULT_SOURCE_RESULTS))
    parser.add_argument("--rerun-output-dir", default=str(DEFAULT_RERUN_OUTPUT))
    parser.add_argument("--corrected-output-dir", default=str(DEFAULT_CORRECTED_OUTPUT))
    parser.add_argument(
        "--selection-only",
        action="store_true",
        help="Validate and write the selection manifest without calling APIs.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    dataset_path = Path(args.dataset).resolve()
    source_results_path = Path(args.source_results).resolve()
    rerun_output_dir = Path(args.rerun_output_dir).resolve()
    corrected_output_dir = Path(args.corrected_output_dir).resolve()
    dataset_records = load_jsonl(dataset_path)
    source_records = load_jsonl(source_results_path)
    selection = select_affected_cases(dataset_records, source_records)
    validate_expected_selection(selection)

    base_graph = json.loads(DEFAULT_ROOM_GRAPH.read_text(encoding="utf-8"))
    visual_graph = json.loads(DEFAULT_VISUAL_ROOM_GRAPH.read_text(encoding="utf-8"))
    validate_corrected_room_edge(base_graph)
    validate_corrected_room_edge(visual_graph)
    graph_without_profiles = {
        room_id: {key: value for key, value in room.items() if key != "visual_profile"}
        for room_id, room in visual_graph.items()
    }
    if graph_without_profiles != base_graph:
        raise ValueError("Base and visual-profile room graphs differ beyond visual_profile.")

    selection_manifest = {
        "schema": "room_edge_affected_navigation_selection",
        "schema_version": 1,
        "created_at": _utc_now(),
        **selection,
        "sources": {
            "dataset": str(dataset_path),
            "dataset_sha256": _sha256(dataset_path),
            "source_results": str(source_results_path),
            "source_results_sha256": _sha256(source_results_path),
            "room_graph": str(DEFAULT_ROOM_GRAPH),
            "room_graph_sha256": _sha256(DEFAULT_ROOM_GRAPH),
            "visual_room_graph": str(DEFAULT_VISUAL_ROOM_GRAPH),
            "visual_room_graph_sha256": _sha256(DEFAULT_VISUAL_ROOM_GRAPH),
            "pano_graph": str(DEFAULT_PANO_GRAPH),
            "pano_graph_sha256": _sha256(DEFAULT_PANO_GRAPH),
        },
        "git": _git_snapshot(),
    }
    manifest_path = rerun_output_dir / "selection_manifest.json"
    _atomic_write_json(manifest_path, selection_manifest)
    selected_ids = selection["affected_test_ids"]
    print(
        json.dumps(
            {
                "reference_edge_count": selection["reference_edge_count"],
                "runtime_edge_count": selection["runtime_edge_count"],
                "affected_count": selection["affected_count"],
                "selection_manifest": str(manifest_path),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    if args.selection_only:
        return 0

    status_path = rerun_output_dir / "run_status.json"
    _atomic_write_json(
        status_path,
        {
            "schema": "room4_room9_rerun_status",
            "schema_version": 1,
            "status": "running",
            "phase": "rerun",
            "updated_at": _utc_now(),
            "expected_case_count": len(selected_ids),
        },
    )
    try:
        _run(
            _evaluator_command(
                dataset_path=dataset_path,
                output_dir=rerun_output_dir,
                test_ids=selected_ids,
            )
        )
        effective_replacements, retry_audit = _load_effective_reruns(
            dataset_path=dataset_path,
            rerun_output_dir=rerun_output_dir,
            selected_ids=selected_ids,
        )
        pano_graph = json.loads(DEFAULT_PANO_GRAPH.read_text(encoding="utf-8"))
        dataset = _index(dataset_records, label="dataset")
        standardized_replacements = standardize_spl_metrics(
            dataset,
            effective_replacements,
            pano_graph,
        )
        merged, replacement_audit = merge_authorized_replacements(
            dataset_records,
            source_records,
            standardized_replacements,
            authorized_test_ids=selected_ids,
        )
        summary = build_experiment_summary(dataset, merged)
        summary.update(
            {
                "expected_case_count": len(dataset_records),
                "completed_case_count": len(merged),
                "pending_case_count": 0,
                "replacement_case_count": len(selected_ids),
                "source_results_sha256": _sha256(source_results_path),
            }
        )
        corrected_output_dir.mkdir(parents=True, exist_ok=True)
        corrected_results_path = corrected_output_dir / "results.jsonl"
        _atomic_write_jsonl(corrected_results_path, merged)
        corrected_results_sha256 = _sha256(corrected_results_path)
        summary["corrected_results_sha256"] = corrected_results_sha256
        _atomic_write_json(corrected_output_dir / "summary.json", summary)
        _atomic_write_text(
            corrected_output_dir / "meeting_summary.md",
            build_meeting_markdown(
                title="LLM-Nav Floor 0 Corrected 485 Navigation Evaluation",
                summary=summary,
            ),
        )
        _atomic_write_json(
            corrected_output_dir / "replacement_audit.json",
            {
                "schema": "navigation_case_replacement_audit",
                "schema_version": 1,
                "replacement_case_count": len(replacement_audit),
                "retry_audit": retry_audit,
                "cases": replacement_audit,
            },
        )
        provenance = {
            "schema": "room4_room9_corrected_navigation_provenance",
            "schema_version": 1,
            "created_at": _utc_now(),
            "selection_manifest": str(manifest_path),
            "selection_manifest_sha256": _sha256(manifest_path),
            "rerun_results": str(rerun_output_dir / "results.jsonl"),
            "source_results": str(source_results_path),
            "source_results_sha256": _sha256(source_results_path),
            "corrected_results": str(corrected_results_path),
            "corrected_results_sha256": corrected_results_sha256,
            "replacement_case_count": len(selected_ids),
            "unchanged_case_count": len(merged) - len(selected_ids),
            "retry_audit": retry_audit,
            "git": selection_manifest["git"],
        }
        _atomic_write_json(corrected_output_dir / "report_provenance.json", provenance)

        _run(
            [
                sys.executable,
                str(DISTANCE_REPORTER),
                "--dataset",
                str(dataset_path),
                "--results",
                str(corrected_results_path),
                "--pano-graph",
                str(DEFAULT_PANO_GRAPH),
                "--output-dir",
                str(corrected_output_dir / "shortest_distance_report"),
            ]
        )
        _run(
            [
                sys.executable,
                str(LOCALIZATION_REPORTER),
                "--results",
                str(corrected_results_path),
                "--output-dir",
                str(corrected_output_dir / "final_error_localization_report"),
            ]
        )
        _atomic_write_json(
            status_path,
            {
                "schema": "room4_room9_rerun_status",
                "schema_version": 1,
                "status": "complete",
                "phase": "complete",
                "updated_at": _utc_now(),
                "expected_case_count": len(selected_ids),
                "completed_case_count": len(selected_ids),
                "corrected_results": str(corrected_results_path),
                "corrected_results_sha256": corrected_results_sha256,
            },
        )
        print(
            json.dumps(
                {
                    "status": "complete",
                    "replacement_case_count": len(selected_ids),
                    "corrected_case_count": len(merged),
                    "corrected_results": str(corrected_results_path),
                    "corrected_results_sha256": corrected_results_sha256,
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
        return 0
    except BaseException as error:
        _atomic_write_json(
            status_path,
            {
                "schema": "room4_room9_rerun_status",
                "schema_version": 1,
                "status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                "phase": "failed",
                "updated_at": _utc_now(),
                "expected_case_count": len(selected_ids),
                "error": {"type": type(error).__name__, "message": str(error)},
            },
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
