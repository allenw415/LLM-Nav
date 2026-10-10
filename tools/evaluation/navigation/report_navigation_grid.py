from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sa_nav.evaluation.navigation_batch import (
    compute_navigation_spl_metrics,
    load_jsonl,
    pano_path_distance_m,
    summarize_batch_results,
)


DATASET_DIR = (
    "outputs/navigation_paths/floor0_semantic_targets_topology_constrained/"
    "room_count_2to6_ratio_1p0_2p0_per20_sparse485"
)
EVALUATION_DIR = (
    "outputs/navigation_evaluation/floor0_semantic_targets_topology_constrained/"
    "room_count_2to6_ratio_1p0_2p0_per20_sparse485"
)
LEGACY_DIR = (
    "outputs/navigation_evaluation/floor0_semantic_targets_topology_constrained/"
    "room_count_2to6_ratio_1p0_2p0_per10_sparse249/llm_nav_combined249"
)
DEFAULT_DATASET = f"{DATASET_DIR}/test_set.jsonl"
DEFAULT_EXTENSION_TEST_SET = f"{DATASET_DIR}/extension_test_set.jsonl"
DEFAULT_LEGACY_RESULTS = f"{LEGACY_DIR}/results.jsonl"
DEFAULT_LEGACY_SUMMARY = f"{LEGACY_DIR}/summary.json"
DEFAULT_EXTENSION_RESULTS = f"{EVALUATION_DIR}/llm_nav_extension236/results.jsonl"
DEFAULT_EXTENSION_OUTPUT = f"{EVALUATION_DIR}/llm_nav_extension236"
DEFAULT_COMBINED_OUTPUT = f"{EVALUATION_DIR}/llm_nav_combined485"
DEFAULT_PANO_GRAPH = "dataset/sites/british_museum/normalized/pano_graph.json"

ROOM_COUNTS = (2, 3, 4, 5, 6)
RATIO_BINS = ("1.0-1.2", "1.2-1.4", "1.4-1.6", "1.6-1.8", "1.8-2.0")


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (PROJECT_ROOT / value).resolve()


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


def _atomic_write_jsonl(path: Path, records: list[dict]) -> None:
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


def _records_by_test_id(records: list[dict], *, label: str) -> dict[str, dict]:
    indexed: dict[str, dict] = {}
    for record in records:
        test_id = record.get("test_id")
        if not isinstance(test_id, str) or not test_id:
            raise ValueError(f"{label} contains a record without test_id.")
        if test_id in indexed:
            raise ValueError(f"{label} contains duplicate test_id {test_id}.")
        indexed[test_id] = record
    return indexed


def _require_test_ids(indexed: dict[str, dict], expected: tuple[str, ...], *, label: str) -> None:
    actual = set(indexed)
    wanted = set(expected)
    if actual != wanted:
        missing = sorted(wanted - actual)
        extra = sorted(actual - wanted)
        raise ValueError(f"{label} test IDs differ: missing={missing}, extra={extra}")


def derive_dataset_partitions(
    dataset: dict[str, dict],
    extension_dataset: dict[str, dict],
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    all_test_ids = tuple(dataset)
    extension_test_ids = tuple(extension_dataset)
    if not extension_test_ids:
        raise ValueError("extension dataset is empty.")
    unknown = sorted(set(extension_test_ids) - set(all_test_ids))
    if unknown:
        raise ValueError(f"extension dataset contains unknown test IDs: {unknown}")
    expected_extension_order = tuple(
        test_id for test_id in all_test_ids if test_id in extension_dataset
    )
    if extension_test_ids != expected_extension_order:
        raise ValueError("extension dataset order differs from the combined dataset.")
    legacy_test_ids = tuple(
        test_id for test_id in all_test_ids if test_id not in extension_dataset
    )
    if not legacy_test_ids:
        raise ValueError("legacy dataset partition is empty.")
    return legacy_test_ids, extension_test_ids, all_test_ids


def _validate_result_identity(dataset: dict[str, dict], results: dict[str, dict], *, label: str) -> None:
    for test_id, result in results.items():
        test = dataset.get(test_id)
        if test is None:
            raise ValueError(f"{label} includes unknown test ID {test_id}.")
        if result.get("path_id") != test.get("path_id"):
            raise ValueError(
                f"{label} path_id differs for {test_id}: "
                f"{result.get('path_id')} != {test.get('path_id')}"
            )


def standardize_spl_metrics(
    dataset: dict[str, dict],
    records: list[dict],
    pano_graph: dict[str, dict],
) -> list[dict]:
    """Return records whose canonical ``spl`` is metre-based.

    Historical evaluator outputs stored a pano-step approximation under
    ``spl``.  The raw files remain untouched; derived final/combined outputs
    preserve that value as ``step_spl_legacy`` and replace ``spl`` with the
    baseline-compatible distance metric.
    """

    standardized: list[dict] = []
    for record in records:
        test_id = record["test_id"]
        test = dataset[test_id]
        pano_path = record.get("pano_path")
        if not isinstance(pano_path, list) or not all(
            isinstance(pano_id, str) and pano_id for pano_id in pano_path
        ):
            raise ValueError(f"{test_id} has no valid pano_path for SPL migration.")
        metrics = compute_navigation_spl_metrics(
            success=bool(record.get("success")),
            shortest_path_distance_m=float(test["path_distance_m"]),
            actual_path_distance_m=pano_path_distance_m(pano_path, pano_graph),
            shortest_path_pano_steps=int(test["pano_step_count"]),
            movement_steps=int(record.get("movement_steps", max(len(pano_path) - 1, 0))),
        )
        if record.get("spl_metric") != "distance_m":
            stored_spl = float(record.get("spl", 0.0))
            expected_legacy = float(metrics["step_spl_legacy"] or 0.0)
            if abs(stored_spl - expected_legacy) > 1e-12:
                raise ValueError(
                    f"{test_id} historical spl={stored_spl} does not match "
                    f"recalculated step_spl_legacy={expected_legacy}."
                )
        updated = dict(record)
        updated.update(metrics)
        standardized.append(updated)
    return standardized


def apply_error_replacements(
    extension_results: dict[str, dict],
    replacement_records: list[dict],
) -> tuple[dict[str, dict], list[dict]]:
    effective = dict(extension_results)
    audit: list[dict] = []
    seen: set[str] = set()
    for replacement in replacement_records:
        test_id = replacement.get("test_id")
        if not isinstance(test_id, str) or test_id not in effective:
            raise ValueError(f"Replacement result has unknown extension test ID: {test_id}")
        if test_id in seen:
            raise ValueError(f"Duplicate replacement result for {test_id}.")
        seen.add(test_id)
        original = effective[test_id]
        if not original.get("error"):
            raise ValueError(f"Replacement is only allowed for an original error result: {test_id}")
        accepted = not bool(replacement.get("error"))
        audit.append(
            {
                "test_id": test_id,
                "accepted": accepted,
                "original_error": original.get("error"),
                "replacement_error": replacement.get("error"),
                "replacement_success": bool(replacement.get("success")),
            }
        )
        if accepted:
            effective[test_id] = replacement
    return effective, audit


def _average(records: list[dict], key: str) -> float:
    values = [float(record.get(key, 0.0)) for record in records]
    return sum(values) / len(values) if values else 0.0


def _case_group(records: list[dict]) -> dict:
    return {
        "case_count": len(records),
        "average_elapsed_seconds": _average(records, "elapsed_seconds"),
        "average_movement_steps": _average(records, "movement_steps"),
    }


def _cell_summary(records: list[dict]) -> dict:
    count = len(records)
    successes = sum(bool(record.get("success")) for record in records)
    return {
        "case_count": count,
        "success_count": successes,
        "success_rate": successes / count if count else None,
        "average_spl": _average(records, "spl") if count else None,
    }


def build_grid_summary(dataset: dict[str, dict], results: list[dict]) -> dict:
    grouped: dict[tuple[int, str], list[dict]] = {
        (room_count, ratio_bin): []
        for room_count in ROOM_COUNTS
        for ratio_bin in RATIO_BINS
    }
    for result in results:
        test = dataset[result["test_id"]]
        key = (int(test["visited_room_count"]), str(test["ratio_stratum"]))
        if key not in grouped:
            raise ValueError(f"Unexpected grid cell for {result['test_id']}: {key}")
        grouped[key].append(result)

    cells = {
        str(room_count): {
            ratio_bin: _cell_summary(grouped[(room_count, ratio_bin)])
            for ratio_bin in RATIO_BINS
        }
        for room_count in ROOM_COUNTS
    }
    row_overall = {
        str(room_count): _cell_summary(
            [record for ratio_bin in RATIO_BINS for record in grouped[(room_count, ratio_bin)]]
        )
        for room_count in ROOM_COUNTS
    }
    column_overall = {
        ratio_bin: _cell_summary(
            [record for room_count in ROOM_COUNTS for record in grouped[(room_count, ratio_bin)]]
        )
        for ratio_bin in RATIO_BINS
    }
    return {
        "room_counts": list(ROOM_COUNTS),
        "ratio_bins": list(RATIO_BINS),
        "cells": cells,
        "row_overall": row_overall,
        "column_overall": column_overall,
        "overall": _cell_summary(results),
    }


def build_experiment_summary(dataset: dict[str, dict], results: list[dict]) -> dict:
    base = summarize_batch_results(results)
    successes = [record for record in results if bool(record.get("success"))]
    failures = [record for record in results if not bool(record.get("success"))]
    base.update(
        {
            "failure_count": len(failures),
            "failure_rate": len(failures) / len(results) if results else 0.0,
            "average_elapsed_seconds": _average(results, "elapsed_seconds"),
            "average_movement_steps": _average(results, "movement_steps"),
            "total_elapsed_seconds": sum(float(record.get("elapsed_seconds", 0.0)) for record in results),
            "success_cases": _case_group(successes),
            "failure_cases": _case_group(failures),
            "error_test_ids": [record["test_id"] for record in results if record.get("error")],
            "grid": build_grid_summary(dataset, results),
        }
    )
    return base


def _format_percent(value: float | None, digits: int = 1) -> str:
    return "—" if value is None else f"{100.0 * value:.{digits}f}%"


def _format_duration(seconds: float) -> str:
    rounded = int(round(seconds))
    minutes, remainder = divmod(rounded, 60)
    if minutes:
        return f"{seconds:.1f} 秒（{minutes} 分 {remainder} 秒）"
    return f"{seconds:.1f} 秒"


def _format_sr_cell(cell: dict) -> str:
    if not cell["case_count"]:
        return "—"
    return f"{_format_percent(cell['success_rate'])}（{cell['success_count']}/{cell['case_count']}）"


def _format_spl_cell(cell: dict) -> str:
    value = cell.get("average_spl")
    return "—" if value is None else f"{value:.3f}"


def _grid_markdown(grid: dict, *, metric: str) -> list[str]:
    formatter = _format_sr_cell if metric == "sr" else _format_spl_cell
    lines = [
        "| 展廳數 / Detour Ratio | " + " | ".join(bin_.replace("-", "–") for bin_ in RATIO_BINS) + " | Overall |",
        "|---:|" + "---:|" * (len(RATIO_BINS) + 1),
    ]
    for room_count in ROOM_COUNTS:
        row = [formatter(grid["cells"][str(room_count)][bin_]) for bin_ in RATIO_BINS]
        row.append(formatter(grid["row_overall"][str(room_count)]))
        lines.append(f"| {room_count} | " + " | ".join(row) + " |")
    overall = [formatter(grid["column_overall"][bin_]) for bin_ in RATIO_BINS]
    overall.append(formatter(grid["overall"]))
    lines.append("| Overall | " + " | ".join(overall) + " |")
    return lines


def _model_call_markdown(model_calls: dict) -> list[str]:
    lines = [
        "| 階段 | Logical calls | 含圖 VLM calls | 純文字 calls | HTTP attempts | HTTP retry/failed attempts |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for stage in ("query_parser", "detection", "spatial_alignment", "action_policy"):
        counters = model_calls.get("stages", {}).get(stage)
        if not isinstance(counters, dict):
            continue
        lines.append(
            f"| {stage} | {counters.get('logical_call_count', 0)} | "
            f"{counters.get('image_logical_call_count', 0)} | "
            f"{counters.get('text_logical_call_count', 0)} | "
            f"{counters.get('http_attempt_count', 0)} | "
            f"{counters.get('failed_http_attempt_count', 0)} |"
        )
    total = model_calls.get("total", {})
    lines.append(
        f"| Total | {total.get('logical_call_count', 0)} | "
        f"{total.get('image_logical_call_count', 0)} | "
        f"{total.get('text_logical_call_count', 0)} | "
        f"{total.get('http_attempt_count', 0)} | "
        f"{total.get('failed_http_attempt_count', 0)} |"
    )
    return lines


def build_meeting_markdown(
    *,
    title: str,
    summary: dict,
    comparison: list[tuple[str, dict]] | None = None,
    legacy_vlm_audit: dict | None = None,
) -> str:
    lines = [
        f"# {title}",
        "",
        "## Overall Navigation Statistics",
        "",
        "| 指標 | 結果 |",
        "|---|---:|",
        f"| 測試案例 | {summary['case_count']} |",
        f"| 成功案例 | {summary['success_count']}/{summary['case_count']}（{_format_percent(summary['success_rate'])}） |",
        f"| 失敗案例 | {summary['failure_count']}/{summary['case_count']}（{_format_percent(summary['failure_rate'])}） |",
        f"| 執行錯誤 | {summary['error_count']}/{summary['case_count']} |",
        f"| Overall SPL | {_format_percent(summary['average_spl'])} |",
        f"| 平均執行時間 | {_format_duration(summary['average_elapsed_seconds'])} |",
        f"| 平均導航步數 | {summary['average_movement_steps']:.1f} steps |",
        f"| 累計實驗時間 | {summary['total_elapsed_seconds'] / 3600.0:.2f} 小時 |",
        "",
        "| 指標 | 成功案例 | 失敗案例 |",
        "|---|---:|---:|",
        f"| 案例數 | {summary['success_cases']['case_count']} | {summary['failure_cases']['case_count']} |",
        f"| 平均執行時間 | {_format_duration(summary['success_cases']['average_elapsed_seconds'])} | {_format_duration(summary['failure_cases']['average_elapsed_seconds'])} |",
        f"| 平均導航步數 | {summary['success_cases']['average_movement_steps']:.1f} steps | {summary['failure_cases']['average_movement_steps']:.1f} steps |",
    ]
    if comparison:
        comparison_heading = " vs ".join(label for label, _ in comparison)
        lines.extend(
            [
                "",
                f"## {comparison_heading}",
                "",
                "| 資料範圍 | 案例數 | 成功案例 | SR | SPL | 平均步數 |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for label, item in comparison:
            lines.append(
                f"| {label} | {item['case_count']} | {item['success_count']}/{item['case_count']} | "
                f"{_format_percent(item['success_rate'])} | {item['average_spl']:.3f} | "
                f"{item['average_movement_steps']:.1f} |"
            )
    lines.extend(
        [
            "",
            "## Navigation Success Rate by Visited Room Count and Detour Ratio",
            "",
            *_grid_markdown(summary["grid"], metric="sr"),
            "",
            "## Navigation SPL by Visited Room Count and Detour Ratio",
            "",
            *_grid_markdown(summary["grid"], metric="spl"),
            "",
            "## Model / VLM Calls",
            "",
            f"Telemetry coverage: {summary['model_calls']['coverage_case_count']}/{summary['case_count']} cases.",
            "",
            *_model_call_markdown(summary["model_calls"]),
            "",
            "## Render Cost",
            "",
            "| 指標 | 結果 |",
            "|---|---:|",
            f"| Cache hit events | {summary['render']['cache_hit_events']} |",
            f"| Cache miss events | {summary['render']['cache_miss_events']} |",
            f"| New render panos | {summary['render']['new_render_pano_count']} |",
            f"| Street View image requests | {summary['render']['new_image_requests']} |",
        ]
    )
    if legacy_vlm_audit:
        lines.extend(
            [
                "",
                "## Historical VLM Audit Caveat",
                "",
                f"- Historical cases have {legacy_vlm_audit.get('logical_call_count')} auditable detection logical calls.",
                "- Historical spatial-alignment and HTTP-attempt totals were not persisted, so complete combined VLM totals must use the telemetry coverage shown above.",
            ]
        )
    if summary["error_test_ids"]:
        lines.extend(
            [
                "",
                "## Error Cases",
                "",
                "Errors are counted as failures and remain in the denominator: "
                + ", ".join(summary["error_test_ids"])
                + ".",
            ]
        )
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate, merge, and report an additive navigation-grid extension."
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--legacy-results", default=DEFAULT_LEGACY_RESULTS)
    parser.add_argument("--extension-test-set", default=DEFAULT_EXTENSION_TEST_SET)
    parser.add_argument("--legacy-summary", default=DEFAULT_LEGACY_SUMMARY)
    parser.add_argument("--extension-results", default=DEFAULT_EXTENSION_RESULTS)
    parser.add_argument("--pano-graph", default=DEFAULT_PANO_GRAPH)
    parser.add_argument("--replacement-results", action="append", default=[])
    parser.add_argument("--extension-output-dir", default=DEFAULT_EXTENSION_OUTPUT)
    parser.add_argument("--combined-output-dir", default=DEFAULT_COMBINED_OUTPUT)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    dataset_path = resolve(args.dataset)
    extension_test_set_path = resolve(args.extension_test_set)
    legacy_results_path = resolve(args.legacy_results)
    legacy_summary_path = resolve(args.legacy_summary)
    extension_results_path = resolve(args.extension_results)
    pano_graph_path = resolve(args.pano_graph)
    extension_output_dir = resolve(args.extension_output_dir)
    combined_output_dir = resolve(args.combined_output_dir)
    replacement_paths = [resolve(path) for path in args.replacement_results]

    dataset = _records_by_test_id(load_jsonl(dataset_path), label="dataset")
    extension_dataset = _records_by_test_id(
        load_jsonl(extension_test_set_path),
        label="extension dataset",
    )
    legacy = _records_by_test_id(load_jsonl(legacy_results_path), label="legacy results")
    extension = _records_by_test_id(load_jsonl(extension_results_path), label="extension results")
    legacy_test_ids, extension_test_ids, all_test_ids = derive_dataset_partitions(
        dataset,
        extension_dataset,
    )
    _require_test_ids(legacy, legacy_test_ids, label="legacy results")
    _require_test_ids(extension, extension_test_ids, label="extension results")
    _validate_result_identity(dataset, extension_dataset, label="extension dataset")
    _validate_result_identity(dataset, legacy, label="legacy results")
    _validate_result_identity(dataset, extension, label="extension results")

    replacements = [record for path in replacement_paths for record in load_jsonl(path)]
    effective_extension, rerun_audit = apply_error_replacements(extension, replacements)
    _validate_result_identity(dataset, effective_extension, label="effective extension results")
    combined = {**legacy, **effective_extension}
    _require_test_ids(combined, all_test_ids, label="combined results")

    pano_graph = json.loads(pano_graph_path.read_text(encoding="utf-8"))
    if not isinstance(pano_graph, dict):
        raise ValueError("Pano graph must be a JSON object.")
    legacy_records = standardize_spl_metrics(
        dataset,
        [legacy[test_id] for test_id in legacy_test_ids],
        pano_graph,
    )
    extension_records = standardize_spl_metrics(
        dataset,
        [effective_extension[test_id] for test_id in extension_test_ids],
        pano_graph,
    )
    standardized_combined = {
        record["test_id"]: record for record in [*legacy_records, *extension_records]
    }
    combined_records = [standardized_combined[test_id] for test_id in all_test_ids]
    legacy_summary = build_experiment_summary(dataset, legacy_records)
    extension_summary = build_experiment_summary(dataset, extension_records)
    combined_summary = build_experiment_summary(dataset, combined_records)
    legacy_payload = json.loads(legacy_summary_path.read_text(encoding="utf-8"))
    legacy_vlm_audit = legacy_payload.get("vlm_call_audit")

    provenance = {
        "schema": "llm_nav_grid_report_provenance",
        "schema_version": 3,
        "dataset": str(dataset_path),
        "dataset_sha256": _sha256(dataset_path),
        "extension_test_set": str(extension_test_set_path),
        "extension_test_set_sha256": _sha256(extension_test_set_path),
        "legacy_results": str(legacy_results_path),
        "legacy_results_sha256": _sha256(legacy_results_path),
        "legacy_summary": str(legacy_summary_path),
        "legacy_summary_sha256": _sha256(legacy_summary_path),
        "extension_results": str(extension_results_path),
        "extension_results_sha256": _sha256(extension_results_path),
        "pano_graph": str(pano_graph_path),
        "pano_graph_sha256": _sha256(pano_graph_path),
        "spl_metric": "distance_m",
        "spl_formula": (
            "success * shortest_path_distance_m / "
            "max(actual_path_distance_m, shortest_path_distance_m)"
        ),
        "historical_step_spl_field": "step_spl_legacy",
        "replacement_results": [
            {"path": str(path), "sha256": _sha256(path)}
            for path in replacement_paths
        ],
        "rerun_audit": rerun_audit,
        "legacy_full_model_telemetry_available": (
            legacy_summary["model_calls"]["coverage_case_count"] == len(legacy_records)
        ),
        "extension_full_model_telemetry_available": (
            extension_summary["model_calls"]["coverage_case_count"] == len(extension_records)
        ),
    }
    extension_summary["expected_case_count"] = len(extension_test_ids)
    extension_summary["completed_case_count"] = len(extension_records)
    extension_summary["pending_case_count"] = 0
    extension_summary["rerun_audit"] = rerun_audit
    extension_summary["provenance"] = provenance
    combined_summary["expected_case_count"] = len(all_test_ids)
    combined_summary["completed_case_count"] = len(combined_records)
    combined_summary["pending_case_count"] = 0
    combined_summary["model_call_reporting"] = {
        "complete_case_telemetry_available": (
            combined_summary["model_calls"]["coverage_case_count"] == len(combined_records)
        ),
        "coverage_case_count": combined_summary["model_calls"]["coverage_case_count"],
        "missing_telemetry_case_count": combined_summary["model_calls"][
            "missing_telemetry_case_count"
        ],
        "legacy_case_count": len(legacy_records),
        "legacy_exact_telemetry": legacy_summary["model_calls"],
        "legacy_historical_audit": legacy_vlm_audit,
        "extension_case_count": len(extension_records),
        "extension_exact_telemetry": extension_summary["model_calls"],
    }
    combined_summary["provenance"] = provenance

    comparison = [
        (f"Existing {len(legacy_records)}", legacy_summary),
        (f"Extension {len(extension_records)}", extension_summary),
        (f"Combined {len(combined_records)}", combined_summary),
    ]
    _atomic_write_jsonl(extension_output_dir / "final_results.jsonl", extension_records)
    _atomic_write_json(extension_output_dir / "summary.json", extension_summary)
    _atomic_write_json(extension_output_dir / "report_provenance.json", provenance)
    _atomic_write_text(
        extension_output_dir / "meeting_summary.md",
        build_meeting_markdown(
            title=f"SA-Nav Floor 0 Extension {len(extension_records)} Navigation Evaluation",
            summary=extension_summary,
        ),
    )
    _atomic_write_jsonl(combined_output_dir / "results.jsonl", combined_records)
    _atomic_write_json(combined_output_dir / "summary.json", combined_summary)
    _atomic_write_json(combined_output_dir / "report_provenance.json", provenance)
    _atomic_write_text(
        combined_output_dir / "meeting_summary.md",
        build_meeting_markdown(
            title=f"SA-Nav Floor 0 Combined {len(combined_records)} Navigation Evaluation",
            summary=combined_summary,
            comparison=comparison,
            legacy_vlm_audit=legacy_vlm_audit,
        ),
    )
    print(
        json.dumps(
            {
                "extension": {
                    "case_count": extension_summary["case_count"],
                    "success_count": extension_summary["success_count"],
                    "error_count": extension_summary["error_count"],
                },
                "combined": {
                    "case_count": combined_summary["case_count"],
                    "success_count": combined_summary["success_count"],
                    "error_count": combined_summary["error_count"],
                },
                "combined_output_dir": str(combined_output_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
