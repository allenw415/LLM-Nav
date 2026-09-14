#!/usr/bin/env python3
"""Report navigation performance by shortest-path distance in metres."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import tempfile
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from st_nav.evaluation.navigation_batch import (
    compute_navigation_spl_metrics,
    pano_path_distance_m,
)


DATASET_DIR = (
    "outputs/navigation_paths/floor0_semantic_targets_topology_constrained/"
    "room_count_2to6_ratio_1p0_2p0_per10_sparse249"
)
EVALUATION_DIR = (
    "outputs/navigation_evaluation/floor0_semantic_targets_topology_constrained/"
    "room_count_2to6_ratio_1p0_2p0_per10_sparse249/llm_nav_combined249"
)
DEFAULT_DATASET = PROJECT_ROOT / DATASET_DIR / "test_set.jsonl"
DEFAULT_RESULTS = PROJECT_ROOT / EVALUATION_DIR / "results.jsonl"
DEFAULT_GRAPH = (
    PROJECT_ROOT / "dataset/sites/british_museum/normalized/pano_graph.json"
)
DEFAULT_OUTPUT = PROJECT_ROOT / EVALUATION_DIR / "shortest_distance_report"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--pano-graph", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bin-width-m", type=float, default=20.0)
    parser.add_argument("--max-distance-m", type=float, default=None)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            records.append(record)
    return records


def index_unique(
    records: Iterable[dict[str, Any]], *, label: str
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for record in records:
        test_id = record.get("test_id")
        if not isinstance(test_id, str) or not test_id:
            raise ValueError(f"{label} contains a record without test_id")
        if test_id in indexed:
            raise ValueError(f"{label} contains duplicate test_id {test_id}")
        indexed[test_id] = record
    return indexed


def build_distance_bins(
    maximum_observed_m: float,
    *,
    width_m: float,
    maximum_requested_m: float | None = None,
) -> list[dict[str, Any]]:
    if width_m <= 0.0:
        raise ValueError("bin width must be positive")
    maximum = maximum_observed_m
    if maximum_requested_m is not None:
        if maximum_requested_m <= 0.0:
            raise ValueError("maximum distance must be positive")
        if maximum_requested_m + 1e-9 < maximum_observed_m:
            raise ValueError(
                "maximum distance is lower than the largest observed shortest path"
            )
        maximum = maximum_requested_m
    upper_limit = max(width_m, math.ceil(maximum / width_m) * width_m)
    bin_count = int(round(upper_limit / width_m))
    bins: list[dict[str, Any]] = []
    for index in range(bin_count):
        lower = index * width_m
        upper = (index + 1) * width_m
        bins.append(
            {
                "label": f"{_format_number(lower)}-{_format_number(upper)} m",
                "lower_m": lower,
                "upper_m": upper,
                "lower_inclusive": True,
                "upper_inclusive": False,
            }
        )
    return bins


def distance_bin_label(
    distance_m: float, bins: Sequence[dict[str, Any]]
) -> str:
    for distance_bin in bins:
        if (
            float(distance_bin["lower_m"])
            <= distance_m
            < float(distance_bin["upper_m"])
        ):
            return str(distance_bin["label"])
    raise ValueError(f"Distance {distance_m:.6f} m is outside configured bins")


def enrich_results(
    dataset_records: Sequence[dict[str, Any]],
    result_records: Sequence[dict[str, Any]],
    pano_graph: dict[str, Any],
    bins: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], float]:
    dataset = index_unique(dataset_records, label="dataset")
    results = index_unique(result_records, label="results")
    if set(dataset) != set(results):
        raise ValueError(
            "Dataset/result test IDs differ: "
            f"missing={sorted(set(dataset) - set(results))}, "
            f"extra={sorted(set(results) - set(dataset))}"
        )

    maximum_reference_difference_m = 0.0
    enriched: list[dict[str, Any]] = []
    for test_id in sorted(dataset):
        test = dataset[test_id]
        result = results[test_id]
        if test.get("path_id") != result.get("path_id"):
            raise ValueError(f"Dataset/result path_id mismatch for {test_id}")
        shortest_distance = float(test["path_distance_m"])
        reference_path = test.get("shortest_path_panos")
        actual_path = result.get("pano_path")
        if not isinstance(reference_path, list) or not all(
            isinstance(value, str) for value in reference_path
        ):
            raise ValueError(f"{test_id} has no valid shortest_path_panos")
        if not isinstance(actual_path, list) or not all(
            isinstance(value, str) for value in actual_path
        ):
            raise ValueError(f"{test_id} has no valid result pano_path")

        recalculated_shortest = pano_path_distance_m(reference_path, pano_graph)
        reference_difference = abs(shortest_distance - recalculated_shortest)
        maximum_reference_difference_m = max(
            maximum_reference_difference_m, reference_difference
        )
        if reference_difference > 0.001:
            raise ValueError(
                f"{test_id} path_distance_m differs from its reference path by "
                f"{reference_difference:.6f} m"
            )

        spl_metrics = compute_navigation_spl_metrics(
            success=bool(result.get("success")),
            shortest_path_distance_m=shortest_distance,
            actual_path_distance_m=pano_path_distance_m(actual_path, pano_graph),
            shortest_path_pano_steps=int(test["pano_step_count"]),
            movement_steps=int(result.get("movement_steps", max(len(actual_path) - 1, 0))),
        )
        enriched_record = dict(result)
        enriched_record.update(spl_metrics)
        enriched_record.update(
            {
                "visited_room_count": int(test["visited_room_count"]),
                "shortest_distance_bin": distance_bin_label(
                    shortest_distance, bins
                ),
            }
        )
        enriched.append(enriched_record)
    return enriched, maximum_reference_difference_m


def _average(records: Sequence[dict[str, Any]], key: str) -> float | None:
    if not records:
        return None
    return sum(float(record.get(key, 0.0)) for record in records) / len(records)


def cell_summary(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    count = len(records)
    successes = sum(bool(record.get("success")) for record in records)
    return {
        "case_count": count,
        "success_count": successes,
        "failure_count": count - successes,
        "error_count": sum(bool(record.get("error")) for record in records),
        "success_rate": successes / count if count else None,
        "average_spl": _average(records, "spl"),
        "average_step_spl_legacy": _average(records, "step_spl_legacy"),
        "average_shortest_path_distance_m": _average(
            records, "shortest_path_distance_m"
        ),
        "average_actual_path_distance_m": _average(
            records, "actual_path_distance_m"
        ),
        "average_movement_steps": _average(records, "movement_steps"),
        "average_elapsed_seconds": _average(records, "elapsed_seconds"),
    }


def build_summary(
    records: Sequence[dict[str, Any]],
    bins: Sequence[dict[str, Any]],
    *,
    maximum_reference_difference_m: float,
) -> dict[str, Any]:
    room_counts = sorted({int(record["visited_room_count"]) for record in records})
    bin_labels = [str(item["label"]) for item in bins]
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = {
        (room_count, label): []
        for room_count in room_counts
        for label in bin_labels
    }
    for record in records:
        grouped[(int(record["visited_room_count"]), record["shortest_distance_bin"])].append(record)

    cells = {
        str(room_count): {
            label: cell_summary(grouped[(room_count, label)])
            for label in bin_labels
        }
        for room_count in room_counts
    }
    row_overall = {
        str(room_count): cell_summary(
            [
                record
                for label in bin_labels
                for record in grouped[(room_count, label)]
            ]
        )
        for room_count in room_counts
    }
    column_overall = {
        label: cell_summary(
            [
                record
                for room_count in room_counts
                for record in grouped[(room_count, label)]
            ]
        )
        for label in bin_labels
    }
    return {
        "schema": "llm_nav_shortest_distance_navigation_report",
        "schema_version": 2,
        "case_count": len(records),
        "distance_definition": (
            "Haversine metres summed over adjacent panorama graph edges"
        ),
        "grouping_metric": "dataset path_distance_m",
        "primary_spl_metric": "spl",
        "spl_formula": (
            "success * shortest_path_distance_m / "
            "max(actual_path_distance_m, shortest_path_distance_m)"
        ),
        "legacy_comparison_metric": "step_spl_legacy",
        "distance_bins": list(bins),
        "overall": cell_summary(records),
        "by_shortest_distance": {
            label: column_overall[label] for label in bin_labels
        },
        "grid": {
            "room_counts": room_counts,
            "distance_bin_labels": bin_labels,
            "cells": cells,
            "row_overall": row_overall,
            "column_overall": column_overall,
        },
        "validation": {
            "result_count": len(records),
            "unique_test_id_count": len({record["test_id"] for record in records}),
            "error_count": sum(bool(record.get("error")) for record in records),
            "max_reference_path_distance_difference_m": (
                maximum_reference_difference_m
            ),
        },
    }


def _format_number(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:g}"


def _format_optional(value: float | None, digits: int = 3) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def _format_sr(cell: dict[str, Any]) -> str:
    count = int(cell["case_count"])
    if not count:
        return "—"
    return (
        f"{cell['success_count']}/{count} "
        f"({100.0 * float(cell['success_rate']):.1f}%)"
    )


def markdown_grid(
    summary: dict[str, Any], *, metric: str
) -> list[str]:
    grid = summary["grid"]
    labels = grid["distance_bin_labels"]
    headers = ["展廳數", *labels, "Overall"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for room_count in grid["room_counts"]:
        cells = [grid["cells"][str(room_count)][label] for label in labels]
        overall = grid["row_overall"][str(room_count)]
        if metric == "count":
            values = [str(cell["case_count"]) if cell["case_count"] else "—" for cell in cells]
            overall_value = str(overall["case_count"])
        elif metric == "sr":
            values = [_format_sr(cell) for cell in cells]
            overall_value = _format_sr(overall)
        else:
            values = [_format_optional(cell.get(metric)) for cell in cells]
            overall_value = _format_optional(overall.get(metric))
        lines.append(
            "| "
            + " | ".join([str(room_count), *values, overall_value])
            + " |"
        )

    column_cells = [grid["column_overall"][label] for label in labels]
    overall = summary["overall"]
    if metric == "count":
        values = [str(cell["case_count"]) if cell["case_count"] else "—" for cell in column_cells]
        overall_value = str(overall["case_count"])
    elif metric == "sr":
        values = [_format_sr(cell) for cell in column_cells]
        overall_value = _format_sr(overall)
    else:
        values = [_format_optional(cell.get(metric)) for cell in column_cells]
        overall_value = _format_optional(overall.get(metric))
    lines.append("| Overall | " + " | ".join([*values, overall_value]) + " |")
    return lines


def build_markdown(summary: dict[str, Any]) -> str:
    overall = summary["overall"]
    lines = [
        "# Navigation Results by Shortest-Path Distance",
        "",
        "- Grouping: dataset `path_distance_m`, lower-inclusive and upper-exclusive 20 m bins.",
        "- SPL uses shortest-path and actual-path distance in metres.",
        "- API errors remain failures and stay in the denominator.",
        "",
        "## Overall",
        "",
        "| 指標 | 結果 |",
        "|---|---:|",
        f"| 測試案例 | {overall['case_count']} |",
        f"| 成功案例 | {_format_sr(overall)} |",
        f"| SPL | {overall['average_spl']:.3f} |",
        f"| Step SPL（舊版對照） | {overall['average_step_spl_legacy']:.3f} |",
        f"| 平均最短路徑距離 | {overall['average_shortest_path_distance_m']:.1f} m |",
        f"| 平均實際行走距離 | {overall['average_actual_path_distance_m']:.1f} m |",
        f"| 錯誤案例 | {overall['error_count']} |",
        "",
        "## Overall Results by Shortest-Path Distance",
        "",
        "| 最短距離 | 案例數 | 成功案例 | SR | SPL | Step SPL（舊） | 平均實際距離 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, cell in summary["by_shortest_distance"].items():
        if not cell["case_count"]:
            continue
        lines.append(
            f"| {label} | {cell['case_count']} | {cell['success_count']} | "
            f"{100.0 * cell['success_rate']:.1f}% | "
            f"{cell['average_spl']:.3f} | "
            f"{cell['average_step_spl_legacy']:.3f} | "
            f"{cell['average_actual_path_distance_m']:.1f} m |"
        )
    lines.extend(
        [
            "",
            "## Case Count by Visited Room Count and Shortest-Path Distance",
            "",
            *markdown_grid(summary, metric="count"),
            "",
            "## Navigation Success Rate by Visited Room Count and Shortest-Path Distance",
            "",
            *markdown_grid(summary, metric="sr"),
            "",
            "## Navigation SPL by Visited Room Count and Shortest-Path Distance",
            "",
            *markdown_grid(summary, metric="average_spl"),
            "",
            "## Step SPL by Visited Room Count and Shortest-Path Distance (Legacy Comparison)",
            "",
            *markdown_grid(summary, metric="average_step_spl_legacy"),
            "",
        ]
    )
    return "\n".join(lines)


def _atomic_write(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        stream.write(contents)
        temporary_path = Path(stream.name)
    os.replace(temporary_path, path)


def write_outputs(
    output_dir: Path,
    records: Sequence[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    _atomic_write(
        output_dir / "results_with_distance_metrics.jsonl",
        "".join(
            json.dumps(record, ensure_ascii=False) + "\n" for record in records
        ),
    )
    _atomic_write(
        output_dir / "summary.json",
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
    )
    _atomic_write(output_dir / "meeting_summary.md", build_markdown(summary))

    csv_rows = []
    for label, cell in summary["by_shortest_distance"].items():
        csv_rows.append({"shortest_distance_bin": label, **cell})
    fieldnames = list(csv_rows[0]) if csv_rows else ["shortest_distance_bin"]
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        dir=output_dir,
        prefix=".distance_bin_summary.csv.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
        temporary_path = Path(stream.name)
    os.replace(temporary_path, output_dir / "distance_bin_summary.csv")


def main() -> int:
    args = parse_args()
    dataset_records = load_jsonl(args.dataset)
    result_records = load_jsonl(args.results)
    with args.pano_graph.open("r", encoding="utf-8") as stream:
        pano_graph = json.load(stream)
    if not isinstance(pano_graph, dict):
        raise ValueError("Pano graph must be a JSON object")
    if not dataset_records:
        raise ValueError("Dataset is empty")

    maximum_observed = max(
        float(record["path_distance_m"]) for record in dataset_records
    )
    bins = build_distance_bins(
        maximum_observed,
        width_m=args.bin_width_m,
        maximum_requested_m=args.max_distance_m,
    )
    enriched, maximum_reference_difference = enrich_results(
        dataset_records, result_records, pano_graph, bins
    )
    summary = build_summary(
        enriched,
        bins,
        maximum_reference_difference_m=maximum_reference_difference,
    )
    write_outputs(args.output_dir, enriched, summary)
    print(
        f"Wrote shortest-distance report for {len(enriched)} cases to "
        f"{args.output_dir}"
    )
    print(
        f"SR={summary['overall']['success_rate']:.6f} "
        f"SPL={summary['overall']['average_spl']:.6f} "
        f"step_SPL_legacy={summary['overall']['average_step_spl_legacy']:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
