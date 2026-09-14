#!/usr/bin/env python3
"""Create thesis figures for navigation SR by distance and detour ratio."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


DISTANCE_BINS = tuple(f"{lower}-{lower + 20} m" for lower in range(0, 180, 20))
DETOUR_BINS = ("1.0-1.2", "1.2-1.4", "1.4-1.6", "1.6-1.8", "1.8-2.0")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number} is not an object")
            records.append(record)
    return records


def index_by_test_id(records: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for record in records:
        test_id = record.get("test_id")
        if not isinstance(test_id, str) or not test_id:
            raise ValueError(f"{label} contains a record without test_id")
        if test_id in indexed:
            raise ValueError(f"{label} contains duplicate test_id {test_id}")
        indexed[test_id] = record
    return indexed


def wilson_interval(successes: int, count: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if count == 0:
        return (math.nan, math.nan)
    rate = successes / count
    denominator = 1.0 + z * z / count
    centre = (rate + z * z / (2.0 * count)) / denominator
    half_width = (
        z
        * math.sqrt(rate * (1.0 - rate) / count + z * z / (4.0 * count * count))
        / denominator
    )
    return (centre - half_width, centre + half_width)


def summarize(
    results: list[dict[str, Any]], dataset: list[dict[str, Any]]
) -> dict[str, Any]:
    result_index = index_by_test_id(results, "results")
    dataset_index = index_by_test_id(dataset, "dataset")
    if set(result_index) != set(dataset_index):
        raise ValueError("Dataset and results do not contain identical test IDs")

    grouped: dict[tuple[str, str], list[bool]] = defaultdict(list)
    for test_id, result in result_index.items():
        test = dataset_index[test_id]
        distance_bin = result.get("shortest_distance_bin")
        detour_bin = test.get("ratio_stratum")
        if distance_bin not in DISTANCE_BINS:
            raise ValueError(f"Unexpected distance bin for {test_id}: {distance_bin!r}")
        if detour_bin not in DETOUR_BINS:
            raise ValueError(f"Unexpected detour bin for {test_id}: {detour_bin!r}")
        grouped[(str(distance_bin), str(detour_bin))].append(bool(result.get("success")))

    distance_rows = []
    heatmap_rows = []
    for distance_bin in DISTANCE_BINS:
        distance_values = [
            value
            for detour_bin in DETOUR_BINS
            for value in grouped[(distance_bin, detour_bin)]
        ]
        successes = sum(distance_values)
        count = len(distance_values)
        lower, upper = wilson_interval(successes, count)
        distance_rows.append(
            {
                "distance_bin": distance_bin,
                "case_count": count,
                "success_count": successes,
                "success_rate": successes / count if count else None,
                "wilson_95_lower": lower,
                "wilson_95_upper": upper,
            }
        )
        for detour_bin in DETOUR_BINS:
            values = grouped[(distance_bin, detour_bin)]
            cell_count = len(values)
            cell_successes = sum(values)
            heatmap_rows.append(
                {
                    "distance_bin": distance_bin,
                    "detour_bin": detour_bin,
                    "case_count": cell_count,
                    "success_count": cell_successes,
                    "success_rate": cell_successes / cell_count if cell_count else None,
                }
            )

    if sum(row["case_count"] for row in distance_rows) != len(results):
        raise AssertionError("Distance-bin counts do not sum to the result count")
    return {
        "schema": "llm_nav_navigation_distance_detour_figures",
        "case_count": len(results),
        "api_errors_in_denominator": True,
        "distance_bins": list(DISTANCE_BINS),
        "detour_bins": list(DETOUR_BINS),
        "by_distance": distance_rows,
        "distance_by_detour": heatmap_rows,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_distance_trend(summary: dict[str, Any], output_path: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    rows = summary["by_distance"]
    x = np.arange(len(rows))
    rates = np.array([row["success_rate"] * 100.0 for row in rows])
    lower = np.array([row["wilson_95_lower"] * 100.0 for row in rows])
    upper = np.array([row["wilson_95_upper"] * 100.0 for row in rows])
    yerr = np.vstack((rates - lower, upper - rates))

    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    ax.errorbar(
        x,
        rates,
        yerr=yerr,
        color="#2878B5",
        linewidth=2.0,
        marker="o",
        markersize=6.5,
        capsize=4,
        capthick=1.2,
        zorder=3,
    )
    for index, row in enumerate(rows):
        ax.annotate(
            f"{100.0 * row['success_rate']:.1f}%\n(n={row['case_count']})",
            (index, rates[index]),
            xytext=(0, 11),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_xticks(x, [row["distance_bin"].replace(" m", "") for row in rows])
    ax.set_xlabel("Ground-Truth Shortest-Path Distance (m)")
    ax.set_ylabel("Navigation Success Rate (%)")
    ax.set_ylim(0, 105)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    # Keep the plotting area horizontally centred on the exported canvas so it
    # remains visually centred after insertion at full text width in Word.
    fig.subplots_adjust(left=0.10, right=0.90, bottom=0.17, top=0.95)
    fig.savefig(output_path, dpi=300, facecolor="white")
    plt.close(fig)


def plot_distance_detour_heatmap(summary: dict[str, Any], output_path: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap

    cell_index = {
        (row["distance_bin"], row["detour_bin"]): row
        for row in summary["distance_by_detour"]
    }
    rates = np.full((len(DISTANCE_BINS), len(DETOUR_BINS)), np.nan)
    counts = np.zeros_like(rates, dtype=int)
    for row_index, distance_bin in enumerate(DISTANCE_BINS):
        for column_index, detour_bin in enumerate(DETOUR_BINS):
            row = cell_index[(distance_bin, detour_bin)]
            counts[row_index, column_index] = row["case_count"]
            if row["success_rate"] is not None:
                rates[row_index, column_index] = row["success_rate"] * 100.0

    cmap = LinearSegmentedColormap.from_list(
        "thesis_success_rate", ["#F2F6FA", "#9CC9E2", "#2878B5", "#174A7E"]
    )
    cmap.set_bad("#EEEEEE")
    fig, ax = plt.subplots(figsize=(8.8, 7.6))
    image = ax.imshow(rates, cmap=cmap, vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(np.arange(len(DETOUR_BINS)), [label.replace("-", "–") for label in DETOUR_BINS])
    ax.set_yticks(np.arange(len(DISTANCE_BINS)), [label.replace("-", "–") for label in DISTANCE_BINS])
    ax.set_xlabel("Detour Ratio")
    ax.set_ylabel("Ground-Truth Shortest-Path Distance")
    ax.set_xticks(np.arange(-0.5, len(DETOUR_BINS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(DISTANCE_BINS), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)
    for row_index in range(len(DISTANCE_BINS)):
        for column_index in range(len(DETOUR_BINS)):
            count = counts[row_index, column_index]
            value = rates[row_index, column_index]
            label = "—\n(n=0)" if count == 0 else f"{value:.1f}%\n(n={count})"
            color = "white" if count and value >= 67.0 else "#222222"
            ax.text(
                column_index,
                row_index,
                label,
                ha="center",
                va="center",
                fontsize=8.5,
                color=color,
            )
    colorbar = fig.colorbar(image, ax=ax, pad=0.025)
    colorbar.set_label("Navigation Success Rate (%)")
    fig.subplots_adjust(left=0.20, right=0.88, bottom=0.11, top=0.97)
    fig.savefig(output_path, dpi=300, facecolor="white")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(load_jsonl(args.results), load_jsonl(args.dataset))
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_csv(args.output_dir / "distance_success_rate.csv", summary["by_distance"])
    write_csv(
        args.output_dir / "distance_detour_success_rate.csv",
        summary["distance_by_detour"],
    )
    plot_distance_trend(
        summary,
        args.output_dir / "figure_4_7a_distance_success_rate_trend.png",
    )
    plot_distance_detour_heatmap(
        summary,
        args.output_dir / "figure_4_7b_distance_detour_success_rate_heatmap.png",
    )
    print(json.dumps(summary["by_distance"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
