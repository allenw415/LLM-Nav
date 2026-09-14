#!/usr/bin/env python3
"""Plot thesis-ready navigation SR trends by estimated mean room area."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-summary", type=Path, required=True)
    parser.add_argument("--actual-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return value


def extract_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for group in summary["groups"]:
        result = group["with_trajectory"]
        rows.append(
            {
                "group_label": group["group_label"],
                "case_count": int(result["case_count"]),
                "success_rate": float(result["success_rate"]),
                "wilson_95_ci_low": float(result["wilson_95_ci_low"]),
                "wilson_95_ci_high": float(result["wilson_95_ci_high"]),
            }
        )
    return rows


def plot(reference_rows: list[dict[str, Any]], actual_rows: list[dict[str, Any]], output: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    if [row["group_label"] for row in reference_rows] != [
        row["group_label"] for row in actual_rows
    ]:
        raise ValueError("Reference and actual summaries use different area bins")

    x = np.arange(len(reference_rows))
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    series = (
        (
            "Reference Shortest Path",
            reference_rows,
            "#2878B5",
            "o",
        ),
        (
            "Actual Agent Path",
            actual_rows,
            "#E26A1B",
            "^",
        ),
    )
    all_rates = [
        np.array([row["success_rate"] * 100.0 for row in rows])
        for _, rows, _, _ in series
    ]
    for series_index, (label, rows, color, marker) in enumerate(series):
        rates = np.array([row["success_rate"] * 100.0 for row in rows])
        lower = np.array([row["wilson_95_ci_low"] * 100.0 for row in rows])
        upper = np.array([row["wilson_95_ci_high"] * 100.0 for row in rows])
        yerr = np.vstack((rates - lower, upper - rates))
        ax.errorbar(
            x,
            rates,
            yerr=yerr,
            label=label,
            color=color,
            linewidth=2.0,
            marker=marker,
            markersize=7,
            capsize=4,
            capthick=1.2,
            zorder=3,
        )
        for index, row in enumerate(rows):
            other_rate = all_rates[1 - series_index][index]
            is_higher = rates[index] >= other_rate
            ax.annotate(
                f"{100.0 * row['success_rate']:.1f}%\n(n={row['case_count']})",
                (index, rates[index]),
                xytext=(0, 12 if is_higher else -16),
                textcoords="offset points",
                ha="center",
                va="bottom" if is_higher else "top",
                fontsize=9,
            )

    ax.set_xticks(x, [row["group_label"].replace("-", "–") for row in reference_rows])
    ax.set_xlabel("Estimated Mean Room Area (m²)")
    ax.set_ylabel("Navigation Success Rate (%)")
    ax.set_ylim(0, 105)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.09), ncol=2, frameon=False)
    fig.subplots_adjust(left=0.11, right=0.89, bottom=0.17, top=0.88)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, facecolor="white")
    plt.close(fig)


def plot_single(
    rows: list[dict[str, Any]],
    *,
    color: str,
    marker: str,
    output: Path,
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    x = np.arange(len(rows))
    rates = np.array([row["success_rate"] * 100.0 for row in rows])
    lower = np.array([row["wilson_95_ci_low"] * 100.0 for row in rows])
    upper = np.array([row["wilson_95_ci_high"] * 100.0 for row in rows])
    yerr = np.vstack((rates - lower, upper - rates))

    fig, ax = plt.subplots(figsize=(8.6, 5.1))
    ax.errorbar(
        x,
        rates,
        yerr=yerr,
        color=color,
        linewidth=2.0,
        marker=marker,
        markersize=7,
        capsize=4,
        capthick=1.2,
        zorder=3,
    )
    for index, row in enumerate(rows):
        ax.annotate(
            f"{100.0 * row['success_rate']:.1f}%\n(n={row['case_count']})",
            (index, rates[index]),
            xytext=(0, 12),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_xticks(x, [row["group_label"].replace("-", "–") for row in rows])
    ax.set_xlabel("Estimated Mean Room Area (m²)")
    ax.set_ylabel("Navigation Success Rate (%)")
    ax.set_ylim(0, 105)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(left=0.11, right=0.89, bottom=0.17, top=0.95)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, facecolor="white")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    reference_rows = extract_rows(load_json(args.reference_summary))
    actual_rows = extract_rows(load_json(args.actual_summary))
    plot(reference_rows, actual_rows, args.output)
    plot_single(
        reference_rows,
        color="#2878B5",
        marker="o",
        output=args.output.parent
        / "figure_4_9_reference_shortest_path_mean_room_area_sr.png",
    )
    plot_single(
        actual_rows,
        color="#E26A1B",
        marker="^",
        output=args.output.parent
        / "figure_4_10_actual_agent_path_mean_room_area_sr.png",
    )
    print(
        json.dumps(
            {"reference": reference_rows, "actual": actual_rows},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
