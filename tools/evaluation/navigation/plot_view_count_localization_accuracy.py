#!/usr/bin/env python3
"""Plot paired step-0 room-localization accuracy for 4-view and 8-view observations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


STAGES = (
    ("visual", "initial_visual_correct", "Visual Localization", "#4C9BE8"),
    (
        "bayesian",
        "initial_bayesian_correct",
        "Visual Localization\n+ Bayesian Filtering",
        "#F28E3B",
    ),
    (
        "spatial",
        "initial_spatial_correct",
        "Visual Localization + Bayesian Filtering\n+ Spatial Alignment",
        "#59C777",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--condition", default="with_trajectory")
    return parser.parse_args()


def load_plot_data(path: Path, condition: str) -> tuple[dict[str, list[float]], int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    condition_data = payload["conditions"][condition]
    values: dict[str, list[float]] = {}
    comparable_counts = set()
    for stage, metric, _, _ in STAGES:
        summary = condition_data["boolean_metrics"][metric]
        comparable_counts.add(int(summary["paired_comparable_count"]))
        values[stage] = [
            100.0 * float(summary["four_view_rate"]),
            100.0 * float(summary["eight_view_rate"]),
        ]
    if len(comparable_counts) != 1:
        raise ValueError(f"Stage denominators differ: {sorted(comparable_counts)}")
    return values, comparable_counts.pop()


def plot(values: dict[str, list[float]], paired_count: int, output_dir: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    x = np.arange(3)
    width = 0.32
    fig, ax = plt.subplots(figsize=(10.2, 5.9))

    view_series = (
        ("4-view", [values[stage][0] for stage, _, _, _ in STAGES], "#4C9BE8"),
        ("8-view", [values[stage][1] for stage, _, _, _ in STAGES], "#F28E3B"),
    )
    for index, (label, stage_values, color) in enumerate(view_series):
        bars = ax.bar(
            x + (index - 0.5) * width,
            stage_values,
            width,
            color=color,
            label=label,
            zorder=3,
        )
        ax.bar_label(
            bars,
            labels=[f"{value:.1f}%" for value in stage_values],
            padding=4,
            fontsize=10,
        )

    ax.set_xticks(
        x,
        [
            "Visual Localization",
            "Visual Localization\n+ Bayesian Filtering",
            "Visual Localization + Bayesian Filtering\n+ Spatial Alignment",
        ],
        fontsize=10,
    )
    ax.set_xlabel("Localization Stage", fontsize=11, labelpad=9)
    ax.set_ylabel("Hit@1 (%)", fontsize=11)
    ax.set_ylim(0, 105)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.85, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#333333")
    ax.tick_params(axis="both", colors="#222222", labelsize=10)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),
        ncol=2,
        frameon=False,
        fontsize=9.5,
        handlelength=1.9,
        columnspacing=2.4,
    )
    fig.subplots_adjust(left=0.105, right=0.98, bottom=0.14, top=0.86)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "initial_localization_accuracy_4view_vs_8view"
    fig.savefig(stem.with_suffix(".png"), dpi=300, facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), facecolor="white")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    values, paired_count = load_plot_data(args.comparison_summary, args.condition)
    plot(values, paired_count, args.output_dir)
    print(
        json.dumps(
            {"paired_case_count": paired_count, "accuracy_percent": values},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
