from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


STAGES = ("visual", "bayesian", "spatial")
STAGE_LABELS = {
    "visual": "Visual Localization",
    "bayesian": "Visual Localization + Bayesian Filtering",
    "spatial": "Visual Localization + Bayesian Filtering + Spatial Alignment",
}
PLOT_STAGE_LABELS = {
    "visual": "Visual Localization",
    "bayesian": "Visual Localization\n+ Bayesian Filtering",
    "spatial": "Visual Localization + Bayesian Filtering\n+ Spatial Alignment",
}
COLORS = {
    "visual": "#4C9BE8",
    "bayesian": "#F28E3B",
    "spatial": "#59C777",
}


def _room_sort_key(room_id: str) -> tuple[int, str]:
    match = re.fullmatch(r"Room\s+(\d+)([a-z]?)", room_id, re.IGNORECASE)
    if match:
        return (int(match.group(1)), match.group(2).lower())
    return (10_000, room_id.lower())


def _score_ranking(
    scores: object,
    *,
    predicted_room_id: object = None,
) -> list[str]:
    if not isinstance(scores, dict):
        return [predicted_room_id] if isinstance(predicted_room_id, str) else []
    numeric_scores = {
        str(room_id): float(score)
        for room_id, score in scores.items()
        if isinstance(score, (int, float))
    }
    ranking = sorted(
        numeric_scores,
        key=lambda room_id: (-numeric_scores[room_id], _room_sort_key(room_id)),
    )
    if isinstance(predicted_room_id, str) and predicted_room_id:
        ranking = [predicted_room_id, *[room for room in ranking if room != predicted_room_id]]
    return ranking


def _event_rankings(event: dict[str, Any]) -> dict[str, list[str]]:
    visual = _score_ranking(
        event.get("visual_room_scores") or event.get("visual_room_belief"),
        predicted_room_id=event.get("visual_predicted_room_id"),
    )
    bayesian = _score_ranking(
        event.get("bayesian_room_belief"),
        predicted_room_id=event.get("bayesian_predicted_room_id"),
    )

    spatial_prefix: list[str] = []
    if event.get("alignment_applied") is True:
        top_k = event.get("alignment_top_k")
        if isinstance(top_k, list):
            spatial_prefix = [
                str(item["room_id"])
                for item in top_k
                if isinstance(item, dict) and isinstance(item.get("room_id"), str)
            ]
    spatial_prediction = event.get("spatial_predicted_room_id")
    if isinstance(spatial_prediction, str) and spatial_prediction:
        spatial_prefix = [
            spatial_prediction,
            *[room for room in spatial_prefix if room != spatial_prediction],
        ]
    spatial = [*spatial_prefix, *[room for room in bayesian if room not in spatial_prefix]]
    return {"visual": visual, "bayesian": bayesian, "spatial": spatial}


def _load_events(results_path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with results_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            result = json.loads(line)
            if not isinstance(result, dict):
                raise ValueError(f"{results_path}:{line_number} is not an object")
            for event in result.get("localization_events") or []:
                if isinstance(event, dict) and isinstance(event.get("ground_truth_room_id"), str):
                    events.append(event)
    return events


def summarize(events: list[dict[str, Any]]) -> dict[str, Any]:
    hits = {stage: {1: 0, 3: 0, 5: 0} for stage in STAGES}
    by_room: dict[str, dict[str, list[bool]]] = defaultdict(
        lambda: {stage: [] for stage in STAGES}
    )
    applied_count = 0
    for event in events:
        ground_truth = str(event["ground_truth_room_id"])
        rankings = _event_rankings(event)
        applied_count += event.get("alignment_applied") is True
        for stage in STAGES:
            ranking = rankings[stage]
            for k in (1, 3, 5):
                hits[stage][k] += ground_truth in ranking[:k]
            by_room[ground_truth][stage].append(ground_truth in ranking[:1])

    event_count = len(events)
    hit_rows = []
    for stage in STAGES:
        row: dict[str, Any] = {"stage": stage, "label": STAGE_LABELS[stage]}
        for k in (1, 3, 5):
            row[f"hit_at_{k}_count"] = hits[stage][k]
            row[f"hit_at_{k}"] = hits[stage][k] / event_count if event_count else None
        hit_rows.append(row)

    room_rows = []
    for room_id in sorted(by_room, key=_room_sort_key):
        if not room_id.startswith("Room "):
            continue
        row = {"room_id": room_id, "event_count": len(by_room[room_id]["visual"])}
        for stage in STAGES:
            values = by_room[room_id][stage]
            row[f"{stage}_correct_count"] = sum(values)
            row[f"{stage}_accuracy"] = sum(values) / len(values) if values else None
        room_rows.append(row)

    return {
        "schema": "llm_nav_full_system_localization_figures",
        "event_count": event_count,
        "alignment_applied_count": applied_count,
        "alignment_skipped_count": event_count - applied_count,
        "ranking_definition": {
            "visual": "Logged visual prediction followed by visual room scores in descending order.",
            "bayesian": "Logged Bayesian prediction followed by posterior room belief in descending order.",
            "spatial": (
                "When alignment is applied, logged spatial prediction and alignment_top_k are "
                "followed by remaining Bayesian-ranked rooms; when skipped, the Bayesian ranking "
                "is retained. Equal scores use natural room-id order."
            ),
        },
        "hit_at_k": hit_rows,
        "by_gallery": room_rows,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot_hit_at_k(summary: dict[str, Any], output_path: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    rows = {row["stage"]: row for row in summary["hit_at_k"]}
    x = np.arange(3)
    width = 0.24
    fig, ax = plt.subplots(figsize=(8.4, 4.8), constrained_layout=True)
    for index, stage in enumerate(STAGES):
        values = [rows[stage][f"hit_at_{k}"] * 100 for k in (1, 3, 5)]
        bars = ax.bar(
            x + (index - 1) * width,
            values,
            width,
            color=COLORS[stage],
            label=PLOT_STAGE_LABELS[stage],
        )
        ax.bar_label(bars, labels=[f"{value:.1f}%" for value in values], padding=3, fontsize=8)
    ax.set_xticks(x, ["Hit@1", "Hit@3", "Hit@5"])
    ax.set_ylabel("Hit Rate (%)")
    ax.set_ylim(0, 105)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=3, frameon=False, fontsize=8)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_by_gallery(summary: dict[str, Any], output_path: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    rows = summary["by_gallery"]
    midpoint = (len(rows) + 1) // 2
    panels = (rows[:midpoint], rows[midpoint:])
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.2), constrained_layout=True)
    width = 0.25
    for ax, panel in zip(axes, panels):
        x = np.arange(len(panel))
        for index, stage in enumerate(STAGES):
            values = [row[f"{stage}_accuracy"] * 100 for row in panel]
            ax.bar(
                x + (index - 1) * width,
                values,
                width,
                color=COLORS[stage],
                label=PLOT_STAGE_LABELS[stage],
            )
        ax.set_xticks(x, [row["room_id"].replace("Room ", "R") for row in panel], fontsize=8)
        ax.set_ylim(0, 105)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_ylabel("Hit@1 (%)")
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        for tick, row in zip(ax.get_xticklabels(), panel):
            tick.set_text(f"{tick.get_text()}\n(n={row['event_count']})")
        ax.set_xticklabels([f"{row['room_id'].replace('Room ', 'R')}\n(n={row['event_count']})" for row in panel])
    axes[0].legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.25),
        ncol=3,
        frameon=False,
        fontsize=8,
    )
    axes[-1].set_xlabel("Room")
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(_load_events(args.results))
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_csv(args.output_dir / "runtime_hit_at_k.csv", summary["hit_at_k"])
    _write_csv(args.output_dir / "runtime_hit_at_1_by_gallery.csv", summary["by_gallery"])
    _plot_hit_at_k(summary, args.output_dir / "runtime_localization_hit_at_k.png")
    _plot_by_gallery(summary, args.output_dir / "runtime_localization_hit_at_1_by_gallery.png")
    print(json.dumps(summary["hit_at_k"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
