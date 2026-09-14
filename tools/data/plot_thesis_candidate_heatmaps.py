from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    "outputs/navigation_paths/floor0_semantic_targets_topology_constrained/"
    "thesis_24909/candidate_tasks_24909.psv"
)
DEFAULT_EXPECTED_TASK_COUNT = 24_909
DEFAULT_BIN_WIDTH = 0.2

PERCENTAGE_HEATMAP_FILENAME = (
    "detour_ratio_by_visited_room_count_percentage_heatmap_24909.png"
)
COUNT_HEATMAP_FILENAME = (
    "detour_ratio_by_visited_room_count_count_heatmap_24909.png"
)
PERCENTAGE_MATRIX_FILENAME = (
    "detour_ratio_by_visited_room_count_percentage_matrix_24909.csv"
)
COUNT_MATRIX_FILENAME = "detour_ratio_by_visited_room_count_count_matrix_24909.csv"
SUMMARY_FILENAME = "detour_ratio_by_visited_room_count_heatmap_summary_24909.json"


@dataclass(frozen=True)
class CandidateMetric:
    task_id: str
    visited_room_count: int
    detour_ratio: float


@dataclass(frozen=True)
class HeatmapData:
    room_counts: tuple[int, ...]
    detour_ratio_edges: np.ndarray
    detour_ratio_labels: tuple[str, ...]
    counts: np.ndarray
    percentages: np.ndarray
    minimum: float
    p99: float
    maximum: float
    display_minimum: float
    display_maximum: float


def resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate.resolve() if candidate.is_absolute() else (PROJECT_ROOT / candidate).resolve()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_candidate_metrics(path: str | Path) -> list[CandidateMetric]:
    source = Path(path)
    delimiter = "|" if source.suffix.lower() == ".psv" else ","
    metrics: list[CandidateMetric] = []
    task_ids: set[str] = set()
    with source.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter=delimiter)
        required_fields = {"task_id", "visited_room_count", "detour_ratio"}
        missing_fields = required_fields - set(reader.fieldnames or ())
        if missing_fields:
            raise ValueError(
                f"{source} is missing required fields: {sorted(missing_fields)!r}."
            )
        for line_number, row in enumerate(reader, start=2):
            task_id = str(row.get("task_id", "")).strip()
            if not task_id:
                raise ValueError(f"{source}:{line_number} has an empty task_id.")
            if task_id in task_ids:
                raise ValueError(f"{source}:{line_number} repeats task_id {task_id!r}.")
            task_ids.add(task_id)
            try:
                visited_room_count = int(row["visited_room_count"])
                detour_ratio = float(row["detour_ratio"])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"{source}:{line_number} has invalid heatmap fields."
                ) from error
            if visited_room_count <= 0:
                raise ValueError(
                    f"{source}:{line_number} has a non-positive visited_room_count."
                )
            if not math.isfinite(detour_ratio) or detour_ratio <= 0.0:
                raise ValueError(
                    f"{source}:{line_number} has an invalid detour_ratio."
                )
            metrics.append(
                CandidateMetric(
                    task_id=task_id,
                    visited_room_count=visited_room_count,
                    detour_ratio=detour_ratio,
                )
            )
    if not metrics:
        raise ValueError(f"{source} contains no candidate tasks.")
    return metrics


def build_heatmap_data(
    metrics: Sequence[CandidateMetric],
    *,
    bin_width: float = DEFAULT_BIN_WIDTH,
) -> HeatmapData:
    if not metrics:
        raise ValueError("Cannot build a heatmap without candidate tasks.")
    if not math.isfinite(bin_width) or bin_width <= 0.0:
        raise ValueError("bin_width must be a positive finite number.")

    room_counts = tuple(sorted({metric.visited_room_count for metric in metrics}))
    all_ratios = np.asarray([metric.detour_ratio for metric in metrics], dtype=np.float64)
    minimum = float(np.min(all_ratios))
    p99 = float(np.percentile(all_ratios, 99))
    maximum = float(np.max(all_ratios))
    display_minimum = math.floor(minimum / bin_width + 1e-12) * bin_width
    display_maximum = math.ceil(p99 / bin_width - 1e-12) * bin_width
    display_maximum = max(display_maximum, display_minimum + bin_width)
    regular_bin_count = int(
        round((display_maximum - display_minimum) / bin_width)
    )
    edges = np.linspace(
        display_minimum,
        display_maximum,
        regular_bin_count + 1,
        dtype=np.float64,
    )

    counts = np.zeros((len(edges), len(room_counts)), dtype=np.int64)
    for column, room_count in enumerate(room_counts):
        values = np.asarray(
            [
                metric.detour_ratio
                for metric in metrics
                if metric.visited_room_count == room_count
            ],
            dtype=np.float64,
        )
        visible = values[values <= display_maximum]
        regular_counts, _ = np.histogram(visible, bins=edges)
        overflow_count = int(values.size - visible.size)
        counts[:, column] = np.concatenate(
            [regular_counts, np.asarray([overflow_count], dtype=np.int64)]
        )

    column_totals = counts.sum(axis=0)
    if np.any(column_totals <= 0):
        raise ValueError("At least one visited-room-count column is empty.")
    percentages = 100.0 * counts / column_totals[np.newaxis, :]
    labels = tuple(
        [
            f"{lower:.1f}–{upper:.1f}"
            for lower, upper in zip(edges[:-1], edges[1:], strict=True)
        ]
        + [f">{display_maximum:.1f}"]
    )
    return HeatmapData(
        room_counts=room_counts,
        detour_ratio_edges=edges,
        detour_ratio_labels=labels,
        counts=counts,
        percentages=percentages,
        minimum=minimum,
        p99=p99,
        maximum=maximum,
        display_minimum=display_minimum,
        display_maximum=display_maximum,
    )


def _atomic_write_matrix_csv(
    path: Path,
    *,
    data: HeatmapData,
    matrix: np.ndarray,
    percentage: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    fieldnames = ["detour_ratio_bin", *(str(value) for value in data.room_counts), "row_total"]
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            for row_index, label in enumerate(data.detour_ratio_labels):
                row = {"detour_ratio_bin": label}
                for column, room_count in enumerate(data.room_counts):
                    value = matrix[row_index, column]
                    row[str(room_count)] = f"{float(value):.10g}" if percentage else int(value)
                row["row_total"] = (
                    "" if percentage else int(data.counts[row_index, :].sum())
                )
                writer.writerow(row)
            total_row: dict[str, object] = {"detour_ratio_bin": "TOTAL"}
            for column, room_count in enumerate(data.room_counts):
                total_row[str(room_count)] = (
                    "100" if percentage else int(data.counts[:, column].sum())
                )
            total_row["row_total"] = "" if percentage else int(data.counts.sum())
            writer.writerow(total_row)
            temporary_path = Path(stream.name)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
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
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _write_heatmap(
    path: Path,
    *,
    data: HeatmapData,
    matrix: np.ndarray,
    title: str,
    colorbar_label: str,
    percentage: bool,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    fig_width = max(12.0, 0.95 * len(data.room_counts) + 3.0)
    fig_height = max(8.0, 0.48 * len(data.detour_ratio_labels) + 2.2)
    fig = plt.figure(figsize=(fig_width, fig_height))
    grid = fig.add_gridspec(
        1,
        2,
        width_ratios=(1.0, 0.038),
        wspace=0.025,
        left=0.10,
        right=0.90,
        bottom=0.08,
        top=0.92,
    )
    ax = fig.add_subplot(grid[0, 0])
    colorbar_ax = fig.add_subplot(grid[0, 1])
    image = ax.imshow(
        matrix,
        origin="lower",
        aspect="auto",
        cmap="YlOrRd",
        vmin=0.0,
    )

    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = float(matrix[row, column])
            if value <= 0.0:
                continue
            if percentage:
                label = "<0.1" if value < 0.05 else f"{value:.1f}"
            else:
                label = str(int(value))
            color = "white" if image.norm(value) >= 0.55 else "#222222"
            ax.text(
                column,
                row,
                label,
                ha="center",
                va="center",
                fontsize=6.5,
                color=color,
            )

    ax.set_xticks(np.arange(len(data.room_counts)))
    ax.set_xticklabels(data.room_counts)
    ax.set_yticks(np.arange(len(data.detour_ratio_labels)))
    ax.set_yticklabels(data.detour_ratio_labels)
    ax.set_xticks(np.arange(-0.5, len(data.room_counts), 1.0), minor=True)
    ax.set_yticks(np.arange(-0.5, len(data.detour_ratio_labels), 1.0), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    fig.suptitle(title, fontsize=14)
    ax.set_xlabel("Visited Room Count")
    ax.set_ylabel("Detour Ratio")
    colorbar = fig.colorbar(image, cax=colorbar_ax)
    colorbar.set_label(colorbar_label)

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".png",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
        fig.savefig(temporary_path, dpi=180)
        os.replace(temporary_path, path)
    finally:
        plt.close(fig)
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def generate_heatmaps(
    *,
    input_path: Path,
    output_dir: Path,
    expected_task_count: int = DEFAULT_EXPECTED_TASK_COUNT,
    bin_width: float = DEFAULT_BIN_WIDTH,
) -> dict:
    metrics = load_candidate_metrics(input_path)
    if len(metrics) != expected_task_count:
        raise ValueError(
            f"Expected {expected_task_count} candidate tasks, found {len(metrics)}."
        )
    data = build_heatmap_data(metrics, bin_width=bin_width)
    if int(data.counts.sum()) != expected_task_count:
        raise ValueError("Heatmap cell counts do not add up to the candidate task count.")
    if not np.allclose(data.percentages.sum(axis=0), 100.0):
        raise ValueError("Percentage heatmap columns do not add up to 100%.")

    output_dir.mkdir(parents=True, exist_ok=True)
    percentage_heatmap_path = output_dir / PERCENTAGE_HEATMAP_FILENAME
    count_heatmap_path = output_dir / COUNT_HEATMAP_FILENAME
    percentage_matrix_path = output_dir / PERCENTAGE_MATRIX_FILENAME
    count_matrix_path = output_dir / COUNT_MATRIX_FILENAME
    _write_heatmap(
        percentage_heatmap_path,
        data=data,
        matrix=data.percentages,
        title=(
            "Detour Ratio Distribution by Visited Room Count "
            f"({expected_task_count:,} Tasks)"
        ),
        colorbar_label="Task Percentage (%)",
        percentage=True,
    )
    _write_heatmap(
        count_heatmap_path,
        data=data,
        matrix=data.counts,
        title=(
            "Task Count by Visited Room Count and Detour Ratio "
            f"({expected_task_count:,} Tasks)"
        ),
        colorbar_label="Task Count",
        percentage=False,
    )
    _atomic_write_matrix_csv(
        percentage_matrix_path,
        data=data,
        matrix=data.percentages,
        percentage=True,
    )
    _atomic_write_matrix_csv(
        count_matrix_path,
        data=data,
        matrix=data.counts,
        percentage=False,
    )

    output_paths = (
        percentage_heatmap_path,
        count_heatmap_path,
        percentage_matrix_path,
        count_matrix_path,
    )
    summary = {
        "schema": "llm_nav_thesis_candidate_heatmaps",
        "schema_version": 1,
        "input": {
            "path": str(input_path),
            "sha256": sha256_file(input_path),
            "task_count": len(metrics),
        },
        "binning": {
            "bin_width": bin_width,
            "minimum": data.minimum,
            "p99": data.p99,
            "maximum": data.maximum,
            "display_minimum": data.display_minimum,
            "display_maximum": data.display_maximum,
            "overflow_label": data.detour_ratio_labels[-1],
        },
        "visited_room_counts": list(data.room_counts),
        "task_count_by_visited_room_count": {
            str(room_count): int(data.counts[:, column].sum())
            for column, room_count in enumerate(data.room_counts)
        },
        "validation": {
            "cell_count_total": int(data.counts.sum()),
            "percentage_column_sums": {
                str(room_count): float(data.percentages[:, column].sum())
                for column, room_count in enumerate(data.room_counts)
            },
            "status": "valid",
        },
        "output_files": {
            path.name: {"path": str(path), "sha256": sha256_file(path)}
            for path in output_paths
        },
    }
    summary_path = output_dir / SUMMARY_FILENAME
    _atomic_write_json(summary_path, summary)
    summary["output_files"][SUMMARY_FILENAME] = {
        "path": str(summary_path),
        "sha256": sha256_file(summary_path),
    }
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate percentage and raw-count Visited Room Count x Detour Ratio "
            "heatmaps for the thesis-defined 24,909 candidate tasks."
        )
    )
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--expected-task-count",
        type=int,
        default=DEFAULT_EXPECTED_TASK_COUNT,
    )
    parser.add_argument("--bin-width", type=float, default=DEFAULT_BIN_WIDTH)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    input_path = resolve_project_path(args.input)
    output_dir = (
        resolve_project_path(args.output_dir)
        if args.output_dir
        else input_path.parent
    )
    result = generate_heatmaps(
        input_path=input_path,
        output_dir=output_dir,
        expected_task_count=args.expected_task_count,
        bin_width=args.bin_width,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
