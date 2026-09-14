from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import io
import json
import math
import os
import re
import statistics
import tempfile
from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image, ImageColor, ImageDraw, ImageFont
from PIL.PngImagePlugin import PngInfo


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_DIR = (
    "outputs/navigation_paths/floor0_semantic_targets_topology_constrained/"
    "room_count_2to6_ratio_1p0_2p0_per30_sparse715"
)
EVALUATION_DIR = (
    "outputs/navigation_evaluation/floor0_semantic_targets_topology_constrained/"
    "room_count_2to6_ratio_1p0_2p0_per30_sparse715"
)
DEFAULT_DATASET = f"{DATASET_DIR}/test_set.jsonl"
DEFAULT_WITHOUT_RESULTS = (
    f"{EVALUATION_DIR}/geometry_localization_stages_combined715/results.jsonl"
)
DEFAULT_WITH_RESULTS = (
    f"{EVALUATION_DIR}/geometry_edge_visit_localization_stages_combined715/results.jsonl"
)
DEFAULT_ROOM_AREAS = (
    "dataset/sites/british_museum/analysis/floor0_estimated_gallery_areas.json"
)
DEFAULT_PANO_ROOM_GROUNDING = (
    "dataset/sites/british_museum/normalized/pano_room_grounding.json"
)
DEFAULT_OUTPUT_DIR = f"{EVALUATION_DIR}/mean_room_area_fixed_bins"
QUARTILE_LABELS = ("Q1", "Q2", "Q3", "Q4")
SUPPLEMENTARY_GROUP_COUNTS = (6, 8, 10)
FIXED_WIDTHS_M2 = (100.0, 150.0)
FILTERED_FIXED_WIDTH_M2 = 200.0
FILTERED_MAXIMUM_AREA_M2 = 800.0
OPEN_ENDED_FIXED_WIDTH_M2 = 200.0
OPEN_ENDED_FROM_M2 = 600.0
PRIMARY_FIXED_INTERVAL_EDGES_M2 = (0.0, 200.0, 400.0, 600.0, 800.0)
SHORTEST_PATH_DISTANCE_BIN_WIDTH_M = 20.0
UNIQUE_GALLERY_AREA_SUM_BIN_WIDTH_M2 = 500.0
WILSON_Z_95 = 1.959963984540054


def resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate.resolve() if candidate.is_absolute() else (PROJECT_ROOT / candidate).resolve()


def load_json(path: str | Path) -> dict:
    resolved = Path(path)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {resolved}")
    return payload


def load_jsonl(path: str | Path) -> list[dict]:
    records: list[dict] = []
    resolved = Path(path)
    for line_number, raw_line in enumerate(
        resolved.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw_line.strip():
            continue
        payload = json.loads(raw_line)
        if not isinstance(payload, dict):
            raise ValueError(f"Expected a JSON object at {resolved}:{line_number}.")
        records.append(payload)
    return records


def index_unique(records: Iterable[dict], *, label: str) -> dict[str, dict]:
    indexed: dict[str, dict] = {}
    for record in records:
        test_id = record.get("test_id")
        if not isinstance(test_id, str) or not test_id:
            raise ValueError(f"{label} contains a record without test_id.")
        if test_id in indexed:
            raise ValueError(f"{label} contains duplicate test_id {test_id}.")
        indexed[test_id] = record
    return indexed


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_paired_inputs(
    dataset: dict[str, dict],
    without_results: dict[str, dict],
    with_results: dict[str, dict],
) -> None:
    dataset_ids = set(dataset)
    for label, results in (
        ("without-trajectory results", without_results),
        ("with-trajectory results", with_results),
    ):
        result_ids = set(results)
        if result_ids != dataset_ids:
            raise ValueError(
                f"{label} test IDs differ from the dataset: "
                f"missing={sorted(dataset_ids - result_ids)}, "
                f"extra={sorted(result_ids - dataset_ids)}"
            )
        for test_id, result in results.items():
            if result.get("path_id") != dataset[test_id].get("path_id"):
                raise ValueError(f"{label} path_id differs for {test_id}.")
            if not isinstance(result.get("success"), bool):
                raise ValueError(f"{label} has no Boolean success for {test_id}.")


def load_area_catalog(payload: dict) -> tuple[dict[str, dict], set[str]]:
    if payload.get("schema") != "st_nav.estimated_gallery_areas":
        raise ValueError("Room-area data has an unsupported schema.")
    if payload.get("unit") != "m2":
        raise ValueError("Room-area data must use square metres.")
    raw_rooms = payload.get("rooms")
    if not isinstance(raw_rooms, list):
        raise ValueError("Room-area data has no rooms list.")
    rooms: dict[str, dict] = {}
    for record in raw_rooms:
        if not isinstance(record, dict):
            raise ValueError("Room-area data contains a non-object room record.")
        room_id = record.get("room_id")
        area = record.get("effective_area_m2")
        if not isinstance(room_id, str) or not room_id:
            raise ValueError("Room-area data contains a room without room_id.")
        if room_id in rooms:
            raise ValueError(f"Room-area data contains duplicate room_id {room_id}.")
        if not isinstance(area, (int, float)) or not math.isfinite(area) or area <= 0:
            raise ValueError(f"Room-area data has invalid effective_area_m2 for {room_id}.")
        rooms[room_id] = record
    raw_excluded = payload.get("excluded_route_room_ids", [])
    if not isinstance(raw_excluded, list) or not all(
        isinstance(room_id, str) and room_id for room_id in raw_excluded
    ):
        raise ValueError("excluded_route_room_ids must be a list of room IDs.")
    excluded = set(raw_excluded)
    overlap = sorted(excluded & set(rooms))
    if overlap:
        raise ValueError(f"Area-bearing rooms cannot also be excluded: {overlap}")
    return rooms, excluded


def load_pano_room_grounding(payload: dict) -> dict[str, str | None]:
    raw_mappings = payload.get("mappings")
    if not isinstance(raw_mappings, dict):
        raise ValueError("Pano-room grounding must contain an object-valued mappings field.")
    null_sentinels = {"", "null", "none", "unknown", "unlabeled"}
    mappings: dict[str, str | None] = {}
    for pano_id, raw_room_id in raw_mappings.items():
        if not isinstance(pano_id, str) or not pano_id:
            raise ValueError("Pano-room grounding contains an invalid pano ID.")
        if raw_room_id is None:
            mappings[pano_id] = None
        elif not isinstance(raw_room_id, str):
            raise ValueError(f"Pano-room grounding has a non-string room for {pano_id}.")
        elif raw_room_id.strip().lower() in null_sentinels:
            mappings[pano_id] = None
        else:
            mappings[pano_id] = raw_room_id
    return mappings


def _unique_in_order(values: Iterable[str]) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value not in seen:
            unique.append(value)
            seen.add(value)
    return unique


def route_gallery_area(
    shortest_path_rooms: object,
    room_areas: dict[str, dict],
    excluded_room_ids: set[str],
) -> dict:
    if not isinstance(shortest_path_rooms, list) or not all(
        isinstance(room_id, str) and room_id for room_id in shortest_path_rooms
    ):
        raise ValueError("shortest_path_rooms must be a non-empty list of room IDs.")
    route_rooms = _unique_in_order(shortest_path_rooms)
    unknown = sorted(
        room_id
        for room_id in route_rooms
        if room_id not in room_areas and room_id not in excluded_room_ids
    )
    if unknown:
        raise ValueError(f"Route contains rooms without an area or exclusion: {unknown}")
    gallery_room_ids = [room_id for room_id in route_rooms if room_id in room_areas]
    excluded_route_room_ids = [
        room_id for room_id in route_rooms if room_id in excluded_room_ids
    ]
    if not gallery_room_ids:
        raise ValueError("Route contains no area-bearing gallery.")
    values = [float(room_areas[room_id]["effective_area_m2"]) for room_id in gallery_room_ids]
    return {
        "gallery_room_ids": gallery_room_ids,
        "excluded_route_room_ids": excluded_route_room_ids,
        "gallery_count_for_mean": len(gallery_room_ids),
        "estimated_mean_gallery_area_m2": statistics.fmean(values),
        "estimated_total_unique_gallery_area_m2": sum(values),
        "area_sources": sorted(
            {str(room_areas[room_id]["area_source"]) for room_id in gallery_room_ids}
        ),
    }


def actual_ground_truth_room_sequence(
    pano_path: object,
    pano_room_grounding: dict[str, str | None],
) -> tuple[list[str], int]:
    if not isinstance(pano_path, list) or not pano_path or not all(
        isinstance(pano_id, str) and pano_id for pano_id in pano_path
    ):
        raise ValueError("Actual pano_path must be a non-empty list of pano IDs.")
    unknown_pano_ids = sorted({pano_id for pano_id in pano_path if pano_id not in pano_room_grounding})
    if unknown_pano_ids:
        raise ValueError(f"Actual pano_path contains unknown pano IDs: {unknown_pano_ids}")
    sequence: list[str] = []
    unlabeled_pano_count = 0
    for pano_id in pano_path:
        room_id = pano_room_grounding[pano_id]
        if room_id is None:
            unlabeled_pano_count += 1
            continue
        if not sequence or sequence[-1] != room_id:
            sequence.append(room_id)
    if not sequence:
        raise ValueError("Actual pano_path contains no ground-truth room labels.")
    return sequence, unlabeled_pano_count


def actual_path_gallery_area(
    result: dict,
    pano_room_grounding: dict[str, str | None],
    room_areas: dict[str, dict],
    excluded_room_ids: set[str],
) -> dict:
    pano_path = result.get("pano_path")
    room_sequence, unlabeled_pano_count = actual_ground_truth_room_sequence(
        pano_path, pano_room_grounding
    )
    area = route_gallery_area(room_sequence, room_areas, excluded_room_ids)
    return {
        "pano_count": len(pano_path),
        "unlabeled_pano_count": unlabeled_pano_count,
        "ground_truth_room_sequence": room_sequence,
        **area,
    }


def build_route_rows(
    dataset_records: Sequence[dict],
    without_results: dict[str, dict],
    with_results: dict[str, dict],
    room_areas: dict[str, dict],
    excluded_room_ids: set[str],
    pano_room_grounding: dict[str, str | None] | None = None,
) -> list[dict]:
    rows: list[dict] = []
    for test in dataset_records:
        test_id = str(test["test_id"])
        reference_area = route_gallery_area(
            test.get("shortest_path_rooms"), room_areas, excluded_room_ids
        )
        without_actual_area = (
            actual_path_gallery_area(
                without_results[test_id],
                pano_room_grounding,
                room_areas,
                excluded_room_ids,
            )
            if pano_room_grounding is not None
            else None
        )
        with_actual_area = (
            actual_path_gallery_area(
                with_results[test_id],
                pano_room_grounding,
                room_areas,
                excluded_room_ids,
            )
            if pano_room_grounding is not None
            else None
        )
        rows.append(
            {
                "test_id": test_id,
                "path_id": test.get("path_id"),
                "shortest_path_rooms": list(test["shortest_path_rooms"]),
                **reference_area,
                "reference_gallery_room_ids": reference_area["gallery_room_ids"],
                "reference_excluded_route_room_ids": reference_area[
                    "excluded_route_room_ids"
                ],
                "reference_gallery_count_for_mean": reference_area[
                    "gallery_count_for_mean"
                ],
                "reference_estimated_mean_gallery_area_m2": reference_area[
                    "estimated_mean_gallery_area_m2"
                ],
                "reference_estimated_total_unique_gallery_area_m2": reference_area[
                    "estimated_total_unique_gallery_area_m2"
                ],
                "visited_room_count": test.get("visited_room_count"),
                "shortest_path_distance_m": test.get("path_distance_m"),
                "detour_ratio": test.get("detour_ratio"),
                "without_trajectory_success": without_results[test_id]["success"],
                "with_trajectory_success": with_results[test_id]["success"],
                **(
                    {
                        "without_actual_pano_count": without_actual_area["pano_count"],
                        "without_actual_unlabeled_pano_count": without_actual_area[
                            "unlabeled_pano_count"
                        ],
                        "without_actual_ground_truth_room_sequence": without_actual_area[
                            "ground_truth_room_sequence"
                        ],
                        "without_actual_gallery_room_ids": without_actual_area[
                            "gallery_room_ids"
                        ],
                        "without_actual_excluded_route_room_ids": without_actual_area[
                            "excluded_route_room_ids"
                        ],
                        "without_actual_gallery_count_for_mean": without_actual_area[
                            "gallery_count_for_mean"
                        ],
                        "without_actual_estimated_mean_gallery_area_m2": without_actual_area[
                            "estimated_mean_gallery_area_m2"
                        ],
                        "without_actual_estimated_total_unique_gallery_area_m2": without_actual_area[
                            "estimated_total_unique_gallery_area_m2"
                        ],
                        "with_actual_pano_count": with_actual_area["pano_count"],
                        "with_actual_unlabeled_pano_count": with_actual_area[
                            "unlabeled_pano_count"
                        ],
                        "with_actual_ground_truth_room_sequence": with_actual_area[
                            "ground_truth_room_sequence"
                        ],
                        "with_actual_gallery_room_ids": with_actual_area[
                            "gallery_room_ids"
                        ],
                        "with_actual_excluded_route_room_ids": with_actual_area[
                            "excluded_route_room_ids"
                        ],
                        "with_actual_gallery_count_for_mean": with_actual_area[
                            "gallery_count_for_mean"
                        ],
                        "with_actual_estimated_mean_gallery_area_m2": with_actual_area[
                            "estimated_mean_gallery_area_m2"
                        ],
                        "with_actual_estimated_total_unique_gallery_area_m2": with_actual_area[
                            "estimated_total_unique_gallery_area_m2"
                        ],
                    }
                    if without_actual_area is not None and with_actual_area is not None
                    else {}
                ),
            }
        )
    return rows


def quantile_boundaries(values: Sequence[float], group_count: int) -> list[float]:
    if group_count < 2:
        raise ValueError("Quantile analysis requires at least two groups.")
    if len(values) < group_count:
        raise ValueError(
            f"At least {group_count} routes are required for {group_count}-group analysis."
        )
    return list(statistics.quantiles(values, n=group_count, method="inclusive"))


def quartile_boundaries(values: Sequence[float]) -> list[float]:
    return quantile_boundaries(values, 4)


def assign_quantile_groups(
    rows: list[dict],
    group_count: int,
    *,
    field: str,
    label_prefix: str = "G",
) -> tuple[list[float], tuple[str, ...]]:
    boundaries = quantile_boundaries(
        [float(row["estimated_mean_gallery_area_m2"]) for row in rows],
        group_count,
    )
    labels = tuple(f"{label_prefix}{index}" for index in range(1, group_count + 1))
    for row in rows:
        index = bisect.bisect_left(
            boundaries, float(row["estimated_mean_gallery_area_m2"])
        )
        row[field] = labels[index]
    counts = {label: 0 for label in labels}
    for row in rows:
        counts[str(row[field])] += 1
    empty = [label for label, count in counts.items() if count == 0]
    if empty:
        raise ValueError(f"Mean-area quantile groups are empty: {empty}")
    return boundaries, labels


def assign_quartiles(rows: list[dict]) -> list[float]:
    boundaries, _ = assign_quantile_groups(
        rows,
        4,
        field="area_quartile",
        label_prefix="Q",
    )
    return boundaries


def wilson_interval(
    success_count: int, case_count: int, *, z: float = WILSON_Z_95
) -> tuple[float, float]:
    if case_count <= 0:
        raise ValueError("Wilson interval requires a positive case count.")
    if success_count < 0 or success_count > case_count:
        raise ValueError("success_count must be between zero and case_count.")
    proportion = success_count / case_count
    denominator = 1.0 + z * z / case_count
    centre = (proportion + z * z / (2.0 * case_count)) / denominator
    half_width = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / case_count
            + z * z / (4.0 * case_count * case_count)
        )
        / denominator
    )
    return max(0.0, centre - half_width), min(1.0, centre + half_width)


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        inverse = math.exp(-value)
        return 1.0 / (1.0 + inverse)
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def fit_logistic_regression(rows: Sequence[dict], success_key: str) -> dict:
    if len(rows) < 3:
        raise ValueError("Logistic regression requires at least three routes.")
    x_values = [float(row["estimated_mean_gallery_area_m2"]) for row in rows]
    outcomes = [1.0 if bool(row[success_key]) else 0.0 for row in rows]
    if len(set(outcomes)) < 2:
        raise ValueError("Logistic regression requires both successes and failures.")
    x_mean = statistics.fmean(x_values)
    x_scale = statistics.stdev(x_values)
    if not math.isfinite(x_scale) or x_scale <= 0.0:
        raise ValueError("Logistic regression requires varying route areas.")
    standardized = [(value - x_mean) / x_scale for value in x_values]
    beta0 = math.log(statistics.fmean(outcomes) / (1.0 - statistics.fmean(outcomes)))
    beta1 = 0.0
    for _ in range(100):
        information00 = 0.0
        information01 = 0.0
        information11 = 0.0
        score0 = 0.0
        score1 = 0.0
        for x_value, outcome in zip(standardized, outcomes):
            probability = _sigmoid(beta0 + beta1 * x_value)
            weight = max(probability * (1.0 - probability), 1e-12)
            information00 += weight
            information01 += weight * x_value
            information11 += weight * x_value * x_value
            residual = outcome - probability
            score0 += residual
            score1 += residual * x_value
        determinant = information00 * information11 - information01 * information01
        if determinant <= 1e-15:
            raise ValueError("Logistic regression information matrix is singular.")
        delta0 = (
            information11 * score0 - information01 * score1
        ) / determinant
        delta1 = (
            -information01 * score0 + information00 * score1
        ) / determinant
        beta0 += delta0
        beta1 += delta1
        if max(abs(delta0), abs(delta1)) < 1e-10:
            break
    else:
        raise ValueError("Logistic regression did not converge.")

    information00 = 0.0
    information01 = 0.0
    information11 = 0.0
    for x_value in standardized:
        probability = _sigmoid(beta0 + beta1 * x_value)
        weight = max(probability * (1.0 - probability), 1e-12)
        information00 += weight
        information01 += weight * x_value
        information11 += weight * x_value * x_value
    determinant = information00 * information11 - information01 * information01
    covariance00 = information11 / determinant
    covariance01 = -information01 / determinant
    covariance11 = information00 / determinant
    slope_per_m2 = beta1 / x_scale
    slope_standard_error_per_m2 = math.sqrt(covariance11) / x_scale
    return {
        "case_count": len(rows),
        "success_count": int(sum(outcomes)),
        "x_mean_m2": x_mean,
        "x_standard_deviation_m2": x_scale,
        "standardized_intercept": beta0,
        "standardized_slope": beta1,
        "slope_log_odds_per_m2": slope_per_m2,
        "slope_standard_error_per_m2": slope_standard_error_per_m2,
        "odds_ratio_per_100_m2": math.exp(100.0 * slope_per_m2),
        "odds_ratio_per_100_m2_95_ci_low": math.exp(
            100.0 * (slope_per_m2 - WILSON_Z_95 * slope_standard_error_per_m2)
        ),
        "odds_ratio_per_100_m2_95_ci_high": math.exp(
            100.0 * (slope_per_m2 + WILSON_Z_95 * slope_standard_error_per_m2)
        ),
        "covariance": [
            [covariance00, covariance01],
            [covariance01, covariance11],
        ],
    }


def logistic_prediction(model: dict, area_m2: float) -> tuple[float, float, float]:
    standardized = (
        float(area_m2) - float(model["x_mean_m2"])
    ) / float(model["x_standard_deviation_m2"])
    eta = float(model["standardized_intercept"]) + float(
        model["standardized_slope"]
    ) * standardized
    covariance = model["covariance"]
    variance = (
        float(covariance[0][0])
        + 2.0 * standardized * float(covariance[0][1])
        + standardized * standardized * float(covariance[1][1])
    )
    standard_error = math.sqrt(max(variance, 0.0))
    probability = _sigmoid(eta)
    lower = _sigmoid(eta - WILSON_Z_95 * standard_error)
    upper = _sigmoid(eta + WILSON_Z_95 * standard_error)
    return probability, lower, upper


def _success_summary(rows: Sequence[dict], key: str) -> dict:
    case_count = len(rows)
    success_count = sum(bool(row[key]) for row in rows)
    lower, upper = wilson_interval(success_count, case_count)
    return {
        "case_count": case_count,
        "success_count": success_count,
        "failure_count": case_count - success_count,
        "success_rate": success_count / case_count,
        "wilson_95_ci_low": lower,
        "wilson_95_ci_high": upper,
    }


def build_quantile_group_rows(
    rows: Sequence[dict],
    boundaries: Sequence[float],
    labels: Sequence[str],
    *,
    field: str,
) -> list[dict]:
    if len(boundaries) != len(labels) - 1:
        raise ValueError("Quantile boundaries and labels have incompatible lengths.")
    grouped_rows: list[dict] = []
    for index, label in enumerate(labels):
        group = [row for row in rows if row[field] == label]
        if not group:
            raise ValueError(f"Mean-area quantile group is empty: {label}")
        areas = [float(row["estimated_mean_gallery_area_m2"]) for row in group]
        without = _success_summary(group, "without_trajectory_success")
        with_trajectory = _success_summary(group, "with_trajectory_success")
        grouped_rows.append(
            {
                "group_label": label,
                "group_index": index + 1,
                "case_count": len(group),
                "quantile_interval": {
                    "lower_m2": None if index == 0 else boundaries[index - 1],
                    "lower_inclusive": index == 0,
                    "upper_m2": None if index == len(labels) - 1 else boundaries[index],
                    "upper_inclusive": index < len(labels) - 1,
                },
                "observed_min_mean_gallery_area_m2": min(areas),
                "observed_max_mean_gallery_area_m2": max(areas),
                "mean_of_route_mean_gallery_area_m2": statistics.fmean(areas),
                "without_trajectory": without,
                "with_trajectory": with_trajectory,
                "delta_with_minus_without_success_rate": (
                    with_trajectory["success_rate"] - without["success_rate"]
                ),
                "delta_with_minus_without_percentage_points": 100.0
                * (with_trajectory["success_rate"] - without["success_rate"]),
            }
        )
    return grouped_rows


def build_summary(
    rows: Sequence[dict],
    boundaries: Sequence[float],
    *,
    source_paths: dict[str, Path] | None = None,
) -> dict:
    quartiles = build_quantile_group_rows(
        rows,
        boundaries,
        QUARTILE_LABELS,
        field="area_quartile",
    )
    for group in quartiles:
        group["quartile"] = group["group_label"]
    all_areas = [float(row["estimated_mean_gallery_area_m2"]) for row in rows]
    without_overall = _success_summary(rows, "without_trajectory_success")
    with_overall = _success_summary(rows, "with_trajectory_success")
    payload = {
        "schema": "llm_nav_mean_gallery_area_success_rate",
        "schema_version": 1,
        "case_count": len(rows),
        "definitions": {
            "x": (
                "Equal-weight mean effective_area_m2 over unique gallery rooms in "
                "the reference shortest_path_rooms sequence."
            ),
            "y": "Success count divided by case count within a shared X quartile.",
            "area_caveat": (
                "Gallery areas are estimates, not architectural floor-plan measurements."
            ),
            "excluded_route_nodes": (
                "Circulation nodes are excluded from the gallery-area mean."
            ),
            "association_caveat": (
                "The plot shows association, not a causal effect of gallery area."
            ),
            "confidence_interval": "Two-sided 95% Wilson score interval.",
        },
        "mean_gallery_area_distribution_m2": {
            "min": min(all_areas),
            "q25": boundaries[0],
            "median": boundaries[1],
            "q75": boundaries[2],
            "max": max(all_areas),
        },
        "overall": {
            "without_trajectory": without_overall,
            "with_trajectory": with_overall,
            "delta_with_minus_without_success_rate": (
                with_overall["success_rate"] - without_overall["success_rate"]
            ),
            "delta_with_minus_without_percentage_points": 100.0
            * (with_overall["success_rate"] - without_overall["success_rate"]),
        },
        "route_level_logistic_regression": {
            "without_trajectory": fit_logistic_regression(
                rows, "without_trajectory_success"
            ),
            "with_trajectory": fit_logistic_regression(
                rows, "with_trajectory_success"
            ),
            "interpretation": (
                "Odds ratios describe association per +100 m2 in estimated mean "
                "gallery area; they are not causal effects."
            ),
        },
        "quartiles": quartiles,
    }
    if source_paths:
        payload["provenance"] = {
            label: {"path": str(path), "sha256": sha256_file(path)}
            for label, path in source_paths.items()
        }
    return payload


def build_binned_summary(
    rows: Sequence[dict],
    boundaries: Sequence[float],
    labels: Sequence[str],
    *,
    field: str,
    source_paths: dict[str, Path] | None = None,
) -> dict:
    groups = build_quantile_group_rows(
        rows,
        boundaries,
        labels,
        field=field,
    )
    without_overall = _success_summary(rows, "without_trajectory_success")
    with_overall = _success_summary(rows, "with_trajectory_success")
    payload = {
        "schema": "llm_nav_mean_gallery_area_binned_success_rate",
        "schema_version": 1,
        "case_count": len(rows),
        "group_count": len(labels),
        "grouping": {
            "method": "equal-frequency quantile boundaries with identical X values kept together",
            "boundaries_m2": list(boundaries),
            "labels": list(labels),
        },
        "definitions": {
            "x": "Mean route-level estimated gallery area within each quantile group.",
            "y": "Success count divided by case count within the shared quantile group.",
            "confidence_interval": "Two-sided 95% Wilson score interval.",
            "area_caveat": (
                "Gallery areas are estimates, not architectural floor-plan measurements."
            ),
            "association_caveat": (
                "The plot shows association, not a causal effect of gallery area."
            ),
        },
        "overall": {
            "without_trajectory": without_overall,
            "with_trajectory": with_overall,
            "delta_with_minus_without_success_rate": (
                with_overall["success_rate"] - without_overall["success_rate"]
            ),
            "delta_with_minus_without_percentage_points": 100.0
            * (with_overall["success_rate"] - without_overall["success_rate"]),
        },
        "groups": groups,
    }
    if source_paths:
        payload["provenance"] = {
            label: {"path": str(path), "sha256": sha256_file(path)}
            for label, path in source_paths.items()
        }
    return payload


def assign_fixed_width_groups(
    rows: list[dict],
    width_m2: float,
    *,
    field: str,
    maximum_m2: float | None = None,
    open_ended_from_m2: float | None = None,
) -> list[dict]:
    if not math.isfinite(width_m2) or width_m2 <= 0.0:
        raise ValueError("Fixed group width must be positive and finite.")
    if not rows:
        raise ValueError("Fixed-width grouping requires at least one route.")
    if maximum_m2 is not None and open_ended_from_m2 is not None:
        raise ValueError(
            "maximum_m2 and open_ended_from_m2 cannot be used together."
        )
    observed_maximum = max(
        float(row["estimated_mean_gallery_area_m2"]) for row in rows
    )
    if open_ended_from_m2 is not None:
        if not math.isfinite(open_ended_from_m2) or open_ended_from_m2 <= 0.0:
            raise ValueError("Open-ended threshold must be positive and finite.")
        closed_group_count = int(round(open_ended_from_m2 / width_m2))
        if closed_group_count < 1 or not math.isclose(
            closed_group_count * width_m2, open_ended_from_m2
        ):
            raise ValueError(
                "Open-ended threshold must be a positive multiple of group width."
            )
        group_count = closed_group_count + 1
    elif maximum_m2 is None:
        group_count = max(1, int(math.floor(observed_maximum / width_m2)) + 1)
    else:
        if not math.isfinite(maximum_m2) or maximum_m2 <= 0.0:
            raise ValueError("Fixed group maximum must be positive and finite.")
        if observed_maximum > maximum_m2:
            raise ValueError("A route exceeds the requested fixed group maximum.")
        group_count = max(1, int(math.ceil(maximum_m2 / width_m2)))
    specifications: list[dict] = [
        {
            "group_label": f"{int(index * width_m2)}-{int((index + 1) * width_m2)}",
            "group_index": index + 1,
            "lower_m2": index * width_m2,
            "upper_m2": (index + 1) * width_m2,
            "midpoint_m2": (index + 0.5) * width_m2,
        }
        for index in range(
            group_count - 1 if open_ended_from_m2 is not None else group_count
        )
    ]
    if open_ended_from_m2 is not None:
        specifications.append(
            {
                "group_label": f">{int(open_ended_from_m2)}",
                "group_index": group_count,
                "lower_m2": open_ended_from_m2,
                "upper_m2": None,
                "midpoint_m2": None,
            }
        )
    for row in rows:
        area = float(row["estimated_mean_gallery_area_m2"])
        if open_ended_from_m2 is not None and area > open_ended_from_m2:
            index = group_count - 1
        elif open_ended_from_m2 is not None and math.isclose(
            area, open_ended_from_m2
        ):
            index = group_count - 2
        else:
            index = int(math.floor(area / width_m2))
        if maximum_m2 is not None and math.isclose(area, maximum_m2):
            index = group_count - 1
        if index < 0 or index >= len(specifications):
            raise ValueError(f"Route area is outside fixed-width groups: {area}")
        row[field] = specifications[index]["group_label"]
    return specifications


def primary_fixed_interval_specifications() -> list[dict]:
    edges = PRIMARY_FIXED_INTERVAL_EDGES_M2
    return [
        {
            "group_label": f"{int(edges[index])}-{int(edges[index + 1])}",
            "group_index": index + 1,
            "lower_m2": edges[index],
            "upper_m2": edges[index + 1],
            "midpoint_m2": (edges[index] + edges[index + 1]) / 2.0,
            "includes_above_upper_bound": index == len(edges) - 2,
        }
        for index in range(len(edges) - 1)
    ]


def assign_primary_fixed_interval_groups(
    rows: Sequence[dict],
    *,
    area_key: str,
    field: str,
) -> list[dict]:
    specifications = primary_fixed_interval_specifications()
    edges = PRIMARY_FIXED_INTERVAL_EDGES_M2
    for row in rows:
        raw_area = row.get(area_key)
        if not isinstance(raw_area, (int, float)) or not math.isfinite(raw_area):
            raise ValueError(f"Route has invalid {area_key}: {raw_area!r}")
        area = float(raw_area)
        if area < edges[0]:
            raise ValueError(f"Route area is below the fixed interval range: {area}")
        index = bisect.bisect_right(edges, area) - 1
        index = min(index, len(specifications) - 1)
        row[field] = specifications[index]["group_label"]
    return specifications


def _fixed_interval_method_summary(
    rows: Sequence[dict],
    *,
    group_field: str,
    group_label: str,
    area_key: str,
    success_key: str,
) -> dict | None:
    group = [row for row in rows if row[group_field] == group_label]
    if not group:
        return None
    summary = _success_summary(group, success_key)
    areas = [float(row[area_key]) for row in group]
    return {
        **summary,
        "observed_min_mean_gallery_area_m2": min(areas),
        "observed_max_mean_gallery_area_m2": max(areas),
        "observed_mean_gallery_area_m2": statistics.fmean(areas),
        "above_800_m2_case_count": sum(
            area > PRIMARY_FIXED_INTERVAL_EDGES_M2[-1] for area in areas
        ),
    }


def build_primary_fixed_interval_summary(
    rows: Sequence[dict],
    *,
    path_kind: str,
    source_paths: dict[str, Path] | None = None,
) -> dict:
    if path_kind == "reference_shortest_path":
        without_area_key = "reference_estimated_mean_gallery_area_m2"
        with_area_key = without_area_key
        without_group_field = "reference_fixed_200m2_group"
        with_group_field = without_group_field
        shared_membership = True
    elif path_kind == "actual_agent_path":
        without_area_key = "without_actual_estimated_mean_gallery_area_m2"
        with_area_key = "with_actual_estimated_mean_gallery_area_m2"
        without_group_field = "without_actual_fixed_200m2_group"
        with_group_field = "with_actual_fixed_200m2_group"
        shared_membership = False
    else:
        raise ValueError(f"Unsupported path_kind: {path_kind}")

    specifications = assign_primary_fixed_interval_groups(
        rows, area_key=without_area_key, field=without_group_field
    )
    if with_group_field != without_group_field:
        assign_primary_fixed_interval_groups(
            rows, area_key=with_area_key, field=with_group_field
        )

    groups: list[dict] = []
    for specification in specifications:
        label = str(specification["group_label"])
        without = _fixed_interval_method_summary(
            rows,
            group_field=without_group_field,
            group_label=label,
            area_key=without_area_key,
            success_key="without_trajectory_success",
        )
        with_trajectory = _fixed_interval_method_summary(
            rows,
            group_field=with_group_field,
            group_label=label,
            area_key=with_area_key,
            success_key="with_trajectory_success",
        )
        without_ids = {
            str(row["test_id"])
            for row in rows
            if row[without_group_field] == label
        }
        with_ids = {
            str(row["test_id"])
            for row in rows
            if row[with_group_field] == label
        }
        delta_rate = (
            None
            if without is None or with_trajectory is None
            else with_trajectory["success_rate"] - without["success_rate"]
        )
        groups.append(
            {
                **specification,
                "plot_x_m2": specification["midpoint_m2"],
                "mean_of_route_mean_gallery_area_m2": specification["midpoint_m2"],
                "case_count": len(without_ids) if shared_membership else None,
                "shared_test_count": len(without_ids & with_ids),
                "without_trajectory": without,
                "with_trajectory": with_trajectory,
                "delta_with_minus_without_success_rate": delta_rate,
                "delta_with_minus_without_percentage_points": (
                    None if delta_rate is None else 100.0 * delta_rate
                ),
            }
        )

    without_overall = _success_summary(rows, "without_trajectory_success")
    with_overall = _success_summary(rows, "with_trajectory_success")
    payload = {
        "schema": "llm_nav_reference_actual_mean_gallery_area_fixed_interval_success_rate",
        "schema_version": 1,
        "path_kind": path_kind,
        "case_count_per_method": len(rows),
        "group_count": len(specifications),
        "grouping": {
            "method": "fixed 200 m2 intervals",
            "intervals_m2": [
                [specification["lower_m2"], specification["upper_m2"]]
                for specification in specifications
            ],
            "x_position": "interval midpoint",
            "above_800_assignment": (
                "Retained in the 600-800 display bin; counts are reported explicitly."
            ),
            "shared_route_membership_between_methods": shared_membership,
        },
        "definitions": {
            "route_area": (
                "Equal-weight mean effective_area_m2 over unique gallery rooms in "
                "the reference shortest_path_rooms sequence."
                if path_kind == "reference_shortest_path"
                else "Equal-weight mean effective_area_m2 over unique ground-truth "
                "gallery rooms traversed by each agent pano_path."
            ),
            "x": "Midpoint of each fixed estimated mean gallery-area interval.",
            "y": "Success count divided by the method-specific case count in the interval.",
            "actual_path_room_source": (
                None
                if path_kind == "reference_shortest_path"
                else "Ground-truth pano-to-room mapping, not the agent's estimated room state."
            ),
            "revisit_weighting": "Each distinct gallery is counted once; movement steps and dwell time are not used as weights.",
            "excluded_route_nodes": "Circulation nodes are excluded from the gallery-area mean.",
            "confidence_interval": (
                "Two-sided 95% Wilson score interval, retained in CSV/JSON tables "
                "and drawn on the current plots."
            ),
            "comparison_caveat": (
                "Both methods share the same routes in every bin."
                if shared_membership
                else "Without and With are binned by their own actual paths, so routes in the same displayed bin can differ between methods."
            ),
            "area_caveat": "Gallery areas are estimates, not architectural floor-plan measurements.",
            "association_caveat": "The plot shows association, not a causal effect of gallery area.",
        },
        "overall": {
            "without_trajectory": without_overall,
            "with_trajectory": with_overall,
            "delta_with_minus_without_success_rate": (
                with_overall["success_rate"] - without_overall["success_rate"]
            ),
            "delta_with_minus_without_percentage_points": 100.0
            * (with_overall["success_rate"] - without_overall["success_rate"]),
        },
        "groups": groups,
    }
    if source_paths:
        payload["provenance"] = {
            label: {"path": str(path), "sha256": sha256_file(path)}
            for label, path in source_paths.items()
        }
    return payload


def build_filtered_actual_fixed_interval_summary(
    rows: Sequence[dict],
    *,
    maximum_m2: float = PRIMARY_FIXED_INTERVAL_EDGES_M2[-1],
    source_paths: dict[str, Path] | None = None,
) -> dict:
    if maximum_m2 != PRIMARY_FIXED_INTERVAL_EDGES_M2[-1]:
        raise ValueError(
            "Filtered actual-path summary currently requires the fixed 800 m2 maximum."
        )
    without_area_key = "without_actual_estimated_mean_gallery_area_m2"
    with_area_key = "with_actual_estimated_mean_gallery_area_m2"
    without_group_field = "without_actual_fixed_200m2_group"
    with_group_field = "with_actual_fixed_200m2_group"
    specifications = assign_primary_fixed_interval_groups(
        rows,
        area_key=without_area_key,
        field=without_group_field,
    )
    assign_primary_fixed_interval_groups(
        rows,
        area_key=with_area_key,
        field=with_group_field,
    )
    specifications = [
        {**specification, "includes_above_upper_bound": False}
        for specification in specifications
    ]
    without_retained = [
        row for row in rows if float(row[without_area_key]) <= maximum_m2
    ]
    with_retained = [
        row for row in rows if float(row[with_area_key]) <= maximum_m2
    ]
    without_excluded = [
        row for row in rows if float(row[without_area_key]) > maximum_m2
    ]
    with_excluded = [
        row for row in rows if float(row[with_area_key]) > maximum_m2
    ]

    groups: list[dict] = []
    for specification in specifications:
        label = str(specification["group_label"])
        without = _fixed_interval_method_summary(
            without_retained,
            group_field=without_group_field,
            group_label=label,
            area_key=without_area_key,
            success_key="without_trajectory_success",
        )
        with_trajectory = _fixed_interval_method_summary(
            with_retained,
            group_field=with_group_field,
            group_label=label,
            area_key=with_area_key,
            success_key="with_trajectory_success",
        )
        without_ids = {
            str(row["test_id"])
            for row in without_retained
            if row[without_group_field] == label
        }
        with_ids = {
            str(row["test_id"])
            for row in with_retained
            if row[with_group_field] == label
        }
        delta_rate = (
            None
            if without is None or with_trajectory is None
            else with_trajectory["success_rate"] - without["success_rate"]
        )
        groups.append(
            {
                **specification,
                "plot_x_m2": specification["midpoint_m2"],
                "mean_of_route_mean_gallery_area_m2": specification[
                    "midpoint_m2"
                ],
                "case_count": None,
                "shared_test_count": len(without_ids & with_ids),
                "without_trajectory": without,
                "with_trajectory": with_trajectory,
                "delta_with_minus_without_success_rate": delta_rate,
                "delta_with_minus_without_percentage_points": (
                    None if delta_rate is None else 100.0 * delta_rate
                ),
            }
        )

    without_overall = _success_summary(
        without_retained, "without_trajectory_success"
    )
    with_overall = _success_summary(with_retained, "with_trajectory_success")
    payload = {
        "schema": "llm_nav_actual_mean_gallery_area_max800_success_rate",
        "schema_version": 1,
        "path_kind": "actual_agent_path",
        "input_case_count_per_method": len(rows),
        "retained_case_count": {
            "without_trajectory": len(without_retained),
            "with_trajectory": len(with_retained),
        },
        "excluded_case_count": {
            "without_trajectory": len(without_excluded),
            "with_trajectory": len(with_excluded),
        },
        "excluded_test_ids": {
            "without_trajectory": [str(row["test_id"]) for row in without_excluded],
            "with_trajectory": [str(row["test_id"]) for row in with_excluded],
        },
        "group_count": len(specifications),
        "grouping": {
            "method": "fixed 200 m2 intervals after method-specific filtering",
            "intervals_m2": [
                [specification["lower_m2"], specification["upper_m2"]]
                for specification in specifications
            ],
            "x_position": "interval midpoint",
            "maximum_included_mean_gallery_area_m2": maximum_m2,
            "above_800_assignment": "Excluded before binning.",
            "shared_route_membership_between_methods": False,
        },
        "definitions": {
            "route_area": (
                "Equal-weight mean effective_area_m2 over unique ground-truth "
                "gallery rooms traversed by each agent pano_path."
            ),
            "filter": (
                "Each method excludes cases whose own actual-path estimated mean "
                "gallery area is greater than 800 m2."
            ),
            "x": "Midpoint of each fixed estimated mean gallery-area interval.",
            "y": "Success count divided by the retained method-specific case count in the interval.",
            "confidence_interval": "Two-sided 95% Wilson score interval.",
            "comparison_caveat": (
                "Without and With retain different route sets because filtering is "
                "based on each method's own actual path."
            ),
            "area_caveat": (
                "Gallery areas are VLM-derived estimates, not architectural "
                "floor-plan measurements."
            ),
        },
        "overall_after_filtering": {
            "without_trajectory": without_overall,
            "with_trajectory": with_overall,
            "delta_with_minus_without_success_rate": (
                with_overall["success_rate"] - without_overall["success_rate"]
            ),
            "delta_with_minus_without_percentage_points": 100.0
            * (with_overall["success_rate"] - without_overall["success_rate"]),
        },
        "groups": groups,
    }
    if source_paths:
        payload["provenance"] = {
            label: {"path": str(path), "sha256": sha256_file(path)}
            for label, path in source_paths.items()
        }
    return payload


def primary_fixed_interval_csv_rows(summary: dict) -> list[dict]:
    rows: list[dict] = []
    shared_membership = bool(
        summary["grouping"]["shared_route_membership_between_methods"]
    )
    for group in summary["groups"]:
        without = group["without_trajectory"]
        with_trajectory = group["with_trajectory"]
        rows.append(
            {
                "area_interval_m2": group["group_label"],
                "interval_midpoint_m2": group["midpoint_m2"],
                "shared_route_membership": shared_membership,
                "shared_test_count": group["shared_test_count"],
                "without_case_count": None if without is None else without["case_count"],
                "without_observed_min_mean_gallery_area_m2": (
                    None if without is None else without["observed_min_mean_gallery_area_m2"]
                ),
                "without_observed_max_mean_gallery_area_m2": (
                    None if without is None else without["observed_max_mean_gallery_area_m2"]
                ),
                "without_above_800_m2_case_count": (
                    None if without is None else without["above_800_m2_case_count"]
                ),
                "without_success_count": None if without is None else without["success_count"],
                "without_success_rate": None if without is None else without["success_rate"],
                "without_wilson_95_ci_low": (
                    None if without is None else without["wilson_95_ci_low"]
                ),
                "without_wilson_95_ci_high": (
                    None if without is None else without["wilson_95_ci_high"]
                ),
                "with_case_count": None if with_trajectory is None else with_trajectory["case_count"],
                "with_observed_min_mean_gallery_area_m2": (
                    None if with_trajectory is None else with_trajectory["observed_min_mean_gallery_area_m2"]
                ),
                "with_observed_max_mean_gallery_area_m2": (
                    None if with_trajectory is None else with_trajectory["observed_max_mean_gallery_area_m2"]
                ),
                "with_above_800_m2_case_count": (
                    None if with_trajectory is None else with_trajectory["above_800_m2_case_count"]
                ),
                "with_success_count": None if with_trajectory is None else with_trajectory["success_count"],
                "with_success_rate": None if with_trajectory is None else with_trajectory["success_rate"],
                "with_wilson_95_ci_low": (
                    None if with_trajectory is None else with_trajectory["wilson_95_ci_low"]
                ),
                "with_wilson_95_ci_high": (
                    None if with_trajectory is None else with_trajectory["wilson_95_ci_high"]
                ),
                "delta_with_minus_without_percentage_points": group[
                    "delta_with_minus_without_percentage_points"
                ],
            }
        )
    return rows


def build_fixed_width_summary(
    rows: Sequence[dict],
    specifications: Sequence[dict],
    *,
    field: str,
    width_m2: float,
    source_paths: dict[str, Path] | None = None,
) -> dict:
    groups: list[dict] = []
    for specification in specifications:
        label = str(specification["group_label"])
        group = [row for row in rows if row[field] == label]
        if group:
            without = _success_summary(group, "without_trajectory_success")
            with_trajectory = _success_summary(group, "with_trajectory_success")
            areas = [float(row["estimated_mean_gallery_area_m2"]) for row in group]
            delta_rate = with_trajectory["success_rate"] - without["success_rate"]
            observed_min = min(areas)
            observed_max = max(areas)
            plot_x_m2 = (
                statistics.fmean(areas)
                if specification["midpoint_m2"] is None
                else specification["midpoint_m2"]
            )
        else:
            without = None
            with_trajectory = None
            delta_rate = None
            observed_min = None
            observed_max = None
            plot_x_m2 = specification["midpoint_m2"]
        groups.append(
            {
                **specification,
                "case_count": len(group),
                "observed_min_mean_gallery_area_m2": observed_min,
                "observed_max_mean_gallery_area_m2": observed_max,
                "mean_of_route_mean_gallery_area_m2": plot_x_m2,
                "without_trajectory": without,
                "with_trajectory": with_trajectory,
                "delta_with_minus_without_success_rate": delta_rate,
                "delta_with_minus_without_percentage_points": (
                    None if delta_rate is None else 100.0 * delta_rate
                ),
            }
        )
    without_overall = _success_summary(rows, "without_trajectory_success")
    with_overall = _success_summary(rows, "with_trajectory_success")
    has_open_ended_group = any(
        specification["upper_m2"] is None for specification in specifications
    )
    payload = {
        "schema": "llm_nav_mean_gallery_area_fixed_width_success_rate",
        "schema_version": 1,
        "case_count": len(rows),
        "group_count": len(specifications),
        "grouping": {
            "method": (
                "fixed-width half-open intervals with an open-ended final interval"
                if has_open_ended_group
                else "fixed-width half-open intervals"
            ),
            "width_m2": width_m2,
            "x_position": (
                "closed-interval midpoint; observed group mean for open-ended final interval"
                if has_open_ended_group
                else "interval midpoint"
            ),
            "empty_bins": "retained in outputs but omitted from plotted marks",
        },
        "definitions": {
            "x": (
                "Midpoint of each closed estimated mean gallery-area interval; "
                "the observed group mean for the open-ended final interval."
                if has_open_ended_group
                else "Midpoint of each fixed-width estimated mean gallery-area interval."
            ),
            "y": "Success count divided by case count within the shared interval.",
            "confidence_interval": "Two-sided 95% Wilson score interval.",
            "area_caveat": (
                "Gallery areas are estimates, not architectural floor-plan measurements."
            ),
            "association_caveat": (
                "The plot shows association, not a causal effect of gallery area."
            ),
        },
        "overall": {
            "without_trajectory": without_overall,
            "with_trajectory": with_overall,
            "delta_with_minus_without_success_rate": (
                with_overall["success_rate"] - without_overall["success_rate"]
            ),
            "delta_with_minus_without_percentage_points": 100.0
            * (with_overall["success_rate"] - without_overall["success_rate"]),
        },
        "groups": groups,
    }
    if source_paths:
        payload["provenance"] = {
            label: {"path": str(path), "sha256": sha256_file(path)}
            for label, path in source_paths.items()
        }
    return payload


def grouped_summary_csv_rows(groups: Sequence[dict]) -> list[dict]:
    rows: list[dict] = []
    for group in groups:
        without = group["without_trajectory"]
        with_trajectory = group["with_trajectory"]
        rows.append(
            {
                "group": group["group_label"],
                "case_count": group["case_count"],
                "observed_min_mean_gallery_area_m2": group[
                    "observed_min_mean_gallery_area_m2"
                ],
                "observed_max_mean_gallery_area_m2": group[
                    "observed_max_mean_gallery_area_m2"
                ],
                "mean_of_route_mean_gallery_area_m2": group[
                    "mean_of_route_mean_gallery_area_m2"
                ],
                "without_success_count": without["success_count"],
                "without_success_rate": without["success_rate"],
                "without_wilson_95_ci_low": without["wilson_95_ci_low"],
                "without_wilson_95_ci_high": without["wilson_95_ci_high"],
                "with_success_count": with_trajectory["success_count"],
                "with_success_rate": with_trajectory["success_rate"],
                "with_wilson_95_ci_low": with_trajectory["wilson_95_ci_low"],
                "with_wilson_95_ci_high": with_trajectory["wilson_95_ci_high"],
                "delta_with_minus_without_percentage_points": group[
                    "delta_with_minus_without_percentage_points"
                ],
            }
        )
    return rows


def fixed_width_summary_csv_rows(groups: Sequence[dict]) -> list[dict]:
    rows: list[dict] = []
    for group in groups:
        without = group["without_trajectory"]
        with_trajectory = group["with_trajectory"]
        rows.append(
            {
                "area_interval_m2": group["group_label"],
                "interval_midpoint_m2": group["midpoint_m2"],
                "plot_x_m2": group["mean_of_route_mean_gallery_area_m2"],
                "case_count": group["case_count"],
                "observed_min_mean_gallery_area_m2": group[
                    "observed_min_mean_gallery_area_m2"
                ],
                "observed_max_mean_gallery_area_m2": group[
                    "observed_max_mean_gallery_area_m2"
                ],
                "without_success_count": (
                    None if without is None else without["success_count"]
                ),
                "without_success_rate": (
                    None if without is None else without["success_rate"]
                ),
                "without_wilson_95_ci_low": (
                    None if without is None else without["wilson_95_ci_low"]
                ),
                "without_wilson_95_ci_high": (
                    None if without is None else without["wilson_95_ci_high"]
                ),
                "with_success_count": (
                    None if with_trajectory is None else with_trajectory["success_count"]
                ),
                "with_success_rate": (
                    None if with_trajectory is None else with_trajectory["success_rate"]
                ),
                "with_wilson_95_ci_low": (
                    None if with_trajectory is None else with_trajectory["wilson_95_ci_low"]
                ),
                "with_wilson_95_ci_high": (
                    None if with_trajectory is None else with_trajectory["wilson_95_ci_high"]
                ),
                "delta_with_minus_without_percentage_points": group[
                    "delta_with_minus_without_percentage_points"
                ],
            }
        )
    return rows


def summary_csv_rows(summary: dict) -> list[dict]:
    rows = grouped_summary_csv_rows(summary["quartiles"])
    return [
        {"quartile": row["group"], **{key: value for key, value in row.items() if key != "group"}}
        for row in rows
    ]


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


def write_json(path: Path, payload: object) -> None:
    _atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def write_csv(path: Path, records: Sequence[dict]) -> None:
    if not records:
        _atomic_write_text(path, "")
        return
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(records[0]))
    writer.writeheader()
    for record in records:
        row = dict(record)
        for key, value in row.items():
            if isinstance(value, list):
                row[key] = " -> ".join(str(item) for item in value)
        writer.writerow(row)
    _atomic_write_text(path, stream.getvalue())


def _room_sort_key(room_id: str) -> tuple[int, str, str]:
    match = re.fullmatch(r"Room (\d+)([A-Za-z]*)", room_id)
    if match is None:
        return (10**9, room_id.casefold(), room_id)
    return (int(match.group(1)), match.group(2).casefold(), room_id)


def numeric_distribution_summary(values: Sequence[float]) -> dict:
    numbers = [float(value) for value in values]
    if not numbers or any(not math.isfinite(value) for value in numbers):
        raise ValueError("Distribution values must be non-empty and finite.")
    quartiles = statistics.quantiles(numbers, n=4, method="inclusive")
    return {
        "case_count": len(numbers),
        "min": min(numbers),
        "q25": quartiles[0],
        "median": statistics.median(numbers),
        "mean": statistics.fmean(numbers),
        "q75": quartiles[2],
        "max": max(numbers),
        "population_standard_deviation": statistics.pstdev(numbers),
    }


def fixed_histogram_edges(values: Sequence[float], width: float) -> list[float]:
    numbers = [float(value) for value in values]
    if width <= 0.0 or not math.isfinite(width):
        raise ValueError("Histogram width must be positive and finite.")
    if not numbers or any(not math.isfinite(value) or value < 0.0 for value in numbers):
        raise ValueError("Histogram values must be non-empty, finite, and non-negative.")
    bin_count = max(1, int(math.ceil(max(numbers) / width)))
    return [index * width for index in range(bin_count + 1)]


def histogram_counts(values: Sequence[float], edges: Sequence[float]) -> list[int]:
    boundaries = [float(edge) for edge in edges]
    if len(boundaries) < 2 or any(
        boundaries[index] >= boundaries[index + 1]
        for index in range(len(boundaries) - 1)
    ):
        raise ValueError("Histogram edges must be strictly increasing.")
    counts = [0] * (len(boundaries) - 1)
    for raw_value in values:
        value = float(raw_value)
        if not math.isfinite(value) or value < boundaries[0] or value > boundaries[-1]:
            raise ValueError(f"Histogram value is outside the plotted range: {value}")
        index = bisect.bisect_right(boundaries, value) - 1
        counts[min(index, len(counts) - 1)] += 1
    return counts


def build_trajectory_distribution_report(
    rows: Sequence[dict],
    room_areas: dict[str, dict],
    *,
    expected_navigable_gallery_count: int | None = 25,
) -> dict:
    if not rows:
        raise ValueError("Trajectory distribution report requires route rows.")
    navigable_room_ids = sorted(
        {
            str(room_id)
            for row in rows
            for room_id in row["reference_gallery_room_ids"]
        },
        key=_room_sort_key,
    )
    if (
        expected_navigable_gallery_count is not None
        and len(navigable_room_ids) != expected_navigable_gallery_count
    ):
        raise ValueError(
            "Unexpected navigable gallery count: "
            f"expected {expected_navigable_gallery_count}, got {len(navigable_room_ids)}."
        )
    unknown = [room_id for room_id in navigable_room_ids if room_id not in room_areas]
    if unknown:
        raise ValueError(f"Navigable galleries lack area estimates: {unknown}")

    distance_values = [float(row["shortest_path_distance_m"]) for row in rows]
    reference_area_values = [
        float(row["reference_estimated_total_unique_gallery_area_m2"])
        for row in rows
    ]
    without_actual_area_values = [
        float(row["without_actual_estimated_total_unique_gallery_area_m2"])
        for row in rows
    ]
    with_actual_area_values = [
        float(row["with_actual_estimated_total_unique_gallery_area_m2"])
        for row in rows
    ]
    distance_edges = fixed_histogram_edges(
        distance_values, SHORTEST_PATH_DISTANCE_BIN_WIDTH_M
    )
    shared_area_edges = fixed_histogram_edges(
        reference_area_values
        + without_actual_area_values
        + with_actual_area_values,
        UNIQUE_GALLERY_AREA_SUM_BIN_WIDTH_M2,
    )

    def histogram_payload(values: Sequence[float], edges: Sequence[float]) -> dict:
        counts = histogram_counts(values, edges)
        return {
            "edges": list(edges),
            "counts": counts,
            "percentages": [count / len(values) for count in counts],
        }

    gallery_records = [
        {
            "gallery_room_id": room_id,
            "effective_area_m2": float(room_areas[room_id]["effective_area_m2"]),
            "area_source": str(room_areas[room_id]["area_source"]),
        }
        for room_id in navigable_room_ids
    ]
    return {
        "schema": "llm_nav_trajectory_distance_and_unique_gallery_area_distributions",
        "schema_version": 1,
        "case_count_per_series": len(rows),
        "definitions": {
            "shortest_path_distance": (
                "Dataset path_distance_m for each of the 715 reference shortest paths."
            ),
            "total_unique_gallery_area": (
                "Sum of effective_area_m2 over unique gallery rooms in one path; "
                "a gallery visited multiple times contributes its area once."
            ),
            "reference_path": "Unique galleries in shortest_path_rooms.",
            "actual_path": (
                "Unique ground-truth galleries along each method's actual pano_path; "
                "unlabeled panoramas and circulation nodes are excluded."
            ),
            "area_caveat": (
                "Gallery areas are estimates, not architectural floor-plan measurements."
            ),
        },
        "navigable_galleries": {
            "gallery_count": len(gallery_records),
            "estimated_total_area_m2": sum(
                float(record["effective_area_m2"]) for record in gallery_records
            ),
            "gallery_room_ids": navigable_room_ids,
            "area_catalog_rooms_not_in_navigable_set": sorted(
                set(room_areas) - set(navigable_room_ids), key=_room_sort_key
            ),
            "rooms": gallery_records,
        },
        "shortest_path_distance_m": {
            "summary": numeric_distribution_summary(distance_values),
            "histogram": histogram_payload(distance_values, distance_edges),
        },
        "total_unique_gallery_area_m2": {
            "shared_histogram_edges": shared_area_edges,
            "reference_shortest_path": {
                "summary": numeric_distribution_summary(reference_area_values),
                "histogram": histogram_payload(
                    reference_area_values, shared_area_edges
                ),
            },
            "without_trajectory_actual_path": {
                "summary": numeric_distribution_summary(without_actual_area_values),
                "histogram": histogram_payload(
                    without_actual_area_values, shared_area_edges
                ),
            },
            "with_trajectory_actual_path": {
                "summary": numeric_distribution_summary(with_actual_area_values),
                "histogram": histogram_payload(
                    with_actual_area_values, shared_area_edges
                ),
            },
        },
    }


def navigable_gallery_area_csv_rows(report: dict) -> list[dict]:
    rows = [dict(record) for record in report["navigable_galleries"]["rooms"]]
    rows.append(
        {
            "gallery_room_id": "TOTAL",
            "effective_area_m2": report["navigable_galleries"][
                "estimated_total_area_m2"
            ],
            "area_source": "sum_of_25_navigable_galleries",
        }
    )
    return rows


def route_distance_and_area_csv_rows(rows: Sequence[dict]) -> list[dict]:
    return [
        {
            "test_id": row["test_id"],
            "path_id": row["path_id"],
            "shortest_path_distance_m": row["shortest_path_distance_m"],
            "reference_unique_gallery_count": row[
                "reference_gallery_count_for_mean"
            ],
            "reference_unique_gallery_room_ids": row[
                "reference_gallery_room_ids"
            ],
            "reference_total_unique_gallery_area_m2": row[
                "reference_estimated_total_unique_gallery_area_m2"
            ],
            "without_actual_unique_gallery_count": row[
                "without_actual_gallery_count_for_mean"
            ],
            "without_actual_unique_gallery_room_ids": row[
                "without_actual_gallery_room_ids"
            ],
            "without_actual_total_unique_gallery_area_m2": row[
                "without_actual_estimated_total_unique_gallery_area_m2"
            ],
            "with_actual_unique_gallery_count": row[
                "with_actual_gallery_count_for_mean"
            ],
            "with_actual_unique_gallery_room_ids": row[
                "with_actual_gallery_room_ids"
            ],
            "with_actual_total_unique_gallery_area_m2": row[
                "with_actual_estimated_total_unique_gallery_area_m2"
            ],
        }
        for row in rows
    ]


def distribution_bin_csv_rows(report: dict) -> list[dict]:
    records: list[dict] = []

    def append_series(
        metric: str,
        series: str,
        unit: str,
        histogram: dict,
    ) -> None:
        edges = histogram["edges"]
        counts = histogram["counts"]
        percentages = histogram["percentages"]
        for index, count in enumerate(counts):
            lower = float(edges[index])
            upper = float(edges[index + 1])
            records.append(
                {
                    "metric": metric,
                    "series": series,
                    "unit": unit,
                    "bin_lower": lower,
                    "bin_upper": upper,
                    "bin_label": f"{lower:g}–{upper:g}",
                    "route_count": count,
                    "route_percentage": percentages[index],
                }
            )

    append_series(
        "shortest_path_distance",
        "Reference shortest path",
        "m",
        report["shortest_path_distance_m"]["histogram"],
    )
    area_section = report["total_unique_gallery_area_m2"]
    for key, label in (
        ("reference_shortest_path", "Reference shortest path"),
        ("without_trajectory_actual_path", "Without trajectory actual path"),
        ("with_trajectory_actual_path", "With trajectory actual path"),
    ):
        append_series(
            "total_unique_gallery_area",
            label,
            "m2",
            area_section[key]["histogram"],
        )
    return records


def _font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    names = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    )
    for name in names:
        if Path(name).exists():
            return ImageFont.truetype(name, size=size)
    return ImageFont.load_default()


def _nice_step(span: float, target_ticks: int = 8) -> float:
    rough = span / max(target_ticks, 1)
    exponent = 10.0 ** math.floor(math.log10(rough))
    fraction = rough / exponent
    nice_fraction = 1.0 if fraction <= 1.0 else 2.0 if fraction <= 2.0 else 5.0
    return nice_fraction * exponent


def render_histogram_plot(
    series: Sequence[tuple[str, Sequence[float], str]],
    edges: Sequence[float],
    path: Path,
    *,
    x_axis_label: str,
    y_axis_label: str = "Number of routes",
) -> None:
    if not series:
        raise ValueError("Histogram plot requires at least one series.")
    boundaries = [float(edge) for edge in edges]
    series_counts = [
        (label, histogram_counts(values, boundaries), color)
        for label, values, color in series
    ]
    max_count = max(max(counts) for _, counts, _ in series_counts)
    y_step = _nice_step(float(max_count), target_ticks=6)
    y_max = max(y_step, math.ceil(max_count / y_step) * y_step)

    width, height = 1600, 1000
    plot_left = 165
    plot_top = 105 if len(series_counts) > 1 else 55
    plot_right = width - 90
    plot_bottom = 830
    background = "#FFFFFF"
    foreground = "#202124"
    muted = "#5F6368"
    grid = "#DADCE0"
    frame = "#9AA0A6"
    image = Image.new("RGB", (width, height), background)
    draw = ImageDraw.Draw(image)
    axis_font = _font(23)
    tick_font = _font(20)
    legend_font = _font(19)

    x_min = boundaries[0]
    x_max = boundaries[-1]

    def x_pixel(value: float) -> float:
        return plot_left + (value - x_min) / (x_max - x_min) * (
            plot_right - plot_left
        )

    def y_pixel(value: float) -> float:
        return plot_bottom - value / y_max * (plot_bottom - plot_top)

    y_tick = 0.0
    while y_tick <= y_max + 1e-9:
        y = y_pixel(y_tick)
        draw.line((plot_left, y, plot_right, y), fill=grid, width=2)
        label = f"{y_tick:g}"
        box = draw.textbbox((0, 0), label, font=tick_font)
        draw.text(
            (plot_left - 24 - (box[2] - box[0]), y - (box[3] - box[1]) / 2),
            label,
            fill=muted,
            font=tick_font,
        )
        y_tick += y_step

    tick_stride = max(1, math.ceil((len(boundaries) - 1) / 9))
    tick_indices = list(range(0, len(boundaries), tick_stride))
    if tick_indices[-1] != len(boundaries) - 1:
        tick_indices.append(len(boundaries) - 1)
    for index in tick_indices:
        value = boundaries[index]
        x = x_pixel(value)
        draw.line((x, plot_bottom, x, plot_bottom + 9), fill=frame, width=2)
        label = f"{value:g}"
        box = draw.textbbox((0, 0), label, font=tick_font)
        label_x = x - (box[2] - box[0]) / 2
        if index == 0:
            label_x = x
        elif index == len(boundaries) - 1:
            label_x = x - (box[2] - box[0])
        draw.text(
            (label_x, plot_bottom + 18),
            label,
            fill=muted,
            font=tick_font,
        )

    series_count = len(series_counts)
    for bin_index in range(len(boundaries) - 1):
        bin_left = x_pixel(boundaries[bin_index])
        bin_right = x_pixel(boundaries[bin_index + 1])
        available_width = (bin_right - bin_left) * 0.82
        start_x = bin_left + (bin_right - bin_left - available_width) / 2.0
        bar_width = available_width / series_count
        for series_index, (_, counts, color) in enumerate(series_counts):
            count = counts[bin_index]
            if count == 0:
                continue
            left = start_x + series_index * bar_width + 1
            right = start_x + (series_index + 1) * bar_width - 1
            draw.rectangle(
                (left, y_pixel(float(count)), right, plot_bottom - 1),
                fill=color,
            )

    draw.rectangle(
        (plot_left, plot_top, plot_right, plot_bottom), outline=frame, width=2
    )

    if len(series_counts) > 1:
        legend_widths = []
        for label, _, _ in series_counts:
            box = draw.textbbox((0, 0), label, font=legend_font)
            legend_widths.append(24 + 12 + (box[2] - box[0]))
        total_legend_width = sum(legend_widths) + 45 * (len(series_counts) - 1)
        legend_x = (width - total_legend_width) / 2
        legend_y = 55
        for (label, _, color), item_width in zip(series_counts, legend_widths):
            draw.rectangle(
                (legend_x, legend_y - 10, legend_x + 20, legend_y + 10),
                fill=color,
            )
            draw.text(
                (legend_x + 32, legend_y - 14),
                label,
                fill=foreground,
                font=legend_font,
            )
            legend_x += item_width + 45

    box = draw.textbbox((0, 0), x_axis_label, font=axis_font)
    draw.text(
        ((width - (box[2] - box[0])) / 2, 916),
        x_axis_label,
        fill=foreground,
        font=axis_font,
    )
    y_label_box = draw.textbbox((0, 0), y_axis_label, font=axis_font)
    y_label_image = Image.new(
        "RGBA",
        (y_label_box[2] - y_label_box[0] + 8, y_label_box[3] - y_label_box[1] + 8),
        (255, 255, 255, 0),
    )
    ImageDraw.Draw(y_label_image).text(
        (4, 0), y_axis_label, fill=foreground, font=axis_font
    )
    rotated = y_label_image.rotate(90, expand=True)
    image.paste(
        rotated,
        (35, int((plot_top + plot_bottom - rotated.height) / 2)),
        rotated,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = PngInfo()
    metadata.add_text("Title", x_axis_label)
    metadata.add_text(
        "Description",
        f"Histogram with {len(boundaries) - 1} equal-width bins. "
        f"X: {x_axis_label}. Y: {y_axis_label}. Series: "
        + ", ".join(label for label, _, _ in series_counts)
        + ".",
    )
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as stream:
        temporary_path = Path(stream.name)
    try:
        image.save(temporary_path, format="PNG", pnginfo=metadata, optimize=True)
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def render_scatter_plot(
    summary: dict,
    path: Path,
    *,
    groups: Sequence[dict] | None = None,
    title: str = "Estimated Mean Gallery Area × Navigation Success Rate",
    subtitle: str | None = None,
    minimal: bool = False,
    show_legend: bool | None = None,
    connect_series: bool | None = None,
    x_domain: tuple[float, float] | None = None,
    x_axis_label: str = "Estimated mean gallery area along reference route (m²)",
    x_tick_labels: Sequence[tuple[float, str]] | None = None,
    show_confidence_intervals: bool = True,
) -> None:
    plotted_groups = [
        group
        for group in (groups if groups is not None else summary["quartiles"])
        if group.get("without_trajectory") is not None
        and group.get("with_trajectory") is not None
    ]
    group_count = len(plotted_groups)
    legend_is_visible = (not minimal) if show_legend is None else show_legend
    series_are_connected = (
        (group_count > 4 and not minimal)
        if connect_series is None
        else connect_series
    )
    width, height = (
        (1600, 1000)
        if minimal
        else ((1800, 1000) if group_count > 4 else (1600, 1000))
    )
    plot_left = 165
    plot_top = 105 if minimal and legend_is_visible else 55 if minimal else 205
    plot_right = width - 90
    plot_bottom = 830
    background = "#FFFFFF"
    foreground = "#202124"
    muted = "#5F6368"
    grid = "#DADCE0"
    frame = "#9AA0A6"
    without_color = "#0072B2"
    with_color = "#D55E00"
    image = Image.new("RGB", (width, height), background)
    draw = ImageDraw.Draw(image)
    title_font = _font(40, bold=True)
    subtitle_font = _font(23)
    axis_font = _font(23)
    tick_font = _font(20)
    small_font = _font(19)
    label_font = _font(18 if group_count > 6 else 20, bold=True)

    if not minimal:
        draw.text(
            (plot_left, 55),
            title,
            fill=foreground,
            font=title_font,
        )
        draw.text(
            (plot_left, 113),
            subtitle
            or "715 paired routes · shared area quartiles · error bars show 95% Wilson CI",
            fill=muted,
            font=subtitle_font,
        )

    group_x_values = [
        float(group["mean_of_route_mean_gallery_area_m2"])
        for group in plotted_groups
    ]
    x_min_observed = min(group_x_values)
    x_max_observed = max(group_x_values)
    x_step = _nice_step(x_max_observed - x_min_observed)
    if x_domain is None:
        x_min = max(
            0.0, math.floor((x_min_observed - x_step * 0.25) / x_step) * x_step
        )
        x_max = math.ceil((x_max_observed + x_step * 0.25) / x_step) * x_step
    else:
        x_min, x_max = x_domain
        if x_max <= x_min:
            raise ValueError("x_domain must have upper > lower.")

    def x_pixel(value: float) -> float:
        return plot_left + (value - x_min) / (x_max - x_min) * (plot_right - plot_left)

    def y_pixel(rate: float) -> float:
        return plot_bottom - rate * (plot_bottom - plot_top)

    for percent in range(0, 101, 20):
        y = y_pixel(percent / 100.0)
        draw.line((plot_left, y, plot_right, y), fill=grid, width=2)
        label = f"{percent}%"
        box = draw.textbbox((0, 0), label, font=tick_font)
        draw.text(
            (plot_left - 24 - (box[2] - box[0]), y - (box[3] - box[1]) / 2),
            label,
            fill=muted,
            font=tick_font,
        )
    if x_tick_labels is None:
        displayed_x_ticks = []
        x_tick = math.ceil(x_min / x_step) * x_step
        while x_tick <= x_max + 1e-9:
            displayed_x_ticks.append((x_tick, f"{x_tick:g}"))
            x_tick += x_step
    else:
        displayed_x_ticks = [
            (float(value), str(label))
            for value, label in x_tick_labels
            if x_min <= float(value) <= x_max
        ]
    for x_tick, label in displayed_x_ticks:
        x = x_pixel(x_tick)
        draw.line((x, plot_bottom, x, plot_bottom + 9), fill=frame, width=2)
        box = draw.textbbox((0, 0), label, font=tick_font)
        draw.text(
            (x - (box[2] - box[0]) / 2, plot_bottom + 18),
            label,
            fill=muted,
            font=tick_font,
        )
    draw.rectangle(
        (plot_left, plot_top, plot_right, plot_bottom), outline=frame, width=2
    )

    legend_y = 55 if minimal else 158
    legend_start_x = plot_left if not minimal else width // 2 - 245
    legend_x = legend_start_x + 270
    if legend_is_visible:
        draw.ellipse(
            (
                legend_start_x,
                legend_y - 9,
                legend_start_x + 18,
                legend_y + 9,
            ),
            fill=without_color,
        )
        draw.text(
            (legend_start_x + 30, legend_y - 14),
            "Without trajectory",
            fill=foreground,
            font=small_font,
        )
        draw.polygon(
            [(legend_x + 10, legend_y - 11), (legend_x, legend_y + 10), (legend_x + 20, legend_y + 10)],
            fill=with_color,
        )
        draw.text(
            (legend_x + 32, legend_y - 14),
            "With trajectory",
            fill=foreground,
            font=small_font,
        )

    if group_count > 4 and not minimal:
        group_sizes = "   ·   ".join(
            (
                f"{group['group_label']}:{group['case_count']}"
                if group_count > 8
                else f"{group['group_label']} n={group['case_count']}"
            )
            for group in plotted_groups
        )
        draw.text(
            (legend_x + 270, legend_y - 14),
            group_sizes,
            fill=muted,
            font=_font(16),
        )
    if series_are_connected:
        for method, color in (
            ("without_trajectory", without_color),
            ("with_trajectory", with_color),
        ):
            series_points = [
                (
                    x_pixel(float(group["mean_of_route_mean_gallery_area_m2"])),
                    y_pixel(float(group[method]["success_rate"])),
                )
                for group in plotted_groups
            ]
            draw.line(series_points, fill=color, width=4, joint="curve")

    for group in plotted_groups:
        point_x = x_pixel(float(group["mean_of_route_mean_gallery_area_m2"]))
        group_label = (
            f"{group['group_label']} · n={group['case_count']}"
            if group_count <= 4
            else str(group["group_label"])
        )
        box = draw.textbbox((0, 0), group_label, font=small_font)
        group_label_y = plot_top + 15 if group_count <= 4 else plot_bottom - 34
        if not minimal:
            draw.text(
                (point_x - (box[2] - box[0]) / 2, group_label_y),
                group_label,
                fill=muted,
                font=small_font,
            )
        for method, color, shape in (
            ("without_trajectory", without_color, "circle"),
            ("with_trajectory", with_color, "triangle"),
        ):
            result = group[method]
            rate = float(result["success_rate"])
            y = y_pixel(rate)
            if show_confidence_intervals:
                low_y = y_pixel(float(result["wilson_95_ci_low"]))
                high_y = y_pixel(float(result["wilson_95_ci_high"]))
                draw.line((point_x, high_y, point_x, low_y), fill=color, width=5)
                draw.line(
                    (point_x - 12, high_y, point_x + 12, high_y),
                    fill=color,
                    width=5,
                )
                draw.line(
                    (point_x - 12, low_y, point_x + 12, low_y),
                    fill=color,
                    width=5,
                )
            if shape == "circle":
                draw.ellipse(
                    (point_x - 13, y - 13, point_x + 13, y + 13),
                    fill=color,
                    outline=background,
                    width=3,
                )
                label_y = y + 17
            else:
                draw.polygon(
                    [(point_x, y - 16), (point_x - 15, y + 13), (point_x + 15, y + 13)],
                    fill=color,
                )
                label_y = y - 39
            if not minimal and group_count <= 6:
                if label_y < plot_top + 8:
                    label_y = y + 17
                draw.text(
                    (point_x + 18, label_y),
                    f"{100.0 * rate:.1f}%",
                    fill=color,
                    font=label_font,
                )

    box = draw.textbbox((0, 0), x_axis_label, font=axis_font)
    draw.text(
        ((width - (box[2] - box[0])) / 2, 916),
        x_axis_label,
        fill=foreground,
        font=axis_font,
    )
    y_axis_label = "Navigation success rate (%)"
    y_label_box = draw.textbbox((0, 0), y_axis_label, font=axis_font)
    y_label_image = Image.new(
        "RGBA",
        (y_label_box[2] - y_label_box[0] + 8, y_label_box[3] - y_label_box[1] + 8),
        (255, 255, 255, 0),
    )
    ImageDraw.Draw(y_label_image).text((4, 0), y_axis_label, fill=foreground, font=axis_font)
    rotated = y_label_image.rotate(90, expand=True)
    image.paste(
        rotated,
        (35, int((plot_top + plot_bottom - rotated.height) / 2)),
        rotated,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = PngInfo()
    metadata.add_text("Title", title)
    metadata.add_text(
        "Description",
        f"X: {x_axis_label}. Y: navigation success rate. Includes Without trajectory "
        f"and With trajectory with {group_count} fixed-interval group points. "
        + (
            "X-axis interval labels: "
            + ", ".join(label for _, label in displayed_x_ticks)
            + ". "
            if x_tick_labels is not None
            else ""
        )
        + (
            "Error bars show 95% Wilson confidence intervals."
            if show_confidence_intervals
            else "Confidence intervals are retained in the accompanying tables but are not drawn."
        ),
    )
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as stream:
        temporary_path = Path(stream.name)
    try:
        image.save(temporary_path, format="PNG", pnginfo=metadata, optimize=True)
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _display_jitter(test_id: str, method: str, amplitude: float = 0.025) -> float:
    digest = hashlib.sha256(f"{test_id}:{method}".encode("utf-8")).digest()
    fraction = int.from_bytes(digest[:8], "big") / float(2**64 - 1)
    return (2.0 * fraction - 1.0) * amplitude


def render_route_level_scatter_plot(
    rows: Sequence[dict], summary: dict, path: Path
) -> None:
    width, height = 1800, 1000
    background = "#FFFFFF"
    foreground = "#202124"
    muted = "#5F6368"
    grid = "#DADCE0"
    frame = "#9AA0A6"
    without_color = "#0072B2"
    with_color = "#D55E00"
    image = Image.new("RGB", (width, height), background)
    draw = ImageDraw.Draw(image)
    title_font = _font(38, bold=True)
    subtitle_font = _font(21)
    panel_font = _font(27, bold=True)
    axis_font = _font(21)
    tick_font = _font(18)
    small_font = _font(18)

    draw.text(
        (135, 42),
        "Route-level Mean Gallery Area × Navigation Success",
        fill=foreground,
        font=title_font,
    )
    draw.text(
        (135, 95),
        "Each route is one 0/100% outcome; curves show fitted success probability with 95% CI",
        fill=muted,
        font=subtitle_font,
    )
    draw.text(
        (135, 130),
        "Vertical jitter is display-only and does not change any route outcome.",
        fill=muted,
        font=small_font,
    )

    panels = (
        {
            "title": "Without trajectory",
            "key": "without_trajectory_success",
            "model_key": "without_trajectory",
            "color": without_color,
            "left": 135,
            "right": 860,
        },
        {
            "title": "With trajectory",
            "key": "with_trajectory_success",
            "model_key": "with_trajectory",
            "color": with_color,
            "left": 985,
            "right": 1710,
        },
    )
    plot_top, plot_bottom = 245, 815
    all_x = [float(row["estimated_mean_gallery_area_m2"]) for row in rows]
    x_step = _nice_step(max(all_x) - min(all_x), target_ticks=10)
    x_min = max(0.0, math.floor((min(all_x) - 0.2 * x_step) / x_step) * x_step)
    x_max = math.ceil((max(all_x) + 0.2 * x_step) / x_step) * x_step

    for panel in panels:
        left = int(panel["left"])
        right = int(panel["right"])
        color = str(panel["color"])

        def x_pixel(value: float) -> float:
            return left + (value - x_min) / (x_max - x_min) * (right - left)

        def y_pixel(rate: float) -> float:
            return plot_bottom - rate * (plot_bottom - plot_top)

        model = summary["route_level_logistic_regression"][panel["model_key"]]
        draw.text(
            (left, 180), str(panel["title"]), fill=foreground, font=panel_font
        )
        draw.text(
            (left, 216),
            "Odds ratio per +100 m²: "
            f"{float(model['odds_ratio_per_100_m2']):.2f} "
            f"[{float(model['odds_ratio_per_100_m2_95_ci_low']):.2f}, "
            f"{float(model['odds_ratio_per_100_m2_95_ci_high']):.2f}]",
            fill=muted,
            font=small_font,
        )
        for percent in range(0, 101, 20):
            y = y_pixel(percent / 100.0)
            draw.line((left, y, right, y), fill=grid, width=2)
            if left == panels[0]["left"]:
                label = f"{percent}%"
                box = draw.textbbox((0, 0), label, font=tick_font)
                draw.text(
                    (left - 19 - (box[2] - box[0]), y - (box[3] - box[1]) / 2),
                    label,
                    fill=muted,
                    font=tick_font,
                )
        x_tick = math.ceil(x_min / x_step) * x_step
        while x_tick <= x_max + 1e-9:
            x = x_pixel(x_tick)
            draw.line((x, plot_bottom, x, plot_bottom + 8), fill=frame, width=2)
            label = f"{x_tick:g}"
            box = draw.textbbox((0, 0), label, font=tick_font)
            draw.text(
                (x - (box[2] - box[0]) / 2, plot_bottom + 17),
                label,
                fill=muted,
                font=tick_font,
            )
            x_tick += x_step
        draw.rectangle((left, plot_top, right, plot_bottom), outline=frame, width=2)

        confidence_overlay = Image.new("RGBA", (width, height), (255, 255, 255, 0))
        confidence_draw = ImageDraw.Draw(confidence_overlay)
        curve_samples: list[tuple[float, float, float, float]] = []
        for sample_index in range(161):
            area = x_min + sample_index / 160.0 * (x_max - x_min)
            probability, lower, upper = logistic_prediction(model, area)
            curve_samples.append((area, probability, lower, upper))
        band = [
            (x_pixel(area), y_pixel(upper)) for area, _, _, upper in curve_samples
        ] + [
            (x_pixel(area), y_pixel(lower))
            for area, _, lower, _ in reversed(curve_samples)
        ]
        confidence_draw.polygon(band, fill=(*ImageColor.getrgb(color), 40))
        image = Image.alpha_composite(image.convert("RGBA"), confidence_overlay).convert("RGB")
        draw = ImageDraw.Draw(image)

        points_overlay = Image.new("RGBA", (width, height), (255, 255, 255, 0))
        points_draw = ImageDraw.Draw(points_overlay)
        point_rgb = ImageColor.getrgb(color)
        for row in rows:
            outcome = 1.0 if bool(row[panel["key"]]) else 0.0
            display_rate = min(
                0.985,
                max(
                    0.015,
                    outcome
                    + _display_jitter(
                        str(row["test_id"]), str(panel["model_key"])
                    ),
                ),
            )
            x = x_pixel(float(row["estimated_mean_gallery_area_m2"]))
            y = y_pixel(display_rate)
            points_draw.ellipse(
                (x - 4, y - 4, x + 4, y + 4),
                fill=(*point_rgb, 70),
                outline=(*point_rgb, 100),
                width=1,
            )
        image = Image.alpha_composite(image.convert("RGBA"), points_overlay).convert("RGB")
        draw = ImageDraw.Draw(image)
        curve_points = [
            (x_pixel(area), y_pixel(probability))
            for area, probability, _, _ in curve_samples
        ]
        draw.line(curve_points, fill=color, width=6, joint="curve")

    x_axis_label = "Estimated mean gallery area along reference route (m²)"
    box = draw.textbbox((0, 0), x_axis_label, font=axis_font)
    draw.text(
        ((width - (box[2] - box[0])) / 2, 914),
        x_axis_label,
        fill=foreground,
        font=axis_font,
    )
    y_axis_label = "Route outcome / fitted success probability (%)"
    y_label_box = draw.textbbox((0, 0), y_axis_label, font=axis_font)
    y_label_image = Image.new(
        "RGBA",
        (y_label_box[2] - y_label_box[0] + 8, y_label_box[3] - y_label_box[1] + 8),
        (255, 255, 255, 0),
    )
    ImageDraw.Draw(y_label_image).text(
        (4, 0), y_axis_label, fill=foreground, font=axis_font
    )
    rotated = y_label_image.rotate(90, expand=True)
    image.paste(
        rotated,
        (30, int((plot_top + plot_bottom - rotated.height) / 2)),
        rotated,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = PngInfo()
    metadata.add_text("Title", "Route-level Mean Gallery Area × Navigation Success")
    metadata.add_text(
        "Description",
        "Two-panel route-level scatter plot for Without trajectory and With "
        "trajectory. Each route is a binary 0/100% outcome with display-only "
        "vertical jitter. Curves are logistic fitted success probabilities with "
        "95% confidence intervals.",
    )
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as stream:
        temporary_path = Path(stream.name)
    try:
        image.save(temporary_path, format="PNG", pnginfo=metadata, optimize=True)
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Report navigation success rate by fixed estimated mean gallery-area "
            "intervals for both reference shortest paths and actual agent paths."
        )
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--without-results", default=DEFAULT_WITHOUT_RESULTS)
    parser.add_argument("--with-results", default=DEFAULT_WITH_RESULTS)
    parser.add_argument("--room-areas", default=DEFAULT_ROOM_AREAS)
    parser.add_argument("--pano-room-grounding", default=DEFAULT_PANO_ROOM_GROUNDING)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    dataset_path = resolve(args.dataset)
    without_path = resolve(args.without_results)
    with_path = resolve(args.with_results)
    room_areas_path = resolve(args.room_areas)
    pano_room_grounding_path = resolve(args.pano_room_grounding)
    output_dir = resolve(args.output_dir)

    dataset_records = load_jsonl(dataset_path)
    dataset = index_unique(dataset_records, label="dataset")
    without_results = index_unique(
        load_jsonl(without_path), label="without-trajectory results"
    )
    with_results = index_unique(
        load_jsonl(with_path), label="with-trajectory results"
    )
    validate_paired_inputs(dataset, without_results, with_results)
    room_areas, excluded_room_ids = load_area_catalog(load_json(room_areas_path))
    pano_room_grounding = load_pano_room_grounding(
        load_json(pano_room_grounding_path)
    )
    route_rows = build_route_rows(
        dataset_records,
        without_results,
        with_results,
        room_areas,
        excluded_room_ids,
        pano_room_grounding,
    )
    source_paths = {
        "dataset": dataset_path,
        "without_trajectory_results": without_path,
        "with_trajectory_results": with_path,
        "estimated_gallery_areas": room_areas_path,
        "pano_room_grounding": pano_room_grounding_path,
    }
    reference_summary = build_primary_fixed_interval_summary(
        route_rows,
        path_kind="reference_shortest_path",
        source_paths=source_paths,
    )
    actual_summary = build_primary_fixed_interval_summary(
        route_rows,
        path_kind="actual_agent_path",
        source_paths=source_paths,
    )
    actual_max800_summary = build_filtered_actual_fixed_interval_summary(
        route_rows,
        source_paths=source_paths,
    )
    distribution_report = build_trajectory_distribution_report(
        route_rows,
        room_areas,
    )
    distribution_report["provenance"] = {
        label: {"path": str(path), "sha256": sha256_file(path)}
        for label, path in source_paths.items()
    }

    write_csv(output_dir / "route_mean_gallery_area.csv", route_rows)
    write_csv(
        output_dir / "navigable_gallery_areas.csv",
        navigable_gallery_area_csv_rows(distribution_report),
    )
    write_csv(
        output_dir / "trajectory_distance_and_unique_gallery_area.csv",
        route_distance_and_area_csv_rows(route_rows),
    )
    write_csv(
        output_dir / "trajectory_distribution_bins.csv",
        distribution_bin_csv_rows(distribution_report),
    )
    write_json(
        output_dir / "trajectory_distance_and_unique_gallery_area_distributions.json",
        distribution_report,
    )
    reference_stem = "reference_shortest_path_fixed_200m2"
    actual_stem = "actual_agent_path_fixed_200m2"
    actual_max800_stem = "actual_agent_path_fixed_200m2_max800m2"
    write_csv(
        output_dir / f"{reference_stem}.csv",
        primary_fixed_interval_csv_rows(reference_summary),
    )
    write_json(output_dir / f"{reference_stem}.json", reference_summary)
    write_csv(
        output_dir / f"{actual_stem}.csv",
        primary_fixed_interval_csv_rows(actual_summary),
    )
    write_json(output_dir / f"{actual_stem}.json", actual_summary)
    write_csv(
        output_dir / f"{actual_max800_stem}.csv",
        primary_fixed_interval_csv_rows(actual_max800_summary),
    )
    write_json(
        output_dir / f"{actual_max800_stem}.json",
        actual_max800_summary,
    )
    write_json(
        output_dir / "summary.json",
        {
            "schema": "llm_nav_reference_and_actual_mean_gallery_area_report",
            "schema_version": 1,
            "case_count_per_method": len(route_rows),
            "fixed_intervals_m2": [
                list(interval)
                for interval in zip(
                    PRIMARY_FIXED_INTERVAL_EDGES_M2[:-1],
                    PRIMARY_FIXED_INTERVAL_EDGES_M2[1:],
                )
            ],
            "reference_shortest_path": reference_summary,
            "actual_agent_path": actual_summary,
            "actual_agent_path_max800m2": actual_max800_summary,
            "trajectory_distributions": distribution_report,
        },
    )
    fixed_interval_ticks = [
        (
            float(group["midpoint_m2"]),
            str(group["group_label"]).replace("-", "–"),
        )
        for group in reference_summary["groups"]
    ]
    render_scatter_plot(
        reference_summary,
        output_dir / f"{reference_stem}.png",
        groups=reference_summary["groups"],
        title="Reference Shortest Path Mean Gallery Area × Navigation Success Rate",
        minimal=True,
        show_legend=True,
        connect_series=True,
        x_domain=(0.0, PRIMARY_FIXED_INTERVAL_EDGES_M2[-1]),
        x_axis_label="Reference shortest-path mean gallery area interval (m²)",
        x_tick_labels=fixed_interval_ticks,
        show_confidence_intervals=True,
    )
    render_scatter_plot(
        actual_summary,
        output_dir / f"{actual_stem}.png",
        groups=actual_summary["groups"],
        title="Actual Agent Path Mean Gallery Area × Navigation Success Rate",
        minimal=True,
        show_legend=True,
        connect_series=True,
        x_domain=(0.0, PRIMARY_FIXED_INTERVAL_EDGES_M2[-1]),
        x_axis_label="Actual agent-path mean gallery area interval (m²)",
        x_tick_labels=fixed_interval_ticks,
        show_confidence_intervals=True,
    )
    render_scatter_plot(
        actual_max800_summary,
        output_dir / f"{actual_max800_stem}.png",
        groups=actual_max800_summary["groups"],
        title="Actual Agent Path Mean Gallery Area × Navigation Success Rate (≤800 m²)",
        minimal=True,
        show_legend=True,
        connect_series=True,
        x_domain=(0.0, PRIMARY_FIXED_INTERVAL_EDGES_M2[-1]),
        x_axis_label="Actual agent-path mean gallery area interval (m²)",
        x_tick_labels=fixed_interval_ticks,
        show_confidence_intervals=True,
    )
    distance_histogram = distribution_report["shortest_path_distance_m"][
        "histogram"
    ]
    render_histogram_plot(
        (
            (
                "Reference shortest path",
                [float(row["shortest_path_distance_m"]) for row in route_rows],
                "#0072B2",
            ),
        ),
        distance_histogram["edges"],
        output_dir / "shortest_path_distance_distribution.png",
        x_axis_label="Reference shortest-path distance (m)",
    )
    area_histogram_edges = distribution_report["total_unique_gallery_area_m2"][
        "shared_histogram_edges"
    ]
    render_histogram_plot(
        (
            (
                "Reference shortest path",
                [
                    float(row["reference_estimated_total_unique_gallery_area_m2"])
                    for row in route_rows
                ],
                "#0072B2",
            ),
        ),
        area_histogram_edges,
        output_dir / "reference_unique_gallery_area_sum_distribution.png",
        x_axis_label="Sum of unique gallery areas along reference shortest path (m²)",
    )
    render_histogram_plot(
        (
            (
                "Without trajectory",
                [
                    float(
                        row[
                            "without_actual_estimated_total_unique_gallery_area_m2"
                        ]
                    )
                    for row in route_rows
                ],
                "#0072B2",
            ),
            (
                "With trajectory",
                [
                    float(
                        row["with_actual_estimated_total_unique_gallery_area_m2"]
                    )
                    for row in route_rows
                ],
                "#D55E00",
            ),
        ),
        area_histogram_edges,
        output_dir / "actual_unique_gallery_area_sum_distribution.png",
        x_axis_label="Sum of unique gallery areas along actual agent path (m²)",
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "case_count_per_method": len(route_rows),
                "without_success_rate": reference_summary["overall"][
                    "without_trajectory"
                ]["success_rate"],
                "with_success_rate": reference_summary["overall"][
                    "with_trajectory"
                ]["success_rate"],
                "fixed_intervals_m2": [
                    group["group_label"] for group in reference_summary["groups"]
                ],
                "reference_case_counts": [
                    group["without_trajectory"]["case_count"]
                    for group in reference_summary["groups"]
                ],
                "actual_without_case_counts": [
                    group["without_trajectory"]["case_count"]
                    for group in actual_summary["groups"]
                ],
                "actual_with_case_counts": [
                    group["with_trajectory"]["case_count"]
                    for group in actual_summary["groups"]
                ],
                "actual_above_800_case_counts": {
                    "without_trajectory": sum(
                        group["without_trajectory"]["above_800_m2_case_count"]
                        for group in actual_summary["groups"]
                    ),
                    "with_trajectory": sum(
                        group["with_trajectory"]["above_800_m2_case_count"]
                        for group in actual_summary["groups"]
                    ),
                },
                "actual_max800_retained_case_counts": actual_max800_summary[
                    "retained_case_count"
                ],
                "actual_max800_excluded_case_counts": actual_max800_summary[
                    "excluded_case_count"
                ],
                "navigable_gallery_count": distribution_report[
                    "navigable_galleries"
                ]["gallery_count"],
                "estimated_total_navigable_gallery_area_m2": distribution_report[
                    "navigable_galleries"
                ]["estimated_total_area_m2"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
