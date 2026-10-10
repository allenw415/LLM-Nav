#!/usr/bin/env python3
"""Benchmark uncached localization latency on a small stratified case set.

The benchmark keeps the user's existing render and detection caches untouched.
It copies only each manifest into the output directory, calls the visual VLM
with cache use disabled, times the model-client boundary, and then benchmarks
the Bayesian update and map-grounded Spatial Alignment on the same case set.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from eval_map_grounded_spatial_alignment import (  # noqa: E402
    _run_case,
    bayesian_top5,
    source_change_category,
)
from sa_nav import (  # noqa: E402
    EvidenceScoreLocalizer,
    GroundingIndex,
    PerceptionPipeline,
    ViewDetector,
    load_dotenv,
    resolve_model_environment,
    resolve_task_num_ctx,
)
from sa_nav.cli._common import load_json, load_normalized_artifacts, render_json  # noqa: E402
from sa_nav.common.room_profiles import preferred_room_graph_path  # noqa: E402
from sa_nav.experiments.map_grounded_alignment import (  # noqa: E402
    MapGroundedSpatialAlignmentExperiment,
)


DEFAULT_SOURCE = (
    PROJECT_ROOT
    / "outputs/localization_evaluation/proportional_gallery_three_stage_gemini/summary.json"
)
DEFAULT_SPATIAL_RESULTS = (
    PROJECT_ROOT
    / "outputs/spatial_alignment_map_grounded_bayesian_top5_level0_158_compact_reasoning1024_20260810/results.jsonl"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "outputs/localization_timing/representative12_20260818"
)
DEFAULT_ARTIFACTS = PROJECT_ROOT / "dataset/sites/british_museum/normalized"
DEFAULT_MAP = PROJECT_ROOT / "dataset/sites/british_museum/maps/level_0_allocentric.png"
DEFAULT_REGIONS = PROJECT_ROOT / "dataset/sites/british_museum/maps/level_0_room_regions.json"
CATEGORIES = ("stable_correct", "correction", "unresolved", "regression")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-summary", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--prior-spatial-results", type=Path, default=DEFAULT_SPATIAL_RESULTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--map-image", type=Path, default=DEFAULT_MAP)
    parser.add_argument("--map-regions", type=Path, default=DEFAULT_REGIONS)
    parser.add_argument("--cases-per-category", type=int, default=3)
    parser.add_argument("--delay", type=float, default=2.0)
    parser.add_argument("--quota-retry-delay", type=float, default=60.0)
    parser.add_argument("--max-quota-retries", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--model")
    parser.add_argument("--api-kind")
    parser.add_argument("--api-base")
    parser.add_argument("--request-timeout", type=float)
    parser.add_argument("--max-output-tokens", type=int, default=1024)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_json(payload), encoding="utf-8")


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def timed_create_wrapper(
    original: Callable[[dict], dict],
    durations: list[float],
) -> Callable[[dict], dict]:
    def wrapped(request_body: dict) -> dict:
        started = time.perf_counter()
        try:
            return original(request_body)
        finally:
            durations.append(time.perf_counter() - started)

    return wrapped


def quantile(values: list[float], proportion: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Cannot calculate a quantile of an empty list.")
    position = (len(ordered) - 1) * proportion
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def select_cases(
    source_cases: list[dict[str, Any]],
    prior_spatial_records: list[dict[str, Any]],
    *,
    cases_per_category: int,
) -> list[dict[str, Any]]:
    prior_by_pano = {
        str(record.get("pano_id")): record
        for record in prior_spatial_records
        if record.get("status") == "completed"
        and isinstance(record.get("latency_seconds"), (int, float))
    }
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in source_cases:
        pano_id = str(case.get("pano_id"))
        prior = prior_by_pano.get(pano_id)
        if prior is None:
            continue
        record = dict(case)
        record["selection_category"] = source_change_category(case)
        record["prior_spatial_latency_seconds"] = float(prior["latency_seconds"])
        buckets[record["selection_category"]].append(record)

    selected: list[dict[str, Any]] = []
    target_quantiles = (
        [0.5]
        if cases_per_category == 1
        else [index / (cases_per_category - 1) for index in range(cases_per_category)]
    )
    for category in CATEGORIES:
        candidates = sorted(
            buckets[category],
            key=lambda item: (
                float(item["prior_spatial_latency_seconds"]),
                str(item.get("expected_room_id")),
                str(item.get("pano_id")),
            ),
        )
        if len(candidates) < cases_per_category:
            raise ValueError(
                f"Category {category} has only {len(candidates)} eligible cases; "
                f"requested {cases_per_category}."
            )
        used_panos: set[str] = set()
        used_rooms: set[str] = set()
        for proportion in target_quantiles:
            target = quantile(
                [float(item["prior_spatial_latency_seconds"]) for item in candidates],
                proportion,
            )
            ranked = sorted(
                candidates,
                key=lambda item: (
                    str(item.get("expected_room_id")) in used_rooms,
                    abs(float(item["prior_spatial_latency_seconds"]) - target),
                    str(item.get("expected_room_id")),
                    str(item.get("pano_id")),
                ),
            )
            choice = next(
                item for item in ranked if str(item.get("pano_id")) not in used_panos
            )
            selected.append(choice)
            used_panos.add(str(choice["pano_id"]))
            used_rooms.add(str(choice.get("expected_room_id")))
    return selected


def usage_from_trace(trace: object) -> dict[str, Any] | None:
    if not isinstance(trace, dict):
        return None
    response = trace.get("response")
    if not isinstance(response, dict):
        return None
    usage = response.get("usageMetadata") or response.get("usage")
    return dict(usage) if isinstance(usage, dict) else None


def predicted_room(distribution: object) -> str | None:
    if not isinstance(distribution, dict):
        return None
    values = [
        (str(room_id), float(probability))
        for room_id, probability in distribution.items()
        if isinstance(room_id, str) and isinstance(probability, (int, float))
    ]
    return sorted(values, key=lambda item: (-item[1], item[0]))[0][0] if values else None


def summarize_method(records: list[dict[str, Any]], method: str) -> dict[str, Any]:
    completed = [record for record in records if record.get("status") == "completed"]
    totals = [float(record["timing"][method]["total_seconds"]) for record in completed]
    modules = [float(record["timing"][method]["module_seconds"]) for record in completed]
    vlms = [float(record["timing"][method]["vlm_seconds"]) for record in completed]
    return {
        "case_count": len(completed),
        "mean_total_seconds": statistics.fmean(totals) if totals else None,
        "median_total_seconds": statistics.median(totals) if totals else None,
        "p25_total_seconds": quantile(totals, 0.25) if totals else None,
        "p75_total_seconds": quantile(totals, 0.75) if totals else None,
        "mean_module_seconds": statistics.fmean(modules) if modules else None,
        "mean_vlm_seconds": statistics.fmean(vlms) if vlms else None,
        "vlm_share_of_mean_total": (
            statistics.fmean(vlms) / statistics.fmean(totals) if totals else None
        ),
    }


def build_summary(
    records: list[dict[str, Any]],
    selected_cases: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    completed = [record for record in records if record.get("status") == "completed"]
    return {
        "schema": "sa_nav.localization_timing_benchmark",
        "schema_version": 1,
        "config": config,
        "selected_count": len(selected_cases),
        "completed_count": len(completed),
        "pending_or_error_count": len(selected_cases) - len(completed),
        "selection_counts": {
            category: sum(case.get("selection_category") == category for case in selected_cases)
            for category in CATEGORIES
        },
        "methods": {
            method: summarize_method(records, method)
            for method in ("visual_only", "bayesian_filter", "spatial_alignment")
        },
        "definitions": {
            "visual_only": "Fresh uncached visual detector from eight panorama views.",
            "bayesian_filter": "Visual Only plus the local room-transition Bayesian update.",
            "spatial_alignment": (
                "End-to-end Visual Only plus Bayesian update plus one compact map-grounded "
                "Spatial Alignment VLM call."
            ),
            "module_seconds": (
                "Wall time outside ModelResponseClient.create, including request construction, "
                "image encoding, response normalization, deterministic scoring, and file IO."
            ),
            "vlm_seconds": (
                "Wall time inside ModelResponseClient.create, including HTTP attempts, provider "
                "generation, response transfer, and retry backoff."
            ),
        },
        "records_path": str(config["results_path"]),
    }


def checkpoint(
    *,
    output_dir: Path,
    records_by_pano: dict[str, dict[str, Any]],
    selected_cases: list[dict[str, Any]],
    config: dict[str, Any],
) -> None:
    ordered = [
        records_by_pano[str(case["pano_id"])]
        for case in selected_cases
        if str(case["pano_id"]) in records_by_pano
    ]
    write_jsonl(output_dir / "results.jsonl", ordered)
    write_json(output_dir / "summary.json", build_summary(ordered, selected_cases, config))


def copy_manifest_for_uncached_call(source_path: Path, destination: Path) -> Path:
    manifest = load_json(source_path)
    manifest["manifest_path"] = str(destination)
    write_json(destination, manifest)
    return destination


def main() -> int:
    args = parse_args()
    if args.cases_per_category < 1:
        raise ValueError("--cases-per-category must be at least 1.")
    load_dotenv(PROJECT_ROOT / ".env")
    model_env = resolve_model_environment(
        default_model="gemma-4-31b-it",
        default_api_base="https://generativelanguage.googleapis.com/v1beta",
        default_api_kind="responses",
    )
    model = args.model or model_env.model_name or "gemma-4-31b-it"
    api_kind = args.api_kind or model_env.api_kind or "responses"
    api_base = args.api_base or model_env.api_base
    request_timeout = float(args.request_timeout or model_env.request_timeout or 300.0)
    if not model_env.api_key:
        raise RuntimeError("No model API key is configured for the active profile.")
    if not api_base:
        raise RuntimeError("No model API base is configured for the active profile.")

    source_payload = load_json(args.source_summary)
    source_cases = [case for case in source_payload.get("results", []) if isinstance(case, dict)]
    prior_spatial_records = read_jsonl(args.prior_spatial_results)
    selected_cases = select_cases(
        source_cases,
        prior_spatial_records,
        cases_per_category=args.cases_per_category,
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        output_dir / "selection.json",
        [
            {
                "pano_id": case.get("pano_id"),
                "expected_room_id": case.get("expected_room_id"),
                "selection_category": case.get("selection_category"),
                "prior_spatial_latency_seconds": case.get("prior_spatial_latency_seconds"),
            }
            for case in selected_cases
        ],
    )

    artifacts = load_normalized_artifacts(
        str(args.artifacts_dir),
        room_graph=True,
        pano_graph=True,
        grounding=True,
    )
    pano_graph = artifacts.pano_graph or {}
    grounding_index = GroundingIndex(artifacts.grounding or {})
    room_graph_path = preferred_room_graph_path(args.artifacts_dir)
    room_graph = load_json(room_graph_path)
    region_payload = load_json(args.map_regions)

    detector = ViewDetector(
        model=model,
        api_key=model_env.api_key,
        api_base=api_base,
        api_kind=api_kind,
        request_timeout=request_timeout,
        use_detection_files=False,
        enable_view_themes=False,
        room_graph=room_graph,
        grounding_index=grounding_index,
    )
    pipeline = PerceptionPipeline(
        pano_graph=pano_graph,
        room_graph=room_graph,
        grounding_index=grounding_index,
        detector=detector,
    )
    visual_vlm_durations: list[float] = []
    detector.model_client.create = timed_create_wrapper(
        detector.model_client.create,
        visual_vlm_durations,
    )

    spatial_experiment = MapGroundedSpatialAlignmentExperiment(
        room_graph=room_graph,
        model=model,
        api_key=model_env.api_key,
        api_base=api_base,
        api_kind=api_kind,
        request_timeout=request_timeout,
        num_ctx=resolve_task_num_ctx(
            "localization",
            fallback_num_ctx=model_env.num_ctx,
            default_num_ctx=16384,
        ),
        max_http_retries=2,
        validation_retries=0,
        max_output_tokens=args.max_output_tokens,
        allow_incomplete_winning_hypothesis=False,
        compact_decision_output=True,
    )
    spatial_vlm_durations: list[float] = []
    spatial_experiment.model_client.create = timed_create_wrapper(
        spatial_experiment.model_client.create,
        spatial_vlm_durations,
    )

    results_path = output_dir / "results.jsonl"
    existing_records = read_jsonl(results_path) if args.resume and results_path.is_file() else []
    records_by_pano = {
        str(record.get("pano_id")): record
        for record in existing_records
        if isinstance(record.get("pano_id"), str)
    }
    config = {
        "source_summary": str(args.source_summary.resolve()),
        "prior_spatial_results": str(args.prior_spatial_results.resolve()),
        "output_dir": str(output_dir),
        "results_path": str(results_path),
        "model": model,
        "provider": model_env.provider,
        "api_kind": api_kind,
        "request_timeout_seconds": request_timeout,
        "spatial_max_output_tokens": args.max_output_tokens,
        "cache_policy": "fresh_visual_call_in_isolated_output; existing_repo_cache_untouched",
        "execution_policy": "sequential",
    }

    for index, case in enumerate(selected_cases, start=1):
        pano_id = str(case["pano_id"])
        existing = records_by_pano.get(pano_id)
        if existing is not None and existing.get("status") == "completed":
            print(f"[{index}/{len(selected_cases)}] {pano_id}: completed (resume)", flush=True)
            continue

        record = existing if isinstance(existing, dict) else {
            "pano_id": pano_id,
            "expected_room_id": case.get("expected_room_id"),
            "selection_category": case.get("selection_category"),
            "prior_spatial_latency_seconds": case.get("prior_spatial_latency_seconds"),
            "status": "pending",
        }
        case_dir = output_dir / "cases" / pano_id
        case_dir.mkdir(parents=True, exist_ok=True)
        try:
            if record.get("status") not in {"visual_completed", "spatial_error"}:
                print(
                    f"[{index}/{len(selected_cases)}] {pano_id}: fresh Visual VLM start "
                    f"({case.get('selection_category')}, {case.get('expected_room_id')})",
                    flush=True,
                )
                source_manifest = Path(str(case.get("manifest_path"))).resolve()
                isolated_manifest = copy_manifest_for_uncached_call(
                    source_manifest,
                    case_dir / f"{pano_id}_manifest.json",
                )
                source_output = load_json(Path(str(case["output_path"])))
                current_heading = float(source_output.get("current_heading", 330.0))
                visual_quota_retries = 0
                while True:
                    visual_call_start = len(visual_vlm_durations)
                    visual_started = time.perf_counter()
                    try:
                        observation = pipeline.observe_from_manifest(
                            isolated_manifest,
                            current_heading=current_heading,
                        )
                    except RuntimeError as exc:
                        if (
                            "HTTP 429" not in str(exc)
                            or visual_quota_retries >= args.max_quota_retries
                            or args.quota_retry_delay <= 0
                        ):
                            raise
                        visual_quota_retries += 1
                        print(
                            f"[{index}/{len(selected_cases)}] {pano_id}: Visual HTTP 429; "
                            f"retry {visual_quota_retries}/{args.max_quota_retries} in "
                            f"{args.quota_retry_delay:.0f}s",
                            flush=True,
                        )
                        time.sleep(args.quota_retry_delay)
                        continue
                    visual_total = time.perf_counter() - visual_started
                    visual_vlm = sum(visual_vlm_durations[visual_call_start:])
                    break
                visual_module = max(0.0, visual_total - visual_vlm)
                visual_localization = observation.metadata.get("visual_localization")
                visual_distribution = (
                    visual_localization.get("room_scores")
                    if isinstance(visual_localization, dict)
                    else None
                )

                localizer = EvidenceScoreLocalizer(
                    room_graph=room_graph,
                    grounding_index=grounding_index,
                    spatial_refiner=None,
                )
                bayesian_started = time.perf_counter()
                bayesian_result = localizer.localize(
                    observation=observation,
                    prior_room_belief={str(case["expected_room_id"]): 1.0},
                    fallback_room_id=str(case["expected_room_id"]),
                )
                bayesian_incremental = time.perf_counter() - bayesian_started
                record.update(
                    {
                        "status": "visual_completed",
                        "visual_predicted_room_id": (
                            visual_localization.get("predicted_room_id")
                            if isinstance(visual_localization, dict)
                            else None
                        ),
                        "bayesian_predicted_room_id": bayesian_result.get("bayesian_predicted_room_id"),
                        "cached_bayesian_top5_for_spatial": bayesian_top5(case),
                        "timing": {
                            "visual_only": {
                                "module_seconds": visual_module,
                                "vlm_seconds": visual_vlm,
                                "total_seconds": visual_total,
                                "logical_vlm_call_count": len(visual_vlm_durations) - visual_call_start,
                                "quota_retry_count": visual_quota_retries,
                            },
                            "bayesian_filter": {
                                "module_seconds": visual_module + bayesian_incremental,
                                "vlm_seconds": visual_vlm,
                                "total_seconds": visual_total + bayesian_incremental,
                                "logical_vlm_call_count": len(visual_vlm_durations) - visual_call_start,
                                "quota_retry_count": visual_quota_retries,
                                "bayesian_incremental_seconds": bayesian_incremental,
                            },
                        },
                        "visual_usage": usage_from_trace(
                            detector.last_traces[-1] if detector.last_traces else None
                        ),
                    }
                )
                write_json(case_dir / "case_summary.json", record)
                records_by_pano[pano_id] = record
                checkpoint(
                    output_dir=output_dir,
                    records_by_pano=records_by_pano,
                    selected_cases=selected_cases,
                    config=config,
                )
                print(
                    f"[{index}/{len(selected_cases)}] {pano_id}: Visual done "
                    f"total={visual_total:.3f}s vlm={visual_vlm:.3f}s module={visual_module:.3f}s",
                    flush=True,
                )

            print(f"[{index}/{len(selected_cases)}] {pano_id}: Spatial Alignment start", flush=True)
            spatial_case = dict(case)
            spatial_case["prior_fused"] = {
                "top1": case.get("prior_fused", {}).get("top1"),
                "top5": list(record["cached_bayesian_top5_for_spatial"]),
            }
            spatial_quota_retries = 0
            while True:
                spatial_call_start = len(spatial_vlm_durations)
                spatial_record = _run_case(
                    case=spatial_case,
                    candidates=sorted(record["cached_bayesian_top5_for_spatial"]),
                    room_graph=room_graph,
                    region_payload=region_payload,
                    map_image_path=args.map_image,
                    output_dir=output_dir,
                    experiment=spatial_experiment,
                )
                spatial_error = spatial_record.get("error")
                spatial_error_message = (
                    spatial_error.get("message") if isinstance(spatial_error, dict) else None
                )
                if not (
                    isinstance(spatial_error_message, str)
                    and "HTTP 429" in spatial_error_message
                ):
                    break
                if (
                    spatial_quota_retries >= args.max_quota_retries
                    or args.quota_retry_delay <= 0
                ):
                    break
                spatial_quota_retries += 1
                print(
                    f"[{index}/{len(selected_cases)}] {pano_id}: Spatial HTTP 429; "
                    f"retry {spatial_quota_retries}/{args.max_quota_retries} in "
                    f"{args.quota_retry_delay:.0f}s",
                    flush=True,
                )
                time.sleep(args.quota_retry_delay)
            spatial_stage_total = float(spatial_record["latency_seconds"])
            spatial_vlm = sum(spatial_vlm_durations[spatial_call_start:])
            spatial_module = max(0.0, spatial_stage_total - spatial_vlm)
            write_json(case_dir / "spatial_result.json", spatial_record)
            if spatial_record.get("status") != "completed":
                raise RuntimeError(
                    spatial_record.get("error", {}).get("message", "Spatial Alignment failed.")
                )
            visual_timing = record["timing"]["visual_only"]
            bayesian_timing = record["timing"]["bayesian_filter"]
            record["timing"]["spatial_alignment_stage_only"] = {
                "module_seconds": spatial_module,
                "vlm_seconds": spatial_vlm,
                "total_seconds": spatial_stage_total,
                "logical_vlm_call_count": len(spatial_vlm_durations) - spatial_call_start,
                "quota_retry_count": spatial_quota_retries,
            }
            record["timing"]["spatial_alignment"] = {
                "module_seconds": float(bayesian_timing["module_seconds"]) + spatial_module,
                "vlm_seconds": float(visual_timing["vlm_seconds"]) + spatial_vlm,
                "total_seconds": float(bayesian_timing["total_seconds"]) + spatial_stage_total,
                "logical_vlm_call_count": (
                    int(visual_timing["logical_vlm_call_count"])
                    + len(spatial_vlm_durations)
                    - spatial_call_start
                ),
            }
            record.update(
                {
                    "status": "completed",
                    "spatial_predicted_room_id": spatial_record.get("raw_map_top1"),
                    "spatial_reasoning_complete": spatial_record.get("spatial_reasoning_complete"),
                    "spatial_usage": spatial_record.get("usage"),
                    "error": None,
                }
            )
            write_json(case_dir / "case_summary.json", record)
            print(
                f"[{index}/{len(selected_cases)}] {pano_id}: Spatial done "
                f"stage_total={spatial_stage_total:.3f}s vlm={spatial_vlm:.3f}s "
                f"module={spatial_module:.3f}s",
                flush=True,
            )
        except Exception as exc:
            has_visual_timing = isinstance(record.get("timing", {}).get("visual_only"), dict)
            record["status"] = "spatial_error" if has_visual_timing else "error"
            record["error"] = {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback_tail": traceback.format_exc().splitlines()[-8:],
            }
            write_json(case_dir / "case_summary.json", record)
            print(
                f"[{index}/{len(selected_cases)}] {pano_id}: {record['status']} "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
        records_by_pano[pano_id] = record
        checkpoint(
            output_dir=output_dir,
            records_by_pano=records_by_pano,
            selected_cases=selected_cases,
            config=config,
        )
        if args.delay > 0 and index < len(selected_cases):
            time.sleep(args.delay)

    final_records = [
        records_by_pano[str(case["pano_id"])]
        for case in selected_cases
        if str(case["pano_id"]) in records_by_pano
    ]
    summary = build_summary(final_records, selected_cases, config)
    write_json(output_dir / "summary.json", summary)
    print(render_json(summary), flush=True)
    return 0 if summary["completed_count"] == summary["selected_count"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
