#!/usr/bin/env python3
"""Benchmark the three-stage localization pipeline on a fixed case selection.

The runner performs a fresh Visual localization request, a local Bayesian
update, and the runtime two-call Spatial refinement for every selected case.
It records ranked predictions, model usage, and program/VLM wall time for each
stage so accuracy and cumulative latency charts can be generated from one
internally consistent run.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import statistics
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from st_nav import (  # noqa: E402
    EvidenceScoreLocalizer,
    GroundingIndex,
    PerceptionPipeline,
    SpatialAlignmentRefiner,
    ViewDetector,
    load_dotenv,
    resolve_model_environment,
    resolve_task_num_ctx,
)
from st_nav.cli._common import load_json, load_normalized_artifacts, render_json  # noqa: E402
from st_nav.common.room_profiles import preferred_room_graph_path  # noqa: E402


DEFAULT_SELECTION = (
    PROJECT_ROOT
    / "outputs/localization_timing/representative12_20260818/selection.json"
)
DEFAULT_SOURCE = (
    PROJECT_ROOT
    / "outputs/localization_evaluation/proportional_gallery_three_stage_gemini/summary.json"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "outputs/localization_timing/representative12_gpt-5.4-nano_20260820"
)
DEFAULT_ARTIFACTS = PROJECT_ROOT / "dataset/sites/british_museum/normalized"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--source-summary", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--model", default="gpt-5.4-nano")
    parser.add_argument("--reasoning-effort", default="none")
    parser.add_argument("--api-kind")
    parser.add_argument("--api-base")
    parser.add_argument("--request-timeout", type=float)
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--quota-retry-delay", type=float, default=30.0)
    parser.add_argument("--max-quota-retries", type=int, default=3)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_json(payload), encoding="utf-8")


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def copy_manifest(source_path: Path, destination: Path) -> Path:
    manifest = load_json(source_path)
    manifest["manifest_path"] = str(destination)
    write_json(destination, manifest)
    return destination


def rank_distribution(distribution: object) -> list[str]:
    if not isinstance(distribution, dict):
        return []
    ranked = [
        (str(room_id), float(score))
        for room_id, score in distribution.items()
        if isinstance(room_id, str) and isinstance(score, (int, float))
    ]
    return [room_id for room_id, _ in sorted(ranked, key=lambda item: (-item[1], item[0]))]


def rank_spatial(refinement: dict[str, Any], bayesian_ranking: list[str]) -> list[str]:
    ranked: list[str] = []
    for record in refinement.get("alignment_top_k", []):
        if not isinstance(record, dict):
            continue
        room_id = record.get("room_id")
        if isinstance(room_id, str) and room_id not in ranked:
            ranked.append(room_id)
    for room_id in bayesian_ranking:
        if room_id not in ranked:
            ranked.append(room_id)
    return ranked


def hit_rate(records: list[dict[str, Any]], ranking_key: str, k: int) -> float | None:
    completed = [record for record in records if record.get("status") == "completed"]
    if not completed:
        return None
    hits = sum(
        str(record.get("expected_room_id")) in list(record.get(ranking_key, []))[:k]
        for record in completed
    )
    return hits / len(completed)


def quantile(values: list[float], proportion: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * proportion
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summarize_stage(records: list[dict[str, Any]], stage: str) -> dict[str, Any]:
    completed = [record for record in records if record.get("status") == "completed"]
    values = [record["timing"][stage] for record in completed]
    totals = [float(value["total_seconds"]) for value in values]
    modules = [float(value["module_seconds"]) for value in values]
    vlms = [float(value["vlm_seconds"]) for value in values]
    return {
        "case_count": len(values),
        "mean_total_seconds": statistics.fmean(totals) if totals else None,
        "median_total_seconds": statistics.median(totals) if totals else None,
        "p25_total_seconds": quantile(totals, 0.25),
        "p75_total_seconds": quantile(totals, 0.75),
        "mean_module_seconds": statistics.fmean(modules) if modules else None,
        "mean_vlm_seconds": statistics.fmean(vlms) if vlms else None,
    }


def usage(payload: object) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    raw = payload.get("usage") or payload.get("usageMetadata") or payload.get("usage_metadata")
    return copy.deepcopy(raw) if isinstance(raw, dict) else None


def usage_totals(records: list[dict[str, Any]]) -> dict[str, int]:
    aliases = {
        "input_tokens": ("input_tokens", "promptTokenCount"),
        "output_tokens": ("output_tokens", "candidatesTokenCount"),
        "total_tokens": ("total_tokens", "totalTokenCount"),
    }
    totals = {key: 0 for key in aliases}
    for record in records:
        if record.get("status") != "completed":
            continue
        for stage_usage in record.get("usage", {}).values():
            if not isinstance(stage_usage, dict):
                continue
            for normalized, names in aliases.items():
                for name in names:
                    value = stage_usage.get(name)
                    if isinstance(value, (int, float)):
                        totals[normalized] += int(value)
                        break
    return totals


def timed_create_wrapper(
    original: Callable[[dict[str, Any]], dict[str, Any]],
    durations: list[float],
    *,
    reasoning_effort: str,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    def wrapped(request_body: dict[str, Any]) -> dict[str, Any]:
        request_body["reasoning"] = {"effort": reasoning_effort}
        started = time.perf_counter()
        try:
            return original(request_body)
        finally:
            durations.append(time.perf_counter() - started)

    return wrapped


def redact_images(payload: object) -> object:
    cloned = copy.deepcopy(payload)

    def visit(value: object) -> None:
        if isinstance(value, dict):
            if value.get("type") == "input_image" and isinstance(value.get("image_url"), str):
                value["image_url"] = "<IMAGE_DATA_URL_OMITTED>"
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(cloned)
    return cloned


def build_summary(
    records: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    completed = [record for record in records if record.get("status") == "completed"]
    stage_summaries = {
        stage: summarize_stage(records, stage)
        for stage in ("visual", "bayesian", "spatial")
    }
    visual = stage_summaries["visual"]
    bayesian = stage_summaries["bayesian"]
    spatial = stage_summaries["spatial"]

    def add(*values: object) -> float | None:
        if any(not isinstance(value, (int, float)) for value in values):
            return None
        return sum(float(value) for value in values)

    return {
        "schema": "st_nav.openai_three_stage_localization_benchmark",
        "schema_version": 1,
        "config": config,
        "selected_count": len(selected),
        "completed_count": len(completed),
        "pending_or_error_count": len(selected) - len(completed),
        "accuracy": {
            method: {
                f"hit_at_{k}": hit_rate(records, ranking_key, k)
                for k in (1, 3, 5)
            }
            for method, ranking_key in (
                ("visual_only", "visual_ranking"),
                ("visual_plus_bayesian_filter", "bayesian_ranking"),
                ("visual_plus_bayesian_plus_spatial_alignment", "spatial_ranking"),
            )
        },
        "stages": stage_summaries,
        "cumulative_methods": {
            "visual_only": {
                "mean_total_seconds": visual["mean_total_seconds"],
                "mean_program_module_seconds": visual["mean_module_seconds"],
                "mean_vlm_response_seconds": visual["mean_vlm_seconds"],
                "logical_vlm_call_count": 1,
            },
            "visual_plus_bayesian_filter": {
                "mean_total_seconds": add(
                    visual["mean_total_seconds"], bayesian["mean_total_seconds"]
                ),
                "mean_program_module_seconds": add(
                    visual["mean_module_seconds"], bayesian["mean_module_seconds"]
                ),
                "mean_vlm_response_seconds": visual["mean_vlm_seconds"],
                "logical_vlm_call_count": 1,
            },
            "visual_plus_bayesian_plus_spatial_alignment": {
                "mean_total_seconds": add(
                    visual["mean_total_seconds"],
                    bayesian["mean_total_seconds"],
                    spatial["mean_total_seconds"],
                ),
                "mean_program_module_seconds": add(
                    visual["mean_module_seconds"],
                    bayesian["mean_module_seconds"],
                    spatial["mean_module_seconds"],
                ),
                "mean_vlm_response_seconds": add(
                    visual["mean_vlm_seconds"], spatial["mean_vlm_seconds"]
                ),
                "logical_vlm_call_count": 3,
            },
        },
        "usage_totals": usage_totals(records),
        "definitions": {
            "sample": "Fixed 12-case representative selection from the prior timing benchmark.",
            "visual_only": "Fresh uncached Visual localization from eight 512x512 panorama views.",
            "bayesian": "Local transition-support update applied to the new Visual distribution; no VLM call.",
            "spatial": "Two-call Spatial refinement: image context extraction followed by text alignment.",
            "program_module_seconds": "Wall time outside ModelResponseClient.create.",
            "vlm_response_seconds": "Wall time inside ModelResponseClient.create, including HTTP and response transfer.",
        },
        "records_path": str(config["results_path"]),
    }


def checkpoint(
    *,
    output_dir: Path,
    selected: list[dict[str, Any]],
    records_by_pano: dict[str, dict[str, Any]],
    config: dict[str, Any],
) -> None:
    records = [
        records_by_pano[str(item["pano_id"])]
        for item in selected
        if str(item["pano_id"]) in records_by_pano
    ]
    write_jsonl(output_dir / "results.jsonl", records)
    write_json(output_dir / "summary.json", build_summary(records, selected, config))


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    selection_payload = json.loads(args.selection.resolve().read_text(encoding="utf-8"))
    if not isinstance(selection_payload, list):
        raise ValueError(f"Expected a JSON array in {args.selection.resolve()}")
    selected = [item for item in selection_payload if isinstance(item, dict)]
    source_payload = load_json(args.source_summary.resolve())
    source_by_pano = {
        str(item["pano_id"]): item
        for item in source_payload.get("results", [])
        if isinstance(item, dict) and isinstance(item.get("pano_id"), str)
    }
    missing = [str(item.get("pano_id")) for item in selected if str(item.get("pano_id")) not in source_by_pano]
    if missing:
        raise ValueError(f"Selected panos missing from source summary: {', '.join(missing)}")
    write_json(output_dir / "selection.json", selected)

    load_dotenv(PROJECT_ROOT / ".env")
    model_env = resolve_model_environment(
        default_model=args.model,
        default_api_base="https://api.openai.com/v1",
        default_api_kind="responses",
    )
    model = args.model or model_env.model_name or "gpt-5.4-nano"
    api_base = args.api_base or model_env.api_base or "https://api.openai.com/v1"
    api_kind = args.api_kind or model_env.api_kind or "responses"
    request_timeout = float(args.request_timeout or model_env.request_timeout or 180.0)
    if model_env.provider != "openai":
        raise RuntimeError(
            f"Expected the OpenAI model profile; resolved provider={model_env.provider!r}."
        )
    if not model_env.api_key:
        raise RuntimeError("The OpenAI model profile does not have an API key.")

    artifacts = load_normalized_artifacts(
        str(args.artifacts_dir.resolve()),
        room_graph=True,
        pano_graph=True,
        grounding=True,
    )
    room_graph = load_json(preferred_room_graph_path(args.artifacts_dir.resolve()))
    grounding_index = GroundingIndex(artifacts.grounding or {})
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
        pano_graph=artifacts.pano_graph or {},
        room_graph=room_graph,
        grounding_index=grounding_index,
        detector=detector,
    )
    refiner = SpatialAlignmentRefiner(
        room_graph=room_graph,
        grounding_index=grounding_index,
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
    )
    localizer = EvidenceScoreLocalizer(
        room_graph=room_graph,
        grounding_index=grounding_index,
        spatial_refiner=None,
    )
    visual_vlm_durations: list[float] = []
    spatial_vlm_durations: list[float] = []
    detector.model_client.create = timed_create_wrapper(
        detector.model_client.create,
        visual_vlm_durations,
        reasoning_effort=args.reasoning_effort,
    )
    refiner.model_client.create = timed_create_wrapper(
        refiner.model_client.create,
        spatial_vlm_durations,
        reasoning_effort=args.reasoning_effort,
    )

    results_path = output_dir / "results.jsonl"
    existing = read_jsonl(results_path) if args.resume else []
    records_by_pano = {
        str(record["pano_id"]): record
        for record in existing
        if isinstance(record.get("pano_id"), str)
    }
    config = {
        "selection_path": str(args.selection.resolve()),
        "source_summary": str(args.source_summary.resolve()),
        "output_dir": str(output_dir),
        "results_path": str(results_path),
        "model": model,
        "provider": model_env.provider,
        "api_base": api_base,
        "api_kind": api_kind,
        "reasoning_effort": args.reasoning_effort,
        "request_timeout_seconds": request_timeout,
        "execution_policy": "sequential",
        "cache_policy": "fresh Visual call in isolated output; existing repository cache untouched",
    }

    pending_seen = 0
    for index, selection_record in enumerate(selected, start=1):
        pano_id = str(selection_record["pano_id"])
        if records_by_pano.get(pano_id, {}).get("status") == "completed":
            print(f"[{index}/{len(selected)}] {pano_id}: completed (resume)", flush=True)
            continue
        if args.limit is not None and pending_seen >= args.limit:
            continue
        pending_seen += 1
        source = source_by_pano[pano_id]
        record: dict[str, Any] = {
            "pano_id": pano_id,
            "expected_room_id": selection_record.get("expected_room_id"),
            "selection_category": selection_record.get("selection_category"),
            "status": "pending",
        }
        case_dir = output_dir / "cases" / pano_id
        case_dir.mkdir(parents=True, exist_ok=True)
        try:
            isolated_manifest = copy_manifest(
                Path(str(source["manifest_path"])).resolve(),
                case_dir / f"{pano_id}_manifest.json",
            )
            source_output = load_json(Path(str(source["output_path"])).resolve())
            current_heading = float(source_output.get("current_heading", 330.0))

            visual_quota_retries = 0
            while True:
                visual_start_index = len(visual_vlm_durations)
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
                        f"[{index}/{len(selected)}] {pano_id}: Visual HTTP 429; "
                        f"retry {visual_quota_retries}/{args.max_quota_retries}",
                        flush=True,
                    )
                    time.sleep(args.quota_retry_delay)
                    continue
                visual_total = time.perf_counter() - visual_started
                visual_call_durations = visual_vlm_durations[visual_start_index:]
                break
            if len(visual_call_durations) != 1:
                raise RuntimeError(
                    f"Expected one Visual VLM call; observed {len(visual_call_durations)}."
                )
            visual_vlm = sum(visual_call_durations)
            visual_module = max(0.0, visual_total - visual_vlm)

            expected_room_id = str(selection_record["expected_room_id"])
            bayesian_started = time.perf_counter()
            bayesian_result = localizer.localize(
                observation=observation,
                prior_room_belief={expected_room_id: 1.0},
                fallback_room_id=expected_room_id,
            )
            bayesian_total = time.perf_counter() - bayesian_started
            visual_ranking = rank_distribution(bayesian_result.get("observation_distribution"))
            bayesian_ranking = rank_distribution(bayesian_result.get("room_belief"))
            spatial_candidates = bayesian_ranking[:5]
            if len(spatial_candidates) < 2:
                raise RuntimeError(
                    f"Spatial refinement requires at least two Bayesian candidates; got {spatial_candidates}."
                )

            spatial_quota_retries = 0
            while True:
                spatial_start_index = len(spatial_vlm_durations)
                spatial_started = time.perf_counter()
                try:
                    refinement = refiner.refine(
                        observation=observation,
                        candidate_room_ids=spatial_candidates,
                    )
                except RuntimeError as exc:
                    if (
                        "HTTP 429" not in str(exc)
                        or spatial_quota_retries >= args.max_quota_retries
                        or args.quota_retry_delay <= 0
                    ):
                        raise
                    spatial_quota_retries += 1
                    print(
                        f"[{index}/{len(selected)}] {pano_id}: Spatial HTTP 429; "
                        f"retry {spatial_quota_retries}/{args.max_quota_retries}",
                        flush=True,
                    )
                    time.sleep(args.quota_retry_delay)
                    continue
                spatial_total = time.perf_counter() - spatial_started
                spatial_call_durations = spatial_vlm_durations[spatial_start_index:]
                break
            if not refinement.get("applied"):
                raise RuntimeError(f"Spatial refinement was not applied: {refinement}")
            if len(spatial_call_durations) != 2:
                raise RuntimeError(
                    f"Expected two Spatial VLM calls; observed {len(spatial_call_durations)}."
                )
            spatial_vlm = sum(spatial_call_durations)
            spatial_module = max(0.0, spatial_total - spatial_vlm)
            spatial_ranking = rank_spatial(refinement, bayesian_ranking)

            visual_trace = detector.last_traces[-1] if detector.last_traces else {}
            visual_response = visual_trace.get("response") if isinstance(visual_trace, dict) else None
            record.update(
                {
                    "status": "completed",
                    "visual_ranking": visual_ranking,
                    "bayesian_ranking": bayesian_ranking,
                    "spatial_candidate_room_ids": spatial_candidates,
                    "spatial_ranking": spatial_ranking,
                    "predictions": {
                        "visual_only": visual_ranking[0] if visual_ranking else None,
                        "visual_plus_bayesian_filter": bayesian_ranking[0] if bayesian_ranking else None,
                        "visual_plus_bayesian_plus_spatial_alignment": (
                            spatial_ranking[0] if spatial_ranking else None
                        ),
                    },
                    "timing": {
                        "visual": {
                            "module_seconds": visual_module,
                            "vlm_seconds": visual_vlm,
                            "total_seconds": visual_total,
                            "logical_vlm_call_count": 1,
                            "quota_retry_count": visual_quota_retries,
                        },
                        "bayesian": {
                            "module_seconds": bayesian_total,
                            "vlm_seconds": 0.0,
                            "total_seconds": bayesian_total,
                            "logical_vlm_call_count": 0,
                        },
                        "spatial": {
                            "module_seconds": spatial_module,
                            "context_extraction_vlm_seconds": spatial_call_durations[0],
                            "alignment_vlm_seconds": spatial_call_durations[1],
                            "vlm_seconds": spatial_vlm,
                            "total_seconds": spatial_total,
                            "logical_vlm_call_count": 2,
                            "quota_retry_count": spatial_quota_retries,
                        },
                    },
                    "usage": {
                        "visual": usage(visual_response),
                        "spatial_context_extraction": usage(
                            refiner.last_extraction_response_payload
                        ),
                        "spatial_alignment": usage(refiner.last_alignment_response_payload),
                    },
                    "error": None,
                }
            )
            write_json(
                case_dir / "model_trace.json",
                {
                    "visual_request": visual_trace.get("request") if isinstance(visual_trace, dict) else None,
                    "visual_response": visual_response,
                    "spatial_context_request": redact_images(refiner.last_extraction_request_body),
                    "spatial_context_response": refiner.last_extraction_response_payload,
                    "spatial_alignment_request": refiner.last_alignment_request_body,
                    "spatial_alignment_response": refiner.last_alignment_response_payload,
                },
            )
            write_json(case_dir / "result.json", record)
            print(
                f"[{index}/{len(selected)}] {pano_id}: done "
                f"visual={visual_total:.3f}s spatial={spatial_total:.3f}s "
                f"top1={record['predictions']['visual_plus_bayesian_plus_spatial_alignment']}",
                flush=True,
            )
        except Exception as exc:
            record.update(
                {
                    "status": "error",
                    "error": {
                        "type": type(exc).__name__,
                        "message": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                }
            )
            write_json(case_dir / "result.json", record)
            print(
                f"[{index}/{len(selected)}] {pano_id}: ERROR {type(exc).__name__}: {exc}",
                flush=True,
            )
        records_by_pano[pano_id] = record
        checkpoint(
            output_dir=output_dir,
            selected=selected,
            records_by_pano=records_by_pano,
            config=config,
        )
        if args.delay > 0 and index < len(selected):
            time.sleep(args.delay)

    checkpoint(
        output_dir=output_dir,
        selected=selected,
        records_by_pano=records_by_pano,
        config=config,
    )
    summary = load_json(output_dir / "summary.json")
    print(render_json(summary), flush=True)
    return 0 if summary["completed_count"] == summary["selected_count"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
