#!/usr/bin/env python3
"""Time the runtime two-call SpatialAlignmentRefiner on a saved timing batch.

This reuses the batch's panorama manifests and Bayesian Top-5 candidates.  It
does not rerun Visual localization and does not alter the original timing
records.  Spatial refinement is deliberately invoked for every selected case
so its two-call cost can be compared on a fixed representative sample.
"""

from __future__ import annotations

import argparse
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
    GroundingIndex,
    Observation,
    RenderedView,
    SpatialAlignmentRefiner,
    load_dotenv,
    resolve_model_environment,
    resolve_task_num_ctx,
)
from st_nav.cli._common import load_json, load_normalized_artifacts, render_json  # noqa: E402
from st_nav.common.room_profiles import preferred_room_graph_path  # noqa: E402
from st_nav.perception.renderer import normalize_heading  # noqa: E402


DEFAULT_BASE = PROJECT_ROOT / "outputs/localization_timing/representative12_20260818"
DEFAULT_ARTIFACTS = PROJECT_ROOT / "dataset/sites/british_museum/normalized"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--delay", type=float, default=5.0)
    parser.add_argument("--quota-retry-delay", type=float, default=60.0)
    parser.add_argument("--max-quota-retries", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--model")
    parser.add_argument("--api-kind")
    parser.add_argument("--api-base")
    parser.add_argument("--request-timeout", type=float)
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


def quantile(values: list[float], proportion: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * proportion
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def method_summary(records: list[dict[str, Any]], path: tuple[str, ...]) -> dict[str, Any]:
    values = []
    for record in records:
        if record.get("status") != "completed":
            continue
        value: Any = record
        for key in path:
            value = value[key]
        values.append(value)
    totals = [float(value["total_seconds"]) for value in values]
    modules = [float(value["module_seconds"]) for value in values]
    vlms = [float(value["vlm_seconds"]) for value in values]
    return {
        "case_count": len(values),
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


def usage(payload: object) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    raw = payload.get("usage") or payload.get("usageMetadata") or payload.get("usage_metadata")
    return dict(raw) if isinstance(raw, dict) else None


def observation_from_manifest(manifest_path: Path) -> Observation:
    manifest = load_json(manifest_path)
    captures = manifest.get("captures")
    if not isinstance(captures, list) or len(captures) != 8:
        raise ValueError(f"Expected eight captures in {manifest_path}")
    views = []
    for index, capture in enumerate(captures):
        image_path = Path(str(capture["path"]))
        if not image_path.is_absolute():
            image_path = (manifest_path.parent / image_path).resolve()
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        views.append(
            RenderedView(
                label=f"view_{index}",
                heading=normalize_heading(float(capture["heading"])),
                path=str(image_path),
            )
        )
    return Observation(
        pano_id=str(manifest["pano_id"]),
        views=views,
        metadata={
            "floor": manifest.get("floor"),
            "heading_mode": manifest.get("heading_mode"),
            "manifest_path": str(manifest_path),
        },
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


def build_summary(
    *,
    base_summary: dict[str, Any],
    base_records: list[dict[str, Any]],
    records: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    completed = [record for record in records if record.get("status") == "completed"]
    refinement = method_summary(records, ("timing", "spatial_refinement_two_stage"))
    context_values = [
        float(record["timing"]["spatial_refinement_two_stage"]["context_extraction_vlm_seconds"])
        for record in completed
    ]
    alignment_values = [
        float(record["timing"]["spatial_refinement_two_stage"]["alignment_vlm_seconds"])
        for record in completed
    ]
    refinement["mean_context_extraction_vlm_seconds"] = (
        statistics.fmean(context_values) if context_values else None
    )
    refinement["mean_alignment_vlm_seconds"] = (
        statistics.fmean(alignment_values) if alignment_values else None
    )
    token_summary: dict[str, dict[str, float | int | None]] = {}
    for stage in ("context_extraction", "alignment"):
        stage_summary: dict[str, float | int | None] = {}
        for field in ("promptTokenCount", "candidatesTokenCount", "totalTokenCount"):
            values = [
                float(value)
                for record in completed
                if isinstance(
                    value := (record.get("usage", {}).get(stage) or {}).get(field),
                    (int, float),
                )
            ]
            stage_summary[f"mean_{field}"] = statistics.fmean(values) if values else None
        token_summary[stage] = stage_summary
    visual = dict(base_summary["methods"]["visual_only"])
    cumulative_bayesian = dict(base_summary["methods"]["bayesian_filter"])
    bayesian_incremental = [
        float(record["timing"]["bayesian_filter"]["bayesian_incremental_seconds"])
        for record in base_records
        if record.get("status") == "completed"
    ]
    bayesian = {
        "case_count": len(bayesian_incremental),
        "mean_total_seconds": statistics.fmean(bayesian_incremental) if bayesian_incremental else None,
        "median_total_seconds": statistics.median(bayesian_incremental) if bayesian_incremental else None,
        "p25_total_seconds": quantile(bayesian_incremental, 0.25) if bayesian_incremental else None,
        "p75_total_seconds": quantile(bayesian_incremental, 0.75) if bayesian_incremental else None,
        "mean_module_seconds": statistics.fmean(bayesian_incremental) if bayesian_incremental else None,
        "mean_vlm_seconds": 0.0,
        "vlm_share_of_mean_total": 0.0,
        "logical_vlm_call_count": 0,
    }
    full_total = None
    full_module = None
    full_vlm = None
    if refinement["mean_total_seconds"] is not None:
        full_total = float(cumulative_bayesian["mean_total_seconds"]) + float(refinement["mean_total_seconds"])
        full_module = float(cumulative_bayesian["mean_module_seconds"]) + float(refinement["mean_module_seconds"])
        full_vlm = float(visual["mean_vlm_seconds"]) + float(refinement["mean_vlm_seconds"])
    return {
        "schema": "st_nav.runtime_spatial_refinement_timing_benchmark",
        "schema_version": 1,
        "config": config,
        "selected_count": len(records),
        "completed_count": len(completed),
        "pending_or_error_count": len(records) - len(completed),
        "selection_counts": base_summary.get("selection_counts"),
        "methods": {
            "visual_only": visual,
            "bayesian_filter": bayesian,
            "spatial_refinement_two_stage": refinement,
        },
        "cumulative_pipeline_estimates": {
            "visual_plus_bayesian_filter": cumulative_bayesian,
            "visual_plus_bayesian_plus_spatial_refinement": {
                "definition": "Visual Only plus Bayesian update plus two-call runtime Spatial refinement; sum of stage means.",
                "mean_total_seconds": full_total,
                "mean_module_seconds": full_module,
                "mean_vlm_seconds": full_vlm,
                "logical_vlm_call_count": 3,
            },
        },
        "spatial_refinement_token_usage": token_summary,
        "definitions": {
            "visual_only": "Fresh uncached visual detector from eight panorama views.",
            "bayesian_filter": "Standalone local Bayesian transition update only; no VLM call.",
            "spatial_refinement_two_stage": (
                "Runtime SpatialAlignmentRefiner.refine forced on the same 12 cases with Bayesian Top-5: "
                "(1) eight-image context extraction VLM and (2) text alignment VLM."
            ),
            "module_seconds": "Wall time outside ModelResponseClient.create.",
            "vlm_seconds": "Wall time inside the two successful ModelResponseClient.create calls.",
            "quota_waits": "External HTTP 429 wait intervals and failed full attempts are excluded.",
        },
        "records_path": str(config["results_path"]),
    }


def checkpoint(
    *,
    output_dir: Path,
    ordered_pano_ids: list[str],
    records_by_pano: dict[str, dict[str, Any]],
    base_summary: dict[str, Any],
    base_records: list[dict[str, Any]],
    config: dict[str, Any],
) -> None:
    records = [records_by_pano[pano_id] for pano_id in ordered_pano_ids if pano_id in records_by_pano]
    write_jsonl(output_dir / "results.jsonl", records)
    write_json(
        output_dir / "summary.json",
        build_summary(
            base_summary=base_summary,
            base_records=base_records,
            records=records,
            config=config,
        ),
    )


def main() -> int:
    args = parse_args()
    base_dir = args.base_dir.resolve()
    output_dir = (args.output_dir or (base_dir / "runtime_two_stage")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    base_records = read_jsonl(base_dir / "results.jsonl")
    base_summary = load_json(base_dir / "summary.json")
    ordered_pano_ids = [str(record["pano_id"]) for record in base_records]

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
    if not model_env.api_key or not api_base:
        raise RuntimeError("The active model profile is not fully configured.")

    artifacts = load_normalized_artifacts(
        str(args.artifacts_dir),
        room_graph=True,
        pano_graph=False,
        grounding=True,
    )
    room_graph = load_json(preferred_room_graph_path(args.artifacts_dir))
    grounding_index = GroundingIndex(artifacts.grounding or {})
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
    vlm_durations: list[float] = []
    refiner.model_client.create = timed_create_wrapper(refiner.model_client.create, vlm_durations)

    results_path = output_dir / "results.jsonl"
    existing = read_jsonl(results_path) if args.resume and results_path.is_file() else []
    records_by_pano = {str(record["pano_id"]): record for record in existing}
    config = {
        "base_timing_dir": str(base_dir),
        "output_dir": str(output_dir),
        "results_path": str(results_path),
        "model": model,
        "provider": model_env.provider,
        "api_kind": api_kind,
        "request_timeout_seconds": request_timeout,
        "candidate_policy": "force cached Bayesian Top-5 on every selected case",
        "execution_policy": "sequential",
    }
    pending_seen = 0
    for index, base_record in enumerate(base_records, start=1):
        pano_id = str(base_record["pano_id"])
        existing_record = records_by_pano.get(pano_id)
        if existing_record is not None and existing_record.get("status") == "completed":
            print(f"[{index}/{len(base_records)}] {pano_id}: completed (resume)", flush=True)
            continue
        if args.limit is not None and pending_seen >= args.limit:
            continue
        pending_seen += 1
        record: dict[str, Any] = {
            "pano_id": pano_id,
            "expected_room_id": base_record.get("expected_room_id"),
            "selection_category": base_record.get("selection_category"),
            "candidate_room_ids": list(base_record["cached_bayesian_top5_for_spatial"]),
            "status": "pending",
        }
        try:
            manifest_path = base_dir / "cases" / pano_id / f"{pano_id}_manifest.json"
            observation = observation_from_manifest(manifest_path)
            retries = 0
            while True:
                start_index = len(vlm_durations)
                started = time.perf_counter()
                try:
                    refinement = refiner.refine(
                        observation=observation,
                        candidate_room_ids=record["candidate_room_ids"],
                    )
                except RuntimeError as exc:
                    if (
                        "HTTP 429" not in str(exc)
                        or retries >= args.max_quota_retries
                        or args.quota_retry_delay <= 0
                    ):
                        raise
                    retries += 1
                    print(
                        f"[{index}/{len(base_records)}] {pano_id}: HTTP 429; "
                        f"retry {retries}/{args.max_quota_retries} in {args.quota_retry_delay:.0f}s",
                        flush=True,
                    )
                    time.sleep(args.quota_retry_delay)
                    continue
                total_seconds = time.perf_counter() - started
                successful_durations = vlm_durations[start_index:]
                break
            if not refinement.get("applied"):
                raise RuntimeError(f"Spatial refinement not applied: {refinement}")
            if len(successful_durations) != 2:
                raise RuntimeError(
                    f"Expected two successful logical VLM calls; observed {len(successful_durations)}"
                )
            vlm_seconds = sum(successful_durations)
            module_seconds = max(0.0, total_seconds - vlm_seconds)
            record.update(
                {
                    "status": "completed",
                    "predicted_room_id": refinement.get("alignment_predicted_room_id"),
                    "timing": {
                        "spatial_refinement_two_stage": {
                            "module_seconds": module_seconds,
                            "context_extraction_vlm_seconds": successful_durations[0],
                            "alignment_vlm_seconds": successful_durations[1],
                            "vlm_seconds": vlm_seconds,
                            "total_seconds": total_seconds,
                            "logical_vlm_call_count": 2,
                            "quota_retry_count": retries,
                        }
                    },
                    "usage": {
                        "context_extraction": usage(refiner.last_extraction_response_payload),
                        "alignment": usage(refiner.last_alignment_response_payload),
                    },
                    "error": None,
                }
            )
            write_json(output_dir / "cases" / pano_id / "result.json", record)
            print(
                f"[{index}/{len(base_records)}] {pano_id}: done total={total_seconds:.3f}s "
                f"context={successful_durations[0]:.3f}s alignment={successful_durations[1]:.3f}s "
                f"module={module_seconds:.3f}s",
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
            print(f"[{index}/{len(base_records)}] {pano_id}: ERROR {exc}", flush=True)
        records_by_pano[pano_id] = record
        checkpoint(
            output_dir=output_dir,
            ordered_pano_ids=ordered_pano_ids,
            records_by_pano=records_by_pano,
            base_summary=base_summary,
            base_records=base_records,
            config=config,
        )
        if args.delay > 0:
            time.sleep(args.delay)

    checkpoint(
        output_dir=output_dir,
        ordered_pano_ids=ordered_pano_ids,
        records_by_pano=records_by_pano,
        base_summary=base_summary,
        base_records=base_records,
        config=config,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
