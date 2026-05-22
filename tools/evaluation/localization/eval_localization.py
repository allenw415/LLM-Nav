from __future__ import annotations

import argparse
import random
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from st_nav.cli._common import (
    PROJECT_ROOT,
    ensure_project_root_on_path,
    load_json,
    load_normalized_artifacts,
    render_json,
    resolve_project_path,
    write_text_if_requested,
)

ensure_project_root_on_path()

from st_nav import (
    EvidenceScoreLocalizer,
    EntityDetection,
    GroundingIndex,
    Observation,
    RenderedView,
    SpatialAlignmentRefiner,
    load_dotenv,
    resolve_model_environment,
    resolve_task_num_ctx,
)

load_dotenv(PROJECT_ROOT / ".env")
MODEL_ENV = resolve_model_environment(
    default_model="gpt-5-mini",
    default_api_base="https://api.openai.com/v1",
    default_api_kind="responses",
)
DEFAULT_LLM_NUM_CTX = resolve_task_num_ctx(
    "localization",
    fallback_num_ctx=MODEL_ENV.num_ctx,
    default_num_ctx=16384,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate integrated pano perception localization by sampling up to N grounded panoramas per room "
            "and comparing visual_localization.predicted_room_id against pano_room_grounding.json."
        )
    )
    parser.add_argument(
        "--pipeline",
        choices=["integrated-visual"],
        default="integrated-visual",
        help="integrated-visual scores visual_localization from run_pano_perception.py.",
    )
    parser.add_argument("--artifacts-dir", default="dataset/sites/british_museum/normalized")
    parser.add_argument("--grounding-path")
    parser.add_argument("--samples-per-room", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--room-id", action="append", default=[], help="Restrict evaluation to specific room ids. Repeatable.")
    parser.add_argument("--max-total", type=int, default=None, help="Optional cap on total sampled panoramas.")
    parser.add_argument("--output-dir", default="outputs/pano_perception_grounding_eval")
    parser.add_argument("--summary-output-path")
    parser.add_argument("--force", action="store_true", help="Rerun perception even when a per-pano output JSON exists.")
    parser.add_argument(
        "--reuse-existing-output",
        action="store_true",
        help=(
            "Score existing per-pano output JSON files. If spatial alignment needs view images and the render "
            "manifest is missing, rerun run_pano_perception.py for that pano to regenerate the render artifacts."
        ),
    )
    parser.add_argument("--render-output-dir", default="renders/pano_perception_grounding_eval")
    parser.add_argument("--render-api-key")
    parser.add_argument("--llm-api-key")
    parser.add_argument("--llm-num-ctx", type=int, default=DEFAULT_LLM_NUM_CTX)
    parser.add_argument("--detector-model", default="gemma-4-31b-it")
    parser.add_argument("--detector-api-kind")
    parser.add_argument("--detector-api-base")
    parser.add_argument("--vlm-timeout", type=float)
    parser.add_argument("--heading-mode", choices=["museum", "cardinal", "graph"], default="museum")
    parser.add_argument("--pitch", type=float, default=0.0)
    parser.add_argument("--fov", type=int, default=45)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--current-heading", type=float, default=330.0)
    parser.add_argument(
        "--no-detection-cache",
        action="store_true",
        help="Pass --no-detection-cache to run_pano_perception.py so the VLM detector is called again.",
    )
    parser.add_argument(
        "--enable-view-themes",
        action="store_true",
        help="Pass --enable-view-themes to run_pano_perception.py when perception is rerun.",
    )
    parser.add_argument("--enable-spatial-alignment", action="store_true")
    parser.add_argument("--alignment-candidate-ratio-threshold", type=float, default=0.5)
    parser.add_argument("--alignment-candidate-max", type=int, default=5)
    parser.add_argument("--alignment-model")
    parser.add_argument("--alignment-timeout", type=float)
    parser.add_argument("--print-failures", action="store_true")
    parser.add_argument("--delay", type=float, default=0, help="Seconds to wait between panorama samples to avoid API rate limits.")
    return parser


def room_samples_from_grounding(
    grounding_payload: dict,
    *,
    room_ids: set[str] | None,
    samples_per_room: int,
    seed: int,
    max_total: int | None = None,
) -> list[dict]:
    mappings = grounding_payload.get("mappings", grounding_payload)
    if not isinstance(mappings, dict):
        raise ValueError("Expected grounding payload to include a mappings object.")

    by_room: dict[str, list[str]] = defaultdict(list)
    for pano_id, room_id in mappings.items():
        if not isinstance(pano_id, str) or not pano_id:
            continue
        if not isinstance(room_id, str) or not room_id or room_id == "null":
            continue
        if room_ids is not None and room_id not in room_ids:
            continue
        by_room[room_id].append(pano_id)

    rng = random.Random(seed)
    records = []
    for room_id in sorted(by_room):
        pano_ids = sorted(by_room[room_id])
        rng.shuffle(pano_ids)
        for pano_id in pano_ids[: max(samples_per_room, 0)]:
            records.append({"room_id": room_id, "pano_id": pano_id})

    if max_total is not None:
        records = records[: max(max_total, 0)]
    return records


def room_samples_from_output_dir(
    output_dir: Path,
    grounding_payload: dict,
    *,
    room_ids: set[str] | None,
    samples_per_room: int,
    max_total: int | None = None,
) -> list[dict]:
    """Build sample list from existing per-pano output JSON files on disk.

    This avoids RNG-dependent sampling so that ``--reuse-existing-output`` is
    not affected by changes to the grounding file (e.g. new rooms added).
    """
    mappings = grounding_payload.get("mappings", grounding_payload)
    pano_to_room: dict[str, str] = {}
    if isinstance(mappings, dict):
        for pano_id, room_id in mappings.items():
            if isinstance(pano_id, str) and pano_id and isinstance(room_id, str) and room_id and room_id != "null":
                pano_to_room[pano_id] = room_id

    by_room: dict[str, list[str]] = defaultdict(list)
    if output_dir.exists():
        for room_dir in sorted(output_dir.iterdir()):
            if not room_dir.is_dir():
                continue
            room_slug = room_dir.name
            for pano_file in sorted(room_dir.iterdir()):
                if pano_file.suffix != ".json":
                    continue
                pano_id = pano_file.stem
                # Determine room_id: prefer grounding mapping, fall back to directory name
                room_id = pano_to_room.get(pano_id)
                if room_id is None:
                    room_id = room_slug.replace("_", " ")
                if room_ids is not None and room_id not in room_ids:
                    continue
                by_room[room_id].append(pano_id)

    records = []
    for room_id in sorted(by_room):
        pano_ids = by_room[room_id][: max(samples_per_room, 0)]
        for pano_id in pano_ids:
            records.append({"room_id": room_id, "pano_id": pano_id})

    if max_total is not None:
        records = records[: max(max_total, 0)]
    return records


def output_path_for_sample(output_dir: Path, sample: dict) -> Path:
    room_slug = str(sample["room_id"]).replace(" ", "_").replace("/", "_")
    pano_slug = str(sample["pano_id"]).replace("/", "_")
    return output_dir / room_slug / f"{pano_slug}.json"


def prediction_from_perception_payload(payload: dict) -> tuple[str | None, float | None]:
    visual_localization = payload.get("visual_localization")
    if not isinstance(visual_localization, dict):
        return None, None
    predicted_room_id = visual_localization.get("predicted_room_id")
    confidence = visual_localization.get("confidence")
    return (
        predicted_room_id if isinstance(predicted_room_id, str) and predicted_room_id else None,
        float(confidence) if isinstance(confidence, (int, float)) else None,
    )


def observation_from_perception_payload(
    payload: dict,
    *,
    include_visual_localization: bool = False,
    manifest_path: Path | None = None,
) -> Observation:
    pano_id = payload.get("pano_id")
    if not isinstance(pano_id, str) or not pano_id:
        raise RuntimeError("Perception payload missing pano_id.")

    entities: list[EntityDetection] = []
    for record in payload.get("entities", []):
        if not isinstance(record, dict):
            continue
        name = record.get("name")
        if not isinstance(name, str) or not name:
            continue
        kind = record.get("kind")
        confidence = record.get("confidence")
        source_views = record.get("source_views")
        normalized_source_views = [
            value
            for value in source_views
            if isinstance(value, str) and value
        ] if isinstance(source_views, list) else []
        location_scope = record.get("location_scope")
        if location_scope not in {"inside", "outside", "unknown"}:
            location_scope = "inside"
        entities.append(
            EntityDetection(
                name=name,
                confidence=float(confidence) if isinstance(confidence, (int, float)) else 0.0,
                kind=kind if isinstance(kind, str) and kind else "other",
                source_view=normalized_source_views[0] if len(normalized_source_views) == 1 else "multiview",
                location_scope=location_scope,
                metadata={
                    "source_views": normalized_source_views,
                    "location_scope": location_scope,
                },
            )
        )

    metadata = {
        "floor": payload.get("floor"),
        "lat": payload.get("lat"),
        "lng": payload.get("lng"),
        "source": "perception-json" if include_visual_localization else "entity-perception-json",
    }
    if include_visual_localization:
        visual_localization = payload.get("visual_localization")
        if isinstance(visual_localization, dict):
            metadata["visual_localization"] = dict(visual_localization)
        candidate_room_ids = payload.get("candidate_room_ids")
        if isinstance(candidate_room_ids, list):
            metadata["candidate_room_ids"] = [value for value in candidate_room_ids if isinstance(value, str)]

    views: list[RenderedView] = []
    if manifest_path is not None:
        try:
            captures = load_manifest_captures(payload, fallback_manifest_path=manifest_path)
        except (FileNotFoundError, RuntimeError, OSError, ValueError):
            captures = []
        for index, capture in enumerate(captures):
            raw_heading = capture.get("heading")
            raw_url = capture.get("url")
            views.append(
                RenderedView(
                    label=str(capture.get("label") or f"view_{index}"),
                    heading=float(raw_heading) if isinstance(raw_heading, (int, float)) else 0.0,
                    path=str(capture["path"]),
                    url=raw_url if isinstance(raw_url, str) and raw_url else None,
                )
            )

    return Observation(
        pano_id=pano_id,
        entities=entities,
        views=views,
        heading_estimate=(
            float(payload["current_heading"])
            if isinstance(payload.get("current_heading"), (int, float))
            else None
        ),
        metadata=metadata,
    )


def ranked_room_ids(distribution: object) -> list[str]:
    if not isinstance(distribution, dict):
        return []
    candidates = [
        (room_id, float(score))
        for room_id, score in distribution.items()
        if isinstance(room_id, str) and isinstance(score, (int, float))
    ]
    return [
        room_id
        for room_id, _ in sorted(candidates, key=lambda item: (-item[1], item[0]))
    ]


def best_room_from_distribution(distribution: dict[str, float]) -> tuple[str | None, float | None]:
    candidates = [
        (room_id, float(score))
        for room_id, score in distribution.items()
        if isinstance(room_id, str) and isinstance(score, (int, float))
    ]
    if not candidates:
        return None, None
    room_id, score = sorted(candidates, key=lambda item: (-item[1], item[0]))[0]
    return room_id, score


def compact_distribution(distribution: object, *, top_k: int = 10) -> dict[str, float]:
    if not isinstance(distribution, dict):
        return {}
    ordered = [
        (room_id, float(score))
        for room_id, score in distribution.items()
        if isinstance(room_id, str) and isinstance(score, (int, float))
    ]
    ordered = sorted(ordered, key=lambda item: (-item[1], item[0]))
    return {room_id: score for room_id, score in ordered[: max(top_k, 0)]}


def ranking_payload(distribution: object, expected_room_id: str) -> dict:
    ranking = ranked_room_ids(distribution)
    rank = ranking.index(expected_room_id) + 1 if expected_room_id in ranking else None
    return {
        "rank": rank,
        "top1": ranking[0] if ranking else None,
        "top3": ranking[:3],
        "top5": ranking[:5],
    }


def ranking_payload_from_order(ranking: list[str], expected_room_id: str) -> dict:
    rank = ranking.index(expected_room_id) + 1 if expected_room_id in ranking else None
    return {
        "rank": rank,
        "top1": ranking[0] if ranking else None,
        "top3": ranking[:3],
        "top5": ranking[:5],
    }



def load_manifest_captures(payload: dict, *, fallback_manifest_path: Path | None = None) -> list[dict]:
    manifest_path = payload.get("manifest_path")
    resolved_manifest_path = Path(manifest_path) if isinstance(manifest_path, str) and manifest_path else None
    if resolved_manifest_path is None or not resolved_manifest_path.exists():
        if fallback_manifest_path is not None and fallback_manifest_path.exists():
            resolved_manifest_path = fallback_manifest_path
        elif resolved_manifest_path is None:
            raise RuntimeError("Perception payload missing manifest_path for spatial alignment.")
        else:
            raise FileNotFoundError(resolved_manifest_path)
    manifest = load_json(resolved_manifest_path)
    captures = [
        capture
        for capture in manifest.get("captures", [])
        if isinstance(capture, dict) and isinstance(capture.get("path"), str) and capture.get("path")
    ]
    if not captures:
        raise RuntimeError(f"Manifest has no capture images for spatial alignment: {resolved_manifest_path}")
    return captures


def manifest_path_candidates(args: argparse.Namespace, *, sample: dict, payload: dict) -> list[Path]:
    candidates: list[Path] = []
    manifest_path = payload.get("manifest_path")
    if isinstance(manifest_path, str) and manifest_path:
        candidates.append(Path(manifest_path))
    pano_id = str(payload.get("pano_id") or sample.get("pano_id"))
    render_output_dir = getattr(args, "render_output_dir", "renders/pano_perception_grounding_eval")
    if pano_id and render_output_dir:
        candidates.append(resolve_project_path(render_output_dir) / pano_id / f"{pano_id}_manifest.json")
    unique: list[Path] = []
    for candidate in candidates:
        if candidate not in unique:
            unique.append(candidate)
    return unique


def has_available_manifest(args: argparse.Namespace, *, sample: dict, payload: dict) -> bool:
    return any(candidate.exists() for candidate in manifest_path_candidates(args, sample=sample, payload=payload))


def first_available_manifest_path(args: argparse.Namespace, *, sample: dict, payload: dict) -> Path | None:
    for candidate in manifest_path_candidates(args, sample=sample, payload=payload):
        if candidate.exists():
            return candidate
    return None


def spatial_ranking_from_localization(localization_payload: dict, base_room_belief: object) -> list[str]:
    seen: set[str] = set()
    ranking: list[str] = []
    alignment_top_k = localization_payload.get("alignment_top_k")
    if isinstance(alignment_top_k, list):
        for record in alignment_top_k:
            if not isinstance(record, dict):
                continue
            room_id = record.get("room_id")
            if isinstance(room_id, str) and room_id and room_id not in seen:
                ranking.append(room_id)
                seen.add(room_id)
    for room_id in ranked_room_ids(base_room_belief):
        if room_id not in seen:
            ranking.append(room_id)
            seen.add(room_id)
    return ranking



def rank_metrics(records: list[dict], key: str) -> dict:
    scored = [record for record in records if record.get("status") == "scored"]
    ranked_records = [
        record
        for record in scored
        if isinstance(record.get(key), dict) and isinstance(record[key].get("rank"), int)
    ]

    def top_k_accuracy(k: int) -> float | None:
        if not scored:
            return None
        matched = 0
        top_key = f"top{k}"
        for record in scored:
            payload = record.get(key)
            top_rooms = payload.get(top_key) if isinstance(payload, dict) else None
            if isinstance(top_rooms, list) and record.get("expected_room_id") in top_rooms:
                matched += 1
            elif k == 1 and isinstance(payload, dict) and record.get("expected_room_id") == payload.get("top1"):
                matched += 1
        return matched / len(scored)

    reciprocal_ranks = [1.0 / record[key]["rank"] for record in ranked_records]
    ranks = [record[key]["rank"] for record in ranked_records]
    return {
        "sample_count": len(records),
        "scored_count": len(scored),
        "ranked_count": len(ranked_records),
        "top1_accuracy": top_k_accuracy(1),
        "top3_accuracy": top_k_accuracy(3),
        "top5_accuracy": top_k_accuracy(5),
        "mrr": (sum(reciprocal_ranks) / len(reciprocal_ranks)) if reciprocal_ranks else None,
        "mean_rank": (sum(ranks) / len(ranks)) if ranks else None,
    }


def rank_metrics_by_room(records: list[dict], key: str) -> dict[str, dict]:
    by_room: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        if record.get("status") == "scored":
            by_room[str(record.get("expected_room_id"))].append(record)
    return {
        room_id: rank_metrics(room_records, key)
        for room_id, room_records in sorted(by_room.items())
    }


def localize_integrated_visual(
    *,
    payload: dict,
    expected_room_id: str,
    room_graph: dict[str, dict],
    grounding_index: GroundingIndex,
    enable_spatial_alignment: bool = False,
    manifest_path: Path | None = None,
    alignment_model: str | None = None,
    llm_api_key: str | None = None,
    detector_api_base: str | None = None,
    detector_api_kind: str | None = None,
    alignment_timeout: float | None = None,
    vlm_timeout: float | None = None,
    alignment_candidate_ratio_threshold: float = 0.5,
    alignment_candidate_max: int = 5,
    llm_num_ctx: int | None = None,
) -> dict:
    observation = observation_from_perception_payload(
        payload,
        include_visual_localization=True,
        manifest_path=manifest_path if enable_spatial_alignment else None,
    )
    spatial_refiner = None
    if enable_spatial_alignment:
        spatial_refiner = SpatialAlignmentRefiner(
            room_graph=room_graph,
            grounding_index=grounding_index,
            model=alignment_model,
            api_key=llm_api_key,
            api_base=detector_api_base,
            api_kind=detector_api_kind,
            request_timeout=alignment_timeout or vlm_timeout,
            num_ctx=llm_num_ctx,
        )
    localizer = EvidenceScoreLocalizer(
        room_graph=room_graph,
        grounding_index=grounding_index,
        alignment_candidate_ratio_threshold=alignment_candidate_ratio_threshold,
        alignment_candidate_max=alignment_candidate_max,
        spatial_refiner=spatial_refiner,
    )
    return localizer.localize(
        observation=observation,
        prior_room_belief={expected_room_id: 1.0},
        fallback_room_id=expected_room_id,
    )


def room_scores_from_payload(payload: dict) -> list[dict]:
    visual_localization = payload.get("visual_localization")
    if not isinstance(visual_localization, dict):
        return []
    room_scores = visual_localization.get("room_scores")
    if not isinstance(room_scores, list):
        return []
    return [record for record in room_scores if isinstance(record, dict)]


def is_failure_record(record: dict) -> bool:
    if record.get("status") != "scored":
        return True
    if isinstance(record.get("observation_only"), dict) or isinstance(record.get("prior_fused"), dict):
        observation_rank = record.get("observation_only", {}).get("rank")
        prior_rank = record.get("prior_fused", {}).get("rank")
        return observation_rank != 1 or prior_rank != 1
    return not record.get("matched")


def score_results(records: list[dict]) -> dict:
    scored = [record for record in records if record.get("status") == "scored"]
    if any(isinstance(record.get("observation_only"), dict) for record in scored):
        summary = {
            "sample_count": len(records),
            "scored_count": len(scored),
            "failed_count": sum(1 for record in records if record.get("status") == "failed"),
            "missing_output_count": sum(1 for record in records if record.get("status") == "missing_output"),
            "observation_only": rank_metrics(records, "observation_only"),
            "prior_fused": rank_metrics(records, "prior_fused"),
            "per_room": {
                "observation_only": rank_metrics_by_room(records, "observation_only"),
                "prior_fused": rank_metrics_by_room(records, "prior_fused"),
            },
        }
        if any(isinstance(record.get("spatial_aligned"), dict) for record in scored):
            summary["spatial_aligned"] = rank_metrics(records, "spatial_aligned")
            summary["per_room"]["spatial_aligned"] = rank_metrics_by_room(records, "spatial_aligned")
        return summary

    matched = [record for record in scored if record.get("matched")]
    per_room: dict[str, dict] = {}
    for record in scored:
        room_id = str(record["expected_room_id"])
        room_stats = per_room.setdefault(room_id, {"total": 0, "matched": 0, "accuracy": 0.0})
        room_stats["total"] += 1
        if record.get("matched"):
            room_stats["matched"] += 1
    for room_stats in per_room.values():
        room_stats["accuracy"] = room_stats["matched"] / room_stats["total"] if room_stats["total"] else 0.0
    return {
        "sample_count": len(records),
        "scored_count": len(scored),
        "matched_count": len(matched),
        "accuracy": (len(matched) / len(scored)) if scored else None,
        "failed_count": sum(1 for record in records if record.get("status") == "failed"),
        "missing_output_count": sum(1 for record in records if record.get("status") == "missing_output"),
        "per_room_accuracy": dict(sorted(per_room.items())),
    }


def run_perception(args: argparse.Namespace, *, sample: dict, output_path: Path) -> dict:
    command = [
        sys.executable,
        "-m",
        "st_nav.cli.run_pano_perception",
        "--artifacts-dir",
        args.artifacts_dir,
        f"--pano-id={sample['pano_id']}",
        "--render-output-dir",
        args.render_output_dir,
        "--heading-mode",
        args.heading_mode,
        "--pitch",
        str(args.pitch),
        "--fov",
        str(args.fov),
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--current-heading",
        str(args.current_heading),
        "--output-path",
        str(output_path),
    ]
    optional_args = [
        ("--render-api-key", args.render_api_key),
        ("--llm-api-key", args.llm_api_key),
        ("--detector-model", args.detector_model),
        ("--detector-api-kind", args.detector_api_kind),
        ("--detector-api-base", args.detector_api_base),
        ("--vlm-timeout", args.vlm_timeout),
    ]
    for flag, value in optional_args:
        if value is not None:
            command.extend([flag, str(value)])
    if getattr(args, "no_detection_cache", False):
        command.append("--no-detection-cache")
    if getattr(args, "enable_view_themes", False):
        command.append("--enable-view-themes")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return {
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-2000:],
        "stderr_tail": completed.stderr[-2000:],
    }


def evaluate_sample(
    args: argparse.Namespace,
    *,
    sample: dict,
    output_dir: Path,
    room_graph: dict[str, dict],
    grounding_index: GroundingIndex,
) -> dict:
    output_path = output_path_for_sample(output_dir, sample)
    if not args.reuse_existing_output and (args.force or not output_path.exists()):
        run_result = run_perception(args, sample=sample, output_path=output_path)
        if run_result["returncode"] != 0:
            return {
                "pano_id": sample["pano_id"],
                "expected_room_id": sample["room_id"],
                "output_path": str(output_path),
                "status": "failed",
                "error": run_result,
            }

    if not output_path.exists():
        return {
            "pano_id": sample["pano_id"],
            "expected_room_id": sample["room_id"],
            "output_path": str(output_path),
            "status": "missing_output",
        }

    payload = load_json(output_path)
    expected_room_id = str(sample["room_id"])
    if args.pipeline == "integrated-visual":
        auto_rerender_result = None
        if getattr(args, "enable_spatial_alignment", False) and not has_available_manifest(args, sample=sample, payload=payload):
            auto_rerender_result = run_perception(args, sample=sample, output_path=output_path)
            if auto_rerender_result["returncode"] != 0:
                return {
                    "pano_id": sample["pano_id"],
                    "expected_room_id": expected_room_id,
                    "output_path": str(output_path),
                    "status": "failed",
                    "error": {
                        "message": "Missing render manifest and automatic rerender failed.",
                        "rerender": auto_rerender_result,
                    },
                }
            payload = load_json(output_path)
        manifest_path = (
            first_available_manifest_path(args, sample=sample, payload=payload)
            if getattr(args, "enable_spatial_alignment", False)
            else None
        )
        try:
            localization_payload = localize_integrated_visual(
                payload=payload,
                expected_room_id=expected_room_id,
                room_graph=room_graph,
                grounding_index=grounding_index,
                enable_spatial_alignment=getattr(args, "enable_spatial_alignment", False),
                manifest_path=manifest_path,
                alignment_model=getattr(args, "alignment_model", None),
                llm_api_key=getattr(args, "llm_api_key", None),
                detector_api_base=getattr(args, "detector_api_base", None),
                detector_api_kind=getattr(args, "detector_api_kind", None),
                alignment_timeout=getattr(args, "alignment_timeout", None),
                vlm_timeout=getattr(args, "vlm_timeout", None),
                alignment_candidate_ratio_threshold=getattr(args, "alignment_candidate_ratio_threshold", 0.5),
                alignment_candidate_max=getattr(args, "alignment_candidate_max", 5),
            )
        except Exception as error:
            return {
                "pano_id": sample["pano_id"],
                "expected_room_id": expected_room_id,
                "output_path": str(output_path),
                "status": "failed",
                "error": {"message": str(error)},
            }
        observation_distribution = localization_payload.get("observation_distribution", {})
        posterior_room_belief = localization_payload.get("base_room_belief") or localization_payload.get("room_belief", {})
        record = {
            "pano_id": sample["pano_id"],
            "expected_room_id": expected_room_id,
            "observation_only": ranking_payload(observation_distribution, expected_room_id),
            "prior_fused": ranking_payload(posterior_room_belief, expected_room_id),
            "observation_distribution": compact_distribution(observation_distribution),
            "transition_support": compact_distribution(localization_payload.get("transition_support", {})),
            "posterior_room_belief": compact_distribution(posterior_room_belief),
            "room_scores": room_scores_from_payload(payload),
            "output_path": str(output_path),
            "manifest_path": payload.get("manifest_path"),
            "base_predicted_room_id": localization_payload.get("base_predicted_room_id"),
            "alignment_candidate_room_ids": list(localization_payload.get("alignment_candidate_room_ids", [])),
            "alignment_top_k": list(localization_payload.get("alignment_top_k", [])),
            "alignment_predicted_room_id": localization_payload.get("alignment_predicted_room_id"),
            "alignment_evidence": list(localization_payload.get("alignment_evidence", [])),
            "alignment_summary": localization_payload.get("alignment_summary"),
            "alignment_applied": bool(localization_payload.get("alignment_applied")),
            "alignment_skipped_reason": localization_payload.get("alignment_skipped_reason"),
            "status": "scored",
        }
        if auto_rerender_result is not None:
            record["auto_rerendered_manifest"] = True
        if getattr(args, "enable_spatial_alignment", False):
            spatial_ranking = spatial_ranking_from_localization(localization_payload, posterior_room_belief)
            record["spatial_alignment"] = (
                localization_payload.get("spatial_alignment")
                if isinstance(localization_payload.get("spatial_alignment"), dict)
                else None
            )
            if isinstance(localization_payload.get("ego_spatial_context"), dict):
                record["ego_spatial_context"] = localization_payload["ego_spatial_context"]
            record["spatial_alignment_status"] = "applied" if localization_payload.get("alignment_applied") else "skipped"
            record["spatial_alignment_skipped_reason"] = localization_payload.get("alignment_skipped_reason")
            record["spatial_aligned_ranked_rooms"] = spatial_ranking
            record["spatial_aligned"] = ranking_payload_from_order(spatial_ranking, expected_room_id)
        return record
    raise RuntimeError(f"Unsupported evaluation pipeline: {args.pipeline}")


def main() -> int:
    args = build_parser().parse_args()
    artifacts = load_normalized_artifacts(args.artifacts_dir, pano_room_grounding=True, room_graph=True, grounding=True)
    grounding_path = resolve_project_path(args.grounding_path) if args.grounding_path else artifacts.artifacts_dir / "pano_room_grounding.json"
    grounding_payload = load_json(grounding_path) if args.grounding_path else artifacts.pano_room_grounding or {}
    room_graph = artifacts.room_graph or {}
    grounding_index = GroundingIndex(artifacts.grounding or {})
    selected_room_ids = set(args.room_id) if args.room_id else None
    output_dir = resolve_project_path(args.output_dir)
    if args.reuse_existing_output:
        samples = room_samples_from_output_dir(
            output_dir,
            grounding_payload,
            room_ids=selected_room_ids,
            samples_per_room=args.samples_per_room,
            max_total=args.max_total,
        )
    else:
        samples = room_samples_from_grounding(
            grounding_payload,
            room_ids=selected_room_ids,
            samples_per_room=args.samples_per_room,
            seed=args.seed,
            max_total=args.max_total,
        )
    if not samples:
        raise RuntimeError("No grounded panorama samples selected.")
    records = []
    for index, sample in enumerate(samples, start=1):
        sample_start_time = time.monotonic()
        output_path = output_path_for_sample(output_dir, sample)
        output_existed = output_path.exists()
        print(
            f"[pano-perception-eval] {index}/{len(samples)} "
            f"room={sample['room_id']} pano={sample['pano_id']}"
            + (" (cached)" if output_existed and not args.force else ""),
            file=sys.stderr,
        )
        record = evaluate_sample(
            args,
            sample=sample,
            output_dir=output_dir,
            room_graph=room_graph,
            grounding_index=grounding_index,
        )
        records.append(record)
        ran_perception = not output_existed or getattr(args, "force", False)
        if args.delay > 0 and index < len(samples) and ran_perception:
            elapsed = time.monotonic() - sample_start_time
            remaining = args.delay - elapsed
            if remaining > 0:
                time.sleep(remaining)
        if record.get("status") == "scored":
            if isinstance(record.get("observation_only"), dict):
                print(
                    f"  expected={record.get('expected_room_id')} "
                    f"obs_top1={record['observation_only'].get('top1')} obs_rank={record['observation_only'].get('rank')} "
                    f"prior_top1={record['prior_fused'].get('top1')} prior_rank={record['prior_fused'].get('rank')}"
                    + (
                        f" spatial_top1={record['spatial_aligned'].get('top1')} "
                        f"spatial_rank={record['spatial_aligned'].get('rank')}"
                        if isinstance(record.get("spatial_aligned"), dict)
                        else ""
                    ),
                    file=sys.stderr,
                )
            else:
                print(
                    f"  expected={record.get('expected_room_id')} predicted={record.get('predicted_room_id')} "
                    f"conf={record.get('confidence')} matched={record.get('matched')}",
                    file=sys.stderr,
                )
        else:
            error = record.get("error")
            error_message = ""
            if isinstance(error, dict):
                message = error.get("message") or error.get("stderr_tail") or error.get("stdout_tail")
                if isinstance(message, str) and message.strip():
                    lines = [line.strip() for line in message.strip().splitlines() if line.strip()]
                    last_line = lines[-1] if lines else message.strip()[:500]
                    error_message = f" error={last_line[:1000]}"
            print(f"  status={record.get('status')}{error_message}", file=sys.stderr)

    summary = score_results(records)
    payload = {
        "config": {
            "pipeline": args.pipeline,
            "artifacts_dir": args.artifacts_dir,
            "grounding_path": str(grounding_path),
            "samples_per_room": args.samples_per_room,
            "seed": args.seed,
            "room_ids": sorted(selected_room_ids) if selected_room_ids else None,
            "max_total": args.max_total,
            "fov": args.fov,
            "heading_mode": args.heading_mode,
            "reuse_existing_output": args.reuse_existing_output,
            "no_detection_cache": args.no_detection_cache,
            "enable_view_themes": args.enable_view_themes,
            "detector_model": args.detector_model,
            "enable_spatial_alignment": args.enable_spatial_alignment,
            "alignment_candidate_ratio_threshold": args.alignment_candidate_ratio_threshold,
            "alignment_candidate_max": args.alignment_candidate_max,
            "alignment_model": args.alignment_model or args.detector_model,
            "alignment_timeout": args.alignment_timeout,
        },
        "summary": summary,
        "results": records,
    }
    summary_output_path = args.summary_output_path or str(output_dir / "summary.json")
    write_text_if_requested(render_json(payload), summary_output_path)

    print(render_json({"summary": summary, "summary_output_path": str(resolve_project_path(summary_output_path))}))
    if args.print_failures:
        failures = [
            record
            for record in records
            if is_failure_record(record)
        ]
        if failures:
            print(render_json({"failures": failures}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
