from __future__ import annotations

import argparse
import random
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from _common import (
    PROJECT_ROOT,
    ensure_project_root_on_path,
    load_json,
    load_normalized_artifacts,
    render_json,
    resolve_project_path,
    write_text_if_requested,
)

ensure_project_root_on_path()

from st_nav import EntityDetection, GroundingIndex, LLMRoomLocalizer, Observation, RoomLocalizer, load_dotenv, resolve_model_environment

load_dotenv(PROJECT_ROOT / ".env")
MODEL_ENV = resolve_model_environment(
    default_model="gpt-5-mini",
    default_api_base="https://api.openai.com/v1",
    default_api_kind="responses",
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
        choices=["integrated-visual", "legacy-entity-llm", "legacy-entity-heuristic"],
        default="integrated-visual",
        help=(
            "integrated-visual scores visual_localization from run_pano_perception.py. "
            "legacy-entity-llm uses original entity-only perception followed by entity-based LLM room localization."
        ),
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
        help="Do not call run_pano_perception.py; only score existing per-pano output JSON files.",
    )
    parser.add_argument("--render-output-dir", default="renders/pano_perception_grounding_eval")
    parser.add_argument("--render-api-key")
    parser.add_argument("--llm-api-key")
    parser.add_argument("--localizer-model", default=MODEL_ENV.model_name)
    parser.add_argument("--localizer-api-kind", default=MODEL_ENV.api_kind)
    parser.add_argument("--localizer-api-base", default=MODEL_ENV.api_base)
    parser.add_argument("--localizer-timeout", type=float, default=MODEL_ENV.request_timeout or 30.0)
    parser.add_argument("--detector-model")
    parser.add_argument("--detector-api-kind")
    parser.add_argument("--detector-api-base")
    parser.add_argument("--vlm-timeout", type=float)
    parser.add_argument("--heading-mode", choices=["museum", "cardinal", "graph"], default="museum")
    parser.add_argument("--pitch", type=float, default=0.0)
    parser.add_argument("--fov", type=int, default=90)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--current-heading", type=float, default=330.0)
    parser.add_argument("--print-failures", action="store_true")
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


def observation_from_perception_payload(payload: dict) -> Observation:
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

    return Observation(
        pano_id=pano_id,
        entities=entities,
        heading_estimate=(
            float(payload["current_heading"])
            if isinstance(payload.get("current_heading"), (int, float))
            else None
        ),
        metadata={
            "floor": payload.get("floor"),
            "lat": payload.get("lat"),
            "lng": payload.get("lng"),
            "source": "legacy-entity-perception-json",
        },
    )


def best_room_from_distribution(distribution: dict[str, float]) -> tuple[str | None, float | None]:
    candidates = [
        (room_id, float(score))
        for room_id, score in distribution.items()
        if isinstance(room_id, str) and isinstance(score, (int, float))
    ]
    if not candidates:
        return None, None
    room_id, score = max(candidates, key=lambda item: (item[1], item[0]))
    return room_id, score


def localize_legacy_entities(
    args: argparse.Namespace,
    *,
    payload: dict,
    room_graph: dict[str, dict],
    grounding_index: GroundingIndex,
) -> tuple[str | None, float | None, dict]:
    cached = payload.get("legacy_localization")
    if isinstance(cached, dict):
        predicted_room_id = cached.get("predicted_room_id")
        confidence = cached.get("confidence")
        return (
            predicted_room_id if isinstance(predicted_room_id, str) and predicted_room_id else None,
            float(confidence) if isinstance(confidence, (int, float)) else None,
            cached,
        )

    observation = observation_from_perception_payload(payload)
    if args.pipeline == "legacy-entity-heuristic":
        localizer = RoomLocalizer(room_graph=room_graph, grounding_index=grounding_index)
    else:
        localizer = LLMRoomLocalizer(
            room_graph=room_graph,
            grounding_index=grounding_index,
            model=args.localizer_model,
            api_key=args.llm_api_key or MODEL_ENV.api_key,
            api_base=args.localizer_api_base,
            api_kind=args.localizer_api_kind,
            request_timeout=args.localizer_timeout,
        )
    localization = localizer.localize(
        observation=observation,
        prior_room_belief={},
        fallback_room_id=None,
    )
    observation_distribution = localization.get("observation_distribution")
    if not isinstance(observation_distribution, dict):
        observation_distribution = localization.get("room_belief", {})
    predicted_room_id, confidence = best_room_from_distribution(observation_distribution)
    legacy_payload = {
        "pipeline": args.pipeline,
        "predicted_room_id": predicted_room_id,
        "confidence": confidence,
        "observation_distribution": observation_distribution,
        "evidence": localization.get("evidence", []),
        "summary": localization.get("summary"),
    }
    return predicted_room_id, confidence, legacy_payload


def score_results(records: list[dict]) -> dict:
    scored = [record for record in records if record.get("status") == "scored"]
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
        str(PROJECT_ROOT / "scripts/demo/run_pano_perception.py"),
        "--artifacts-dir",
        args.artifacts_dir,
        "--pano-id",
        str(sample["pano_id"]),
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
    if args.pipeline in {"legacy-entity-llm", "legacy-entity-heuristic"}:
        command.append("--legacy-entity-only")
        command.append("--no-detection-cache")

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
    if args.pipeline == "integrated-visual":
        predicted_room_id, confidence = prediction_from_perception_payload(payload)
        localization_payload = None
    else:
        try:
            predicted_room_id, confidence, localization_payload = localize_legacy_entities(
                args,
                payload=payload,
                room_graph=room_graph,
                grounding_index=grounding_index,
            )
        except Exception as error:
            return {
                "pano_id": sample["pano_id"],
                "expected_room_id": sample["room_id"],
                "output_path": str(output_path),
                "status": "failed",
                "error": {"message": str(error)},
            }
        if isinstance(localization_payload, dict):
            payload["legacy_localization"] = localization_payload
            output_path.write_text(render_json(payload), encoding="utf-8")
    expected_room_id = str(sample["room_id"])
    record = {
        "pano_id": sample["pano_id"],
        "expected_room_id": expected_room_id,
        "predicted_room_id": predicted_room_id,
        "confidence": confidence,
        "matched": predicted_room_id == expected_room_id,
        "output_path": str(output_path),
        "manifest_path": payload.get("manifest_path"),
        "status": "scored",
    }
    if args.pipeline != "integrated-visual":
        record["legacy_localization"] = localization_payload
    return record


def main() -> int:
    args = build_parser().parse_args()
    artifacts = load_normalized_artifacts(args.artifacts_dir, pano_room_grounding=True, room_graph=True, grounding=True)
    grounding_path = resolve_project_path(args.grounding_path) if args.grounding_path else artifacts.artifacts_dir / "pano_room_grounding.json"
    grounding_payload = load_json(grounding_path) if args.grounding_path else artifacts.pano_room_grounding or {}
    room_graph = artifacts.room_graph or {}
    grounding_index = GroundingIndex(artifacts.grounding or {})
    selected_room_ids = set(args.room_id) if args.room_id else None
    samples = room_samples_from_grounding(
        grounding_payload,
        room_ids=selected_room_ids,
        samples_per_room=args.samples_per_room,
        seed=args.seed,
        max_total=args.max_total,
    )
    if not samples:
        raise RuntimeError("No grounded panorama samples selected.")

    output_dir = resolve_project_path(args.output_dir)
    records = []
    for index, sample in enumerate(samples, start=1):
        print(
            f"[pano-perception-eval] {index}/{len(samples)} "
            f"room={sample['room_id']} pano={sample['pano_id']}",
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
        if record.get("status") == "scored":
            print(
                f"  expected={record.get('expected_room_id')} predicted={record.get('predicted_room_id')} "
                f"conf={record.get('confidence')} matched={record.get('matched')}",
                file=sys.stderr,
            )
        else:
            print(f"  status={record.get('status')}", file=sys.stderr)

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
            if record.get("status") != "scored" or not record.get("matched")
        ]
        if failures:
            print(render_json({"failures": failures}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
