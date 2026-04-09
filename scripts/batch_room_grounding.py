from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from st_nav import PanoramaRenderer
from st_nav.env import load_dotenv
from st_nav.normalize import BRITISH_MUSEUM_EXPERIMENT_ROOM_IDS
from st_nav.room_grounder import (
    GeminiRoomGrounder,
    build_compact_pano_room_mapping,
    build_manual_annotation_records,
    collect_seed_panos_for_rooms,
    collect_manual_seed_panos,
    expand_seed_panos_by_region_growing,
    expand_seed_panos_by_hops,
    merge_seed_panos_by_room,
    merge_records_by_pano_id,
)

load_dotenv(PROJECT_ROOT / ".env")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Batch-generate pano-to-room grounding candidates with Gemini.")
    parser.add_argument("--artifacts-dir", default="dataset/sites/british_museum/normalized")
    parser.add_argument("--room-id", action="append", default=[])
    parser.add_argument(
        "--expansion-strategy",
        choices=["confidence-region-growing", "fixed-hops"],
        default="confidence-region-growing",
    )
    parser.add_argument("--max-hops", type=int, default=1)
    parser.add_argument("--floor", default="0")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output-path", default="dataset/sites/british_museum/normalized/room_grounding.gemini.json")
    parser.add_argument(
        "--review-output-path",
        default="dataset/sites/british_museum/normalized/room_grounding.gemini.review.json",
    )
    parser.add_argument(
        "--manual-output-path",
        default="dataset/sites/british_museum/normalized/room_grounding.manual.json",
    )
    parser.add_argument(
        "--compact-output-path",
        default="dataset/sites/british_museum/normalized/pano_room_grounding.json",
    )
    parser.add_argument("--min-confidence", type=float, default=0.75)
    parser.add_argument("--expansion-confidence", type=float, default=0.8)
    parser.add_argument("--gemini-api-key", default=os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
    parser.add_argument("--gemini-model", default="gemini-2.5-flash")
    parser.add_argument("--vlm-timeout", type=float, default=180.0)
    parser.add_argument("--render-api-key", default=os.environ.get("GMAPS_API_KEY"))
    parser.add_argument("--render-output-dir", default="renders/room_grounding")
    parser.add_argument("--pitch", type=float, default=0.0)
    parser.add_argument("--fov", type=int, default=45)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--candidate-scope", choices=["same-floor", "all"], default="same-floor")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--debug-trace", action="store_true")
    return parser


def load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict JSON: {path}")
    return payload


def load_existing_results(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        payload = load_json(path)
    except (OSError, json.JSONDecodeError, ValueError):
        return []
    results = payload.get("results")
    if not isinstance(results, list):
        return []
    return [record for record in results if isinstance(record, dict)]


def default_room_ids() -> list[str]:
    return sorted(BRITISH_MUSEUM_EXPERIMENT_ROOM_IDS, key=room_sort_key)


def room_sort_key(room_id: str) -> tuple[int, str]:
    if not room_id.startswith("Room "):
        return (10_000, room_id)
    suffix = room_id[5:]
    digits = []
    letters = []
    for ch in suffix:
        if ch.isdigit() and not letters:
            digits.append(ch)
        else:
            letters.append(ch)
    number = int("".join(digits)) if digits else 10_000
    return (number, "".join(letters))


def ensure_manifest(
    *,
    renderer: PanoramaRenderer,
    artifacts_dir: Path,
    render_api_key: str | None,
    render_output_dir: Path,
    pano_id: str,
    pitch: float,
    fov: int,
    width: int,
    height: int,
) -> Path:
    if not render_api_key:
        raise RuntimeError("Missing GMAPS_API_KEY to render pano views.")
    manifest = renderer.render(
        pano_id=pano_id,
        api_key=render_api_key,
        output_dir=str(render_output_dir),
        heading_mode="grounding",
        pitch=pitch,
        fov=fov,
        width=width,
        height=height,
        graph_path=str(artifacts_dir / "pano_graph.json"),
    )
    return Path(str(manifest["manifest_path"])).resolve()


def requires_review(result: dict, *, min_confidence: float) -> bool:
    predicted_room_id = result.get("predicted_room_id")
    confidence = result.get("confidence")
    if not isinstance(predicted_room_id, str) or not predicted_room_id:
        return True
    if not isinstance(confidence, (int, float)) or float(confidence) < min_confidence:
        return True
    alternatives = result.get("alternative_room_ids")
    return isinstance(alternatives, list) and len(alternatives) > 0


def flatten_region_record(record: dict) -> dict:
    classification = record.get("classification", {})
    if not isinstance(classification, dict):
        classification = {}
    return {
        "pano_id": record.get("pano_id"),
        "floor": record.get("floor"),
        "region_depth": record.get("region_depth"),
        "frontier_room_ids": record.get("frontier_room_ids", []),
        "expansion_room_ids": record.get("expansion_room_ids", []),
        "manifest_path": classification.get("manifest_path"),
        "predicted_room_id": classification.get("predicted_room_id"),
        "confidence": classification.get("confidence"),
        "evidence": classification.get("evidence"),
        "alternative_room_ids": classification.get("alternative_room_ids"),
        "summary": classification.get("summary"),
        "candidate_room_ids": classification.get("candidate_room_ids"),
    }


def main() -> int:
    args = build_parser().parse_args()

    artifacts_dir = (PROJECT_ROOT / args.artifacts_dir).resolve()
    room_graph = load_json(artifacts_dir / "room_graph.json")
    pano_graph = load_json(artifacts_dir / "pano_graph.json")
    room_grounding = load_json(artifacts_dir / "room_grounding.template.json")
    manual_output_path = (PROJECT_ROOT / args.manual_output_path).resolve()
    existing_manual_records = load_existing_results(manual_output_path)

    selected_room_ids = list(args.room_id) if args.room_id else default_room_ids()
    template_seed_panos_by_room, missing_seed_rooms = collect_seed_panos_for_rooms(room_grounding, selected_room_ids)
    manual_seed_panos_by_room = collect_manual_seed_panos(existing_manual_records, room_ids=selected_room_ids)
    seed_panos_by_room = merge_seed_panos_by_room(template_seed_panos_by_room, manual_seed_panos_by_room)

    summary = {
        "room_ids": selected_room_ids,
        "floor": args.floor,
        "expansion_strategy": args.expansion_strategy,
        "max_hops": args.max_hops,
        "seed_room_count": len(seed_panos_by_room),
        "template_seed_room_count": len(template_seed_panos_by_room),
        "manual_seed_room_count": len(manual_seed_panos_by_room),
        "manual_seed_pano_count": sum(len(pano_ids) for pano_ids in manual_seed_panos_by_room.values()),
        "missing_seed_rooms": missing_seed_rooms,
        "candidate_scope": args.candidate_scope,
        "gemini_model": args.gemini_model,
        "min_confidence": args.min_confidence,
        "expansion_confidence": args.expansion_confidence,
    }

    if args.dry_run:
        if args.expansion_strategy == "fixed-hops":
            expanded = expand_seed_panos_by_hops(
                pano_graph,
                seed_panos_by_room,
                max_hops=max(args.max_hops, 0),
                floor=args.floor,
            )
            candidate_pano_ids = sorted(expanded.keys())
            if args.limit is not None:
                candidate_pano_ids = candidate_pano_ids[: max(args.limit, 0)]
            preview = {
                "summary": {**summary, "candidate_pano_count": len(candidate_pano_ids)},
                "template_seed_panos_by_room": template_seed_panos_by_room,
                "manual_seed_panos_by_room": manual_seed_panos_by_room,
                "seed_panos_by_room": seed_panos_by_room,
                "candidate_panos": [expanded[pano_id] for pano_id in candidate_pano_ids],
            }
        else:
            seed_records = []
            for room_id, pano_ids in seed_panos_by_room.items():
                for pano_id in pano_ids:
                    seed_records.append({"room_id": room_id, "pano_id": pano_id, "region_depth": 0})
            if args.limit is not None:
                seed_records = seed_records[: max(args.limit, 0)]
            preview = {
                "summary": {
                    **summary,
                    "candidate_pano_count": len(seed_records),
                    "dry_run_note": "Region growing expands only after real grounding results are available.",
                },
                "template_seed_panos_by_room": template_seed_panos_by_room,
                "manual_seed_panos_by_room": manual_seed_panos_by_room,
                "seed_panos_by_room": seed_panos_by_room,
                "initial_frontier": seed_records,
            }
        print(json.dumps(preview, ensure_ascii=False, indent=2))
        return 0

    renderer = PanoramaRenderer(pano_graph)
    grounder = GeminiRoomGrounder(
        model=args.gemini_model,
        api_key=args.gemini_api_key,
        request_timeout=args.vlm_timeout,
        same_floor_only=(args.candidate_scope == "same-floor"),
        max_captures=4,
    )

    classification_cache: dict[str, dict] = {}

    def classify_pano(pano_id: str) -> dict:
        cached = classification_cache.get(pano_id)
        if cached is not None:
            return cached
        manifest_path = ensure_manifest(
            renderer=renderer,
            artifacts_dir=artifacts_dir,
            render_api_key=args.render_api_key,
            render_output_dir=(PROJECT_ROOT / args.render_output_dir).resolve(),
            pano_id=pano_id,
            pitch=args.pitch,
            fov=args.fov,
            width=args.width,
            height=args.height,
        )
        grounding_result = grounder.ground(
            manifest_path,
            room_graph=room_graph,
            room_grounding=room_grounding,
        )
        result = {
            **grounding_result,
            "manifest_path": str(manifest_path),
        }
        if args.debug_trace:
            result["trace"] = grounder.last_traces
        classification_cache[pano_id] = result
        return result

    if args.expansion_strategy == "fixed-hops":
        expanded = expand_seed_panos_by_hops(
            pano_graph,
            seed_panos_by_room,
            max_hops=max(args.max_hops, 0),
            floor=args.floor,
        )
        candidate_pano_ids = sorted(expanded.keys())
        if args.limit is not None:
            candidate_pano_ids = candidate_pano_ids[: max(args.limit, 0)]
        results = []
        review_queue = []
        for pano_id in candidate_pano_ids:
            grounding_result = classify_pano(pano_id)
            record = {
                **expanded[pano_id],
                "manifest_path": grounding_result.get("manifest_path"),
                "predicted_room_id": grounding_result.get("predicted_room_id"),
                "confidence": grounding_result.get("confidence"),
                "evidence": grounding_result.get("evidence"),
                "alternative_room_ids": grounding_result.get("alternative_room_ids"),
                "summary": grounding_result.get("summary"),
                "candidate_room_ids": grounding_result.get("candidate_room_ids"),
            }
            if args.debug_trace:
                record["trace"] = grounding_result.get("trace")
            results.append(record)
            if requires_review(record, min_confidence=args.min_confidence):
                review_queue.append(record)
    else:
        expanded = expand_seed_panos_by_region_growing(
            pano_graph,
            seed_panos_by_room,
            classify_pano=classify_pano,
            max_depth=max(args.max_hops, 0),
            floor=args.floor,
            min_confidence=args.expansion_confidence,
            limit=args.limit,
        )
        results = []
        review_queue = []
        for pano_id in sorted(expanded.keys()):
            record = flatten_region_record(expanded[pano_id])
            if args.debug_trace:
                record["trace"] = expanded[pano_id]["classification"].get("trace")
            results.append(record)
            if requires_review(record, min_confidence=args.min_confidence):
                review_queue.append(record)

    output_path = (PROJECT_ROOT / args.output_path).resolve()
    review_output_path = (PROJECT_ROOT / args.review_output_path).resolve()
    compact_output_path = (PROJECT_ROOT / args.compact_output_path).resolve()
    merged_results = merge_records_by_pano_id(load_existing_results(output_path), results)
    merged_review_queue = [record for record in merged_results if requires_review(record, min_confidence=args.min_confidence)]
    manual_records = build_manual_annotation_records(
        merged_results,
        existing_manual_records=existing_manual_records,
        min_confidence=args.min_confidence,
    )
    compact_mapping = build_compact_pano_room_mapping(
        merged_results,
        manual_records=manual_records,
    )

    output_payload = {
        "summary": {
            **summary,
            "latest_run_result_count": len(results),
            "candidate_pano_count": len(merged_results),
            "result_count": len(merged_results),
            "review_count": len(merged_review_queue),
            "compact_mapping_count": len(compact_mapping["mappings"]),
        },
        "results": merged_results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    review_output_path.parent.mkdir(parents=True, exist_ok=True)
    review_output_path.write_text(
        json.dumps({"summary": output_payload["summary"], "results": merged_review_queue}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    manual_output_path.parent.mkdir(parents=True, exist_ok=True)
    manual_output_path.write_text(
        json.dumps({"summary": output_payload["summary"], "results": manual_records}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    compact_output_path.parent.mkdir(parents=True, exist_ok=True)
    compact_output_path.write_text(
        json.dumps({"summary": output_payload["summary"], **compact_mapping}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "output_path": str(output_path),
                "review_output_path": str(review_output_path),
                "manual_output_path": str(manual_output_path),
                "compact_output_path": str(compact_output_path),
                "summary": output_payload["summary"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
