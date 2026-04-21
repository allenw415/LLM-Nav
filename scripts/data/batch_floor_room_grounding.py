from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from st_nav import PanoramaRenderer, load_dotenv
from st_nav_data.room_grounder import ModelRoomGrounder, build_manual_annotation_records

load_dotenv(PROJECT_ROOT / ".env")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ground arbitrary floor panos in fixed-size batches.")
    parser.add_argument("--artifacts-dir", default="dataset/sites/british_museum/normalized")
    parser.add_argument("--floor", default="0")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--output-path")
    parser.add_argument("--review-output-path")
    parser.add_argument("--manual-output-path")
    parser.add_argument("--profile")
    parser.add_argument("--model-provider")
    parser.add_argument("--model-name")
    parser.add_argument("--api-key")
    parser.add_argument("--api-base")
    parser.add_argument("--api-kind")
    parser.add_argument("--gemini-api-key", default=None)
    parser.add_argument("--gemini-model", default=None)
    parser.add_argument("--vlm-timeout", type=float, default=180.0)
    parser.add_argument("--render-api-key", default=os.environ.get("GMAPS_API_KEY"))
    parser.add_argument("--render-output-dir", default="renders/room_grounding")
    parser.add_argument("--render-seed", type=int)
    parser.add_argument("--heading-mode", choices=["museum", "cardinal", "grounding", "graph"], default="grounding")
    parser.add_argument("--max-captures", type=int, default=4)
    parser.add_argument("--pitch", type=float, default=0.0)
    parser.add_argument("--fov", type=int, default=45)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--render-timeout", type=float, default=60.0)
    parser.add_argument("--candidate-scope", choices=["same-floor", "all"], default="same-floor")
    parser.add_argument("--min-confidence", type=float, default=0.75)
    parser.add_argument("--debug-trace", action="store_true")
    return parser


def load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict JSON: {path}")
    return payload


def select_floor_pano_ids(pano_graph: dict[str, dict], floor: str) -> list[str]:
    floor_text = str(floor)
    return sorted(
        pano_id
        for pano_id, record in pano_graph.items()
        if isinstance(record, dict) and str(record.get("floor")) == floor_text
    )


def default_batch_output_path(*, artifacts_dir: Path, floor: str, offset: int, limit: int) -> Path:
    batch_dir = artifacts_dir / "room_grounding_batches"
    return batch_dir / f"floor{floor}_batch_{offset:04d}_{limit:03d}.json"


def emit_progress(message: str) -> None:
    sys.stderr.write(f"{message}\n")
    sys.stderr.flush()


def format_eta(seconds: float) -> str:
    if seconds <= 0:
        return "0s"
    whole_seconds = int(round(seconds))
    minutes, secs = divmod(whole_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def build_render_progress_callback(*, index: int, total: int, pano_id: str):
    percent = (index - 1) / total * 100 if total else 100.0

    def _callback(event: dict) -> None:
        event_name = event.get("event")
        capture_index = event.get("capture_index")
        capture_count = event.get("capture_count")
        label = event.get("label")

        if event_name == "render_cached":
            emit_progress(
                f"[{index}/{total} {percent:5.1f}%] render  pano={pano_id} cache-hit captures={capture_count}"
            )
            return
        if event_name == "render_capture_start":
            emit_progress(
                f"[{index}/{total} {percent:5.1f}%] render  pano={pano_id} "
                f"capture={capture_index}/{capture_count} label={label}"
            )
            return
        if event_name == "render_capture_done":
            emit_progress(
                f"[{index}/{total} {percent:5.1f}%] render  pano={pano_id} "
                f"capture={capture_index}/{capture_count} done label={label}"
            )

    return _callback


def ensure_manifest(
    *,
    renderer: PanoramaRenderer,
    artifacts_dir: Path,
    render_api_key: str | None,
    render_output_dir: Path,
    pano_id: str,
    heading_mode: str,
    pitch: float,
    fov: int,
    width: int,
    height: int,
    progress_callback=None,
) -> Path:
    if not render_api_key:
        raise RuntimeError("Missing GMAPS_API_KEY to render pano views.")
    manifest = renderer.render(
        pano_id=pano_id,
        api_key=render_api_key,
        output_dir=str(render_output_dir),
        heading_mode=heading_mode,
        pitch=pitch,
        fov=fov,
        width=width,
        height=height,
        graph_path=str(artifacts_dir / "pano_graph.json"),
        progress_callback=progress_callback,
    )
    return Path(str(manifest["manifest_path"])).resolve()


def main() -> int:
    args = build_parser().parse_args()

    artifacts_dir = (PROJECT_ROOT / args.artifacts_dir).resolve()
    room_graph = load_json(artifacts_dir / "room_graph.json")
    pano_graph = load_json(artifacts_dir / "pano_graph.json")
    room_grounding = load_json(artifacts_dir / "room_grounding.template.json")

    floor_pano_ids = select_floor_pano_ids(pano_graph, args.floor)
    start = max(args.offset, 0)
    end = start + max(args.limit, 0)
    batch_pano_ids = floor_pano_ids[start:end]
    if not batch_pano_ids:
        raise RuntimeError(f"No floor panos selected for floor={args.floor}, offset={args.offset}, limit={args.limit}.")

    renderer = PanoramaRenderer(
        pano_graph,
        image_timeout=args.render_timeout,
        rng=random.Random(args.render_seed) if args.render_seed is not None else None,
    )
    grounder = ModelRoomGrounder(
        profile=args.profile,
        provider=args.model_provider,
        model=args.model_name or args.gemini_model,
        api_key=args.api_key or args.gemini_api_key,
        api_base=args.api_base,
        api_kind=args.api_kind,
        request_timeout=args.vlm_timeout,
        same_floor_only=(args.candidate_scope == "same-floor"),
        max_captures=max(args.max_captures, 1),
    )

    results: list[dict] = []
    total = len(batch_pano_ids)
    batch_started_at = time.time()
    emit_progress(
        f"[room-grounding] floor={args.floor} batch={start}:{end} total={total} "
        f"model={grounder.model} heading={args.heading_mode} captures={max(args.max_captures, 1)} fov={args.fov}"
    )
    for index, pano_id in enumerate(batch_pano_ids, start=1):
        percent = (index - 1) / total * 100 if total else 100.0
        emit_progress(f"[{index}/{total} {percent:5.1f}%] render  pano={pano_id} start")
        manifest_path = ensure_manifest(
            renderer=renderer,
            artifacts_dir=artifacts_dir,
            render_api_key=args.render_api_key,
            render_output_dir=(PROJECT_ROOT / args.render_output_dir).resolve(),
            pano_id=pano_id,
            heading_mode=args.heading_mode,
            pitch=args.pitch,
            fov=args.fov,
            width=args.width,
            height=args.height,
            progress_callback=build_render_progress_callback(index=index, total=total, pano_id=pano_id),
        )
        emit_progress(f"[{index}/{total} {percent:5.1f}%] ground  pano={pano_id}")
        result = grounder.ground(
            manifest_path,
            room_graph=room_graph,
            room_grounding=room_grounding,
        )
        record = {
            "pano_id": result.get("pano_id"),
            "floor": result.get("floor"),
            "manifest_path": str(manifest_path),
            "predicted_room_id": result.get("predicted_room_id"),
            "confidence": result.get("confidence"),
            "alternative_room_ids": result.get("alternative_room_ids"),
            "evidence": result.get("evidence"),
            "summary": result.get("summary"),
            "candidate_room_ids": result.get("candidate_room_ids"),
        }
        if args.debug_trace:
            record["trace"] = grounder.last_traces
        results.append(record)

        elapsed = time.time() - batch_started_at
        avg_seconds = elapsed / index
        remaining = max(total - index, 0) * avg_seconds
        percent = index / total * 100 if total else 100.0
        predicted_room_id = result.get("predicted_room_id") or "null"
        confidence = result.get("confidence")
        confidence_text = f"{float(confidence):.2f}" if isinstance(confidence, (int, float)) else "n/a"
        emit_progress(
            f"[{index}/{total} {percent:5.1f}%] done    pano={pano_id} "
            f"pred={predicted_room_id} conf={confidence_text} "
            f"elapsed={format_eta(elapsed)} eta={format_eta(remaining)}"
        )

    manual_records = build_manual_annotation_records(
        results,
        min_confidence=args.min_confidence,
        prefill_manual_room_id_from_prediction=True,
    )
    review_queue = [record for record in manual_records if record.get("needs_review")]

    payload = {
        "summary": {
            "floor": str(args.floor),
            "offset": start,
            "limit": max(args.limit, 0),
            "batch_count": len(results),
            "floor_pano_count": len(floor_pano_ids),
            "candidate_scope": args.candidate_scope,
            "heading_mode": args.heading_mode,
            "max_captures": max(args.max_captures, 1),
            "fov": args.fov,
            "render_seed": args.render_seed,
            "review_count": len(review_queue),
            "model_name": grounder.model,
            "provider": grounder.provider,
            "profile": grounder.profile,
        },
        "results": results,
    }

    output_path = (
        (PROJECT_ROOT / args.output_path).resolve()
        if args.output_path
        else default_batch_output_path(
            artifacts_dir=artifacts_dir,
            floor=str(args.floor),
            offset=start,
            limit=max(args.limit, 0),
        )
    )
    review_output_path = (
        (PROJECT_ROOT / args.review_output_path).resolve()
        if args.review_output_path
        else output_path.with_name(f"{output_path.stem}.review.json")
    )
    manual_output_path = (
        (PROJECT_ROOT / args.manual_output_path).resolve()
        if args.manual_output_path
        else output_path.with_name(f"{output_path.stem}.manual.json")
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    review_output_path.parent.mkdir(parents=True, exist_ok=True)
    review_output_path.write_text(
        json.dumps({"summary": payload["summary"], "results": review_queue}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    manual_output_path.parent.mkdir(parents=True, exist_ok=True)
    manual_output_path.write_text(
        json.dumps({"summary": payload["summary"], "results": manual_records}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    emit_progress(
        f"[room-grounding] wrote raw={output_path.name} review={review_output_path.name} "
        f"manual={manual_output_path.name} review_count={len(review_queue)} "
        f"elapsed={format_eta(time.time() - batch_started_at)}"
    )

    print(
        json.dumps(
            {
                "output_path": str(output_path),
                "review_output_path": str(review_output_path),
                "manual_output_path": str(manual_output_path),
                "summary": payload["summary"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
