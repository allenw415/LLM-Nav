from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from st_nav import PanoramaRenderer, load_dotenv
from st_nav_data.room_grounder import ModelRoomGrounder, aggregate_model_usage_from_traces, invert_room_grounding

load_dotenv(PROJECT_ROOT / ".env")


def format_usage(usage: dict) -> dict:
    return {
        "requests": usage.get("request_count", 0),
        "input_tokens": usage.get("prompt_token_count", 0),
        "output_tokens": usage.get("candidates_token_count", 0),
        "total_tokens": usage.get("total_token_count", 0),
        "thinking_tokens": usage.get("thoughts_token_count", 0),
        "cached_tokens": usage.get("cached_content_token_count", 0),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run room grounding from rendered pano views using the active model profile.")
    parser.add_argument("--artifacts-dir", default="dataset/sites/british_museum/normalized")
    parser.add_argument("--manifest-path")
    parser.add_argument("--pano-id")
    parser.add_argument("--room-id", action="append", default=[])
    parser.add_argument("--limit", type=int)
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
    parser.add_argument("--candidate-scope", choices=["same-floor", "all"], default="same-floor")
    parser.add_argument("--no-grounding-cache", action="store_true")
    parser.add_argument("--debug-trace", action="store_true")
    parser.add_argument("--full-output", action="store_true")
    return parser


def load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict JSON: {path}")
    return payload


def evaluate_targets(
    *,
    manifest_path: str | None,
    pano_id: str | None,
    room_ids: list[str],
    limit: int | None,
    room_grounding: dict[str, dict],
) -> list[dict]:
    if manifest_path:
        return [{"manifest_path": manifest_path}]
    if pano_id:
        return [{"pano_id": pano_id}]

    pano_to_rooms = invert_room_grounding(room_grounding)
    allowed_room_ids = set(room_ids)
    targets: list[dict] = []
    for grounded_pano_id in sorted(pano_to_rooms.keys()):
        expected_room_ids = pano_to_rooms[grounded_pano_id]
        if allowed_room_ids and not any(room_id in allowed_room_ids for room_id in expected_room_ids):
            continue
        targets.append({"pano_id": grounded_pano_id, "expected_room_ids": expected_room_ids})

    if limit is not None:
        return targets[: max(limit, 0)]
    return targets


def ensure_manifest(
    target: dict,
    *,
    renderer: PanoramaRenderer,
    artifacts_dir: Path,
    render_api_key: str | None,
    render_output_dir: Path,
    heading_mode: str,
    pitch: float,
    fov: int,
    width: int,
    height: int,
) -> Path:
    manifest_path = target.get("manifest_path")
    if isinstance(manifest_path, str) and manifest_path:
        return Path(manifest_path).resolve()

    pano_id = target.get("pano_id")
    if not isinstance(pano_id, str) or not pano_id:
        raise RuntimeError("Evaluation target is missing both manifest_path and pano_id.")
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
    )
    return Path(str(manifest["manifest_path"])).resolve()


def main() -> int:
    args = build_parser().parse_args()

    artifacts_dir = (PROJECT_ROOT / args.artifacts_dir).resolve()
    room_graph = load_json(artifacts_dir / "room_graph.json")
    pano_graph = load_json(artifacts_dir / "pano_graph.json")
    room_grounding = load_json(artifacts_dir / "room_grounding.template.json")

    targets = evaluate_targets(
        manifest_path=args.manifest_path,
        pano_id=args.pano_id,
        room_ids=args.room_id,
        limit=args.limit,
        room_grounding=room_grounding,
    )
    if not targets:
        raise RuntimeError("No grounding evaluation targets found.")

    renderer = PanoramaRenderer(
        pano_graph,
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
        use_grounding_files=not args.no_grounding_cache,
        same_floor_only=(args.candidate_scope == "same-floor"),
        max_captures=max(args.max_captures, 1),
    )

    results = []
    matched_count = 0
    scored_count = 0
    usage_totals = {
        "request_count": 0,
        "prompt_token_count": 0,
        "candidates_token_count": 0,
        "total_token_count": 0,
        "thoughts_token_count": 0,
        "cached_content_token_count": 0,
    }
    for target in targets:
        manifest_path = ensure_manifest(
            target,
            renderer=renderer,
            artifacts_dir=artifacts_dir,
            render_api_key=args.render_api_key,
            render_output_dir=(PROJECT_ROOT / args.render_output_dir).resolve(),
            heading_mode=args.heading_mode,
            pitch=args.pitch,
            fov=args.fov,
            width=args.width,
            height=args.height,
        )

        result = grounder.ground(
            manifest_path,
            room_graph=room_graph,
            room_grounding=room_grounding,
        )
        usage = aggregate_model_usage_from_traces(grounder.last_traces)
        expected_room_ids = target.get("expected_room_ids")
        if not isinstance(expected_room_ids, list):
            expected_room_ids = invert_room_grounding(room_grounding).get(str(result.get("pano_id")), [])

        is_match = None
        predicted_room_id = result.get("predicted_room_id")
        if expected_room_ids:
            scored_count += 1
            is_match = isinstance(predicted_room_id, str) and predicted_room_id in expected_room_ids
            if is_match:
                matched_count += 1

        payload = {
            "pano_id": result.get("pano_id"),
            "predicted_room_id": predicted_room_id,
            "confidence": result.get("confidence"),
        }
        if expected_room_ids:
            payload["expected_room_ids"] = expected_room_ids
            payload["is_match"] = is_match
        if args.full_output:
            payload.update(
                {
                    "manifest_path": str(manifest_path),
                    "evidence": result.get("evidence"),
                    "alternative_room_ids": result.get("alternative_room_ids"),
                    "summary": result.get("summary"),
                "candidate_count": len(result.get("candidate_room_ids", [])),
                "render_capture_count": len(json.loads(Path(manifest_path).read_text(encoding="utf-8")).get("captures", [])),
                "usage": format_usage(usage),
            }
            )
        if args.debug_trace:
            payload["trace"] = grounder.last_traces
        results.append(payload)
        for key in usage_totals:
            value = usage.get(key)
            if isinstance(value, int):
                usage_totals[key] += value

    summary = {
        "model_name": grounder.model,
        "provider": grounder.provider,
        "profile": grounder.profile,
        "heading_mode": args.heading_mode,
        "max_captures": max(args.max_captures, 1),
        "fov": args.fov,
        "render_seed": args.render_seed,
        "use_grounding_cache": not args.no_grounding_cache,
        "usage": format_usage(usage_totals) if usage_totals["request_count"] > 0 else {},
    }
    if args.full_output:
        summary.update(
            {
                "target_count": len(results),
                "scored_count": scored_count,
                "matched_count": matched_count,
                "accuracy": (matched_count / scored_count) if scored_count else None,
                "candidate_scope": args.candidate_scope,
            }
        )
    print(json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
