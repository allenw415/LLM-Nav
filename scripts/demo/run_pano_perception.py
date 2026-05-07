from __future__ import annotations

import argparse
import os
import sys
from _common import (
    PROJECT_ROOT,
    ensure_project_root_on_path,
    load_normalized_artifacts,
    render_json,
    write_text_if_requested,
)

ensure_project_root_on_path()

from st_nav import GroundingIndex, PanoramaRenderer, PerceptionPipeline, ViewDetector, load_dotenv, resolve_model_environment

load_dotenv(PROJECT_ROOT / ".env")
MODEL_ENV = resolve_model_environment(
    default_model="gpt-5-mini",
    default_api_base="https://api.openai.com/v1",
    default_api_kind="responses",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run perception directly on a given pano id.")
    parser.add_argument("--artifacts-dir", default="dataset/sites/british_museum/normalized")
    parser.add_argument("--pano-id", required=True)
    parser.add_argument("--debug-request", action="store_true")
    parser.add_argument("--llm-api-key", default=MODEL_ENV.api_key)
    parser.add_argument("--detector-model", default=MODEL_ENV.model_name)
    parser.add_argument("--detector-api-kind", default=MODEL_ENV.api_kind)
    parser.add_argument("--detector-api-base", default=MODEL_ENV.api_base)
    parser.add_argument("--vlm-timeout", type=float, default=MODEL_ENV.request_timeout or 180.0)
    parser.add_argument("--render-api-key", default=os.environ.get("GMAPS_API_KEY"))
    parser.add_argument("--render-output-dir", default="renders/pano_perception")
    parser.add_argument("--heading-mode", choices=["museum", "cardinal", "graph"], default="museum")
    parser.add_argument("--pitch", type=float, default=0.0)
    parser.add_argument("--fov", type=int, default=90)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--current-heading", type=float, default=330.0)
    parser.add_argument("--demo-trace", action="store_true")
    parser.add_argument(
        "--legacy-entity-only",
        action="store_true",
        help="Use the original entity-only VLM detection prompt instead of integrated visual localization.",
    )
    parser.add_argument(
        "--no-detection-cache",
        action="store_true",
        help="Ignore sibling *_detections.json files and call the detector model again.",
    )
    parser.add_argument("--output-path")
    return parser


def _resolve_endpoint(detector: ViewDetector, request_body: dict) -> str:
    client = detector.model_client
    if client.provider in {"gemini", "gemini_api", "google_gemma_api"}:
        return client._gemini_endpoint(request_body)
    if client.provider == "ollama":
        return f"{client._ollama_api_base()}/api/chat"
    if client.api_kind == "responses":
        return f"{client.api_base}/responses"
    return f"{client.api_base}/chat/completions"


def _resolve_transport_payload(detector: ViewDetector, request_body: dict) -> dict:
    client = detector.model_client
    if client.provider in {"gemini", "gemini_api", "google_gemma_api"}:
        return client._responses_to_gemini_generate_content_payload(request_body)
    if client.provider == "ollama":
        return client._responses_to_ollama_chat_payload(request_body)
    if client.api_kind == "responses":
        return request_body
    return client._responses_to_chat_completions_payload(request_body)


def main() -> int:
    args = build_parser().parse_args()
    if not args.render_api_key:
        raise RuntimeError("Missing GMAPS_API_KEY.")

    artifacts = load_normalized_artifacts(args.artifacts_dir, room_graph=True, pano_graph=True, grounding=True)
    room_graph = artifacts.room_graph or {}
    pano_graph = artifacts.pano_graph or {}
    grounding_index = GroundingIndex(artifacts.grounding or {})

    detector = ViewDetector(
        api_key=args.llm_api_key,
        api_base=args.detector_api_base,
        api_kind=args.detector_api_kind,
        model=args.detector_model,
        request_timeout=args.vlm_timeout,
        use_detection_files=not args.no_detection_cache,
        room_graph=None if args.legacy_entity_only else room_graph,
        grounding_index=None if args.legacy_entity_only else grounding_index,
    )
    pipeline = PerceptionPipeline(
        pano_graph=pano_graph,
        room_graph=room_graph,
        grounding_index=grounding_index,
        renderer=PanoramaRenderer(pano_graph),
        detector=detector,
    )

    manifest = pipeline.render_views(
        pano_id=args.pano_id,
        api_key=args.render_api_key,
        output_dir=str((PROJECT_ROOT / args.render_output_dir).resolve()),
        heading_mode=args.heading_mode,
        pitch=args.pitch,
        fov=args.fov,
        width=args.width,
        height=args.height,
        graph_path=str(artifacts.artifacts_dir / "pano_graph.json"),
    )
    if args.debug_request:
        request_body = detector._build_request_body(
            [
                capture
                for capture in manifest.get("captures", [])
                if isinstance(capture, dict) and isinstance(capture.get("path"), str) and capture.get("path")
            ]
        )
        debug_payload = {
            "provider": detector.model_client.provider,
            "model": detector.model,
            "api_kind": detector.api_kind,
            "api_base": detector.api_base,
            "endpoint": _resolve_endpoint(detector, request_body),
            "request_timeout": detector.request_timeout,
            "has_api_key": bool(detector.api_key),
            "request_body": detector._redact_request_body(request_body),
            "transport_payload": detector._redact_request_body(_resolve_transport_payload(detector, request_body)),
        }
        print(render_json(debug_payload), file=sys.stderr)
    observation = pipeline.observe_from_manifest(
        manifest["manifest_path"],
        current_heading=args.current_heading,
    )

    payload = {
        "pano_id": observation.pano_id,
        "manifest_path": manifest["manifest_path"],
        "floor": observation.metadata.get("floor"),
        "lat": observation.metadata.get("lat"),
        "lng": observation.metadata.get("lng"),
        "current_heading": args.current_heading,
        "view_count": len(observation.views),
        "entities": [
            {
                "name": entity.name,
                "kind": entity.kind,
                "confidence": entity.confidence,
                "source_views": entity.metadata.get("source_views"),
                "location_scope": entity.location_scope,
            }
            for entity in observation.entities
        ],
        "inside_entities": list(observation.metadata.get("inside_entities", [])),
        "outside_entities": list(observation.metadata.get("outside_entities", [])),
        "visual_localization": observation.metadata.get("visual_localization"),
        "candidate_room_ids": observation.metadata.get("candidate_room_ids"),
    }
    if args.demo_trace:
        payload["render_manifest"] = manifest
        payload["vlm_trace"] = {
            "model": args.detector_model,
            "requests_and_responses": detector.last_traces,
        }

    output_text = render_json(payload)
    write_text_if_requested(output_text, args.output_path)
    print(output_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
