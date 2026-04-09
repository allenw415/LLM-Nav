from __future__ import annotations

import argparse
import os
from _common import (
    PROJECT_ROOT,
    ensure_project_root_on_path,
    load_normalized_artifacts,
    render_json,
    write_text_if_requested,
)

ensure_project_root_on_path()

from st_nav import PanoramaRenderer, PerceptionPipeline, ViewDetector, load_dotenv

load_dotenv(PROJECT_ROOT / ".env")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run perception directly on a given pano id.")
    parser.add_argument("--artifacts-dir", default="dataset/sites/british_museum/normalized")
    parser.add_argument("--pano-id", required=True)
    parser.add_argument("--llm-api-key", default=os.environ.get("OPENAI_API_KEY"))
    parser.add_argument("--detector-model", default="gpt-5-mini")
    parser.add_argument("--vlm-timeout", type=float, default=180.0)
    parser.add_argument("--render-api-key", default=os.environ.get("GMAPS_API_KEY"))
    parser.add_argument("--render-output-dir", default="renders/pano_perception")
    parser.add_argument("--heading-mode", choices=["museum", "cardinal", "graph"], default="museum")
    parser.add_argument("--pitch", type=float, default=0.0)
    parser.add_argument("--fov", type=int, default=45)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--current-heading", type=float, default=330.0)
    parser.add_argument("--demo-trace", action="store_true")
    parser.add_argument("--output-path")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not args.llm_api_key:
        raise RuntimeError("Missing OPENAI_API_KEY.")
    if not args.render_api_key:
        raise RuntimeError("Missing GMAPS_API_KEY.")

    artifacts = load_normalized_artifacts(args.artifacts_dir, pano_graph=True)
    pano_graph = artifacts.pano_graph or {}

    detector = ViewDetector(
        api_key=args.llm_api_key,
        model=args.detector_model,
        request_timeout=args.vlm_timeout,
    )
    pipeline = PerceptionPipeline(
        pano_graph=pano_graph,
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
            }
            for entity in observation.entities
        ],
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
