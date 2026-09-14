from __future__ import annotations

import argparse
import time
from pathlib import Path

from ._common import (
    PROJECT_ROOT,
    ensure_project_root_on_path,
    load_json,
    render_json,
    resolve_project_path,
)

ensure_project_root_on_path()

from st_nav import (  # noqa: E402
    GroundingIndex,
    Observation,
    RenderedView,
    SpatialAlignmentRefiner,
    load_dotenv,
    resolve_model_environment,
    resolve_task_num_ctx,
)
from st_nav.common.model_client import extract_output_text  # noqa: E402
from st_nav.common.room_profiles import preferred_room_graph_path  # noqa: E402
from st_nav.perception.renderer import normalize_heading, sanitize_name  # noqa: E402
from st_nav_data.pano_room_grounding import build_room_grounding_from_pano_room_mapping  # noqa: E402


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
CARDINAL_HEADINGS = {"north": 0.0, "east": 90.0, "south": 180.0, "west": 270.0}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one-call, eight-image VLM spatial alignment independently from navigation."
    )
    parser.add_argument("--pano-id", required=True)
    parser.add_argument("--render-output-dir", default="renders/room_grounding_fov90")
    parser.add_argument("--manifest-path")
    parser.add_argument("--scores-json-path")
    parser.add_argument("--artifacts-dir", default="dataset/sites/british_museum/normalized")
    parser.add_argument("--candidate-top-k", type=int, default=5)
    parser.add_argument("--expected-room-id")
    parser.add_argument("--llm-model", default=MODEL_ENV.model_name)
    parser.add_argument("--llm-api-key", default=MODEL_ENV.api_key)
    parser.add_argument("--llm-api-kind", default=MODEL_ENV.api_kind)
    parser.add_argument("--llm-api-base", default=MODEL_ENV.api_base)
    parser.add_argument("--llm-timeout", type=float, default=MODEL_ENV.request_timeout or 180.0)
    parser.add_argument("--llm-num-ctx", type=int, default=DEFAULT_LLM_NUM_CTX)
    parser.add_argument("--output-path")
    parser.add_argument("--json", action="store_true")
    return parser


def resolve_input_paths(
    *,
    pano_id: str,
    render_output_dir: str,
    manifest_path: str | None,
    scores_json_path: str | None,
) -> tuple[Path, Path]:
    if manifest_path:
        resolved_manifest_path = resolve_project_path(manifest_path)
    else:
        pano_slug = sanitize_name(pano_id)
        resolved_manifest_path = (
            resolve_project_path(render_output_dir)
            / pano_slug
            / f"{pano_slug}_manifest.json"
        )
    if scores_json_path:
        resolved_scores_path = resolve_project_path(scores_json_path)
    else:
        resolved_scores_path = resolved_manifest_path.with_name(
            f"{resolved_manifest_path.stem}_detections.json"
        )
    return resolved_manifest_path, resolved_scores_path


def default_output_path(pano_id: str) -> Path:
    return resolve_project_path(
        Path("outputs/spatial_alignment") / f"{sanitize_name(pano_id)}_single_call.json"
    )


def load_manifest_observation(
    *,
    manifest_path: Path,
    pano_id: str,
) -> tuple[dict, Observation, list[dict]]:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    manifest = load_json(manifest_path)
    manifest_pano_id = manifest.get("pano_id")
    if manifest_pano_id != pano_id:
        raise ValueError(
            f"Manifest pano_id mismatch: expected {pano_id}, found {manifest_pano_id}."
        )

    captures = manifest.get("captures")
    if not isinstance(captures, list) or len(captures) != 8:
        count = len(captures) if isinstance(captures, list) else 0
        raise ValueError(f"Spatial alignment requires exactly 8 manifest captures; found {count}.")

    rendered_views: list[RenderedView] = []
    diagnostics: list[dict] = []
    for index, capture in enumerate(captures):
        if not isinstance(capture, dict):
            raise ValueError(f"Manifest capture {index} is not an object.")
        raw_path = capture.get("path")
        raw_heading = capture.get("heading")
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError(f"Manifest capture {index} is missing path.")
        if not isinstance(raw_heading, (int, float)):
            raise ValueError(f"Manifest capture {index} is missing numeric heading.")
        image_path = Path(raw_path)
        if not image_path.is_absolute():
            image_path = (manifest_path.parent / image_path).resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f"Panorama image not found: {image_path}")

        view_id = f"view_{index}"
        heading = normalize_heading(float(raw_heading))
        rendered_views.append(
            RenderedView(
                label=view_id,
                heading=heading,
                path=str(image_path),
            )
        )
        diagnostics.append(
            {
                "view_id": view_id,
                "geocentric_heading_deg": heading,
                "image_path": str(image_path),
            }
        )

    observation = Observation(
        pano_id=pano_id,
        views=rendered_views,
        metadata={
            "floor": manifest.get("floor"),
            "heading_mode": manifest.get("heading_mode"),
            "manifest_path": str(manifest_path),
        },
    )
    return manifest, observation, diagnostics


def select_candidate_rooms(
    *,
    scores_path: Path,
    room_graph: dict[str, dict],
    top_k: int,
) -> list[dict]:
    if top_k < 2:
        raise ValueError("--candidate-top-k must be at least 2.")
    if not scores_path.is_file():
        raise FileNotFoundError(
            f"Detection score cache not found: {scores_path}. "
            "The single-call experiment will not run perception automatically."
        )
    payload = load_json(scores_path)
    visual_localization = payload.get("visual_localization")
    room_scores = visual_localization.get("room_scores") if isinstance(visual_localization, dict) else None
    if not isinstance(room_scores, list):
        raise ValueError(f"Detection score cache has no visual_localization.room_scores: {scores_path}")

    best_by_room: dict[str, dict] = {}
    for record in room_scores:
        if not isinstance(record, dict):
            continue
        room_id = record.get("room_id")
        score = record.get("score")
        if room_id not in room_graph or not isinstance(score, (int, float)):
            continue
        normalized = {
            "room_id": str(room_id),
            "score": float(score),
            "evidence_type": record.get("evidence_type"),
            "reason": record.get("reason"),
        }
        previous = best_by_room.get(str(room_id))
        if previous is None or normalized["score"] > previous["score"]:
            best_by_room[str(room_id)] = normalized

    ranked = sorted(
        best_by_room.values(),
        key=lambda record: (-float(record["score"]), str(record["room_id"])),
    )
    selected = ranked[:top_k]
    if len(selected) < 2:
        raise ValueError(
            f"Detection score cache produced only {len(selected)} valid candidate rooms."
        )
    for rank, record in enumerate(selected, start=1):
        record["rank"] = rank
    return selected


def expected_room_from_grounding(payload: dict, pano_id: str) -> str | None:
    mappings = payload.get("mappings")
    if not isinstance(mappings, dict):
        mappings = payload
    room_id = mappings.get(pano_id)
    if not isinstance(room_id, str) or not room_id or room_id.lower() == "null":
        return None
    return room_id


def expected_orientation_from_manifest(manifest: dict) -> tuple[str | None, float | None]:
    captures = manifest.get("captures")
    if not isinstance(captures, list) or not captures or not isinstance(captures[0], dict):
        return None, None
    label = captures[0].get("label")
    heading = captures[0].get("heading")
    if manifest.get("heading_mode") != "museum" or label not in CARDINAL_HEADINGS:
        return None, None
    if not isinstance(heading, (int, float)):
        return None, None
    offset = normalize_heading(CARDINAL_HEADINGS[str(label)] - float(heading))
    return str(label), offset


def angular_error_deg(left: float | None, right: float | None) -> float | None:
    if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
        return None
    return abs(((float(left) - float(right) + 180.0) % 360.0) - 180.0)


def response_diagnostics(response_payload: object) -> dict:
    if not isinstance(response_payload, dict):
        return {"output_text": None, "usage": None}
    usage = response_payload.get("usage")
    if not isinstance(usage, dict):
        usage = response_payload.get("usageMetadata")
    if not isinstance(usage, dict):
        usage = response_payload.get("usage_metadata")
    return {
        "output_text": extract_output_text(response_payload),
        "usage": usage if isinstance(usage, dict) else None,
    }


def request_diagnostics(request_body: object) -> dict:
    if not isinstance(request_body, dict):
        return {
            "image_count": 0,
            "instructions": None,
            "prompt_text": None,
            "view_text_blocks": [],
            "structured_output_schema": None,
        }
    text_blocks: list[str] = []
    image_count = 0
    raw_input = request_body.get("input")
    if isinstance(raw_input, list):
        for message in raw_input:
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "input_image":
                    image_count += 1
                elif block.get("type") == "input_text" and isinstance(block.get("text"), str):
                    text_blocks.append(str(block["text"]))
    text_format = request_body.get("text")
    schema = None
    if isinstance(text_format, dict) and isinstance(text_format.get("format"), dict):
        schema = text_format["format"].get("schema")
    return {
        "image_count": image_count,
        "instructions": request_body.get("instructions"),
        "prompt_text": text_blocks[0] if text_blocks else None,
        "view_text_blocks": text_blocks[1:],
        "structured_output_schema": schema,
    }


def write_output(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_json(payload), encoding="utf-8")


def main() -> int:
    args = build_parser().parse_args()
    output_path = resolve_project_path(args.output_path) if args.output_path else default_output_path(args.pano_id)
    manifest_path, scores_path = resolve_input_paths(
        pano_id=args.pano_id,
        render_output_dir=args.render_output_dir,
        manifest_path=args.manifest_path,
        scores_json_path=args.scores_json_path,
    )
    artifacts_dir = resolve_project_path(args.artifacts_dir)
    room_graph_path = preferred_room_graph_path(artifacts_dir)
    grounding_path = artifacts_dir / "pano_room_grounding.json"
    started_at = time.perf_counter()
    payload: dict = {
        "schema": "st_nav.single_call_spatial_alignment",
        "schema_version": 1,
        "status": "error",
        "mode": "direct_images_single_call",
        "pano_id": args.pano_id,
        "paths": {
            "manifest": str(manifest_path),
            "scores": str(scores_path),
            "room_graph": str(room_graph_path),
            "grounding": str(grounding_path),
            "output": str(output_path),
        },
        "model": {
            "name": args.llm_model,
            "api_kind": args.llm_api_kind,
        },
        "request_policy": {
            "logical_vlm_call_limit": 1,
            "http_retry_limit": 1,
            "perception_fallback_enabled": False,
        },
    }
    refiner: SpatialAlignmentRefiner | None = None

    try:
        room_graph = load_json(room_graph_path)
        grounding_payload = load_json(grounding_path)
        manifest, observation, views = load_manifest_observation(
            manifest_path=manifest_path,
            pano_id=args.pano_id,
        )
        candidates = select_candidate_rooms(
            scores_path=scores_path,
            room_graph=room_graph,
            top_k=args.candidate_top_k,
        )
        candidate_room_ids = [record["room_id"] for record in candidates]
        payload["views"] = views
        payload["candidate_selection"] = {
            "source": "visual_localization.room_scores",
            "top_k": args.candidate_top_k,
            "candidates": candidates,
            "scores_sent_to_alignment_vlm": False,
        }

        grounding = build_room_grounding_from_pano_room_mapping(room_graph, grounding_payload)
        pano_to_room = grounding_payload.get("mappings")
        if not isinstance(pano_to_room, dict):
            pano_to_room = grounding_payload
        refiner = SpatialAlignmentRefiner(
            room_graph=room_graph,
            grounding_index=GroundingIndex(grounding, pano_to_room=pano_to_room),
            model=args.llm_model,
            api_key=args.llm_api_key,
            api_base=args.llm_api_base,
            api_kind=args.llm_api_kind,
            request_timeout=args.llm_timeout,
            num_ctx=args.llm_num_ctx,
            max_http_retries=1,
        )
        refinement = refiner.refine_single_call(
            observation=observation,
            candidate_room_ids=candidate_room_ids,
        )
        raw_response = refinement.pop("alignment_response_payload", None)
        payload["request"] = request_diagnostics(refiner.last_alignment_request_body)
        payload["request"].update(
            {
                "manifest_cardinal_labels_sent": False,
                "ground_truth_sent": False,
                "base64_images_saved": False,
            }
        )
        payload["response"] = response_diagnostics(raw_response)
        payload["result"] = refinement

        expected_room_id = args.expected_room_id or expected_room_from_grounding(
            grounding_payload,
            args.pano_id,
        )
        expected_direction, expected_offset = expected_orientation_from_manifest(manifest)
        spatial_alignment = refinement.get("spatial_alignment", {})
        predicted_room_id = refinement.get("alignment_predicted_room_id")
        predicted_direction = spatial_alignment.get("view_0_allocentric_direction")
        predicted_offset = spatial_alignment.get("geocentric_to_allocentric_offset_deg")
        payload["evaluation"] = {
            "expected_room_id": expected_room_id,
            "predicted_room_id": predicted_room_id,
            "room_correct": predicted_room_id == expected_room_id if expected_room_id else None,
            "expected_view_0_allocentric_direction": expected_direction,
            "predicted_view_0_allocentric_direction": predicted_direction,
            "orientation_correct": (
                predicted_direction == expected_direction if expected_direction else None
            ),
            "expected_geocentric_to_allocentric_offset_deg": expected_offset,
            "predicted_geocentric_to_allocentric_offset_deg": predicted_offset,
            "offset_error_deg": angular_error_deg(predicted_offset, expected_offset),
        }
        payload["status"] = "completed"
    except Exception as exc:
        payload["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
        if refiner is not None:
            payload.setdefault("request", request_diagnostics(refiner.last_alignment_request_body))
            payload.setdefault(
                "response", response_diagnostics(refiner.last_alignment_response_payload)
            )

    payload["latency_seconds"] = round(time.perf_counter() - started_at, 6)
    write_output(output_path, payload)

    if args.json:
        print(render_json(payload))
    elif payload["status"] == "completed":
        evaluation = payload.get("evaluation", {})
        result = payload.get("result", {})
        spatial_alignment = result.get("spatial_alignment", {})
        print(
            "\n".join(
                [
                    "Single-call spatial alignment completed",
                    f"Output: {output_path}",
                    f"Candidates: {', '.join(record['room_id'] for record in payload['candidate_selection']['candidates'])}",
                    f"Predicted room: {result.get('alignment_predicted_room_id')}",
                    f"Expected room: {evaluation.get('expected_room_id')}",
                    f"view_0 allocentric direction: {spatial_alignment.get('view_0_allocentric_direction')}",
                    f"Geocentric-to-allocentric offset: {spatial_alignment.get('geocentric_to_allocentric_offset_deg')}",
                    f"Latency: {payload['latency_seconds']:.2f}s",
                ]
            )
        )
    else:
        error = payload.get("error", {})
        print(f"Single-call spatial alignment failed: {error.get('type')}: {error.get('message')}")
        print(f"Diagnostic output: {output_path}")
    return 0 if payload["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
