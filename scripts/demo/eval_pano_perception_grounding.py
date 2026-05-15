from __future__ import annotations

import argparse
import mimetypes
import random
import subprocess
import sys
from base64 import b64encode
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

from st_nav import (
    EntityDetection,
    GroundingIndex,
    LLMRoomLocalizer,
    Observation,
    RoomLocalizer,
    VisualObservationLocalizer,
    load_dotenv,
    ModelResponseClient,
    parse_json_output,
    resolve_model_environment,
)
from st_nav.common.prompts import canonical_room_themes_text

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
        help=(
            "Score existing per-pano output JSON files. If spatial alignment needs view images and the render "
            "manifest is missing, rerun run_pano_perception.py for that pano to regenerate the render artifacts."
        ),
    )
    parser.add_argument("--render-output-dir", default="renders/pano_perception_grounding_eval")
    parser.add_argument("--render-api-key")
    parser.add_argument("--llm-api-key")
    parser.add_argument("--localizer-model", default=MODEL_ENV.model_name)
    parser.add_argument("--localizer-api-kind", default=MODEL_ENV.api_kind)
    parser.add_argument("--localizer-api-base", default=MODEL_ENV.api_base)
    parser.add_argument("--localizer-timeout", type=float, default=MODEL_ENV.request_timeout or 30.0)
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
    parser.add_argument(
        "--alignment-mode",
        choices=["textual-map", "textual-map-two-stage", "map-image"],
        default="textual-map",
        help=(
            "textual-map uses local room graph direction text plus panorama images. "
            "textual-map-two-stage first extracts per-view themes, then aligns from text only. "
            "map-image sends the allocentric map image plus panorama images."
        ),
    )
    parser.add_argument("--alignment-map-path", default="dataset/sites/british_museum/maps/level_0_allocentric.png")
    parser.add_argument("--alignment-candidate-ratio-threshold", type=float, default=0.5)
    parser.add_argument("--alignment-candidate-max", type=int, default=5)
    parser.add_argument("--alignment-context-max", type=int, default=8)
    parser.add_argument("--alignment-model")
    parser.add_argument("--alignment-timeout", type=float)
    parser.add_argument(
        "--alignment-reasoning-effort",
        choices=["minimal", "low", "medium", "high"],
        default="low",
        help="Reasoning effort for the spatial alignment VLM call. View-theme extraction keeps its existing effort.",
    )
    parser.add_argument(
        "--alignment-cache-dir",
        help=(
            "Directory for map-based spatial alignment cache. "
            "Defaults to <output-dir>/_map_spatial_alignment_cache."
        ),
    )
    parser.add_argument(
        "--view-theme-cache-dir",
        help=(
            "Directory for textual-map-two-stage per-view theme cache. "
            "Defaults to <output-dir>/_textual_map_view_theme_cache."
        ),
    )
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


def observation_from_perception_payload(payload: dict, *, include_visual_localization: bool = False) -> Observation:
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
        "source": "perception-json" if include_visual_localization else "legacy-entity-perception-json",
    }
    if include_visual_localization:
        visual_localization = payload.get("visual_localization")
        if isinstance(visual_localization, dict):
            metadata["visual_localization"] = dict(visual_localization)
        candidate_room_ids = payload.get("candidate_room_ids")
        if isinstance(candidate_room_ids, list):
            metadata["candidate_room_ids"] = [value for value in candidate_room_ids if isinstance(value, str)]

    return Observation(
        pano_id=pano_id,
        entities=entities,
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


def select_alignment_candidate_rooms(
    posterior_room_belief: object,
    *,
    ratio_threshold: float,
    max_candidates: int,
) -> list[str]:
    if not isinstance(posterior_room_belief, dict):
        return []
    ranked = [
        (room_id, float(probability))
        for room_id, probability in posterior_room_belief.items()
        if isinstance(room_id, str) and isinstance(probability, (int, float)) and float(probability) > 0.0
    ]
    if not ranked:
        return []
    ranked = sorted(ranked, key=lambda item: (-item[1], item[0]))
    top_probability = ranked[0][1]
    threshold = max(0.0, float(ratio_threshold)) * top_probability
    selected = [room_id for room_id, probability in ranked if probability >= threshold]
    return selected[: max(0, int(max_candidates))]


def alignment_context_rooms(
    candidate_room_ids: list[str],
    room_graph: dict[str, dict],
    *,
    max_context_rooms: int,
) -> list[str]:
    candidate_set = set(candidate_room_ids)
    context: list[str] = []
    for candidate_room_id in candidate_room_ids:
        node = room_graph.get(candidate_room_id, {})
        for neighbor in node.get("neighbors", []):
            if not isinstance(neighbor, dict):
                continue
            room_id = neighbor.get("target_room_id")
            if not isinstance(room_id, str) or not room_id:
                continue
            if room_id in candidate_set or room_id in context:
                continue
            context.append(room_id)
            if len(context) >= max(0, int(max_context_rooms)):
                return context
    return context


def room_theme_payload(room_id: str, room_graph: dict[str, dict], grounding_index: GroundingIndex) -> dict:
    node = room_graph.get(room_id, {})
    entry = grounding_index.room_entry(room_id) or {}
    aliases = []
    for value in list(node.get("aliases") or []) + list(entry.get("aliases") or []):
        if isinstance(value, str) and value and value not in aliases:
            aliases.append(value)
    anchors = [
        value
        for value in entry.get("anchor_entities", [])
        if isinstance(value, str) and value
    ]
    return {
        "room_id": room_id,
        "title": node.get("title"),
        "category": node.get("category"),
        "aliases": aliases[:6],
        "anchor_entities": anchors[:8],
    }


def build_alignment_room_text(
    *,
    candidate_room_ids: list[str],
    context_room_ids: list[str],
    room_graph: dict[str, dict],
    grounding_index: GroundingIndex,
) -> str:
    lines = ["Candidate rooms (valid final answers):"]
    for room_id in candidate_room_ids:
        payload = room_theme_payload(room_id, room_graph, grounding_index)
        lines.append(
            f"- {room_id}: title={payload.get('title') or 'unknown'}; "
            f"category={payload.get('category') or 'unknown'}; "
            f"aliases={payload.get('aliases')}; anchors={payload.get('anchor_entities')}."
        )
    lines.append("")
    lines.append("Context rooms (spatial references only, not valid final answers):")
    if context_room_ids:
        for room_id in context_room_ids:
            payload = room_theme_payload(room_id, room_graph, grounding_index)
            lines.append(
                f"- {room_id}: title={payload.get('title') or 'unknown'}; "
                f"category={payload.get('category') or 'unknown'}; "
                f"aliases={payload.get('aliases')}; anchors={payload.get('anchor_entities')}."
            )
    else:
        lines.append("- none")
    return "\n".join(lines)


def build_textual_map_relations(
    *,
    candidate_room_ids: list[str],
    context_room_ids: list[str],
    room_graph: dict[str, dict],
    grounding_index: GroundingIndex,
) -> str:
    included_room_ids = []
    for room_id in list(candidate_room_ids) + list(context_room_ids):
        if room_id not in included_room_ids:
            included_room_ids.append(room_id)
    included_set = set(included_room_ids)
    lines = [
        "Textual museum map relations from the local room graph:",
        "Directions are allocentric map directions. They describe room-to-room spatial layout, not the panorama heading.",
    ]
    for room_id in included_room_ids:
        payload = room_theme_payload(room_id, room_graph, grounding_index)
        lines.append(
            f"- {room_id} ({payload.get('title') or 'unknown'}):"
        )
        neighbors = []
        for neighbor in room_graph.get(room_id, {}).get("neighbors", []):
            if not isinstance(neighbor, dict):
                continue
            target_room_id = neighbor.get("target_room_id")
            direction = neighbor.get("allocentric_direction")
            if not isinstance(target_room_id, str) or target_room_id not in included_set:
                continue
            if not isinstance(direction, str) or not direction:
                direction = "unknown"
            target_payload = room_theme_payload(target_room_id, room_graph, grounding_index)
            neighbors.append((direction, target_room_id, target_payload.get("title") or "unknown"))
        if neighbors:
            for direction, target_room_id, title in sorted(neighbors, key=lambda item: (item[0], item[1])):
                lines.append(f"  - {direction}: {target_room_id} ({title})")
        else:
            lines.append("  - no listed candidate/context neighbors")
    return "\n".join(lines)


def apply_spatial_alignment_ranking(
    posterior_room_belief: object,
    alignment: object,
    candidate_room_ids: list[str],
) -> tuple[list[str], str | None]:
    ranking = ranked_room_ids(posterior_room_belief)
    if not isinstance(alignment, dict):
        return ranking, "missing_alignment"
    support = alignment.get("support")
    aligned_room_id = alignment.get("aligned_room_id")
    if support not in {"strong", "moderate"}:
        return ranking, f"support_{support or 'missing'}"
    if aligned_room_id not in set(candidate_room_ids):
        return ranking, "aligned_room_not_candidate"
    aligned_assessment = next(
        (
            assessment
            for assessment in alignment.get("candidate_assessments", [])
            if isinstance(assessment, dict) and assessment.get("candidate_room_id") == aligned_room_id
        ),
        {},
    )
    if aligned_assessment.get("spatial_consistency") == "contradicted":
        return ranking, "aligned_room_contradicted"
    if support == "moderate":
        posterior = posterior_room_belief if isinstance(posterior_room_belief, dict) else {}
        top_probability = max(
            [float(value) for value in posterior.values() if isinstance(value, (int, float))],
            default=0.0,
        )
        aligned_probability = posterior.get(aligned_room_id, 0.0)
        if not isinstance(aligned_probability, (int, float)) or float(aligned_probability) < 0.8 * top_probability:
            return ranking, "moderate_support_not_close_to_top"
    reranked = [aligned_room_id] + [room_id for room_id in ranking if room_id != aligned_room_id]
    return reranked, None


def _image_to_data_url(image_path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(str(image_path))
    if not mime_type:
        mime_type = "image/png"
    return f"data:{mime_type};base64,{b64encode(image_path.read_bytes()).decode('ascii')}"


def _clone_without_image_data(payload: object) -> object:
    if isinstance(payload, dict):
        cloned = {}
        for key, value in payload.items():
            if key == "image_url" and isinstance(value, str) and value.startswith("data:"):
                cloned[key] = "<IMAGE_DATA_URL_OMITTED>"
            else:
                cloned[key] = _clone_without_image_data(value)
        return cloned
    if isinstance(payload, list):
        return [_clone_without_image_data(value) for value in payload]
    return payload


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
    if pano_id:
        candidates.append(resolve_project_path(args.render_output_dir) / pano_id / f"{pano_id}_manifest.json")
    unique: list[Path] = []
    for candidate in candidates:
        if candidate not in unique:
            unique.append(candidate)
    return unique


def has_available_manifest(args: argparse.Namespace, *, sample: dict, payload: dict) -> bool:
    return any(candidate.exists() for candidate in manifest_path_candidates(args, sample=sample, payload=payload))


def build_map_spatial_alignment_schema(candidate_room_ids: list[str], view_ids: list[str]) -> dict:
    direction_values = ["north", "northeast", "east", "southeast", "south", "southwest", "west", "northwest", None]
    sector_schema = {
        "type": "object",
        "properties": {
            "view_id": {"type": "string", "enum": view_ids},
            "allocentric_direction": {"type": ["string", "null"], "enum": direction_values},
            "evidence_type": {
                "type": "string",
                "enum": [
                    "current_room_interior",
                    "adjacent_room_through_opening",
                    "shared_boundary_or_threshold",
                    "map_feature",
                    "ambiguous",
                ],
            },
            "matched_room_or_theme": {"type": "string"},
            "reason": {"type": "string"},
        },
        "required": ["view_id", "allocentric_direction", "evidence_type", "matched_room_or_theme", "reason"],
        "additionalProperties": False,
    }
    candidate_schema = {
        "type": "object",
        "properties": {
            "candidate_room_id": {"type": "string", "enum": candidate_room_ids},
            "spatial_consistency": {"type": "string", "enum": ["strong", "moderate", "weak", "contradicted"]},
            "current_room_evidence": {"type": "array", "items": {"type": "string"}},
            "adjacent_or_visible_only_evidence": {"type": "array", "items": {"type": "string"}},
            "contradictions": {"type": "array", "items": {"type": "string"}},
            "summary": {"type": "string"},
        },
        "required": [
            "candidate_room_id",
            "spatial_consistency",
            "current_room_evidence",
            "adjacent_or_visible_only_evidence",
            "contradictions",
            "summary",
        ],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "aligned_room_id": {"type": ["string", "null"], "enum": candidate_room_ids + [None]},
            "support": {"type": "string", "enum": ["strong", "moderate", "weak", "insufficient"]},
            "view_0_allocentric_direction": {"type": ["string", "null"], "enum": direction_values},
            "sector_alignment": {
                "type": "array",
                "items": sector_schema,
            },
            "candidate_assessments": {
                "type": "array",
                "items": candidate_schema,
            },
            "supporting_evidence": {"type": "array", "items": {"type": "string"}},
            "conflicting_evidence": {"type": "array", "items": {"type": "string"}},
            "summary": {"type": "string"},
        },
        "required": [
            "aligned_room_id",
            "support",
            "view_0_allocentric_direction",
            "sector_alignment",
            "candidate_assessments",
            "supporting_evidence",
            "conflicting_evidence",
            "summary",
        ],
        "additionalProperties": False,
    }


def build_view_theme_extraction_schema(view_ids: list[str]) -> dict:
    observation_schema = {
        "type": "object",
        "properties": {
            "view_id": {"type": "string", "enum": view_ids},
            "observed_theme": {
                "type": "string",
                "description": "Best matching canonical theme, or none. Do not invent labels.",
            },
            "confidence": {"type": "number"},
            "visible_room_label": {"type": ["string", "null"]},
            "evidence": {"type": "array", "items": {"type": "string"}},
            "visual_evidence": {"type": "array", "items": {"type": "string"}},
            "theme_matches": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "room_ids": {"type": "array", "items": {"type": "string"}},
                        "canonical_theme": {"type": "string"},
                        "confidence": {"type": "number"},
                        "reason": {"type": "string"},
                    },
                    "required": ["room_ids", "canonical_theme", "confidence", "reason"],
                    "additionalProperties": False,
                },
            },
            "current_or_adjacent": {"type": "string", "enum": ["current", "adjacent", "both", "ambiguous"]},
            "spatial_boundary_evidence": {"type": "array", "items": {"type": "string"}},
            "reason": {"type": "string"},
        },
        "required": [
            "view_id",
            "observed_theme",
            "confidence",
            "visible_room_label",
            "evidence",
            "visual_evidence",
            "theme_matches",
            "current_or_adjacent",
            "spatial_boundary_evidence",
            "reason",
        ],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "view_theme_observations": {"type": "array", "items": observation_schema},
            "summary": {"type": "string"},
        },
        "required": ["view_theme_observations", "summary"],
        "additionalProperties": False,
    }


def build_view_theme_extraction_request_body(*, model: str, captures: list[dict]) -> dict:
    view_ids = [f"view_{index}" for index, _ in enumerate(captures)]
    instructions = (
        "You extract visual observations from museum panorama sectors. Do not localize the camera or choose "
        "the current room. Room candidates and the museum map are intentionally hidden. For each sector, "
        "identify visible evidence, visible room/gallery label if any, best matching canonical room themes, "
        "and whether the evidence appears current, adjacent, both, or ambiguous. Use only the provided "
        "canonical room themes; do not invent new theme labels. Return JSON only."
    )
    task_text = "\n".join(
        [
            "Per-view theme extraction task:",
            "1. Inspect each panorama sector independently.",
            "2. Do not infer unseen map context or candidate room identity.",
            "3. Use confidence from 0.0 to 1.0 for each canonical theme match.",
            "4. If evidence is generic or could match multiple themes, return multiple low-confidence theme_matches.",
            '5. Use current_or_adjacent="ambiguous" when it is unclear whether the evidence is current-room or adjacent-room.',
            "6. Include one view_theme_observations record for each view when possible.",
            "",
            "Canonical room themes:",
            canonical_room_themes_text(),
            "",
            f"Panorama sectors in clockwise order: {', '.join(view_ids)}.",
        ]
    )
    content: list[dict] = [{"type": "input_text", "text": task_text}]
    for index, capture in enumerate(captures):
        content.extend(
            [
                {
                    "type": "input_text",
                    "text": (
                        f"Panorama sector view_{index}. This is sector {index + 1} of {len(captures)} "
                        "in clockwise order; the absolute allocentric direction is unknown."
                    ),
                },
                {
                    "type": "input_image",
                    "image_url": _image_to_data_url(Path(str(capture["path"]))),
                    "detail": "high",
                },
            ]
        )
    return {
        "model": model,
        "instructions": instructions,
        "input": [{"role": "user", "content": content}],
        "reasoning": {"effort": "low"},
        "text": {
            "format": {
                "type": "json_schema",
                "name": "view_theme_extraction",
                "strict": True,
                "schema": build_view_theme_extraction_schema(view_ids),
            }
        },
    }


def build_map_spatial_alignment_request_body(
    *,
    model: str,
    reasoning_effort: str,
    map_path: Path,
    captures: list[dict],
    candidate_room_ids: list[str],
    context_room_ids: list[str],
    room_graph: dict[str, dict],
    grounding_index: GroundingIndex,
) -> dict:
    view_ids = [f"view_{index}" for index, _ in enumerate(captures)]
    room_text = build_alignment_room_text(
        candidate_room_ids=candidate_room_ids,
        context_room_ids=context_room_ids,
        room_graph=room_graph,
        grounding_index=grounding_index,
    )
    relation_text = build_textual_map_relations(
        candidate_room_ids=candidate_room_ids,
        context_room_ids=context_room_ids,
        room_graph=room_graph,
        grounding_index=grounding_index,
    )
    instructions = (
        "You are a conservative museum spatial alignment verifier. Your job is to decide which candidate "
        "room the camera is physically inside, not which room or theme is merely visible. Use the allocentric "
        "floor map image to test whether one clockwise rotation of all panorama sectors is spatially "
        "consistent with a candidate room's map position, openings, neighboring rooms, and map features. "
        "Use candidate/context themes only as secondary labels for visual evidence; theme similarity alone "
        "is not spatial alignment. The panorama global heading is unknown, and sectors are ordered clockwise. "
        "Final aligned_room_id must be one of the candidate rooms only; context rooms can explain views "
        "through openings but are not valid final answers. Do not output a probability distribution. "
        "Return JSON only."
    )
    task_text = "\n".join(
        [
            "Map-based spatial alignment task:",
            "1. Read the floor map as an allocentric map.",
            "2. Evaluate all 8 panorama sectors in clockwise order. For each sector, decide whether it is "
            "current_room_interior, adjacent_room_through_opening, shared_boundary_or_threshold, map_feature, "
            "or ambiguous.",
            "3. Find a single clockwise rotation from panorama sectors to the allocentric map. Do not cherry-pick "
            "only the sectors that support one answer.",
            "4. Compare every candidate room. For each candidate, separate current-room evidence from adjacent-room "
            "or visible-only evidence, and list contradictions.",
            "5. Choose a candidate only when its map position explains the 8-sector pattern better than the other "
            "candidates. If evidence is mostly theme similarity, mostly visible adjacent-room content, or cannot "
            "support one consistent rotation, use support=weak or support=insufficient.",
            "6. Use support=strong only for distinctive spatial evidence such as multiple map-consistent openings, "
            "thresholds, signs, stairs, court/shop views, or neighboring rooms under the same rotation. Use "
            "support=moderate only when the spatial evidence is plausible but incomplete. Do not mark strong "
            "because artifacts or themes match.",
            "7. sector_alignment must contain exactly one record for each panorama sector. Use allocentric_direction=null "
            "and evidence_type=ambiguous when a sector has no reliable spatial evidence.",
            "",
            room_text,
            "",
            relation_text,
            "",
            f"Panorama sectors in clockwise order: {', '.join(view_ids)}.",
        ]
    )
    content: list[dict] = [
        {"type": "input_text", "text": task_text},
        {"type": "input_text", "text": "Allocentric floor map image:"},
        {"type": "input_image", "image_url": _image_to_data_url(map_path), "detail": "high"},
    ]
    for index, capture in enumerate(captures):
        content.extend(
            [
                {
                    "type": "input_text",
                    "text": (
                        f"Panorama sector view_{index}. This is sector {index + 1} of {len(captures)} "
                        "in clockwise order; the absolute allocentric direction is unknown."
                    ),
                },
                {
                    "type": "input_image",
                    "image_url": _image_to_data_url(Path(str(capture["path"]))),
                    "detail": "high",
                },
            ]
        )
    return {
        "model": model,
        "instructions": instructions,
        "input": [{"role": "user", "content": content}],
        "reasoning": {"effort": reasoning_effort},
        "text": {
            "format": {
                "type": "json_schema",
                "name": "map_spatial_alignment",
                "strict": True,
                "schema": build_map_spatial_alignment_schema(candidate_room_ids, view_ids),
            }
        },
    }


def build_textual_map_spatial_alignment_request_body(
    *,
    model: str,
    reasoning_effort: str,
    captures: list[dict],
    candidate_room_ids: list[str],
    context_room_ids: list[str],
    room_graph: dict[str, dict],
    grounding_index: GroundingIndex,
) -> dict:
    view_ids = [f"view_{index}" for index, _ in enumerate(captures)]
    room_text = build_alignment_room_text(
        candidate_room_ids=candidate_room_ids,
        context_room_ids=context_room_ids,
        room_graph=room_graph,
        grounding_index=grounding_index,
    )
    relation_text = build_textual_map_relations(
        candidate_room_ids=candidate_room_ids,
        context_room_ids=context_room_ids,
        room_graph=room_graph,
        grounding_index=grounding_index,
    )
    instructions = (
        "You are a conservative museum spatial alignment verifier. Determine which candidate room the camera "
        "is physically inside by aligning egocentric panorama-sector observations to textual allocentric room "
        "relations. The panorama global heading is unknown; view_0 is not north. Try possible rotations between "
        "the clockwise view sequence and the allocentric map directions. Use room themes as visual labels, but "
        "theme similarity alone is not spatial alignment. Distinguish current-room interior evidence from "
        "adjacent rooms visible through openings. Final aligned_room_id must be one candidate room only, or null "
        "when the spatial evidence is insufficient. Context rooms are spatial references only, not valid answers. "
        "Do not output a probability distribution. Return JSON only."
    )
    task_text = "\n".join(
        [
            "Textual-map spatial alignment task:",
            "1. Inspect all 8 panorama sectors in clockwise order and summarize visible themes, signs, thresholds, openings, and neighboring rooms.",
            "2. Use the textual museum map relations below as the only allocentric map source.",
            "3. Try rotations from view_0 to north, northeast, east, southeast, south, southwest, west, and northwest.",
            "4. Compare every candidate room under the best rotation. A good candidate should explain both current-room interior cues and adjacent/context rooms seen in the correct relative directions.",
            "5. Penalize candidates whose evidence is mostly an adjacent room seen through an opening, or only a shared theme with no directional support.",
            "6. Use support=strong only when multiple sector observations are consistent under one rotation. Use support=weak or insufficient when the same themes appear in adjacent candidate rooms and direction evidence is not distinctive.",
            "7. sector_alignment should include one record for each view when possible; use allocentric_direction=null and evidence_type=ambiguous when a sector has no reliable spatial evidence.",
            "",
            room_text,
            "",
            relation_text,
            "",
            f"Panorama sectors in clockwise order: {', '.join(view_ids)}.",
        ]
    )
    content: list[dict] = [{"type": "input_text", "text": task_text}]
    for index, capture in enumerate(captures):
        content.extend(
            [
                {
                    "type": "input_text",
                    "text": (
                        f"Panorama sector view_{index}. This is sector {index + 1} of {len(captures)} "
                        "in clockwise order; the absolute allocentric direction is unknown."
                    ),
                },
                {
                    "type": "input_image",
                    "image_url": _image_to_data_url(Path(str(capture["path"]))),
                    "detail": "high",
                },
            ]
        )
    return {
        "model": model,
        "instructions": instructions,
        "input": [{"role": "user", "content": content}],
        "reasoning": {"effort": reasoning_effort},
        "text": {
            "format": {
                "type": "json_schema",
                "name": "textual_map_spatial_alignment",
                "strict": True,
                "schema": build_map_spatial_alignment_schema(candidate_room_ids, view_ids),
            }
        },
    }


def build_two_stage_textual_map_alignment_request_body(
    *,
    model: str,
    reasoning_effort: str,
    view_theme_payload: dict,
    candidate_room_ids: list[str],
    context_room_ids: list[str],
    room_graph: dict[str, dict],
    grounding_index: GroundingIndex,
) -> dict:
    observations = view_theme_payload.get("view_theme_observations")
    if not isinstance(observations, list):
        observations = []
    view_ids = [
        str(observation.get("view_id"))
        for observation in observations
        if isinstance(observation, dict) and isinstance(observation.get("view_id"), str)
    ]
    room_text = build_alignment_room_text(
        candidate_room_ids=candidate_room_ids,
        context_room_ids=context_room_ids,
        room_graph=room_graph,
        grounding_index=grounding_index,
    )
    relation_text = build_textual_map_relations(
        candidate_room_ids=candidate_room_ids,
        context_room_ids=context_room_ids,
        room_graph=room_graph,
        grounding_index=grounding_index,
    )
    instructions = (
        "You are a museum spatial alignment reasoner. You do not see images in this step. "
        "Use the extracted per-view observations, candidate rooms, context rooms, and textual allocentric "
        "room relations to decide which candidate room the camera is physically inside. The panorama heading "
        "is unknown; view_0 is not north. Candidate rooms are the only valid final answers. Return JSON only."
    )
    task_text = "\n".join(
        [
            "Two-stage textual-map spatial alignment task:",
            "1. Treat view_theme_observations as fixed perception output from a previous image-only step.",
            "2. Try possible rotations between the clockwise view sequence and the allocentric map directions.",
            "3. Use the per-view themes as visual evidence, but do not treat theme similarity alone as proof of location.",
            "4. Separate evidence that appears to be inside the current room from evidence that may come from adjacent/context rooms.",
            "5. Prefer the candidate whose room theme and neighboring-room context are most consistent under one rotation.",
            "6. If several candidates explain the observations similarly, return weak or insufficient support.",
            "",
            room_text,
            "",
            relation_text,
            "",
            "Fixed view theme observations:",
            render_json({"view_theme_observations": observations, "summary": view_theme_payload.get("summary", "")}),
        ]
    )
    return {
        "model": model,
        "instructions": instructions,
        "input": [{"role": "user", "content": [{"type": "input_text", "text": task_text}]}],
        "reasoning": {"effort": reasoning_effort},
        "text": {
            "format": {
                "type": "json_schema",
                "name": "two_stage_textual_map_spatial_alignment",
                "strict": True,
                "schema": build_map_spatial_alignment_schema(candidate_room_ids, view_ids),
            }
        },
    }


def spatial_alignment_cache_path(
    output_path: Path,
    *,
    output_root: Path | None = None,
    cache_root: Path | None = None,
) -> Path:
    if cache_root is None:
        return output_path.parent / "_map_spatial_alignment_cache" / f"{output_path.stem}.json"
    if output_root is None:
        relative_path = Path(output_path.name)
    else:
        try:
            relative_path = output_path.relative_to(output_root)
        except ValueError:
            relative_path = Path(output_path.parent.name) / output_path.name
    return cache_root / relative_path.with_suffix(".json")


def map_cache_identity(
    *,
    alignment_mode: str,
    map_path: Path | None,
    model: str,
    api_kind: str,
    reasoning_effort: str,
    candidate_room_ids: list[str],
    context_room_ids: list[str],
    captures: list[dict],
    relation_text: str,
    view_theme_payload: dict | None = None,
) -> dict:
    identity = {
        "cache_version": 8,
        "alignment_mode": alignment_mode,
        "model": model,
        "api_kind": api_kind,
        "reasoning_effort": reasoning_effort,
        "candidate_room_ids": list(candidate_room_ids),
        "context_room_ids": list(context_room_ids),
        "view_paths": [str(capture.get("path")) for capture in captures],
        "relation_text": relation_text,
    }
    if view_theme_payload is not None:
        identity["view_theme_payload"] = view_theme_payload
    if map_path is not None:
        stat = map_path.stat()
        identity.update(
            {
                "map_path": str(map_path),
                "map_size": stat.st_size,
                "map_mtime_ns": stat.st_mtime_ns,
            }
        )
    return identity


def view_theme_cache_path(
    output_path: Path,
    *,
    output_root: Path | None = None,
    cache_root: Path | None = None,
) -> Path:
    root = cache_root or output_path.parent / "_textual_map_view_theme_cache"
    return spatial_alignment_cache_path(output_path, output_root=output_root, cache_root=root)


def view_theme_cache_identity(*, model: str, api_kind: str, captures: list[dict]) -> dict:
    return {
        "cache_version": 2,
        "model": model,
        "api_kind": api_kind,
        "view_paths": [str(capture.get("path")) for capture in captures],
    }


def load_json_cache(cache_path: Path, identity: dict, payload_key: str) -> dict | None:
    if not cache_path.exists():
        return None
    try:
        payload = load_json(cache_path)
    except (OSError, ValueError):
        return None
    if payload.get("identity") != identity:
        return None
    parsed = payload.get(payload_key)
    return parsed if isinstance(parsed, dict) else None


def write_json_cache(
    cache_path: Path,
    *,
    identity: dict,
    request_body: dict,
    response_payload: dict,
    payload_key: str,
    parsed_payload: dict,
) -> None:
    cache_payload = {
        "identity": identity,
        "request_body": _clone_without_image_data(request_body),
        "response_payload": response_payload,
        payload_key: parsed_payload,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(render_json(cache_payload), encoding="utf-8")


def load_spatial_alignment_cache(cache_path: Path, identity: dict) -> dict | None:
    if not cache_path.exists():
        return None
    try:
        payload = load_json(cache_path)
    except (OSError, ValueError):
        return None
    if payload.get("identity") != identity:
        return None
    parsed = payload.get("parsed_alignment")
    return parsed if isinstance(parsed, dict) else None


def write_spatial_alignment_cache(
    cache_path: Path,
    *,
    identity: dict,
    request_body: dict,
    response_payload: dict,
    parsed_alignment: dict,
) -> None:
    cache_payload = {
        "identity": identity,
        "request_body": _clone_without_image_data(request_body),
        "response_payload": response_payload,
        "parsed_alignment": parsed_alignment,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(render_json(cache_payload), encoding="utf-8")


def run_map_spatial_alignment(
    args: argparse.Namespace,
    *,
    payload: dict,
    output_path: Path,
    output_root: Path | None = None,
    cache_root: Path | None = None,
    view_theme_cache_root: Path | None = None,
    posterior_room_belief: dict,
    room_graph: dict[str, dict],
    grounding_index: GroundingIndex,
) -> dict:
    candidate_room_ids = select_alignment_candidate_rooms(
        posterior_room_belief,
        ratio_threshold=args.alignment_candidate_ratio_threshold,
        max_candidates=args.alignment_candidate_max,
    )
    if len(candidate_room_ids) < 2:
        return {
            "status": "skipped",
            "candidate_room_ids": candidate_room_ids,
            "context_room_ids": [],
            "skipped_reason": "fewer_than_two_candidates",
        }
    context_room_ids = alignment_context_rooms(
        candidate_room_ids,
        room_graph,
        max_context_rooms=args.alignment_context_max,
    )
    alignment_mode = getattr(args, "alignment_mode", "textual-map")
    map_path = None
    if alignment_mode == "map-image":
        map_path = resolve_project_path(args.alignment_map_path)
        if not map_path.exists():
            raise FileNotFoundError(f"Spatial alignment map not found: {map_path}")
    pano_id = payload.get("pano_id")
    if not isinstance(pano_id, str) or not pano_id:
        pano_id = output_path.stem
    fallback_manifest_path = resolve_project_path(args.render_output_dir) / pano_id / f"{pano_id}_manifest.json"
    captures = load_manifest_captures(payload, fallback_manifest_path=fallback_manifest_path)
    model = args.alignment_model or args.detector_model
    reasoning_effort = getattr(args, "alignment_reasoning_effort", "low")
    api_kind = args.detector_api_kind or MODEL_ENV.api_kind
    api_base = args.detector_api_base or MODEL_ENV.api_base
    timeout = args.alignment_timeout or args.vlm_timeout or MODEL_ENV.request_timeout or 180.0
    client = ModelResponseClient(
        provider=MODEL_ENV.provider,
        api_key=args.llm_api_key or MODEL_ENV.api_key,
        api_base=api_base,
        api_kind=api_kind,
        request_timeout=float(timeout),
        num_ctx=MODEL_ENV.num_ctx,
        temperature=MODEL_ENV.temperature,
    )
    view_theme_payload = None
    view_theme_cache = None
    if alignment_mode == "textual-map-two-stage":
        payload_observations = payload.get("view_theme_observations")
        if isinstance(payload_observations, list) and payload_observations:
            view_theme_payload = {
                "view_theme_observations": [record for record in payload_observations if isinstance(record, dict)],
                "summary": payload.get("view_theme_summary", ""),
            }
        else:
            view_theme_cache = view_theme_cache_path(output_path, output_root=output_root, cache_root=view_theme_cache_root)
            view_theme_identity = view_theme_cache_identity(model=model, api_kind=api_kind, captures=captures)
            view_theme_payload = load_json_cache(view_theme_cache, view_theme_identity, "view_theme_payload")
        if view_theme_payload is None:
            view_theme_request = build_view_theme_extraction_request_body(model=model, captures=captures)
            view_theme_response = client.create(view_theme_request)
            view_theme_payload = parse_json_output(view_theme_response)
            write_json_cache(
                view_theme_cache,
                identity=view_theme_identity,
                request_body=view_theme_request,
                response_payload=view_theme_response,
                payload_key="view_theme_payload",
                parsed_payload=view_theme_payload,
            )
    relation_text = build_textual_map_relations(
        candidate_room_ids=candidate_room_ids,
        context_room_ids=context_room_ids,
        room_graph=room_graph,
        grounding_index=grounding_index,
    )
    identity = map_cache_identity(
        alignment_mode=alignment_mode,
        map_path=map_path,
        model=model,
        api_kind=api_kind,
        reasoning_effort=reasoning_effort,
        candidate_room_ids=candidate_room_ids,
        context_room_ids=context_room_ids,
        captures=captures,
        relation_text=relation_text,
        view_theme_payload=view_theme_payload,
    )
    cache_path = spatial_alignment_cache_path(output_path, output_root=output_root, cache_root=cache_root)
    cached = load_spatial_alignment_cache(cache_path, identity)
    if cached is not None:
        return {
            "status": "cached",
            "candidate_room_ids": candidate_room_ids,
            "context_room_ids": context_room_ids,
            "alignment": cached,
            "cache_path": str(cache_path),
            "view_theme_cache_path": str(view_theme_cache) if view_theme_cache is not None else None,
            "view_theme_payload": view_theme_payload,
        }

    if alignment_mode == "map-image":
        request_body = build_map_spatial_alignment_request_body(
            model=model,
            reasoning_effort=reasoning_effort,
            map_path=map_path,
            captures=captures,
            candidate_room_ids=candidate_room_ids,
            context_room_ids=context_room_ids,
            room_graph=room_graph,
            grounding_index=grounding_index,
        )
    elif alignment_mode == "textual-map-two-stage":
        request_body = build_two_stage_textual_map_alignment_request_body(
            model=model,
            reasoning_effort=reasoning_effort,
            view_theme_payload=view_theme_payload or {},
            candidate_room_ids=candidate_room_ids,
            context_room_ids=context_room_ids,
            room_graph=room_graph,
            grounding_index=grounding_index,
        )
    else:
        request_body = build_textual_map_spatial_alignment_request_body(
            model=model,
            reasoning_effort=reasoning_effort,
            captures=captures,
            candidate_room_ids=candidate_room_ids,
            context_room_ids=context_room_ids,
            room_graph=room_graph,
            grounding_index=grounding_index,
        )
    response_payload = client.create(request_body)
    parsed = parse_json_output(response_payload)
    write_spatial_alignment_cache(
        cache_path,
        identity=identity,
        request_body=request_body,
        response_payload=response_payload,
        parsed_alignment=parsed,
    )
    return {
        "status": "scored",
        "candidate_room_ids": candidate_room_ids,
        "context_room_ids": context_room_ids,
        "alignment": parsed,
        "cache_path": str(cache_path),
        "view_theme_cache_path": str(view_theme_cache) if view_theme_cache is not None else None,
        "view_theme_payload": view_theme_payload,
    }


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


def localize_integrated_visual(
    *,
    payload: dict,
    expected_room_id: str,
    room_graph: dict[str, dict],
    grounding_index: GroundingIndex,
) -> dict:
    observation = observation_from_perception_payload(payload, include_visual_localization=True)
    localizer = VisualObservationLocalizer(room_graph=room_graph, grounding_index=grounding_index)
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
    if args.no_detection_cache:
        command.append("--no-detection-cache")
    if args.enable_view_themes:
        command.append("--enable-view-themes")
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
    expected_room_id = str(sample["room_id"])
    if args.pipeline == "integrated-visual":
        try:
            localization_payload = localize_integrated_visual(
                payload=payload,
                expected_room_id=expected_room_id,
                room_graph=room_graph,
                grounding_index=grounding_index,
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
        posterior_room_belief = localization_payload.get("room_belief", {})
        posterior_ranking = ranked_room_ids(posterior_room_belief)
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
            try:
                localization_payload = localize_integrated_visual(
                    payload=payload,
                    expected_room_id=expected_room_id,
                    room_graph=room_graph,
                    grounding_index=grounding_index,
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
            posterior_room_belief = localization_payload.get("room_belief", {})
            posterior_ranking = ranked_room_ids(posterior_room_belief)
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
            "status": "scored",
        }
        if auto_rerender_result is not None:
            record["auto_rerendered_manifest"] = True
        if getattr(args, "enable_spatial_alignment", False):
            try:
                alignment_cache_root = (
                    resolve_project_path(args.alignment_cache_dir)
                    if getattr(args, "alignment_cache_dir", None)
                    else output_dir / "_map_spatial_alignment_cache"
                )
                view_theme_cache_root = (
                    resolve_project_path(args.view_theme_cache_dir)
                    if getattr(args, "view_theme_cache_dir", None)
                    else output_dir / "_textual_map_view_theme_cache"
                )
                alignment_result = run_map_spatial_alignment(
                    args,
                    payload=payload,
                    output_path=output_path,
                    output_root=output_dir,
                    cache_root=alignment_cache_root,
                    view_theme_cache_root=view_theme_cache_root,
                    posterior_room_belief=posterior_room_belief,
                    room_graph=room_graph,
                    grounding_index=grounding_index,
                )
            except Exception as error:
                return {
                    "pano_id": sample["pano_id"],
                    "expected_room_id": expected_room_id,
                    "output_path": str(output_path),
                    "status": "failed",
                    "error": {"message": str(error)},
                }
            alignment = alignment_result.get("alignment")
            spatial_ranking = list(posterior_ranking)
            ranking_skipped_reason = alignment_result.get("skipped_reason")
            if isinstance(alignment, dict):
                spatial_ranking, ranking_skipped_reason = apply_spatial_alignment_ranking(
                    posterior_room_belief,
                    alignment,
                    alignment_result.get("candidate_room_ids", []),
                )
            record["spatial_alignment_candidates"] = list(alignment_result.get("candidate_room_ids", []))
            record["spatial_alignment_context_rooms"] = list(alignment_result.get("context_room_ids", []))
            record["spatial_alignment"] = alignment if isinstance(alignment, dict) else None
            record["spatial_alignment_cache_path"] = alignment_result.get("cache_path")
            record["view_theme_cache_path"] = alignment_result.get("view_theme_cache_path")
            if isinstance(alignment_result.get("view_theme_payload"), dict):
                record["view_theme_observations"] = alignment_result["view_theme_payload"].get("view_theme_observations", [])
            record["spatial_alignment_status"] = alignment_result.get("status")
            record["spatial_alignment_skipped_reason"] = ranking_skipped_reason
            record["spatial_aligned_ranked_rooms"] = spatial_ranking
            record["spatial_aligned"] = ranking_payload_from_order(spatial_ranking, expected_room_id)
        return record
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
    if args.enable_spatial_alignment and args.alignment_mode == "map-image":
        alignment_map_path = resolve_project_path(args.alignment_map_path)
        if not alignment_map_path.exists():
            raise RuntimeError(f"Spatial alignment map not found: {alignment_map_path}")
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
                    error_message = f" error={message.strip()[:500]}"
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
            "alignment_mode": args.alignment_mode,
            "alignment_map_path": args.alignment_map_path,
            "alignment_candidate_ratio_threshold": args.alignment_candidate_ratio_threshold,
            "alignment_candidate_max": args.alignment_candidate_max,
            "alignment_context_max": args.alignment_context_max,
            "alignment_model": args.alignment_model or args.detector_model,
            "alignment_timeout": args.alignment_timeout,
            "alignment_reasoning_effort": args.alignment_reasoning_effort,
            "alignment_cache_dir": (
                str(resolve_project_path(args.alignment_cache_dir))
                if args.alignment_cache_dir
                else str(output_dir / "_map_spatial_alignment_cache")
            ),
            "view_theme_cache_dir": (
                str(resolve_project_path(args.view_theme_cache_dir))
                if args.view_theme_cache_dir
                else str(output_dir / "_textual_map_view_theme_cache")
            ),
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
