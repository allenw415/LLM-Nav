from __future__ import annotations

import hashlib
import json
import mimetypes
import re
from base64 import b64encode
from collections import Counter
from pathlib import Path
from typing import Callable

from PIL import Image, ImageDraw, ImageEnhance, ImageFont, ImageOps

from ..common.env import resolve_model_environment, resolve_task_num_ctx
from ..common.model_client import (
    DEFAULT_OPENAI_API_BASE,
    ModelResponseClient,
    extract_output_text,
    parse_json_output,
    resolve_api_kind,
)
from ..common.room_profiles import compact_visual_profile
from ..common.types import Observation
from ..perception.renderer import normalize_heading


ALLOCENTRIC_DIRECTIONS = ("north", "east", "south", "west")
DIRECTION_HEADINGS = {"north": 0.0, "east": 90.0, "south": 180.0, "west": 270.0}
LOCATION_SCOPES = ("current_room", "adjacent_room", "transition", "uncertain")
MATCH_STATUSES = ("specific_match", "broad_match", "different_room", "not_visible", "uncertain")
EVIDENCE_SPECIFICITIES = ("broad", "specific", "exact")
DEFAULT_COMPACT_OUTPUT_TOKENS = 1024
CANDIDATE_COLORS = (
    (230, 25, 75, 105),
    (60, 180, 75, 105),
    (0, 110, 220, 105),
    (245, 130, 48, 105),
    (145, 30, 180, 105),
)


def _normalized_text(value: object) -> str:
    return " ".join(str(value).strip().lower().split()) if isinstance(value, str) else ""


def _profile_fields(node: dict) -> tuple[str, list[str]]:
    profile = compact_visual_profile(node)
    description = profile.get("short_description")
    raw_cues = profile.get("visual_cues")
    cues = [str(value) for value in raw_cues if isinstance(value, str) and value] if isinstance(raw_cues, list) else []
    return description if isinstance(description, str) else "", cues


def _floor_is_on_level_zero(value: object) -> bool:
    return str(value) in {"0", "0/1"}


def validate_map_regions(room_graph: dict[str, dict], region_payload: dict) -> None:
    if region_payload.get("coordinate_system") != "normalized_0_1":
        raise ValueError("Map regions must use coordinate_system=normalized_0_1.")
    if region_payload.get("north_up") is not True:
        raise ValueError("Map regions must declare north_up=true.")
    rooms = region_payload.get("rooms")
    if not isinstance(rooms, dict):
        raise ValueError("Map regions payload must contain a rooms object.")

    missing = [
        room_id
        for room_id, node in room_graph.items()
        if isinstance(node, dict) and _floor_is_on_level_zero(node.get("floor")) and room_id not in rooms
    ]
    if missing:
        raise ValueError("Map regions are missing Level 0 rooms: " + ", ".join(sorted(missing)))

    for room_id, record in rooms.items():
        if not isinstance(record, dict):
            raise ValueError(f"Map region for {room_id} must be an object.")
        polygons = record.get("polygons")
        anchor = record.get("label_anchor")
        if not isinstance(polygons, list) or not polygons:
            raise ValueError(f"Map region for {room_id} must contain at least one polygon.")
        if not isinstance(anchor, list) or len(anchor) != 2:
            raise ValueError(f"Map region for {room_id} must contain a two-value label_anchor.")
        for polygon in polygons:
            if not isinstance(polygon, list) or len(polygon) < 3:
                raise ValueError(f"Map polygon for {room_id} must contain at least three points.")
            for point in polygon:
                if (
                    not isinstance(point, list)
                    or len(point) != 2
                    or any(not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 1.0 for value in point)
                ):
                    raise ValueError(f"Map polygon point for {room_id} must be normalized to [0, 1].")
        if any(not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 1.0 for value in anchor):
            raise ValueError(f"Map label anchor for {room_id} must be normalized to [0, 1].")


def _font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    candidates = (
        Path("/usr/share/fonts/truetype/dejavu") / name,
        Path("/usr/share/fonts/dejavu") / name,
    )
    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _pixel_points(polygon: list[list[float]], width: int, height: int) -> list[tuple[int, int]]:
    return [(round(float(x) * width), round(float(y) * height)) for x, y in polygon]


def _pixel_anchor(anchor: list[float], width: int, height: int) -> tuple[int, int]:
    return round(float(anchor[0]) * width), round(float(anchor[1]) * height)


def _draw_centered_label(
    draw: ImageDraw.ImageDraw,
    position: tuple[int, int],
    text: str,
    *,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int, int],
    background: tuple[int, int, int, int],
) -> None:
    bbox = draw.textbbox((0, 0), text, font=font, stroke_width=1)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    x = position[0] - width // 2
    y = position[1] - height // 2
    draw.rounded_rectangle((x - 5, y - 3, x + width + 5, y + height + 3), radius=4, fill=background)
    draw.text((x, y), text, font=font, fill=fill, stroke_width=1, stroke_fill=(255, 255, 255, 220))


def render_semantic_map(
    *,
    base_map_path: Path,
    region_payload: dict,
    room_graph: dict[str, dict],
    candidate_room_ids: list[str],
    output_path: Path,
) -> dict:
    if len(candidate_room_ids) != 5 or len(set(candidate_room_ids)) != 5:
        raise ValueError("Map-grounded alignment requires exactly five unique candidate rooms.")
    if not base_map_path.is_file():
        raise FileNotFoundError(f"Map image not found: {base_map_path}")
    validate_map_regions(room_graph, region_payload)

    rooms = region_payload["rooms"]
    missing_candidates = [room_id for room_id in candidate_room_ids if room_id not in rooms]
    if missing_candidates:
        raise ValueError("Candidate rooms missing from map regions: " + ", ".join(missing_candidates))

    candidate_room_ids = sorted(candidate_room_ids)
    map_labels = {room_id: f"C{index}" for index, room_id in enumerate(candidate_room_ids, start=1)}
    neighbor_room_ids: set[str] = set()
    for room_id in candidate_room_ids:
        for neighbor in room_graph.get(room_id, {}).get("neighbors", []):
            if not isinstance(neighbor, dict):
                continue
            neighbor_id = neighbor.get("target_room_id")
            neighbor_node = room_graph.get(str(neighbor_id), {})
            if (
                isinstance(neighbor_id, str)
                and neighbor_id not in candidate_room_ids
                and _floor_is_on_level_zero(neighbor_node.get("floor"))
            ):
                if neighbor_id not in rooms:
                    raise ValueError(f"Level 0 neighbor {neighbor_id} is missing from map regions.")
                neighbor_room_ids.add(neighbor_id)

    base = Image.open(base_map_path).convert("RGBA")
    width, height = base.size
    grayscale = ImageOps.grayscale(base.convert("RGB")).convert("RGBA")
    dimmed = ImageEnhance.Brightness(grayscale).enhance(0.78)
    map_canvas = Image.blend(base, dimmed, 0.72)
    overlay = Image.new("RGBA", map_canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    neighbor_color = (255, 215, 0, 255)
    for room_id in sorted(neighbor_room_ids):
        record = rooms[room_id]
        for polygon in record["polygons"]:
            draw.line(_pixel_points(polygon + [polygon[0]], width, height), fill=neighbor_color, width=5, joint="curve")
        _draw_centered_label(
            draw,
            _pixel_anchor(record["label_anchor"], width, height),
            room_id,
            font=_font(22, bold=True),
            fill=(20, 20, 20, 255),
            background=(255, 235, 80, 235),
        )

    legend_records: list[dict] = []
    for index, room_id in enumerate(candidate_room_ids):
        record = rooms[room_id]
        color = CANDIDATE_COLORS[index]
        opaque = (color[0], color[1], color[2], 255)
        for polygon in record["polygons"]:
            points = _pixel_points(polygon, width, height)
            draw.polygon(points, fill=color)
            draw.line(points + [points[0]], fill=opaque, width=9, joint="curve")
        label = map_labels[room_id]
        _draw_centered_label(
            draw,
            _pixel_anchor(record["label_anchor"], width, height),
            f"{label} {room_id}",
            font=_font(25, bold=True),
            fill=(15, 15, 15, 255),
            background=(255, 255, 255, 240),
        )
        legend_records.append({"map_label": label, "room_id": room_id, "color_rgb": list(opaque[:3])})

    map_canvas = Image.alpha_composite(map_canvas, overlay)
    sidebar_width = 500
    result = Image.new("RGBA", (width + sidebar_width, height), (250, 250, 250, 255))
    result.paste(map_canvas, (0, 0))
    sidebar = ImageDraw.Draw(result)
    sidebar.rectangle((width, 0, width + sidebar_width, height), fill=(250, 250, 250, 255))
    sidebar.text((width + 32, 30), "MAP-GROUNDED ALIGNMENT", font=_font(25, bold=True), fill=(20, 20, 20, 255))
    sidebar.text((width + 32, 82), "MAP NORTH", font=_font(25, bold=True), fill=(20, 20, 20, 255))
    sidebar.line((width + 205, 145, width + 205, 92), fill=(20, 20, 20, 255), width=7)
    sidebar.polygon(
        [(width + 205, 82), (width + 187, 108), (width + 223, 108)],
        fill=(20, 20, 20, 255),
    )
    sidebar.text((width + 195, 150), "N", font=_font(25, bold=True), fill=(20, 20, 20, 255))

    y = 215
    sidebar.text((width + 32, y), "CANDIDATE ROOMS", font=_font(24, bold=True), fill=(20, 20, 20, 255))
    y += 48
    for index, record in enumerate(legend_records):
        rgb = tuple(record["color_rgb"])
        sidebar.rounded_rectangle((width + 32, y, width + 72, y + 40), radius=5, fill=rgb + (255,))
        sidebar.text(
            (width + 90, y + 4),
            f"{record['map_label']} = {record['room_id']}",
            font=_font(24, bold=True),
            fill=(20, 20, 20, 255),
        )
        y += 60
    y += 20
    sidebar.line((width + 32, y + 15, width + 82, y + 15), fill=neighbor_color, width=6)
    sidebar.text((width + 98, y), "One-hop neighbor", font=_font(22, bold=True), fill=(20, 20, 20, 255))
    y += 60
    sidebar.multiline_text(
        (width + 32, y),
        "The base map is north-up and schematic.\nUse candidate colors only as IDs;\nthey do not encode prior scores.\nUse topology and relative layout, not scale.",
        font=_font(21),
        fill=(40, 40, 40, 255),
        spacing=10,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(output_path, format="PNG")
    digest = hashlib.sha256(output_path.read_bytes()).hexdigest()
    return {
        "path": str(output_path),
        "sha256": digest,
        "width": result.width,
        "height": result.height,
        "north_up": True,
        "candidate_map_labels": map_labels,
        "neighbor_room_ids": sorted(neighbor_room_ids),
    }


def build_candidate_cards(room_graph: dict[str, dict], candidate_room_ids: list[str]) -> list[dict]:
    candidate_room_ids = sorted(dict.fromkeys(candidate_room_ids))
    if len(candidate_room_ids) != 5:
        raise ValueError("Map-grounded alignment requires exactly five unique candidate rooms.")
    unknown = [room_id for room_id in candidate_room_ids if room_id not in room_graph]
    if unknown:
        raise ValueError("Unknown candidate rooms: " + ", ".join(unknown))

    candidate_cues: dict[str, list[str]] = {}
    cue_counts: Counter[str] = Counter()
    for room_id in candidate_room_ids:
        _, cues = _profile_fields(room_graph[room_id])
        candidate_cues[room_id] = cues
        cue_counts.update({_normalized_text(cue) for cue in cues if _normalized_text(cue)})

    cards: list[dict] = []
    for index, room_id in enumerate(candidate_room_ids, start=1):
        node = room_graph[room_id]
        description, cues = _profile_fields(node)
        shared_cues = [cue for cue in cues if cue_counts[_normalized_text(cue)] > 1]
        distinctive_cues = [cue for cue in cues if cue_counts[_normalized_text(cue)] == 1]
        title = node.get("title") or room_id
        category = node.get("category") or "unknown"
        same_title = [
            other_id
            for other_id in candidate_room_ids
            if other_id != room_id and _normalized_text(room_graph[other_id].get("title")) == _normalized_text(title)
        ]
        same_category = [
            other_id
            for other_id in candidate_room_ids
            if other_id != room_id and _normalized_text(room_graph[other_id].get("category")) == _normalized_text(category)
        ]
        topology = []
        for neighbor in sorted(
            (value for value in node.get("neighbors", []) if isinstance(value, dict)),
            key=lambda value: (ALLOCENTRIC_DIRECTIONS.index(value.get("allocentric_direction")) if value.get("allocentric_direction") in ALLOCENTRIC_DIRECTIONS else 99, str(value.get("target_room_id"))),
        ):
            neighbor_id = neighbor.get("target_room_id")
            if not isinstance(neighbor_id, str) or neighbor_id not in room_graph:
                continue
            neighbor_node = room_graph[neighbor_id]
            neighbor_description, neighbor_cues = _profile_fields(neighbor_node)
            topology.append(
                {
                    "allocentric_direction": neighbor.get("allocentric_direction"),
                    "neighbor_room_id": neighbor_id,
                    "transition_type": neighbor.get("transition_type") or "unknown",
                    "neighbor_title": neighbor_node.get("title") or neighbor_id,
                    "neighbor_category": neighbor_node.get("category") or "unknown",
                    "neighbor_short_description": neighbor_description,
                    "neighbor_visual_cues": neighbor_cues[:3],
                }
            )
        cards.append(
            {
                "map_label": f"C{index}",
                "room_id": room_id,
                "title": title,
                "category": category,
                "short_description": description,
                "distinctive_visual_cues": distinctive_cues,
                "shared_non_decisive_visual_cues": shared_cues,
                "same_title_candidates": same_title,
                "same_category_candidates": same_category,
                "topology_signature": topology,
            }
        )
    return cards


def build_map_grounded_instructions() -> str:
    return " ".join(
        [
            "You are a museum map-grounded spatial localizer.",
            "You must choose exactly one of the five candidate rooms and infer one global rotation between the geocentric panorama frame and the north-up allocentric floor map.",
            "The first image is the north-up semantic map; the next eight images are clockwise panorama sectors.",
            "The eight sectors are captured every 45 degrees with a 90-degree field of view, so adjacent images overlap substantially.",
            "Do not count the same doorway, artifact, sign, opening, or neighboring room twice merely because it appears in overlapping images.",
            "Candidate colors and C1-C5 labels identify rooms only and never encode prior probability or rank.",
            "First inspect every panorama sector independently and freeze its visual observations before testing any candidate hypothesis.",
            "Do not relabel a view differently merely to rescue a preferred candidate.",
            "Distinguish evidence inside the current room from adjacent rooms seen through openings and from transition-space evidence.",
            "Test view_0 as north, east, south, and west for every candidate, propagating one constant rotation to all eight views.",
            "Broad category or theme similarity is weak evidence and cannot by itself distinguish same-theme galleries.",
            "Prefer room-specific artifacts, exact text, exhibit format, multiple directionally consistent neighbor themes, and map topology.",
            "A specific neighbor match counts only when that room is the listed neighbor in the aligned allocentric direction.",
            "A clearly different visible room is a contradiction; not_visible and uncertain are neutral.",
            "Two independent directional matches are stronger than one iconic object.",
            "Independent spatial support must come from different allocentric directions, not two overlapping views of the same feature.",
            "Do not hallucinate unseen doors, neighbors, or map geometry.",
            "Treat the floor map as schematic: use north, topology, room shape, relative position, passages, stairs, and corridor structure, but do not infer exact metric scale or an exact camera coordinate.",
            "Use the supplied text topology as authoritative if a small map label is hard to read.",
            "Use the same scoring rubric for every candidate and every rotation: current_room_semantics 0-3, neighbor_semantics 0-3, directional_layout 0-4, contradiction_penalty 0-4, and total_score equal to the first three values minus the penalty.",
            "Rank every candidate exactly once, explicitly compare the top two, and always return one top-1 candidate even when confidence is low.",
            "Return JSON only and obey the supplied schema exactly.",
        ]
    )


def build_map_grounded_decision_instructions() -> str:
    return " ".join(
        [
            "You are a museum map-grounded spatial localizer.",
            "Choose exactly one of the five candidate rooms from one north-up map and eight clockwise panorama sectors.",
            "The panorama headings are geocentric and must be aligned to the allocentric map with one constant rotation.",
            "Analyze every view, deduplicate overlapping evidence, distinguish the current room from adjacent rooms, and compare all five candidates.",
            "For each candidate, internally test view_0 as north, east, south, and west using room semantics, directional neighbors, map topology, and contradictions.",
            "Broad visual theme alone is weak evidence; prefer room-specific artifacts, exact text, exhibit format, and directionally consistent transitions.",
            "Use a compact score matrix to compare every candidate under all four view_0 rotations; scores range from -4 for strong contradiction to 10 for strong combined semantic and spatial support.",
            "Return exactly one compact JSON object containing the final room, final orientation, all candidate rotation scores, Top-2 room, concise supporting views, map-neighbor matches, and one decisive discriminator.",
            "Do not output per-view observations, long hypotheses, rotation rationales, map traces, or prose summaries for individual candidates.",
            "Return JSON only, with no markdown or prose outside the object.",
        ]
    )


def build_map_grounded_prompt(*, candidate_cards: list[dict], views: list[dict]) -> str:
    heading_lines = [f"- {view['view_id']}: {float(view['heading']):.6f} degrees" for view in views]
    return "\n".join(
        [
            "INPUT ORDER",
            "1. map_reference: a dynamically highlighted, north-up Level 0 map.",
            "2. view_0 through view_7: panorama sectors ordered clockwise.",
            "",
            "FRAME RULES",
            "Map up is allocentric north. Panorama headings are geocentric bearings in an unknown local frame; heading 0 is not map north.",
            "Use one constant rotation for all panorama sectors.",
            "The views are spaced by 45 degrees and each has a 90-degree field of view, so adjacent views overlap by about 45 degrees.",
            "Repeated appearances of the same physical feature across overlapping views are one item of evidence, not independent support.",
            "The map is schematic. Use topology and relative spatial layout; do not infer exact distance, scale, or camera XY.",
            "",
            "PANORAMA HEADINGS",
            *heading_lines,
            "",
            "CANDIDATE ROOM CARDS",
            json.dumps(candidate_cards, ensure_ascii=False, indent=2),
            "",
            "REQUIRED ANALYSIS ORDER",
            "1. Produce exactly one fixed view_observations record for each of the eight views.",
            "2. For every candidate, score all four view_0 rotations: north, east, south, west.",
            "3. Select the best rotation for each candidate and check every one of that candidate's listed neighbors.",
            "4. Rank all five candidates using current-room semantics, neighbor identity, directional layout, map layout, and contradictions.",
            "5. Compare Top-1 against Top-2 and identify the decisive discriminator. Broad shared theme is not a decisive discriminator.",
            "6. Return a map_alignment_trace referencing at least two panorama views.",
            "",
            "EVIDENCE RULES",
            "- Keep each view's room_hints fixed across all candidate hypotheses.",
            "- shared_non_decisive_visual_cues and same-title matches cannot decide the room alone.",
            "- specific_match requires room-specific visual evidence; broad_match means category-level evidence only.",
            "- not_visible and uncertain contribute neither support nor contradiction.",
            "- different_room is a contradiction when the view has a specific fixed room hint incompatible with the hypothesis.",
            "- Independent directional support must use different allocentric directions; adjacent overlapping views of one feature count once.",
            "- All score components and total_score must follow the stated numeric rubric exactly.",
        ]
    )


def build_map_grounded_decision_prompt(*, candidate_cards: list[dict], views: list[dict]) -> str:
    heading_lines = [f"- {view['view_id']}: {float(view['heading']):.6f} degrees" for view in views]
    return "\n".join(
        [
            "INPUT ORDER",
            "1. map_reference: a dynamically highlighted, north-up Level 0 map.",
            "2. view_0 through view_7: panorama sectors ordered clockwise.",
            "",
            "FRAME RULES",
            "Map up is allocentric north. Panorama headings are geocentric bearings in an unknown local frame; heading 0 is not map north.",
            "Use one constant rotation for all panorama sectors.",
            "The views are spaced by 45 degrees with overlapping 90-degree fields of view, so repeated features count only once.",
            "Use topology and relative spatial layout; do not infer exact map scale, distance, or camera XY.",
            "",
            "PANORAMA HEADINGS",
            *heading_lines,
            "",
            "CANDIDATE ROOM CARDS",
            json.dumps(candidate_cards, ensure_ascii=False, indent=2),
            "",
            "DECISION PROCEDURE",
            "1. Inspect all eight views and freeze the visible room and transition evidence.",
            "2. Compare every candidate under all four possible view_0 rotations.",
            "3. Check current-room semantics, directional neighbor identity, map topology, and contradictions.",
            "4. Emit exactly one candidate_rotation_scores record per candidate with numeric north, east, south, and west scores from -4 to 10.",
            "5. predicted_room_id must have the strongest candidate score; view_0_allocentric_direction must select that candidate's strongest rotation.",
            "6. top2_room_id must be the strongest remaining candidate.",
            "7. supporting_views must contain one to three decisive view IDs.",
            "8. matched_neighbors may contain zero to two clearly visible neighbors from the predicted candidate's supplied topology; use an empty array when none is reliable.",
            "9. decisive_discriminator must be one short sentence. Return no other analysis.",
            "",
            "REQUIRED OUTPUT FIELDS",
            "predicted_room_id, confidence, view_0_allocentric_direction, candidate_rotation_scores, top2_room_id, supporting_views, matched_neighbors, decisive_discriminator",
        ]
    )


def _object_schema(properties: dict, required: list[str]) -> dict:
    return {"type": "object", "properties": properties, "required": required, "additionalProperties": False}


def build_map_grounded_decision_schema(
    *,
    candidate_room_ids: list[str],
    context_room_ids: list[str],
    view_ids: list[str],
) -> dict:
    score_schema = _object_schema(
        {
            "room_id": {"type": "string", "enum": sorted(candidate_room_ids)},
            **{
                direction: {"type": "number", "minimum": -4, "maximum": 10}
                for direction in ALLOCENTRIC_DIRECTIONS
            },
        },
        ["room_id", *ALLOCENTRIC_DIRECTIONS],
    )
    neighbor_schema = _object_schema(
        {
            "direction": {"type": "string", "enum": list(ALLOCENTRIC_DIRECTIONS)},
            "room_id": {"type": "string", "enum": list(dict.fromkeys(context_room_ids))},
            "view_id": {"type": "string", "enum": view_ids},
        },
        ["direction", "room_id", "view_id"],
    )
    return _object_schema(
        {
            "predicted_room_id": {"type": "string", "enum": sorted(candidate_room_ids)},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "view_0_allocentric_direction": {"type": "string", "enum": list(ALLOCENTRIC_DIRECTIONS)},
            "candidate_rotation_scores": {
                "type": "array",
                "items": score_schema,
                "minItems": len(candidate_room_ids),
                "maxItems": len(candidate_room_ids),
            },
            "top2_room_id": {"type": "string", "enum": sorted(candidate_room_ids)},
            "supporting_views": {
                "type": "array",
                "items": {"type": "string", "enum": view_ids},
                "minItems": 1,
                "maxItems": 3,
            },
            "matched_neighbors": {
                "type": "array",
                "items": neighbor_schema,
                "maxItems": 2,
            },
            "decisive_discriminator": {"type": "string"},
        },
        [
            "predicted_room_id",
            "confidence",
            "view_0_allocentric_direction",
            "candidate_rotation_scores",
            "top2_room_id",
            "supporting_views",
            "matched_neighbors",
            "decisive_discriminator",
        ],
    )


def build_map_grounded_schema(
    *,
    candidate_room_ids: list[str],
    context_room_ids: list[str],
    view_ids: list[str],
) -> dict:
    candidate_room_ids = sorted(candidate_room_ids)
    nullable_context = list(dict.fromkeys(context_room_ids)) + [None]
    room_hint_schema = _object_schema(
        {
            "room_id": {"type": "string", "enum": list(dict.fromkeys(context_room_ids))},
            "specificity": {"type": "string", "enum": list(EVIDENCE_SPECIFICITIES)},
            "evidence": {"type": "string"},
        },
        ["room_id", "specificity", "evidence"],
    )
    view_observation_schema = _object_schema(
        {
            "view_id": {"type": "string", "enum": view_ids},
            "themes": {"type": "array", "items": {"type": "string"}},
            "location_scope": {"type": "string", "enum": list(LOCATION_SCOPES)},
            "room_hints": {"type": "array", "items": room_hint_schema},
            "opening_or_transition_cues": {"type": "array", "items": {"type": "string"}},
            "summary": {"type": "string"},
        },
        ["view_id", "themes", "location_scope", "room_hints", "opening_or_transition_cues", "summary"],
    )
    rotation_test_schema = _object_schema(
        {
            "view_0_allocentric_direction": {"type": "string", "enum": list(ALLOCENTRIC_DIRECTIONS)},
            "current_room_semantics": {"type": "number", "minimum": 0, "maximum": 3},
            "neighbor_semantics": {"type": "number", "minimum": 0, "maximum": 3},
            "directional_layout": {"type": "number", "minimum": 0, "maximum": 4},
            "contradiction_penalty": {"type": "number", "minimum": 0, "maximum": 4},
            "total_score": {"type": "number", "minimum": -4, "maximum": 10},
            "supporting_view_ids": {"type": "array", "items": {"type": "string", "enum": view_ids}},
            "contradictions": {"type": "array", "items": {"type": "string"}},
            "summary": {"type": "string"},
        },
        [
            "view_0_allocentric_direction",
            "current_room_semantics",
            "neighbor_semantics",
            "directional_layout",
            "contradiction_penalty",
            "total_score",
            "supporting_view_ids",
            "contradictions",
            "summary",
        ],
    )
    neighbor_check_schema = _object_schema(
        {
            "expected_neighbor_room_id": {"type": "string", "enum": list(dict.fromkeys(context_room_ids))},
            "allocentric_direction": {"type": "string", "enum": list(ALLOCENTRIC_DIRECTIONS)},
            "inspected_view_ids": {"type": "array", "items": {"type": "string", "enum": view_ids}},
            "status": {"type": "string", "enum": list(MATCH_STATUSES)},
            "observed_room_hint": {"type": ["string", "null"], "enum": nullable_context},
            "evidence_specificity": {"type": "string", "enum": list(EVIDENCE_SPECIFICITIES)},
            "rationale": {"type": "string"},
        },
        [
            "expected_neighbor_room_id",
            "allocentric_direction",
            "inspected_view_ids",
            "status",
            "observed_room_hint",
            "evidence_specificity",
            "rationale",
        ],
    )
    hypothesis_schema = _object_schema(
        {
            "room_id": {"type": "string", "enum": candidate_room_ids},
            "rotation_tests": {"type": "array", "items": rotation_test_schema, "minItems": 4, "maxItems": 4},
            "selected_view_0_allocentric_direction": {"type": "string", "enum": list(ALLOCENTRIC_DIRECTIONS)},
            "directional_neighbor_checks": {"type": "array", "items": neighbor_check_schema},
            "current_room_support": {"type": "array", "items": {"type": "string"}},
            "neighbor_support": {"type": "array", "items": {"type": "string"}},
            "contradictions": {"type": "array", "items": {"type": "string"}},
            "summary": {"type": "string"},
        },
        [
            "room_id",
            "rotation_tests",
            "selected_view_0_allocentric_direction",
            "directional_neighbor_checks",
            "current_room_support",
            "neighbor_support",
            "contradictions",
            "summary",
        ],
    )
    ranking_schema = _object_schema(
        {"room_id": {"type": "string", "enum": candidate_room_ids}, "score": {"type": "number"}},
        ["room_id", "score"],
    )
    comparison_schema = _object_schema(
        {
            "top1_room_id": {"type": "string", "enum": candidate_room_ids},
            "top2_room_id": {"type": "string", "enum": candidate_room_ids},
            "margin": {"type": "number"},
            "decisive_discriminator": {"type": "string"},
            "comparison_summary": {"type": "string"},
        },
        ["top1_room_id", "top2_room_id", "margin", "decisive_discriminator", "comparison_summary"],
    )
    trace_schema = _object_schema(
        {
            "view_id": {"type": "string", "enum": view_ids},
            "allocentric_direction": {"type": "string", "enum": list(ALLOCENTRIC_DIRECTIONS)},
            "mapped_room_id": {"type": ["string", "null"], "enum": nullable_context},
            "evidence_type": {"type": "string", "enum": ["current_room", "neighbor", "map_layout", "transition"]},
            "evidence_specificity": {"type": "string", "enum": list(EVIDENCE_SPECIFICITIES)},
            "rationale": {"type": "string"},
        },
        ["view_id", "allocentric_direction", "mapped_room_id", "evidence_type", "evidence_specificity", "rationale"],
    )
    return _object_schema(
        {
            "predicted_room_id": {"type": "string", "enum": candidate_room_ids},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "ambiguity_level": {"type": "string", "enum": ["low", "medium", "high"]},
            "view_0_allocentric_direction": {"type": "string", "enum": list(ALLOCENTRIC_DIRECTIONS)},
            "geocentric_to_allocentric_offset_deg": {"type": "number", "minimum": 0, "maximum": 360},
            "view_observations": {
                "type": "array",
                "items": view_observation_schema,
                "minItems": len(view_ids),
                "maxItems": len(view_ids),
            },
            "candidate_hypotheses": {
                "type": "array",
                "items": hypothesis_schema,
                "minItems": len(candidate_room_ids),
                "maxItems": len(candidate_room_ids),
            },
            "room_ranking": {
                "type": "array",
                "items": ranking_schema,
                "minItems": len(candidate_room_ids),
                "maxItems": len(candidate_room_ids),
            },
            "top1_vs_top2": comparison_schema,
            "map_alignment_trace": {"type": "array", "items": trace_schema, "minItems": 2, "maxItems": len(view_ids)},
            "summary": {"type": "string"},
        },
        [
            "predicted_room_id",
            "confidence",
            "ambiguity_level",
            "view_0_allocentric_direction",
            "geocentric_to_allocentric_offset_deg",
            "view_observations",
            "candidate_hypotheses",
            "room_ranking",
            "top1_vs_top2",
            "map_alignment_trace",
            "summary",
        ],
    )


def _numeric(value: object, name: str) -> float:
    if not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric.")
    return float(value)


def _close(left: float, right: float, *, tolerance: float = 1e-5) -> bool:
    return abs(left - right) <= tolerance


def _heading_to_direction(heading: float) -> str:
    normalized = normalize_heading(heading)
    return min(
        DIRECTION_HEADINGS,
        key=lambda direction: abs(((normalized - DIRECTION_HEADINGS[direction] + 180.0) % 360.0) - 180.0),
    )


def validate_map_grounded_output(
    payload: dict,
    *,
    candidate_room_ids: list[str],
    room_graph: dict[str, dict],
    views: list[dict],
) -> None:
    candidates = sorted(candidate_room_ids)
    candidate_set = set(candidates)
    view_ids = [str(view["view_id"]) for view in views]
    view_set = set(view_ids)
    context_room_ids = candidate_set | {
        str(neighbor.get("target_room_id"))
        for room_id in candidates
        for neighbor in room_graph.get(room_id, {}).get("neighbors", [])
        if isinstance(neighbor, dict) and isinstance(neighbor.get("target_room_id"), str)
    }

    predicted_room_id = payload.get("predicted_room_id")
    if predicted_room_id not in candidate_set:
        raise ValueError("predicted_room_id must be one of the five candidates.")
    confidence = _numeric(payload.get("confidence"), "confidence")
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be between 0 and 1.")
    if payload.get("ambiguity_level") not in {"low", "medium", "high"}:
        raise ValueError("ambiguity_level is invalid.")
    selected_direction = payload.get("view_0_allocentric_direction")
    if selected_direction not in ALLOCENTRIC_DIRECTIONS:
        raise ValueError("view_0_allocentric_direction is invalid.")

    observations = payload.get("view_observations")
    if not isinstance(observations, list) or len(observations) != len(view_ids):
        raise ValueError("view_observations must contain exactly eight records.")
    observed_ids = []
    for observation in observations:
        if not isinstance(observation, dict):
            raise ValueError("Every view_observations record must be an object.")
        observed_ids.append(observation.get("view_id"))
        if observation.get("location_scope") not in LOCATION_SCOPES:
            raise ValueError("view_observations location_scope is invalid.")
        hints = observation.get("room_hints")
        if not isinstance(hints, list):
            raise ValueError("view_observations room_hints must be an array.")
        for hint in hints:
            if not isinstance(hint, dict) or hint.get("room_id") not in context_room_ids:
                raise ValueError("view_observations contains an invalid room hint.")
            if hint.get("specificity") not in EVIDENCE_SPECIFICITIES or not isinstance(hint.get("evidence"), str):
                raise ValueError("view_observations contains malformed room-hint evidence.")
    if set(observed_ids) != view_set or len(observed_ids) != len(set(observed_ids)):
        raise ValueError("view_observations must contain every view exactly once.")

    hypotheses = payload.get("candidate_hypotheses")
    if not isinstance(hypotheses, list) or not 1 <= len(hypotheses) <= len(candidates):
        raise ValueError("candidate_hypotheses must contain between one and five records.")
    hypothesis_by_room: dict[str, dict] = {}
    for hypothesis in hypotheses:
        if not isinstance(hypothesis, dict) or hypothesis.get("room_id") not in candidate_set:
            raise ValueError("candidate_hypotheses contains an invalid room.")
        room_id = str(hypothesis["room_id"])
        if room_id in hypothesis_by_room:
            raise ValueError("candidate_hypotheses contains a duplicate room.")
        hypothesis_by_room[room_id] = hypothesis
    winning_hypothesis = hypothesis_by_room.get(str(predicted_room_id))
    if winning_hypothesis is None:
        raise ValueError("candidate_hypotheses must include the winning candidate.")

    # The full five-room ranking is the decision contract. Candidate hypotheses
    # are an audit trail, so only the winning room must carry a completely valid
    # four-rotation and directional-neighbor audit. Some providers truncate or
    # corrupt non-winning diagnostic detail even when their ranking is complete.
    room_id = str(predicted_room_id)
    rotations = winning_hypothesis.get("rotation_tests")
    if not isinstance(rotations, list) or len(rotations) != 4:
        raise ValueError(f"{room_id} must contain exactly four rotation tests.")
    directions = []
    direction_scores: dict[str, float] = {}
    for rotation in rotations:
        if not isinstance(rotation, dict):
            raise ValueError("Every rotation test must be an object.")
        direction = rotation.get("view_0_allocentric_direction")
        if direction not in ALLOCENTRIC_DIRECTIONS:
            raise ValueError("Rotation test contains an invalid direction.")
        directions.append(direction)
        current = _numeric(rotation.get("current_room_semantics"), "current_room_semantics")
        neighbor = _numeric(rotation.get("neighbor_semantics"), "neighbor_semantics")
        layout = _numeric(rotation.get("directional_layout"), "directional_layout")
        penalty = _numeric(rotation.get("contradiction_penalty"), "contradiction_penalty")
        total = _numeric(rotation.get("total_score"), "total_score")
        if not (0 <= current <= 3 and 0 <= neighbor <= 3 and 0 <= layout <= 4 and 0 <= penalty <= 4):
            raise ValueError("Rotation score component is outside the allowed range.")
        expected_total = current + neighbor + layout - penalty
        if not _close(total, expected_total):
            raise ValueError(f"Rotation total_score for {room_id}/{direction} does not match the rubric.")
        supporting_ids = rotation.get("supporting_view_ids")
        if not isinstance(supporting_ids, list) or any(value not in view_set for value in supporting_ids):
            raise ValueError("Rotation test references an invalid supporting view.")
        direction_scores[str(direction)] = total
    if set(directions) != set(ALLOCENTRIC_DIRECTIONS) or len(directions) != len(set(directions)):
        raise ValueError(f"{room_id} must test north, east, south, and west exactly once.")
    chosen = winning_hypothesis.get("selected_view_0_allocentric_direction")
    if chosen not in ALLOCENTRIC_DIRECTIONS:
        raise ValueError(f"{room_id} selected rotation is invalid.")
    winning_score = max(direction_scores.values())
    if not _close(direction_scores[str(chosen)], winning_score):
        raise ValueError(f"{room_id} selected rotation is not one of its highest-scoring rotations.")

    expected_neighbors = {
        str(neighbor.get("target_room_id")): str(neighbor.get("allocentric_direction"))
        for neighbor in room_graph.get(room_id, {}).get("neighbors", [])
        if isinstance(neighbor, dict)
        and isinstance(neighbor.get("target_room_id"), str)
        and neighbor.get("allocentric_direction") in ALLOCENTRIC_DIRECTIONS
    }
    checks = winning_hypothesis.get("directional_neighbor_checks")
    if not isinstance(checks, list):
        raise ValueError("directional_neighbor_checks must be an array.")
    checked_ids = []
    for check in checks:
        if not isinstance(check, dict):
            raise ValueError("Every directional neighbor check must be an object.")
        neighbor_id = check.get("expected_neighbor_room_id")
        checked_ids.append(neighbor_id)
        if neighbor_id not in expected_neighbors:
            raise ValueError(f"{room_id} directional check references a non-neighbor: {neighbor_id}")
        if check.get("allocentric_direction") != expected_neighbors[neighbor_id]:
            raise ValueError(f"{room_id} directional check uses the wrong direction for {neighbor_id}.")
        inspected = check.get("inspected_view_ids")
        if not isinstance(inspected, list) or not inspected or any(value not in view_set for value in inspected):
            raise ValueError("Directional check must inspect at least one valid view.")
        if check.get("status") not in MATCH_STATUSES:
            raise ValueError("Directional check status is invalid.")
        if check.get("observed_room_hint") not in context_room_ids | {None}:
            raise ValueError("Directional check observed_room_hint is invalid.")
        if check.get("evidence_specificity") not in EVIDENCE_SPECIFICITIES:
            raise ValueError("Directional check evidence_specificity is invalid.")
    if len(checked_ids) != len(expected_neighbors) or set(checked_ids) != set(expected_neighbors):
        raise ValueError(f"{room_id} must check every listed neighbor exactly once.")
    ranking = payload.get("room_ranking")
    if not isinstance(ranking, list) or len(ranking) != len(candidates):
        raise ValueError("room_ranking must contain exactly five records.")
    ranked_ids = []
    ranked_scores = []
    for record in ranking:
        if not isinstance(record, dict) or record.get("room_id") not in candidate_set:
            raise ValueError("room_ranking contains an invalid candidate.")
        room_id = str(record["room_id"])
        score = _numeric(record.get("score"), "room_ranking score")
        ranked_ids.append(room_id)
        ranked_scores.append(score)
        if room_id == predicted_room_id and not _close(score, winning_score):
            raise ValueError(f"room_ranking score for {room_id} must equal its selected rotation score.")
    if set(ranked_ids) != candidate_set or len(ranked_ids) != len(set(ranked_ids)):
        raise ValueError("room_ranking must contain every candidate exactly once.")
    if any(left < right for left, right in zip(ranked_scores, ranked_scores[1:])):
        raise ValueError("room_ranking must be sorted by descending score.")
    if predicted_room_id != ranked_ids[0]:
        raise ValueError("predicted_room_id must equal the first room_ranking item.")
    if selected_direction != winning_hypothesis.get("selected_view_0_allocentric_direction"):
        raise ValueError("Top-level direction must equal the winning candidate's selected rotation.")

    offset = _numeric(payload.get("geocentric_to_allocentric_offset_deg"), "geocentric_to_allocentric_offset_deg")
    expected_offset = normalize_heading(DIRECTION_HEADINGS[str(selected_direction)] - float(views[0]["heading"]))
    angular_error = abs(((offset - expected_offset + 180.0) % 360.0) - 180.0)
    if angular_error > 1e-3:
        raise ValueError("geocentric_to_allocentric_offset_deg is inconsistent with view_0 and the selected rotation.")

    comparison = payload.get("top1_vs_top2")
    if not isinstance(comparison, dict):
        raise ValueError("top1_vs_top2 must be an object.")
    if comparison.get("top1_room_id") != ranked_ids[0] or comparison.get("top2_room_id") != ranked_ids[1]:
        raise ValueError("top1_vs_top2 rooms must match room_ranking.")
    margin = _numeric(comparison.get("margin"), "top1_vs_top2 margin")
    if not _close(margin, ranked_scores[0] - ranked_scores[1]):
        raise ValueError("top1_vs_top2 margin must equal the ranking score difference.")
    if not isinstance(comparison.get("decisive_discriminator"), str) or not comparison["decisive_discriminator"].strip():
        raise ValueError("top1_vs_top2 must include a decisive discriminator.")

    trace = payload.get("map_alignment_trace")
    if not isinstance(trace, list) or len(trace) < 2:
        raise ValueError("map_alignment_trace must contain at least two records.")
    heading_by_view = {str(view["view_id"]): float(view["heading"]) for view in views}
    for record in trace:
        if not isinstance(record, dict) or record.get("view_id") not in view_set:
            raise ValueError("map_alignment_trace references an invalid view.")
        expected_direction = _heading_to_direction(heading_by_view[str(record["view_id"])] + offset)
        if record.get("allocentric_direction") != expected_direction:
            raise ValueError("map_alignment_trace direction is inconsistent with the global rotation.")
        if record.get("mapped_room_id") not in context_room_ids | {None}:
            raise ValueError("map_alignment_trace mapped_room_id is invalid.")


def validate_map_grounded_decision_output(
    payload: dict,
    *,
    candidate_room_ids: list[str],
) -> None:
    """Validate the localization decision without requiring its full audit trail."""
    candidates = sorted(candidate_room_ids)
    candidate_set = set(candidates)
    predicted_room_id = payload.get("predicted_room_id")
    if predicted_room_id not in candidate_set:
        raise ValueError("predicted_room_id must be one of the five candidates.")

    ranking = payload.get("room_ranking")
    if not isinstance(ranking, list) or len(ranking) != len(candidates):
        raise ValueError("room_ranking must contain exactly five records.")
    ranked_ids: list[str] = []
    ranked_scores: list[float] = []
    for record in ranking:
        if not isinstance(record, dict) or record.get("room_id") not in candidate_set:
            raise ValueError("room_ranking contains an invalid candidate.")
        ranked_ids.append(str(record["room_id"]))
        ranked_scores.append(_numeric(record.get("score"), "room_ranking score"))
    if set(ranked_ids) != candidate_set or len(ranked_ids) != len(set(ranked_ids)):
        raise ValueError("room_ranking must contain every candidate exactly once.")
    if any(left < right for left, right in zip(ranked_scores, ranked_scores[1:])):
        raise ValueError("room_ranking must be sorted by descending score.")
    if predicted_room_id != ranked_ids[0]:
        raise ValueError("predicted_room_id must equal the first room_ranking item.")

    comparison = payload.get("top1_vs_top2")
    if not isinstance(comparison, dict):
        raise ValueError("top1_vs_top2 must be an object.")
    if comparison.get("top1_room_id") != ranked_ids[0] or comparison.get("top2_room_id") != ranked_ids[1]:
        raise ValueError("top1_vs_top2 rooms must match room_ranking.")
    margin = _numeric(comparison.get("margin"), "top1_vs_top2 margin")
    if not _close(margin, ranked_scores[0] - ranked_scores[1]):
        raise ValueError("top1_vs_top2 margin must equal the ranking score difference.")


def validate_map_grounded_top1_output(
    payload: dict,
    *,
    candidate_room_ids: list[str],
) -> None:
    """Validate only the primary localization answer used for Top-1 accuracy."""
    if not isinstance(payload, dict):
        raise ValueError("Map-grounded output must be a JSON object.")
    if payload.get("predicted_room_id") not in set(candidate_room_ids):
        raise ValueError("predicted_room_id must be one of the five candidates.")


def validate_map_grounded_compact_reasoning_output(
    payload: dict,
    *,
    candidate_room_ids: list[str],
    room_graph: dict[str, dict],
    views: list[dict],
) -> None:
    """Audit compact spatial reasoning without making it a Top-1 scoring requirement."""
    validate_map_grounded_top1_output(payload, candidate_room_ids=candidate_room_ids)
    candidate_set = set(candidate_room_ids)
    predicted_room_id = str(payload["predicted_room_id"])
    selected_direction = payload.get("view_0_allocentric_direction")
    if selected_direction not in ALLOCENTRIC_DIRECTIONS:
        raise ValueError("view_0_allocentric_direction must be north, east, south, or west.")

    confidence = _numeric(payload.get("confidence"), "confidence")
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be between 0 and 1.")

    raw_scores = payload.get("candidate_rotation_scores")
    if not isinstance(raw_scores, list) or len(raw_scores) != len(candidate_room_ids):
        raise ValueError("candidate_rotation_scores must contain exactly five records.")
    score_by_room: dict[str, dict[str, float]] = {}
    for record in raw_scores:
        if not isinstance(record, dict) or record.get("room_id") not in candidate_set:
            raise ValueError("candidate_rotation_scores contains an invalid candidate.")
        room_id = str(record["room_id"])
        if room_id in score_by_room:
            raise ValueError("candidate_rotation_scores contains a duplicate candidate.")
        direction_scores = {
            direction: _numeric(record.get(direction), f"{room_id} {direction} score")
            for direction in ALLOCENTRIC_DIRECTIONS
        }
        if any(not -4.0 <= score <= 10.0 for score in direction_scores.values()):
            raise ValueError("candidate rotation scores must be between -4 and 10.")
        score_by_room[room_id] = direction_scores
    if set(score_by_room) != candidate_set:
        raise ValueError("candidate_rotation_scores must include every candidate exactly once.")

    best_by_room = {room_id: max(scores.values()) for room_id, scores in score_by_room.items()}
    predicted_best = best_by_room[predicted_room_id]
    if not _close(score_by_room[predicted_room_id][str(selected_direction)], predicted_best):
        raise ValueError("view_0_allocentric_direction must select the predicted room's best rotation.")
    if not _close(predicted_best, max(best_by_room.values())):
        raise ValueError("predicted_room_id must have the strongest candidate rotation score.")

    top2_room_id = payload.get("top2_room_id")
    if top2_room_id not in candidate_set or top2_room_id == predicted_room_id:
        raise ValueError("top2_room_id must be a different candidate room.")
    remaining_best = max(score for room_id, score in best_by_room.items() if room_id != predicted_room_id)
    if not _close(best_by_room[str(top2_room_id)], remaining_best):
        raise ValueError("top2_room_id must have the strongest remaining candidate score.")

    view_ids = {str(view["view_id"]) for view in views}
    supporting_views = payload.get("supporting_views")
    if (
        not isinstance(supporting_views, list)
        or not 1 <= len(supporting_views) <= 3
        or len(set(supporting_views)) != len(supporting_views)
        or any(view_id not in view_ids for view_id in supporting_views)
    ):
        raise ValueError("supporting_views must contain one to three unique valid view IDs.")

    valid_neighbors = {
        (neighbor.get("allocentric_direction"), neighbor.get("target_room_id"))
        for neighbor in room_graph.get(predicted_room_id, {}).get("neighbors", [])
        if isinstance(neighbor, dict)
    }
    matched_neighbors = payload.get("matched_neighbors")
    if not isinstance(matched_neighbors, list) or len(matched_neighbors) > 2:
        raise ValueError("matched_neighbors must contain at most two records.")
    for match in matched_neighbors:
        if not isinstance(match, dict):
            raise ValueError("matched_neighbors records must be objects.")
        if (match.get("direction"), match.get("room_id")) not in valid_neighbors:
            raise ValueError("matched_neighbors must reference the predicted room's supplied topology.")
        if match.get("view_id") not in view_ids:
            raise ValueError("matched_neighbors references an invalid view ID.")

    discriminator = payload.get("decisive_discriminator")
    if not isinstance(discriminator, str) or not discriminator.strip():
        raise ValueError("decisive_discriminator must be a non-empty string.")


_TOP1_JSON_PREFIX = re.compile(
    r'^\s*(?:```(?:json)?\s*)?\{\s*"predicted_room_id"\s*:\s*("(?:\\.|[^"\\])*")',
    re.IGNORECASE,
)


def recover_map_grounded_top1_prefix(text: str, *, candidate_room_ids: list[str]) -> str | None:
    """Recover only an exact candidate ID from the first field of malformed JSON."""
    match = _TOP1_JSON_PREFIX.match(text)
    if match is None:
        return None
    try:
        room_id = json.loads(match.group(1))
    except json.JSONDecodeError:
        return None
    return room_id if room_id in set(candidate_room_ids) else None


def _context_room_ids(room_graph: dict[str, dict], candidate_room_ids: list[str]) -> list[str]:
    values = list(sorted(candidate_room_ids))
    for room_id in sorted(candidate_room_ids):
        for neighbor in room_graph.get(room_id, {}).get("neighbors", []):
            if not isinstance(neighbor, dict):
                continue
            neighbor_id = neighbor.get("target_room_id")
            if isinstance(neighbor_id, str) and neighbor_id in room_graph and neighbor_id not in values:
                values.append(neighbor_id)
    return values


def _ordered_views(observation: Observation) -> list[dict]:
    return [
        {"view_id": f"view_{index}", "path": str(view.path), "heading": float(view.heading)}
        for index, view in enumerate(observation.views)
        if isinstance(view.path, str) and view.path
    ]


def _image_to_data_url(path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(path.name)
    return f"data:{mime_type or 'image/png'};base64,{b64encode(path.read_bytes()).decode('ascii')}"


def _relax_json_schema(value: object) -> object:
    if isinstance(value, list):
        return [_relax_json_schema(item) for item in value]
    if not isinstance(value, dict):
        return value
    # Gemma's generateContent endpoint returns HTTP 400 for this deeply nested
    # contract when value/cardinality constraints are included. Preserve the
    # object shape, field types, and required fields at the provider boundary;
    # conservative normalization plus strict local validation enforce the rest.
    ignored = {"enum", "minimum", "maximum", "minItems", "maxItems", "additionalProperties"}
    return {key: _relax_json_schema(item) for key, item in value.items() if key not in ignored}


def normalize_map_grounded_output(
    payload: dict,
    *,
    candidate_room_ids: list[str],
    room_graph: dict[str, dict],
    views: list[dict],
) -> tuple[dict, list[str]]:
    normalized = json.loads(json.dumps(payload))
    changes: list[str] = []
    view_ids = [str(view["view_id"]) for view in views]
    view_set = set(view_ids)
    context_room_ids = set(_context_room_ids(room_graph, candidate_room_ids))

    if normalized.get("ambiguity_level") not in {"low", "medium", "high"}:
        normalized["ambiguity_level"] = "high"
        changes.append("ambiguity_level->high")
    winner_room_id = normalized.get("predicted_room_id")

    raw_observations = normalized.get("view_observations")
    observations_by_id: dict[str, dict] = {}
    for observation in raw_observations if isinstance(raw_observations, list) else []:
        if not isinstance(observation, dict):
            changes.append("dropped_non_object_view_observation")
            continue
        view_id = observation.get("view_id")
        if view_id not in view_set or view_id in observations_by_id:
            changes.append("dropped_invalid_or_duplicate_view_observation")
            continue
        record = json.loads(json.dumps(observation))
        if record.get("location_scope") not in LOCATION_SCOPES:
            record["location_scope"] = "uncertain"
            changes.append(f"{view_id}.location_scope->uncertain")
        normalized_hints = []
        raw_hints = record.get("room_hints")
        for hint in raw_hints if isinstance(raw_hints, list) else []:
            if not isinstance(hint, dict) or hint.get("room_id") not in context_room_ids:
                changes.append(f"{view_id}.dropped_invalid_room_hint")
                continue
            fixed_hint = json.loads(json.dumps(hint))
            if fixed_hint.get("specificity") not in EVIDENCE_SPECIFICITIES:
                fixed_hint["specificity"] = "broad"
                changes.append(f"{view_id}.room_hint_specificity->broad")
            if not isinstance(fixed_hint.get("evidence"), str):
                value = fixed_hint.get("evidence")
                fixed_hint["evidence"] = "" if value is None else json.dumps(value, ensure_ascii=False)
                changes.append(f"{view_id}.room_hint_evidence->string")
            normalized_hints.append(fixed_hint)
        record["room_hints"] = normalized_hints
        observations_by_id[str(view_id)] = record
    for view_id in view_ids:
        if view_id not in observations_by_id:
            observations_by_id[view_id] = {
                "view_id": view_id,
                "themes": [],
                "location_scope": "uncertain",
                "room_hints": [],
                "opening_or_transition_cues": [],
                "summary": "Model omitted this view observation.",
            }
            changes.append(f"added_missing_{view_id}_observation")
    normalized["view_observations"] = [observations_by_id[view_id] for view_id in view_ids]

    for hypothesis in normalized.get("candidate_hypotheses", []):
        if not isinstance(hypothesis, dict):
            continue
        room_id = hypothesis.get("room_id")
        rotations = hypothesis.get("rotation_tests")
        if isinstance(rotations, list):
            for rotation in rotations:
                if not isinstance(rotation, dict):
                    continue
                supporting_ids = rotation.get("supporting_view_ids")
                if isinstance(supporting_ids, list):
                    filtered_ids = []
                    for view_id in supporting_ids:
                        if view_id in view_set and view_id not in filtered_ids:
                            filtered_ids.append(view_id)
                    if filtered_ids != supporting_ids:
                        rotation["supporting_view_ids"] = filtered_ids
                        changes.append(f"{room_id}.supporting_view_ids->valid_unique")

        if room_id == winner_room_id and isinstance(rotations, list):
            score_by_direction = {
                str(rotation.get("view_0_allocentric_direction")): float(rotation["total_score"])
                for rotation in rotations
                if isinstance(rotation, dict)
                and rotation.get("view_0_allocentric_direction") in ALLOCENTRIC_DIRECTIONS
                and isinstance(rotation.get("total_score"), (int, float))
            }
            if len(rotations) == 4 and set(score_by_direction) == set(ALLOCENTRIC_DIRECTIONS):
                best_score = max(score_by_direction.values())
                chosen = hypothesis.get("selected_view_0_allocentric_direction")
                if chosen not in score_by_direction or not _close(score_by_direction[str(chosen)], best_score):
                    chosen = next(
                        direction
                        for direction in ALLOCENTRIC_DIRECTIONS
                        if _close(score_by_direction[direction], best_score)
                    )
                    hypothesis["selected_view_0_allocentric_direction"] = chosen
                    changes.append(f"{room_id}.selected_rotation->highest_score")
                if normalized.get("view_0_allocentric_direction") != chosen:
                    normalized["view_0_allocentric_direction"] = chosen
                    changes.append("view_0_allocentric_direction->winning_rotation")

        checks = hypothesis.get("directional_neighbor_checks")
        if not isinstance(checks, list):
            checks = []
            hypothesis["directional_neighbor_checks"] = checks
        for check in checks:
            if not isinstance(check, dict):
                continue
            if check.get("status") not in MATCH_STATUSES:
                check["status"] = "uncertain"
                changes.append("neighbor_check_status->uncertain")
            if check.get("evidence_specificity") not in EVIDENCE_SPECIFICITIES:
                check["evidence_specificity"] = "broad"
                changes.append("neighbor_check_specificity->broad")
            if check.get("observed_room_hint") not in context_room_ids | {None}:
                check["observed_room_hint"] = None
                changes.append("neighbor_check_room_hint->null")
        selected_direction = hypothesis.get("selected_view_0_allocentric_direction")
        if room_id in room_graph and selected_direction in ALLOCENTRIC_DIRECTIONS:
            checked_neighbor_ids = {
                check.get("expected_neighbor_room_id") for check in checks if isinstance(check, dict)
            }
            offset = normalize_heading(
                DIRECTION_HEADINGS[str(selected_direction)] - float(views[0]["heading"])
            )
            for neighbor in room_graph[str(room_id)].get("neighbors", []):
                if not isinstance(neighbor, dict):
                    continue
                neighbor_id = neighbor.get("target_room_id")
                direction = neighbor.get("allocentric_direction")
                if (
                    not isinstance(neighbor_id, str)
                    or direction not in ALLOCENTRIC_DIRECTIONS
                    or neighbor_id in checked_neighbor_ids
                ):
                    continue
                target_heading = DIRECTION_HEADINGS[str(direction)]
                nearest_view = min(
                    views,
                    key=lambda view: abs(
                        ((normalize_heading(float(view["heading"]) + offset) - target_heading + 180.0) % 360.0)
                        - 180.0
                    ),
                )
                checks.append(
                    {
                        "expected_neighbor_room_id": neighbor_id,
                        "allocentric_direction": direction,
                        "inspected_view_ids": [str(nearest_view["view_id"])],
                        "status": "uncertain",
                        "observed_room_hint": None,
                        "evidence_specificity": "broad",
                        "rationale": "Model omitted this directional neighbor check; treated as neutral uncertainty.",
                    }
                )
                changes.append(f"{room_id}.added_neutral_neighbor_check:{neighbor_id}")

    selected_direction = normalized.get("view_0_allocentric_direction")
    normalized_offset: float | None = None
    if selected_direction in ALLOCENTRIC_DIRECTIONS:
        normalized_offset = normalize_heading(
            DIRECTION_HEADINGS[str(selected_direction)] - float(views[0]["heading"])
        )
        raw_offset = normalized.get("geocentric_to_allocentric_offset_deg")
        if not isinstance(raw_offset, (int, float)) or abs(
            ((float(raw_offset) - normalized_offset + 180.0) % 360.0) - 180.0
        ) > 1e-3:
            normalized["geocentric_to_allocentric_offset_deg"] = normalized_offset
            changes.append("geocentric_to_allocentric_offset_deg->derived")

    for trace in normalized.get("map_alignment_trace", []):
        if not isinstance(trace, dict):
            continue
        if trace.get("mapped_room_id") not in context_room_ids | {None}:
            trace["mapped_room_id"] = None
            changes.append("map_trace_room_id->null")
        if trace.get("evidence_specificity") not in EVIDENCE_SPECIFICITIES:
            trace["evidence_specificity"] = "broad"
            changes.append("map_trace_specificity->broad")
        if trace.get("evidence_type") not in {"current_room", "neighbor", "map_layout", "transition"}:
            trace["evidence_type"] = "transition"
            changes.append("map_trace_evidence_type->transition")
        trace_view_id = trace.get("view_id")
        if trace_view_id in view_set and normalized_offset is not None:
            heading_by_view = {str(view["view_id"]): float(view["heading"]) for view in views}
            expected_direction = _heading_to_direction(heading_by_view[str(trace_view_id)] + normalized_offset)
            if trace.get("allocentric_direction") != expected_direction:
                trace["allocentric_direction"] = expected_direction
                changes.append(f"{trace_view_id}.map_trace_direction->derived")
    return normalized, changes


class MapGroundedSpatialAlignmentExperiment:
    def __init__(
        self,
        *,
        room_graph: dict[str, dict],
        model: str | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
        api_kind: str | None = None,
        request_timeout: float | None = None,
        num_ctx: int | None = None,
        response_client: Callable[[dict], dict] | None = None,
        max_http_retries: int = 1,
        validation_retries: int = 0,
        max_output_tokens: int | None = None,
        allow_incomplete_winning_hypothesis: bool = False,
        compact_decision_output: bool = True,
    ):
        model_env = resolve_model_environment(
            default_model="gpt-5-mini",
            default_api_base=DEFAULT_OPENAI_API_BASE,
            default_api_kind="responses",
        )
        self.room_graph = room_graph
        self.model = model or model_env.model_name or "gpt-5-mini"
        self.model_client = ModelResponseClient(
            provider=model_env.provider,
            api_key=api_key if api_key is not None else model_env.api_key,
            api_base=(api_base or model_env.api_base or DEFAULT_OPENAI_API_BASE).rstrip("/"),
            api_kind=resolve_api_kind(api_kind or model_env.api_kind),
            request_timeout=float(request_timeout if request_timeout is not None else (model_env.request_timeout or 180.0)),
            num_ctx=resolve_task_num_ctx(
                "localization",
                explicit_num_ctx=num_ctx,
                fallback_num_ctx=model_env.num_ctx,
                default_num_ctx=16384,
            ),
            temperature=0.0,
            think=model_env.think,
            response_client=response_client,
            max_http_retries=max_http_retries,
        )
        self.last_request_body: dict | None = None
        self.last_response_payload: dict | None = None
        self.last_parsed_output: dict | None = None
        self.last_normalization_changes: list[str] = []
        self.last_validation_errors: list[str] = []
        self.last_schema_mode: str | None = None
        self.validation_retries = max(0, int(validation_retries))
        self.compact_decision_output = bool(compact_decision_output)
        default_output_tokens = DEFAULT_COMPACT_OUTPUT_TOKENS if self.compact_decision_output else 16384
        self.max_output_tokens = max(1, int(max_output_tokens or default_output_tokens))
        self.allow_incomplete_winning_hypothesis = bool(allow_incomplete_winning_hypothesis)
        self.last_audit_warnings: list[str] = []
        self.last_ranking_complete = False
        self.last_top1_source: str | None = None

    def build_request_body(
        self,
        *,
        observation: Observation,
        manifest: dict,
        candidate_room_ids: list[str],
        semantic_map_path: Path,
    ) -> tuple[dict, list[dict], str]:
        if manifest.get("heading_mode") != "museum":
            raise ValueError("Map-grounded alignment requires heading_mode=museum.")
        views = _ordered_views(observation)
        if len(views) != 8:
            raise ValueError(f"Map-grounded alignment requires exactly 8 panorama views; received {len(views)}.")
        missing = [view["path"] for view in views if not Path(view["path"]).is_file()]
        if missing:
            raise FileNotFoundError(f"Missing panorama image: {missing[0]}")
        if not semantic_map_path.is_file():
            raise FileNotFoundError(f"Semantic map not found: {semantic_map_path}")
        cards = build_candidate_cards(self.room_graph, candidate_room_ids)
        if self.compact_decision_output:
            context_ids = _context_room_ids(self.room_graph, candidate_room_ids)
            schema = build_map_grounded_decision_schema(
                candidate_room_ids=candidate_room_ids,
                context_room_ids=context_ids,
                view_ids=[view["view_id"] for view in views],
            )
            prompt = build_map_grounded_decision_prompt(candidate_cards=cards, views=views)
            instructions = build_map_grounded_decision_instructions()
            format_name = "map_grounded_compact_spatial_reasoning"
        else:
            context_ids = _context_room_ids(self.room_graph, candidate_room_ids)
            schema = build_map_grounded_schema(
                candidate_room_ids=candidate_room_ids,
                context_room_ids=context_ids,
                view_ids=[view["view_id"] for view in views],
            )
            prompt = build_map_grounded_prompt(candidate_cards=cards, views=views)
            instructions = build_map_grounded_instructions()
            format_name = "map_grounded_spatial_alignment"
        if self.model_client.provider in {"gemini", "gemini_api", "google_gemma_api"} and self.model_client.response_client is None:
            response_schema = _relax_json_schema(schema)
            self.last_schema_mode = (
                "gemini_compact_spatial_reasoning_schema_with_local_validation"
                if self.compact_decision_output
                else "gemini_relaxed_schema_with_local_validation"
            )
        else:
            response_schema = schema
            self.last_schema_mode = (
                "compact_spatial_reasoning_json_schema" if self.compact_decision_output else "full_json_schema"
            )
        content: list[dict] = [
            {"type": "input_text", "text": prompt},
            {"type": "input_text", "text": "Image 1 of 9: map_reference. Map up is allocentric north."},
            {"type": "input_image", "image_url": _image_to_data_url(semantic_map_path), "detail": "high"},
        ]
        for index, view in enumerate(views, start=2):
            content.extend(
                [
                    {
                        "type": "input_text",
                        "text": f"Image {index} of 9: {view['view_id']}; geocentric heading={float(view['heading']):.6f} degrees.",
                    },
                    {"type": "input_image", "image_url": _image_to_data_url(Path(view["path"])), "detail": "high"},
                ]
            )
        request_body = {
            "model": self.model,
            "max_output_tokens": self.max_output_tokens,
            "instructions": instructions,
            "input": [{"role": "user", "content": content}],
            "reasoning": {"effort": "medium"},
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": format_name,
                    "strict": True,
                    "schema": response_schema,
                }
            },
        }
        return request_body, cards, prompt

    def run_once(
        self,
        *,
        observation: Observation,
        manifest: dict,
        candidate_room_ids: list[str],
        semantic_map_path: Path,
    ) -> dict:
        if not self.model_client.is_configured():
            raise RuntimeError("Map-grounded alignment requires configured VLM access.")
        request_body, cards, prompt = self.build_request_body(
            observation=observation,
            manifest=manifest,
            candidate_room_ids=candidate_room_ids,
            semantic_map_path=semantic_map_path,
        )
        self.last_request_body = json.loads(json.dumps(request_body))
        views = _ordered_views(observation)
        self.last_response_payload = None
        self.last_parsed_output = None
        self.last_normalization_changes = []
        self.last_validation_errors = []
        self.last_audit_warnings = []
        self.last_ranking_complete = False
        self.last_top1_source = None
        if self.compact_decision_output:
            response_payload = self.model_client.create(request_body)
            self.last_response_payload = json.loads(json.dumps(response_payload))
            try:
                parsed = parse_json_output(response_payload)
                self.last_top1_source = "compact_spatial_reasoning_json"
            except ValueError as exc:
                self.last_validation_errors.append(str(exc))
                output_text = extract_output_text(response_payload)
                recovered_room_id = (
                    recover_map_grounded_top1_prefix(output_text, candidate_room_ids=candidate_room_ids)
                    if isinstance(output_text, str)
                    else None
                )
                if recovered_room_id is None:
                    raise
                parsed = {"predicted_room_id": recovered_room_id}
                self.last_top1_source = "json_prefix_recovery"
                self.last_normalization_changes.append("predicted_room_id<-json_prefix")
            self.last_parsed_output = json.loads(json.dumps(parsed))
            try:
                validate_map_grounded_top1_output(parsed, candidate_room_ids=candidate_room_ids)
            except ValueError as exc:
                self.last_validation_errors.append(str(exc))
                raise
            try:
                validate_map_grounded_compact_reasoning_output(
                    parsed,
                    candidate_room_ids=candidate_room_ids,
                    room_graph=self.room_graph,
                    views=views,
                )
            except ValueError as exc:
                self.last_audit_warnings.append(str(exc))
                spatial_reasoning_complete = False
            else:
                spatial_reasoning_complete = True
            return {
                "applied": True,
                "logical_vlm_call_count": 1,
                "validation_attempt_count": 1,
                "validation_errors_before_success": list(self.last_validation_errors),
                "audit_warnings": list(self.last_audit_warnings),
                "winning_hypothesis_audit_complete": False,
                "ranking_complete": False,
                "spatial_reasoning_complete": spatial_reasoning_complete,
                "top1_valid": True,
                "top1_source": self.last_top1_source,
                "normalization_changes": list(self.last_normalization_changes),
                "structured_output_mode": self.last_schema_mode,
                "candidate_cards": cards,
                "alignment_prompt_text": prompt,
                "result": parsed,
                "alignment_response_payload": json.loads(json.dumps(response_payload)),
            }
        validation_attempt_count = 0
        for validation_attempt_count in range(1, self.validation_retries + 2):
            try:
                response_payload = self.model_client.create(request_body)
            except RuntimeError as exc:
                if "HTTP 400" not in str(exc):
                    raise
                fallback_body = json.loads(json.dumps(request_body))
                fallback_body["text"]["format"]["schema"] = {"type": "object"}
                self.last_schema_mode = "gemini_minimal_object_schema_after_http_400"
                request_body = fallback_body
                response_payload = self.model_client.create(request_body)
            self.last_response_payload = json.loads(json.dumps(response_payload))
            try:
                raw_parsed = parse_json_output(response_payload)
            except ValueError as exc:
                self.last_validation_errors.append(str(exc))
                if validation_attempt_count > self.validation_retries:
                    raise
                request_body = json.loads(json.dumps(request_body))
                request_body["instructions"] = (
                    "Return one compact JSON object for localization scoring. "
                    f"Choose predicted_room_id from exactly {json.dumps(sorted(candidate_room_ids))}. "
                    "The only required field is predicted_room_id. You may also include confidence and summary. "
                    "Do not return view observations, candidate hypotheses, rotation tests, markdown, or prose outside JSON."
                )
                request_body["text"]["format"]["schema"] = {"type": "object"}
                continue
            parsed, normalization_changes = normalize_map_grounded_output(
                raw_parsed,
                candidate_room_ids=candidate_room_ids,
                room_graph=self.room_graph,
                views=views,
            )
            self.last_normalization_changes = list(normalization_changes)
            self.last_parsed_output = json.loads(json.dumps(parsed))
            try:
                validate_map_grounded_output(
                    parsed,
                    candidate_room_ids=candidate_room_ids,
                    room_graph=self.room_graph,
                    views=views,
                )
                self.last_ranking_complete = True
                break
            except ValueError as exc:
                if self.allow_incomplete_winning_hypothesis:
                    try:
                        validate_map_grounded_top1_output(parsed, candidate_room_ids=candidate_room_ids)
                    except ValueError as top1_exc:
                        self.last_validation_errors.append(str(top1_exc))
                        if validation_attempt_count > self.validation_retries:
                            raise
                        request_body = json.loads(json.dumps(request_body))
                        request_body["instructions"] = (
                            "Return JSON only. The response must contain exactly one required field: "
                            f"predicted_room_id, chosen from {json.dumps(sorted(candidate_room_ids))}. "
                            "Example shape: {\"predicted_room_id\": \"one exact listed room\"}. "
                            "Do not omit the field and do not choose any room outside the list."
                        )
                        request_body["text"]["format"]["schema"] = {"type": "object"}
                        continue
                    try:
                        validate_map_grounded_decision_output(parsed, candidate_room_ids=candidate_room_ids)
                    except ValueError:
                        self.last_ranking_complete = False
                    else:
                        self.last_ranking_complete = True
                    self.last_audit_warnings.append(str(exc))
                    break
                self.last_validation_errors.append(str(exc))
                if validation_attempt_count > self.validation_retries:
                    raise
                request_body = json.loads(json.dumps(request_body))
                request_body["instructions"] = (
                    str(request_body.get("instructions", ""))
                    + " Your previous response failed deterministic local validation: "
                    + str(exc)
                    + " Reinspect the supplied map and eight images, then return a complete corrected object. "
                    + "Count before returning: view_observations must have 8 items; candidate_hypotheses must have "
                    + f"{len(candidate_room_ids)} items for exactly {json.dumps(sorted(candidate_room_ids))}; "
                    + "every candidate's rotation_tests must have exactly 4 items named north, east, south, and west; "
                    + f"room_ranking must have {len(candidate_room_ids)} items. "
                    + "Do not omit any listed neighbor check or required evidence field."
                )
        return {
            "applied": True,
            "logical_vlm_call_count": validation_attempt_count,
            "validation_attempt_count": validation_attempt_count,
            "validation_errors_before_success": list(self.last_validation_errors),
            "audit_warnings": list(self.last_audit_warnings),
            "winning_hypothesis_audit_complete": not self.last_audit_warnings,
            "ranking_complete": self.last_ranking_complete,
            "top1_valid": True,
            "normalization_changes": list(self.last_normalization_changes),
            "structured_output_mode": self.last_schema_mode,
            "candidate_cards": cards,
            "alignment_prompt_text": prompt,
            "result": parsed,
            "alignment_response_payload": json.loads(json.dumps(response_payload)),
        }
