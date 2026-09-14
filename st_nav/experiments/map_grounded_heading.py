from __future__ import annotations

import hashlib
import json
import mimetypes
from base64 import b64encode
from pathlib import Path
from typing import Callable

from PIL import Image, ImageDraw, ImageEnhance, ImageFont, ImageOps

from ..common.env import DEFAULT_GEMINI_API_BASE, resolve_model_environment, resolve_task_num_ctx
from ..common.model_client import ModelResponseClient, parse_json_output, resolve_api_kind


DIRECTION_TO_HEADING = {
    "N": 0.0,
    "NE": 45.0,
    "E": 90.0,
    "SE": 135.0,
    "S": 180.0,
    "SW": 225.0,
    "W": 270.0,
    "NW": 315.0,
}
DIRECTIONS = tuple(DIRECTION_TO_HEADING)
MAP_MODES = ("highlighted", "plain")
AMBIGUITY_LEVELS = ("low", "medium", "high")
OUTPUT_FIELDS = {
    "current_room_id",
    "target_room_id",
    "passage_direction",
    "heading_deg",
    "confidence",
    "ambiguity_level",
    "passage_description",
    "summary",
}
CURRENT_ROOM_COLOR = (30, 120, 230, 112)
CURRENT_ROOM_OUTLINE = (10, 75, 180, 255)
TARGET_ROOM_COLOR = (40, 190, 100, 112)
TARGET_ROOM_OUTLINE = (15, 125, 60, 255)


def _font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    candidates = (
        Path("/usr/share/fonts/truetype/dejavu") / name,
        Path("/usr/share/fonts") / name,
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
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = position[0] - text_width // 2
    y = position[1] - text_height // 2
    draw.rounded_rectangle(
        (x - 6, y - 4, x + text_width + 6, y + text_height + 4),
        radius=5,
        fill=background,
    )
    draw.text(
        (x, y),
        text,
        font=font,
        fill=fill,
        stroke_width=1,
        stroke_fill=(255, 255, 255, 220),
    )


def validate_heading_map_regions(region_payload: dict) -> None:
    if region_payload.get("coordinate_system") != "normalized_0_1":
        raise ValueError("Heading map regions must use coordinate_system=normalized_0_1.")
    if region_payload.get("north_up") is not True:
        raise ValueError("Heading map regions must declare north_up=true.")
    rooms = region_payload.get("rooms")
    if not isinstance(rooms, dict) or not rooms:
        raise ValueError("Heading map regions payload must contain a non-empty rooms object.")
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
                    or any(
                        not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 1.0
                        for value in point
                    )
                ):
                    raise ValueError(f"Map polygon point for {room_id} must be normalized to [0, 1].")
        if any(
            not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 1.0
            for value in anchor
        ):
            raise ValueError(f"Map label anchor for {room_id} must be normalized to [0, 1].")


def render_passage_heading_map(
    *,
    base_map_path: Path,
    region_payload: dict,
    current_room_id: str,
    target_room_id: str,
    output_path: Path,
) -> dict:
    validate_heading_map_regions(region_payload)
    if current_room_id == target_room_id:
        raise ValueError("current_room_id and target_room_id must be different.")
    rooms = region_payload["rooms"]
    missing = [room_id for room_id in (current_room_id, target_room_id) if room_id not in rooms]
    if missing:
        raise ValueError("Rooms missing from map regions: " + ", ".join(missing))
    if not base_map_path.is_file():
        raise FileNotFoundError(f"Map image not found: {base_map_path}")

    base = Image.open(base_map_path).convert("RGBA")
    width, height = base.size
    grayscale = ImageOps.grayscale(base.convert("RGB")).convert("RGBA")
    subdued = ImageEnhance.Brightness(grayscale).enhance(0.92)
    map_canvas = Image.blend(base, subdued, 0.36)
    overlay = Image.new("RGBA", map_canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for room_id, fill, outline, marker in (
        (current_room_id, CURRENT_ROOM_COLOR, CURRENT_ROOM_OUTLINE, "C"),
        (target_room_id, TARGET_ROOM_COLOR, TARGET_ROOM_OUTLINE, "T"),
    ):
        record = rooms[room_id]
        for polygon in record["polygons"]:
            points = _pixel_points(polygon, width, height)
            draw.polygon(points, fill=fill)
            draw.line(points + [points[0]], fill=outline, width=9, joint="curve")
        _draw_centered_label(
            draw,
            _pixel_anchor(record["label_anchor"], width, height),
            marker,
            font=_font(23, bold=True),
            fill=(15, 15, 15, 255),
            background=(255, 255, 255, 242),
        )

    map_canvas = Image.alpha_composite(map_canvas, overlay)
    sidebar_width = 530
    result = Image.new("RGBA", (width + sidebar_width, height), (250, 250, 250, 255))
    result.paste(map_canvas, (0, 0))
    sidebar = ImageDraw.Draw(result)
    sidebar.rectangle((width, 0, width + sidebar_width, height), fill=(250, 250, 250, 255))

    x = width + 32
    sidebar.text((x, 30), "MAP-ONLY PASSAGE HEADING", font=_font(24, bold=True), fill=(20, 20, 20, 255))
    sidebar.rounded_rectangle((x, 90, x + 40, 130), radius=5, fill=CURRENT_ROOM_OUTLINE)
    sidebar.text((x + 58, 95), f"CURRENT: {current_room_id}", font=_font(22, bold=True), fill=(20, 20, 20, 255))
    sidebar.rounded_rectangle((x, 150, x + 40, 190), radius=5, fill=TARGET_ROOM_OUTLINE)
    sidebar.text((x + 58, 155), f"TARGET: {target_room_id}", font=_font(22, bold=True), fill=(20, 20, 20, 255))

    sidebar.text((x, 240), "NORTH-UP HEADING FRAME", font=_font(23, bold=True), fill=(20, 20, 20, 255))
    center_x = width + sidebar_width // 2
    center_y = 470
    radius = 150
    sidebar.ellipse(
        (center_x - radius, center_y - radius, center_x + radius, center_y + radius),
        outline=(60, 60, 60, 255),
        width=4,
    )
    direction_positions = {
        "N 0°": (center_x, center_y - radius - 28),
        "NE 45°": (center_x + 112, center_y - 112),
        "E 90°": (center_x + radius + 20, center_y),
        "SE 135°": (center_x + 112, center_y + 112),
        "S 180°": (center_x, center_y + radius + 28),
        "SW 225°": (center_x - 112, center_y + 112),
        "W 270°": (center_x - radius - 24, center_y),
        "NW 315°": (center_x - 112, center_y - 112),
    }
    compass_font = _font(19, bold=True)
    for label, position in direction_positions.items():
        bbox = sidebar.textbbox((0, 0), label, font=compass_font)
        label_width = bbox[2] - bbox[0]
        label_height = bbox[3] - bbox[1]
        sidebar.text(
            (position[0] - label_width // 2, position[1] - label_height // 2),
            label,
            font=compass_font,
            fill=(20, 20, 20, 255),
        )
    sidebar.line((center_x, center_y + 88, center_x, center_y - 98), fill=(20, 20, 20, 255), width=7)
    sidebar.polygon(
        [(center_x, center_y - 125), (center_x - 18, center_y - 92), (center_x + 18, center_y - 92)],
        fill=(20, 20, 20, 255),
    )
    sidebar.text(
        (x, 700),
        "Determine the direction of the connecting\npassage as seen from the CURRENT room.\nAngles increase clockwise from map north.",
        font=_font(21),
        fill=(35, 35, 35, 255),
        spacing=10,
    )
    sidebar.text(
        (x, 850),
        "Use only the visible map geometry and\nhighlighted room identities.",
        font=_font(21, bold=True),
        fill=(35, 35, 35, 255),
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
        "current_room_id": current_room_id,
        "target_room_id": target_room_id,
        "map_mode": "highlighted",
        "spatial_inputs": ["map_image", "current_room_id", "target_room_id"],
        "room_highlights_rendered": True,
        "compass_sidebar_rendered": True,
        "graph_heading_rendered": False,
        "topology_text_rendered": False,
        "panorama_rendered": False,
    }


def describe_plain_heading_map(
    *,
    base_map_path: Path,
    current_room_id: str,
    target_room_id: str,
) -> dict:
    if current_room_id == target_room_id:
        raise ValueError("current_room_id and target_room_id must be different.")
    if not base_map_path.is_file():
        raise FileNotFoundError(f"Map image not found: {base_map_path}")
    with Image.open(base_map_path) as image:
        width, height = image.size
    return {
        "path": str(base_map_path),
        "sha256": hashlib.sha256(base_map_path.read_bytes()).hexdigest(),
        "width": width,
        "height": height,
        "north_up": True,
        "current_room_id": current_room_id,
        "target_room_id": target_room_id,
        "map_mode": "plain",
        "spatial_inputs": ["map_image", "current_room_id", "target_room_id"],
        "room_highlights_rendered": False,
        "compass_sidebar_rendered": False,
        "graph_heading_rendered": False,
        "topology_text_rendered": False,
        "panorama_rendered": False,
    }


def build_heading_instructions(*, map_mode: str = "highlighted") -> str:
    if map_mode not in MAP_MODES:
        raise ValueError(f"map_mode must be one of {MAP_MODES}.")
    image_description = (
        "The supplied image is a north-up floor map with exactly one blue CURRENT room and one green TARGET room."
        if map_mode == "highlighted"
        else "The supplied image is the original unmodified north-up floor map with room numbers printed on it."
    )
    identity_instruction = (
        "Use only the visible map geometry and the supplied current and target room identities."
        if map_mode == "highlighted"
        else (
            "Use only the visible map geometry and the supplied current_room_id and target_room_id. "
            "Locate both rooms by their printed room numbers."
        )
    )
    return " ".join(
        [
            "You are a museum floor-map passage-direction analyst.",
            image_description,
            identity_instruction,
            "Identify the connecting passage or doorway that leads from the CURRENT room toward the TARGET room.",
            "Report the passage direction from the perspective of the CURRENT room in the north-up allocentric map frame.",
            "Choose exactly one of N, NE, E, SE, S, SW, W, or NW.",
            "The matching clockwise headings are N=0, NE=45, E=90, SE=135, S=180, SW=225, W=270, and NW=315 degrees.",
            "Do not infer an agent position, panorama heading, route, graph edge, or navigation action that is not shown.",
            "When the doorway is unclear, still choose the best-supported direction and set ambiguity_level and confidence accordingly.",
            "Return JSON only and obey the supplied schema exactly.",
        ]
    )


def build_heading_prompt(
    *,
    current_room_id: str,
    target_room_id: str,
    map_mode: str = "highlighted",
) -> str:
    if map_mode not in MAP_MODES:
        raise ValueError(f"map_mode must be one of {MAP_MODES}.")
    map_lines = (
        [
            "The blue highlighted region is the current room.",
            "The green highlighted region is the target room.",
        ]
        if map_mode == "highlighted"
        else [
            "The image is the original unmodified floor map.",
            "Locate the rooms using the room numbers printed on the map.",
        ]
    )
    return "\n".join(
        [
            "MAP-ONLY PASSAGE HEADING TASK",
            f"current_room_id: {current_room_id}",
            f"target_room_id: {target_room_id}",
            "",
            *map_lines,
            "Map up is north. Angles increase clockwise from north.",
            "Find the connecting passage and report its direction as seen from the current room.",
        ]
    )


def _object_schema(properties: dict, required: list[str]) -> dict:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def build_heading_schema(*, current_room_id: str, target_room_id: str) -> dict:
    return _object_schema(
        {
            "current_room_id": {"type": "string", "enum": [current_room_id]},
            "target_room_id": {"type": "string", "enum": [target_room_id]},
            "passage_direction": {"type": "string", "enum": list(DIRECTIONS)},
            "heading_deg": {"type": "number", "enum": list(DIRECTION_TO_HEADING.values())},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "ambiguity_level": {"type": "string", "enum": list(AMBIGUITY_LEVELS)},
            "passage_description": {"type": "string"},
            "summary": {"type": "string"},
        },
        sorted(OUTPUT_FIELDS),
    )


def _relax_json_schema(value: object) -> object:
    if isinstance(value, list):
        return [_relax_json_schema(item) for item in value]
    if not isinstance(value, dict):
        return value
    ignored = {"enum", "minimum", "maximum", "additionalProperties"}
    return {key: _relax_json_schema(item) for key, item in value.items() if key not in ignored}


def validate_heading_output(
    payload: dict,
    *,
    current_room_id: str,
    target_room_id: str,
) -> None:
    if not isinstance(payload, dict):
        raise ValueError("Map heading output must be an object.")
    if set(payload) != OUTPUT_FIELDS:
        missing = sorted(OUTPUT_FIELDS - set(payload))
        extra = sorted(set(payload) - OUTPUT_FIELDS)
        raise ValueError(f"Map heading output fields mismatch; missing={missing}, extra={extra}.")
    if payload.get("current_room_id") != current_room_id:
        raise ValueError("VLM current_room_id does not match the request.")
    if payload.get("target_room_id") != target_room_id:
        raise ValueError("VLM target_room_id does not match the request.")
    direction = payload.get("passage_direction")
    if direction not in DIRECTION_TO_HEADING:
        raise ValueError("VLM passage_direction must be one of the eight allowed directions.")
    heading = payload.get("heading_deg")
    if not isinstance(heading, (int, float)):
        raise ValueError("VLM heading_deg must be numeric.")
    expected_heading = DIRECTION_TO_HEADING[str(direction)]
    if abs(float(heading) - expected_heading) > 1e-6:
        raise ValueError("VLM heading_deg is inconsistent with passage_direction.")
    confidence = payload.get("confidence")
    if not isinstance(confidence, (int, float)) or not 0.0 <= float(confidence) <= 1.0:
        raise ValueError("VLM confidence must be between 0 and 1.")
    if payload.get("ambiguity_level") not in AMBIGUITY_LEVELS:
        raise ValueError("VLM ambiguity_level is invalid.")
    for field in ("passage_description", "summary"):
        if not isinstance(payload.get(field), str) or not str(payload[field]).strip():
            raise ValueError(f"VLM {field} must be a non-empty string.")


def _image_to_data_url(path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(path.name)
    return f"data:{mime_type or 'image/png'};base64,{b64encode(path.read_bytes()).decode('ascii')}"


class MapGroundedHeadingExperiment:
    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
        api_kind: str | None = None,
        request_timeout: float | None = None,
        num_ctx: int | None = None,
        response_client: Callable[[dict], dict] | None = None,
        max_http_retries: int = 1,
    ) -> None:
        model_env = resolve_model_environment(
            default_model="gemma-4-31b-it",
            default_api_base=DEFAULT_GEMINI_API_BASE,
            default_api_kind="responses",
        )
        self.model = model or model_env.model_name or "gemma-4-31b-it"
        self.model_client = ModelResponseClient(
            provider=model_env.provider,
            api_key=api_key if api_key is not None else model_env.api_key,
            api_base=(api_base or model_env.api_base or DEFAULT_GEMINI_API_BASE).rstrip("/"),
            api_kind=resolve_api_kind(api_kind or model_env.api_kind),
            request_timeout=float(
                request_timeout if request_timeout is not None else (model_env.request_timeout or 180.0)
            ),
            num_ctx=resolve_task_num_ctx(
                "localization",
                explicit_num_ctx=num_ctx,
                fallback_num_ctx=model_env.num_ctx,
                default_num_ctx=16384,
            ),
            temperature=0.0,
            response_client=response_client,
            max_http_retries=max_http_retries,
        )
        self.last_request_body: dict | None = None
        self.last_response_payload: dict | None = None
        self.last_schema_mode: str | None = None

    def build_request_body(
        self,
        *,
        current_room_id: str,
        target_room_id: str,
        semantic_map_path: Path,
        map_mode: str = "highlighted",
    ) -> tuple[dict, str]:
        if current_room_id == target_room_id:
            raise ValueError("current_room_id and target_room_id must be different.")
        if not semantic_map_path.is_file():
            raise FileNotFoundError(f"Semantic map not found: {semantic_map_path}")
        schema = build_heading_schema(
            current_room_id=current_room_id,
            target_room_id=target_room_id,
        )
        if self.model_client.provider in {"gemini", "gemini_api", "google_gemma_api"}:
            response_schema = _relax_json_schema(schema)
            self.last_schema_mode = "gemini_relaxed_schema_with_local_validation"
        else:
            response_schema = schema
            self.last_schema_mode = "full_json_schema"
        prompt = build_heading_prompt(
            current_room_id=current_room_id,
            target_room_id=target_room_id,
            map_mode=map_mode,
        )
        request_body = {
            "model": self.model,
            "instructions": build_heading_instructions(map_mode=map_mode),
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {
                            "type": "input_image",
                            "image_url": _image_to_data_url(semantic_map_path),
                            "detail": "high",
                        },
                    ],
                }
            ],
            "reasoning": {"effort": "medium"},
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "map_grounded_passage_heading",
                    "strict": True,
                    "schema": response_schema,
                }
            },
        }
        return request_body, prompt

    def run_once(
        self,
        *,
        current_room_id: str,
        target_room_id: str,
        semantic_map_path: Path,
        map_mode: str = "highlighted",
    ) -> dict:
        if not self.model_client.is_configured():
            raise RuntimeError("Map-grounded heading requires configured VLM access.")
        request_body, prompt = self.build_request_body(
            current_room_id=current_room_id,
            target_room_id=target_room_id,
            semantic_map_path=semantic_map_path,
            map_mode=map_mode,
        )
        self.last_request_body = json.loads(json.dumps(request_body))
        response_payload = self.model_client.create(request_body)
        self.last_response_payload = json.loads(json.dumps(response_payload))
        parsed = parse_json_output(response_payload)
        validate_heading_output(
            parsed,
            current_room_id=current_room_id,
            target_room_id=target_room_id,
        )
        return {
            "applied": True,
            "logical_vlm_call_count": 1,
            "structured_output_mode": self.last_schema_mode,
            "prompt_text": prompt,
            "result": parsed,
            "response_payload": json.loads(json.dumps(response_payload)),
        }
