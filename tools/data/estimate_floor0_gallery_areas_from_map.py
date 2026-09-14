from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGIONS = (
    "dataset/sites/british_museum/maps/level_0_gallery_regions_vlm.json"
)
DEFAULT_EXISTING_CATALOG = (
    "dataset/sites/british_museum/analysis/floor0_estimated_gallery_areas.json"
)
DEFAULT_OUTPUT_JSON = DEFAULT_EXISTING_CATALOG
DEFAULT_OUTPUT_CSV = (
    "outputs/dataset_statistics/floor0_gallery_area_vlm/"
    "floor0_gallery_area_estimates.csv"
)
DEFAULT_BAR_PNG = "outputs/figures/gallery_area_by_room.png"
DEFAULT_OVERLAY_PNG = "outputs/figures/gallery_area_map_overlay.png"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate Level 0 gallery areas from VLM-annotated map polygons and "
            "robust anchor calibration."
        )
    )
    parser.add_argument("--regions", default=DEFAULT_REGIONS)
    parser.add_argument("--existing-catalog", default=DEFAULT_EXISTING_CATALOG)
    parser.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-csv", default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--bar-png", default=DEFAULT_BAR_PNG)
    parser.add_argument("--overlay-png", default=DEFAULT_OVERLAY_PNG)
    return parser


def resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate.resolve() if candidate.is_absolute() else (PROJECT_ROOT / candidate).resolve()


def polygon_area(points: Iterable[Iterable[float]]) -> float:
    vertices = [tuple(float(value) for value in point) for point in points]
    if len(vertices) < 3 or any(len(point) != 2 for point in vertices):
        raise ValueError("Each polygon must contain at least three [x, y] points.")
    doubled_area = 0.0
    for index, (x1, y1) in enumerate(vertices):
        x2, y2 = vertices[(index + 1) % len(vertices)]
        doubled_area += x1 * y2 - x2 * y1
    area = abs(doubled_area) / 2.0
    if not math.isfinite(area) or area <= 0.0:
        raise ValueError(f"Polygon has invalid area: {vertices!r}")
    return area


def room_pixel_area(record: dict) -> float:
    polygons = record.get("polygons")
    if not isinstance(polygons, list) or not polygons:
        raise ValueError("Each room must contain a non-empty polygons list.")
    return sum(polygon_area(points) for points in polygons)


def polygons_pixel_area(polygons: object) -> float:
    if not isinstance(polygons, list) or not polygons:
        raise ValueError("Expected a non-empty polygons list.")
    return sum(polygon_area(points) for points in polygons)


def room_number(room_id: str) -> tuple[int, str]:
    suffix = room_id.removeprefix("Room ")
    digits = "".join(character for character in suffix if character.isdigit())
    return (int(digits), suffix) if digits else (10_000, suffix)


def load_titles(path: Path) -> dict[str, str | None]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(record["room_id"]): record.get("room_title")
        for record in payload.get("rooms", [])
        if isinstance(record, dict) and isinstance(record.get("room_id"), str)
    }


def build_estimates(regions: dict, titles: dict[str, str | None]) -> tuple[dict, list[dict]]:
    raw_rooms = regions.get("rooms")
    anchors = regions.get("calibration_anchors")
    if not isinstance(raw_rooms, dict) or not raw_rooms:
        raise ValueError("Region data must contain a rooms object.")
    if not isinstance(anchors, list) or len(anchors) < 3:
        raise ValueError("At least three calibration anchors are required.")

    pixel_areas = {
        room_id: room_pixel_area(record) for room_id, record in raw_rooms.items()
    }
    anchor_rows: list[dict] = []
    for anchor in anchors:
        room_id = anchor.get("room_id")
        observed = anchor.get("observed_area_m2")
        if room_id not in pixel_areas:
            raise ValueError(f"Anchor has no annotated footprint: {room_id!r}")
        if not isinstance(observed, (int, float)) or observed <= 0:
            raise ValueError(f"Anchor has invalid observed area: {anchor!r}")
        scale = float(observed) / pixel_areas[room_id]
        anchor_rows.append(
            {
                "room_id": room_id,
                "role": anchor.get("role", "primary"),
                "observed_area_m2": float(observed),
                "map_pixel_area": pixel_areas[room_id],
                "m2_per_pixel": scale,
            }
        )

    scales = [record["m2_per_pixel"] for record in anchor_rows]
    selected_scale = float(statistics.median(scales))
    low_scale = min(scales)
    high_scale = max(scales)
    anchors_by_room = {record["room_id"]: record for record in anchor_rows}

    alternate_estimates: dict[str, dict] = {}
    raw_alternates = regions.get("alternate_map_room_estimates", {})
    if not isinstance(raw_alternates, dict):
        raise ValueError("alternate_map_room_estimates must be an object.")
    alternate_calibrations: dict[str, dict] = {}
    for room_id, alternate in raw_alternates.items():
        if room_id not in raw_rooms or not isinstance(alternate, dict):
            raise ValueError(f"Invalid alternate-map room estimate: {room_id!r}")
        alternate_pixel_area = polygons_pixel_area(alternate.get("polygons"))
        raw_alternate_anchors = alternate.get("calibration_anchors")
        if not isinstance(raw_alternate_anchors, list) or len(raw_alternate_anchors) < 3:
            raise ValueError(f"Alternate-map estimate for {room_id} needs at least 3 anchors.")
        alternate_anchor_rows = []
        for anchor in raw_alternate_anchors:
            anchor_pixel_area = polygons_pixel_area(anchor.get("polygons"))
            observed = anchor.get("observed_area_m2")
            anchor_room_id = anchor.get("room_id")
            if not isinstance(observed, (int, float)) or observed <= 0:
                raise ValueError(f"Invalid alternate-map anchor for {room_id}: {anchor!r}")
            alternate_anchor_rows.append(
                {
                    "room_id": anchor_room_id,
                    "observed_area_m2": float(observed),
                    "map_pixel_area": anchor_pixel_area,
                    "m2_per_pixel": float(observed) / anchor_pixel_area,
                }
            )
        alternate_scales = [record["m2_per_pixel"] for record in alternate_anchor_rows]
        alternate_scale = float(statistics.median(alternate_scales))
        alternate_estimates[room_id] = {
            "map_pixel_area": alternate_pixel_area,
            "area_source": "vlm_alternate_map_relative_estimate",
            "effective_area_m2": alternate_pixel_area * alternate_scale,
            "estimated_area_low_m2": alternate_pixel_area * min(alternate_scales),
            "estimated_area_high_m2": alternate_pixel_area * max(alternate_scales),
            "annotation_source_image": alternate.get("source_image_filename"),
        }
        alternate_calibrations[room_id] = {
            "source_image_filename": alternate.get("source_image_filename"),
            "source_description": alternate.get("source_description"),
            "image_width": alternate.get("image_width"),
            "image_height": alternate.get("image_height"),
            "selected_m2_per_pixel": alternate_scale,
            "anchor_m2_per_pixel_min": min(alternate_scales),
            "anchor_m2_per_pixel_max": max(alternate_scales),
            "anchors": alternate_anchor_rows,
        }

    room_rows: list[dict] = []
    for room_id in sorted(raw_rooms, key=room_number):
        pixel_area = pixel_areas[room_id]
        estimated = pixel_area * selected_scale
        anchor = anchors_by_room.get(room_id)
        anchor_observed = anchor["observed_area_m2"] if anchor else None
        room_rows.append(
            {
                "room_id": room_id,
                "room_title": titles.get(room_id),
                "map_pixel_area": int(pixel_area) if pixel_area.is_integer() else pixel_area,
                "area_source": "vlm_map_relative_estimate",
                "effective_area_m2": estimated,
                "estimated_area_low_m2": pixel_area * low_scale,
                "estimated_area_high_m2": pixel_area * high_scale,
                "is_anchor": anchor is not None,
                "anchor_role": anchor["role"] if anchor else None,
                "anchor_observed_area_m2": anchor_observed,
                "anchor_residual_percent": (
                    (estimated / anchor_observed - 1.0) * 100.0 if anchor else None
                ),
            }
        )
        if room_id in alternate_estimates:
            room_rows[-1].update(alternate_estimates[room_id])

    calibration = {
        "method": (
            "Median of observed-area-to-VLM-map-footprint ratios for Rooms 1, "
            "15, 27, and 17; Room 17 is retained as a validation outlier and the "
            "median prevents it from dominating the scale."
        ),
        "selected_m2_per_pixel": selected_scale,
        "selected_m_per_pixel": math.sqrt(selected_scale),
        "anchor_m2_per_pixel_min": low_scale,
        "anchor_m2_per_pixel_max": high_scale,
        "anchors": anchor_rows,
        "alternate_map_room_calibrations": alternate_calibrations,
    }
    return calibration, room_rows


def write_json(path: Path, regions: dict, calibration: dict, rows: list[dict]) -> None:
    payload = {
        "schema": "st_nav.estimated_gallery_areas",
        "schema_version": 2,
        "site": "british_museum",
        "floor": "0",
        "unit": "m2",
        "description": (
            "VLM-annotated map-relative gallery footprint estimates. Values are "
            "not architectural floor-plan measurements."
        ),
        "area_methods": {
            "vlm_map_relative_estimate": (
                "Shoelace pixel area of VLM-annotated visitor-map polygons, "
                "scaled by a robust median of user-selected anchors."
            ),
            "vlm_alternate_map_relative_estimate": (
                "Shoelace pixel area from a user-supplied alternate Level 0 map, "
                "calibrated independently with the same four room anchors."
            ),
        },
        "map_annotation": {
            "source": str(resolve(DEFAULT_REGIONS).relative_to(PROJECT_ROOT)),
            "image_width": regions["image_width"],
            "image_height": regions["image_height"],
            "coordinate_system": regions["coordinate_system"],
            "annotation_method": regions["annotation_method"],
        },
        "map_relative_calibration": calibration,
        "excluded_route_room_ids": [
            "East stairs",
            "West stairs",
            "North stairs",
            "North lift",
        ],
        "rooms": rows,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = (
        "room_id",
        "room_title",
        "map_pixel_area",
        "effective_area_m2",
        "estimated_area_low_m2",
        "estimated_area_high_m2",
        "is_anchor",
        "anchor_role",
        "anchor_observed_area_m2",
        "anchor_residual_percent",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def load_font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{name}", size=size)


def render_bar_chart(path: Path, rows: list[dict], selected_scale: float) -> None:
    width = 2400
    row_height = 58
    top = 210
    bottom = 105
    left = 315
    right = 250
    height = top + bottom + row_height * len(rows)
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = load_font(50, bold=True)
    subtitle_font = load_font(27)
    label_font = load_font(27)
    value_font = load_font(25, bold=True)
    small_font = load_font(22)
    draw.text((left, 42), "Estimated map-relative gallery area", fill="#172033", font=title_font)
    alternate_row = next(
        (row for row in rows if row["area_source"] == "vlm_alternate_map_relative_estimate"),
        None,
    )
    alternate_note = ""
    if alternate_row is not None:
        alternate_scale = (
            float(alternate_row["effective_area_m2"])
            / float(alternate_row["map_pixel_area"])
        )
        alternate_note = (
            f"; {alternate_row['room_id']} alternate-map scale = "
            f"{alternate_scale:.5f} m²/pixel"
        )
    draw.text(
        (left, 112),
        f"Primary scale = {selected_scale:.5f} m²/pixel{alternate_note}",
        fill="#536078",
        font=subtitle_font,
    )

    maximum = max(float(row["effective_area_m2"]) for row in rows)
    plot_width = width - left - right
    for tick in range(0, 1601, 200):
        x = left + plot_width * tick / 1600.0
        draw.line((x, top - 8, x, height - bottom), fill="#d8dde8", width=2)
        draw.text((x - 20, height - bottom + 18), str(tick), fill="#536078", font=small_font)
    draw.text(
        (left + plot_width / 2 - 80, height - 45),
        "Estimated area (m²)",
        fill="#172033",
        font=small_font,
    )
    draw.text((40, top - 80), "Gallery", fill="#172033", font=subtitle_font)

    anchors = {row["room_id"] for row in rows if row["is_anchor"]}
    for index, row in enumerate(rows):
        y = top + index * row_height
        room_id = str(row["room_id"])
        value = float(row["effective_area_m2"])
        draw.text((75, y + 9), room_id, fill="#172033", font=label_font)
        bar_right = left + plot_width * value / 1600.0
        color = "#d8892b" if room_id in anchors else "#3478b8"
        draw.rounded_rectangle((left, y + 8, bar_right, y + 42), radius=8, fill=color)
        draw.text((bar_right + 12, y + 10), f"{value:.1f}", fill="#172033", font=value_font)

    legend_y = 126
    draw.rectangle((width - 650, legend_y, width - 620, legend_y + 24), fill="#d8892b")
    draw.text(
        (width - 608, legend_y - 4),
        "Calibration anchor",
        fill="#536078",
        font=small_font,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, dpi=(180, 180))


def render_overlay(path: Path, regions: dict, rows: list[dict], regions_path: Path) -> None:
    map_path = regions_path.parent / regions["map_image"]
    image = Image.open(map_path).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    colors = (
        (38, 139, 210, 92),
        (211, 91, 78, 92),
        (80, 166, 96, 92),
        (206, 139, 41, 92),
        (129, 92, 176, 92),
    )
    row_by_room = {row["room_id"]: row for row in rows}
    for index, (room_id, record) in enumerate(regions["rooms"].items()):
        if row_by_room[room_id]["area_source"] == "vlm_alternate_map_relative_estimate":
            continue
        color = colors[index % len(colors)]
        for polygon in record["polygons"]:
            points = [(float(x), float(y)) for x, y in polygon]
            draw.polygon(points, fill=color, outline=(255, 255, 255, 225), width=2)
        all_points = [point for polygon in record["polygons"] for point in polygon]
        centre_x = sum(float(point[0]) for point in all_points) / len(all_points)
        centre_y = sum(float(point[1]) for point in all_points) / len(all_points)
        value = float(row_by_room[room_id]["effective_area_m2"])
        label = f"{room_id.removeprefix('Room ')}\n{value:.0f}"
        font = load_font(15, bold=True)
        box = draw.multiline_textbbox((centre_x, centre_y), label, font=font, anchor="mm", align="center")
        draw.rounded_rectangle(
            (box[0] - 3, box[1] - 2, box[2] + 3, box[3] + 2),
            radius=3,
            fill=(255, 255, 255, 205),
        )
        draw.multiline_text(
            (centre_x, centre_y),
            label,
            fill=(24, 32, 50, 255),
            font=font,
            anchor="mm",
            align="center",
            spacing=0,
        )
    composite = Image.alpha_composite(image, overlay).convert("RGB")
    path.parent.mkdir(parents=True, exist_ok=True)
    composite.save(path, dpi=(180, 180))


def main() -> None:
    args = build_parser().parse_args()
    regions_path = resolve(args.regions)
    regions = json.loads(regions_path.read_text(encoding="utf-8"))
    map_path = regions_path.parent / regions["map_image"]
    with Image.open(map_path) as image:
        if image.size != (regions["image_width"], regions["image_height"]):
            raise ValueError(
                f"Map size {image.size} differs from annotation size "
                f"{(regions['image_width'], regions['image_height'])}."
            )
    titles = load_titles(resolve(args.existing_catalog))
    calibration, rows = build_estimates(regions, titles)
    write_json(resolve(args.output_json), regions, calibration, rows)
    write_csv(resolve(args.output_csv), rows)
    render_bar_chart(resolve(args.bar_png), rows, calibration["selected_m2_per_pixel"])
    render_overlay(resolve(args.overlay_png), regions, rows, regions_path)
    print(
        json.dumps(
            {
                "room_count": len(rows),
                "selected_m2_per_pixel": calibration["selected_m2_per_pixel"],
                "output_json": str(resolve(args.output_json)),
                "output_csv": str(resolve(args.output_csv)),
                "bar_png": str(resolve(args.bar_png)),
                "overlay_png": str(resolve(args.overlay_png)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
