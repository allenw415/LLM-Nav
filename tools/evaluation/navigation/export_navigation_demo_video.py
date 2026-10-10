#!/usr/bin/env python3
"""Render a successful batch-navigation episode as a presentation-ready MP4."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from PIL import Image, ImageDraw, ImageFont

from tools.evaluation.navigation.export_batch_trajectories_for_pano_viewer import (
    build_envelope,
    build_episode,
    index_unique,
    load_json,
    load_jsonl,
    write_json,
)
from tools.pano_viewer.server import build_panorama_frame_manifest
from sa_nav_data.pano_visualization import build_visualization_payload


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET = PROJECT_ROOT / (
    "outputs/navigation_paths/floor0_semantic_targets_topology_constrained/"
    "room_count_2to6_ratio_1p0_2p0_per30_sparse715/test_set.jsonl"
)
RESULTS = PROJECT_ROOT / (
    "outputs/navigation_evaluation/floor0_semantic_targets_topology_constrained/"
    "room_count_2to6_ratio_1p0_2p0_per30_sparse715/"
    "geometry_edge_visit_localization_stages_combined715/results.jsonl"
)
PANO_GRAPH = PROJECT_ROOT / "dataset/sites/british_museum/normalized/pano_graph.json"
GROUNDING = PROJECT_ROOT / (
    "dataset/sites/british_museum/normalized/pano_room_grounding.json"
)
DEFAULT_TEST_ID = "TEST452"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / (
    "artifacts/pano_viewer/british_museum/demos/TEST452"
)

FONT_REGULAR = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
FONT_BOLD = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")

BACKGROUND = (8, 13, 25)
FOREGROUND = (248, 250, 252)
MUTED = (203, 213, 225)
FUTURE_ROUTE = (148, 163, 184)
TRAVERSED_ROUTE = (37, 99, 235)
AGENT = (245, 158, 11)
START = (34, 197, 94)
GOAL = (239, 68, 68)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--pano-graph", type=Path, default=PANO_GRAPH)
    parser.add_argument("--grounding", type=Path, default=GROUNDING)
    parser.add_argument("--test-id", default=DEFAULT_TEST_ID)
    parser.add_argument("--model-label", default="Gemma-4-31B-IT")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--render-dir",
        type=Path,
        help="Use exact-heading panorama images from this renderer cache.",
    )
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--seconds-per-step", type=float, default=0.55)
    parser.add_argument("--intro-seconds", type=float, default=2.0)
    parser.add_argument("--outro-seconds", type=float, default=2.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT_BOLD if bold else FONT_REGULAR), size=size)


def clean_room(value: object) -> str:
    return "Transit area" if value in {None, "", "null"} else str(value)


def image_path_from_frame(frame: dict[str, Any]) -> Path:
    local_path = frame.get("local_image_path")
    if isinstance(local_path, str) and local_path:
        return Path(local_path)

    query = parse_qs(urlparse(str(frame["image_url"])).query)
    values = query.get("path", [])
    if not values:
        raise ValueError(f"Frame {frame.get('frame_index')} has no image path")
    path = Path(values[0])
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def attach_render_dir(frames: list[dict[str, Any]], render_dir: Path) -> None:
    for frame in frames:
        pano_id = str(frame["pano_id"])
        manifest_path = render_dir / pano_id / f"{pano_id}_manifest.json"
        manifest = load_json(manifest_path)
        captures = manifest.get("captures") if isinstance(manifest, dict) else None
        if not isinstance(captures, list) or len(captures) != 1:
            raise ValueError(f"Expected one trajectory capture in {manifest_path}")
        capture = captures[0]
        if not isinstance(capture, dict) or not isinstance(capture.get("path"), str):
            raise ValueError(f"Invalid trajectory capture in {manifest_path}")
        image_path = Path(capture["path"])
        if not image_path.is_absolute():
            image_path = manifest_path.parent / image_path
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        with Image.open(image_path) as image:
            if image.width < 640 or image.height < 640:
                raise ValueError(
                    f"Expected a high-resolution frame, found {image.size}: {image_path}"
                )
        frame["local_image_path"] = str(image_path.resolve())


def cover(image: Image.Image, width: int, height: int) -> Image.Image:
    scale = max(width / image.width, height / image.height)
    resized = image.resize(
        (math.ceil(image.width * scale), math.ceil(image.height * scale)),
        Image.Resampling.LANCZOS,
    )
    left = max(0, (resized.width - width) // 2)
    top = max(0, (resized.height - height) // 2)
    return resized.crop((left, top, left + width, top + height)).convert("RGB")


def navigation_half(image: Image.Image, width: int, height: int) -> Image.Image:
    left_width = width // 2
    scale = min(left_width / image.width, height / image.height)
    resized = image.resize(
        (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
        Image.Resampling.LANCZOS,
    ).convert("RGB")
    canvas = Image.new("RGB", (width, height), BACKGROUND)
    left = (left_width - resized.width) // 2
    top = (height - resized.height) // 2
    canvas.paste(resized, (left, top))
    return canvas


def smoothstep(value: float) -> float:
    value = max(0.0, min(1.0, value))
    return value * value * (3.0 - 2.0 * value)


def hex_rgb(value: object) -> tuple[int, int, int]:
    text = str(value)
    if len(text) != 7 or not text.startswith("#"):
        return (148, 163, 184)
    return tuple(int(text[index : index + 2], 16) for index in (1, 3, 5))


def build_floor_panorama_graph(
    *,
    graph: dict[str, Any],
    grounding: dict[str, Any],
    pano_path: list[str],
    test_id: str,
    box: tuple[int, int, int, int],
) -> tuple[Image.Image, list[tuple[float, float]]]:
    width = box[2] - box[0]
    height = box[3] - box[1]
    payload = build_visualization_payload(
        graph,
        grounding_payload={"mappings": grounding},
    )
    floor_nodes = {
        str(node["id"]): node
        for node in payload["nodes"]
        if isinstance(node, dict)
        and str(node.get("floor")) == "0"
        and isinstance(node.get("lat"), (int, float))
        and isinstance(node.get("lng"), (int, float))
    }
    route_set = set(pano_path)
    visible_ids = set(floor_nodes)
    visible_nodes = list(floor_nodes.values())
    missing = sorted(route_set - floor_nodes.keys())
    if missing:
        raise ValueError(f"Route panos missing from Level 0 graph: {missing}")

    mean_lat = sum(float(node["lat"]) for node in visible_nodes) / len(visible_nodes)
    mean_lng = sum(float(node["lng"]) for node in visible_nodes) / len(visible_nodes)
    earth_radius_m = 6_371_000.0
    cosine = math.cos(math.radians(mean_lat))
    metric_points = {
        str(node["id"]): (
            math.radians(float(node["lng"]) - mean_lng) * earth_radius_m * cosine,
            -math.radians(float(node["lat"]) - mean_lat) * earth_radius_m,
        )
        for node in visible_nodes
    }
    xs = [point[0] for point in metric_points.values()]
    ys = [point[1] for point in metric_points.values()]
    span_x = max(max(xs) - min(xs), 1e-9)
    span_y = max(max(ys) - min(ys), 1e-9)
    plot_box = (36, 108, width - 36, height - 36)
    scale = min(
        (plot_box[2] - plot_box[0]) / span_x,
        (plot_box[3] - plot_box[1]) / span_y,
    )
    centre_x = (min(xs) + max(xs)) / 2
    centre_y = (min(ys) + max(ys)) / 2
    target_x = (plot_box[0] + plot_box[2]) / 2
    target_y = (plot_box[1] + plot_box[3]) / 2
    positions = {
        pano_id: (
            target_x + (point[0] - centre_x) * scale,
            target_y + (point[1] - centre_y) * scale,
        )
        for pano_id, point in metric_points.items()
    }

    panel = Image.new("RGBA", (width, height), (248, 250, 252, 255))
    draw = ImageDraw.Draw(panel)
    draw.text((36, 28), "LEVEL 0 PANORAMA GRAPH", fill=(15, 23, 42), font=font(32, bold=True))
    draw.text(
        (36, 72),
        f"{test_id} | {len(visible_nodes):,} panorama nodes",
        fill=(71, 85, 105),
        font=font(18),
    )

    for edge in payload["edges"]:
        if not isinstance(edge, dict):
            continue
        source = str(edge.get("source"))
        target = str(edge.get("target"))
        if source not in visible_ids or target not in visible_ids:
            continue
        if source not in positions or target not in positions:
            continue
        draw.line((positions[source], positions[target]), fill=(180, 190, 202, 255), width=1)

    for pano_id in sorted(visible_ids):
        node = floor_nodes.get(pano_id)
        point = positions.get(pano_id)
        if node is None or point is None:
            continue
        red, green, blue = hex_rgb(node.get("color"))
        radius = 3.0 if pano_id in route_set else 2.1
        draw.ellipse(
            (point[0] - radius, point[1] - radius, point[0] + radius, point[1] + radius),
            fill=(red, green, blue, 255),
        )

    arrow_x = width - 42
    draw.text((arrow_x, 38), "N", fill=(71, 85, 105), font=font(14, bold=True), anchor="mm")
    draw.line((arrow_x, 52, arrow_x, 82), fill=(71, 85, 105), width=2)
    draw.polygon(((arrow_x, 48), (arrow_x - 5, 57), (arrow_x + 5, 57)), fill=(71, 85, 105))

    return panel, [positions[pano_id] for pano_id in pano_path]

def draw_line(
    draw: ImageDraw.ImageDraw,
    points: list[tuple[float, float]],
    *,
    fill: tuple[int, int, int] | tuple[int, int, int, int],
    width: int,
) -> None:
    if len(points) >= 2:
        draw.line(points, fill=fill, width=width, joint="curve")


def interpolated_route(
    points: list[tuple[float, float]], step_index: int, progress: float
) -> tuple[list[tuple[float, float]], tuple[float, float]]:
    if step_index >= len(points) - 1:
        return points, points[-1]
    source = points[step_index]
    target = points[step_index + 1]
    agent = (
        source[0] + (target[0] - source[0]) * progress,
        source[1] + (target[1] - source[1]) * progress,
    )
    return [*points[: step_index + 1], agent], agent


def rounded_panel(
    canvas: Image.Image,
    box: tuple[int, int, int, int],
    *,
    fill: tuple[int, int, int, int] = (8, 13, 25, 210),
    radius: int = 22,
) -> ImageDraw.ImageDraw:
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.rounded_rectangle(box, radius=radius, fill=fill)
    canvas.paste(overlay, (0, 0), overlay)
    return ImageDraw.Draw(canvas)


def draw_hud(
    canvas: Image.Image,
    *,
    query: str,
    model_label: str,
    test_id: str,
    frame: dict[str, Any],
    step_index: int,
    move_count: int,
    route_points: list[tuple[float, float]],
    graph_panel: Image.Image,
    progress: float,
    map_box: tuple[int, int, int, int],
) -> None:
    width, height = canvas.size
    left_width = map_box[0]
    rounded_panel(canvas, (28, 28, left_width - 28, 170), fill=(8, 13, 25, 222))
    draw = ImageDraw.Draw(canvas)
    draw.text((54, 48), "USER QUERY", fill=MUTED, font=font(18, bold=True))
    draw.text((54, 78), query, fill=FOREGROUND, font=font(28, bold=True))
    draw.text((54, 128), f"{model_label}  |  {test_id}", fill=MUTED, font=font(18))

    info_box = (28, height - 142, left_width - 28, height - 28)
    rounded_panel(canvas, info_box, fill=(8, 13, 25, 222))
    draw = ImageDraw.Draw(canvas)
    room = clean_room(frame.get("room_id"))
    subgoal = clean_room(frame.get("subgoal_room_id") or frame.get("active_target_room_id"))
    draw.text((54, height - 120), f"Step {step_index:02d} / {move_count}", fill=FOREGROUND, font=font(28, bold=True))
    draw.text((54, height - 75), f"Panorama location: {room}   |   Subgoal: {subgoal}", fill=MUTED, font=font(18))

    map_left, map_top, _, _ = map_box
    canvas.paste(graph_panel, (map_left, map_top), graph_panel)
    draw = ImageDraw.Draw(canvas)
    draw.line((map_left, 0, map_left, height), fill=(15, 23, 42), width=3)
    plot_points = [(map_left + x, map_top + y) for x, y in route_points]
    draw_line(draw, plot_points, fill=(255, 255, 255), width=5)
    draw_line(draw, plot_points, fill=(51, 65, 85), width=3)
    traversed, agent = interpolated_route(plot_points, step_index, progress)
    draw_line(draw, traversed, fill=TRAVERSED_ROUTE, width=4)

    for point, color, label in (
        (plot_points[0], START, "S"),
        (plot_points[-1], GOAL, "G"),
    ):
        x, y = point
        draw.ellipse((x - 11, y - 11, x + 11, y + 11), fill=color, outline=FOREGROUND, width=3)
        draw.text((x, y), label, fill=FOREGROUND, font=font(12, bold=True), anchor="mm")
    x, y = agent
    draw.ellipse((x - 13, y - 13, x + 13, y + 13), fill=AGENT, outline=FOREGROUND, width=4)


def draw_intro(
    background: Image.Image,
    *,
    query: str,
    model_label: str,
    test_id: str,
    distance_m: float,
    move_count: int,
) -> Image.Image:
    canvas = background.copy()
    shade = Image.new("RGBA", canvas.size, (3, 7, 18, 168))
    canvas.paste(shade, (0, 0), shade)
    draw = ImageDraw.Draw(canvas)
    centre_x = canvas.width // 2
    draw.text((centre_x, 350), "INDOOR NAVIGATION DEMO", fill=MUTED, font=font(28, bold=True), anchor="mm")
    draw.multiline_text((centre_x, 480), query, fill=FOREGROUND, font=font(50, bold=True), anchor="mm", align="center", spacing=12)
    draw.text((centre_x, 600), f"{test_id}  |  {model_label}", fill=MUTED, font=font(27), anchor="mm")
    draw.text((centre_x, 660), f"{distance_m:.1f} m  |  {move_count} movement steps", fill=FOREGROUND, font=font(28), anchor="mm")
    return canvas


def draw_outro(background: Image.Image, *, target_room: str) -> Image.Image:
    canvas = background.copy()
    shade = Image.new("RGBA", canvas.size, (3, 7, 18, 128))
    canvas.paste(shade, (0, 0), shade)
    draw = ImageDraw.Draw(canvas)
    centre_x = canvas.width // 2
    draw.rounded_rectangle((centre_x - 330, 420, centre_x + 330, 660), radius=28, fill=(8, 13, 25, 224))
    draw.ellipse((centre_x - 33, 462, centre_x + 33, 528), fill=START)
    draw.text((centre_x, 495), "✓", fill=FOREGROUND, font=font(42, bold=True), anchor="mm")
    draw.text((centre_x, 565), "GOAL REACHED", fill=FOREGROUND, font=font(42, bold=True), anchor="mm")
    draw.text((centre_x, 620), target_room, fill=MUTED, font=font(27), anchor="mm")
    return canvas


def ffmpeg_executable() -> str:
    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise RuntimeError(
            "Install imageio-ffmpeg in the active environment to encode MP4."
        ) from exc
    return imageio_ffmpeg.get_ffmpeg_exe()


def open_encoder(output_path: Path, *, width: int, height: int, fps: int) -> subprocess.Popen[bytes]:
    command = [
        ffmpeg_executable(),
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "16",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    return subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )


def write_video_frame(process: subprocess.Popen[bytes], frame: Image.Image) -> None:
    if process.stdin is None:
        raise RuntimeError("FFmpeg stdin is unavailable")
    process.stdin.write(frame.convert("RGB").tobytes())


def finish_encoder(process: subprocess.Popen[bytes]) -> None:
    if process.stdin is not None:
        process.stdin.close()
    stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"FFmpeg failed with exit code {return_code}: {stderr.strip()}")


def render_video(
    *,
    output_path: Path,
    frames: list[dict[str, Any]],
    pano_path: list[str],
    graph: dict[str, Any],
    grounding: dict[str, Any],
    query: str,
    model_label: str,
    test_id: str,
    target_room: str,
    distance_m: float,
    width: int,
    height: int,
    fps: int,
    seconds_per_step: float,
    intro_seconds: float,
    outro_seconds: float,
) -> None:
    image_paths = [image_path_from_frame(frame) for frame in frames]
    map_box = (width // 2, 0, width, height)
    graph_panel, route_points = build_floor_panorama_graph(
        graph=graph,
        grounding=grounding,
        pano_path=pano_path,
        test_id=test_id,
        box=map_box,
    )
    move_count = len(pano_path) - 1
    intro_count = max(1, round(intro_seconds * fps))
    outro_count = max(1, round(outro_seconds * fps))
    step_frame_count = max(2, round(seconds_per_step * fps))

    first_image = cover(Image.open(image_paths[0]), width, height)
    last_image = cover(Image.open(image_paths[-1]), width, height)
    intro = draw_intro(
        first_image,
        query=query,
        model_label=model_label,
        test_id=test_id,
        distance_m=distance_m,
        move_count=move_count,
    )
    outro = draw_outro(last_image, target_room=target_room)

    process = open_encoder(output_path, width=width, height=height, fps=fps)
    try:
        for _ in range(intro_count):
            write_video_frame(process, intro)

        for step_index in range(move_count):
            current = navigation_half(Image.open(image_paths[step_index]), width, height)
            following = navigation_half(Image.open(image_paths[step_index + 1]), width, height)
            for local_index in range(step_frame_count):
                progress = local_index / max(1, step_frame_count - 1)
                blend_progress = smoothstep((progress - 0.18) / 0.68)
                canvas = Image.blend(current, following, blend_progress)
                display_frame = frames[min(step_index, len(frames) - 1)]
                draw_hud(
                    canvas,
                    query=query,
                    model_label=model_label,
                    test_id=test_id,
                    frame=display_frame,
                    step_index=step_index,
                    move_count=move_count,
                    route_points=route_points,
                    graph_panel=graph_panel,
                    progress=progress,
                    map_box=map_box,
                )
                write_video_frame(process, canvas)
            if (step_index + 1) % 10 == 0 or step_index + 1 == move_count:
                print(f"Rendered {step_index + 1}/{move_count} movement steps", flush=True)

        for _ in range(outro_count):
            write_video_frame(process, outro)
        finish_encoder(process)
    except Exception:
        process.kill()
        process.wait()
        raise


def main() -> int:
    args = parse_args()
    if args.width <= 0 or args.height <= 0 or args.fps <= 0:
        raise ValueError("Video width, height, and fps must be positive")
    if args.seconds_per_step <= 0 or args.intro_seconds < 0 or args.outro_seconds < 0:
        raise ValueError("Video durations must be non-negative and seconds-per-step positive")

    dataset = load_jsonl(args.dataset)
    results = load_jsonl(args.results)
    cases_by_id = index_unique(dataset, key="test_id", source=args.dataset)
    results_by_id = index_unique(results, key="test_id", source=args.results)
    if args.test_id not in cases_by_id or args.test_id not in results_by_id:
        raise ValueError(f"Unknown test ID: {args.test_id}")
    case = cases_by_id[args.test_id]
    result = results_by_id[args.test_id]
    if result.get("success") is not True:
        raise ValueError(f"{args.test_id} is not a successful navigation episode")
    if result.get("path_id") != case.get("path_id"):
        raise ValueError(f"Dataset/result path mismatch for {args.test_id}")

    graph = load_json(args.pano_graph)
    grounding_payload = load_json(args.grounding)
    grounding = grounding_payload.get("mappings", {})
    pano_path = result.get("pano_path")
    if not isinstance(graph, dict) or not isinstance(grounding, dict):
        raise ValueError("Pano graph and grounding must be JSON objects")
    if not isinstance(pano_path, list) or not all(isinstance(item, str) and item for item in pano_path):
        raise ValueError(f"{args.test_id} has no valid pano path")

    episode = build_episode(
        case=case,
        result=result,
        pano_path=pano_path,
        graph=graph,
        grounding=grounding,
        trajectory_kind="actual",
    )
    envelope = build_envelope(
        case=case,
        result=result,
        episode=episode,
        trajectory_kind="actual",
    )
    manifest = build_panorama_frame_manifest(envelope)
    if manifest["missing_count"]:
        raise ValueError(f"Missing panorama frames: {manifest['missing_frames']}")
    frames = manifest["frames"]
    if len(frames) != len(pano_path):
        raise ValueError(f"Expected {len(pano_path)} frames, found {len(frames)}")
    if args.render_dir is not None:
        attach_render_dir(frames, args.render_dir.resolve())

    metadata = {
        "schema": "llm_nav_navigation_demo_video",
        "schema_version": 1,
        "test_id": args.test_id,
        "model": args.model_label,
        "query": case["query"],
        "start_room_id": case["start_room_id"],
        "target_room_id": case["target_room_id"],
        "success": True,
        "movement_steps": result["movement_steps"],
        "shortest_path_distance_m": result["shortest_path_distance_m"],
        "actual_path_distance_m": result["actual_path_distance_m"],
        "spl": result["spl"],
        "frame_count": len(frames),
        "missing_frame_count": manifest["missing_count"],
        "source_dataset": str(args.dataset.resolve()),
        "source_results": str(args.results.resolve()),
        "video": {
            "width": args.width,
            "height": args.height,
            "fps": args.fps,
            "seconds_per_step": args.seconds_per_step,
            "intro_seconds": args.intro_seconds,
            "outro_seconds": args.outro_seconds,
        },
    }
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    if args.dry_run:
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    episode_path = args.output_dir / f"{args.test_id}_episode.json"
    metadata_path = args.output_dir / f"{args.test_id}_demo_metadata.json"
    video_path = args.output_dir / f"{args.test_id}_gemma4_navigation_demo.mp4"
    write_json(episode_path, envelope)
    write_json(metadata_path, metadata)
    render_video(
        output_path=video_path,
        frames=frames,
        pano_path=pano_path,
        graph=graph,
        grounding=grounding,
        query=str(case["query"]),
        model_label=args.model_label,
        test_id=args.test_id,
        target_room=str(case["target_room_id"]),
        distance_m=float(result["actual_path_distance_m"]),
        width=args.width,
        height=args.height,
        fps=args.fps,
        seconds_per_step=args.seconds_per_step,
        intro_seconds=args.intro_seconds,
        outro_seconds=args.outro_seconds,
    )
    print(f"Video: {video_path}")
    print(f"Episode JSON: {episode_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
