from __future__ import annotations

import argparse
import json
import math
import mimetypes
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, urlparse


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts/pano_viewer/british_museum"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
MAX_JSON_BYTES = 16 * 1024 * 1024


class PanoViewerRequestHandler(BaseHTTPRequestHandler):
    output_dir = DEFAULT_OUTPUT_DIR

    def do_GET(self) -> None:  # noqa: N802
        request = urlparse(self.path)
        if request.path == "/api/render-image":
            self._serve_render_image(parse_qs(request.query).get("path", [""])[0])
            return
        relative_path = "index.html" if request.path in {"", "/"} else request.path.lstrip("/")
        self._serve_static(relative_path)

    def do_POST(self) -> None:  # noqa: N802
        request = urlparse(self.path)
        if request.path != "/api/trajectory-panorama-frames":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 0 or length > MAX_JSON_BYTES:
                raise ValueError(f"Request body must be at most {MAX_JSON_BYTES} bytes.")
            payload = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
            if not isinstance(payload, dict):
                raise ValueError("Expected JSON object.")
            self._send_json(build_panorama_frame_manifest(payload))
        except Exception as exc:
            self._send_json(
                {
                    "frames": [],
                    "missing_frames": [],
                    "error": f"{type(exc).__name__}: {exc}",
                },
                status=400,
            )

    def log_message(self, format: str, *args: object) -> None:
        print("[pano-viewer] " + format % args)

    def _serve_static(self, relative_path: str) -> None:
        root = self.output_dir.resolve()
        candidate = (root / relative_path).resolve()
        if not _path_is_relative_to(candidate, root):
            self.send_error(403)
            return
        if candidate.is_dir():
            candidate = candidate / "index.html"
        self._serve_file(candidate)

    def _serve_render_image(self, raw_path: str) -> None:
        try:
            candidate = resolve_project_image_path(raw_path)
        except ValueError as exc:
            self._send_text(str(exc), status=403)
            return
        self._serve_file(candidate)

    def _serve_file(self, path: Path) -> None:
        if not path.exists() or not path.is_file():
            self.send_error(404)
            return
        content = path.read_bytes()
        media_type, _ = mimetypes.guess_type(str(path))
        self.send_response(200)
        self.send_header("Content-Type", media_type or "application/octet-stream")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _send_json(self, payload: object, *, status: int = 200) -> None:
        content = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _send_text(self, message: str, *, status: int) -> None:
        content = message.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)


def build_panorama_frame_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    episode = _episode_payload(payload)
    movements = _trajectory_movements(episode)
    frames: list[dict[str, Any]] = []
    missing_frames: list[dict[str, Any]] = []

    for frame_index, movement in enumerate(movements):
        pano_id = _string_or_none(movement.get("current_pano_id"))
        if not pano_id:
            missing_frames.append({"frame_index": frame_index, "reason": "missing_current_pano_id"})
            continue
        image_path = _image_path_for_movement(movement)
        if image_path is None:
            missing_frames.append({"frame_index": frame_index, "pano_id": pano_id, "reason": "missing_render"})
            continue
        frames.append(
            _panorama_frame_payload(
                frame_index=frame_index,
                pano_id=pano_id,
                room_id=_string_or_none(movement.get("current_room_id")),
                next_room_id=_string_or_none(movement.get("next_room_id")),
                subgoal_room_id=_string_or_none(movement.get("subgoal_room_id")),
                active_target_room_id=_string_or_none(movement.get("active_target_room_id")),
                heading=_finite_number(movement.get("selected_action_heading")),
                image_path=image_path,
            )
        )

    final_pano_id = _final_pano_id(episode, movements)
    if final_pano_id:
        final_index = len(movements)
        last_movement = movements[-1] if movements else {}
        final_image_path = find_render_capture(
            final_pano_id,
            heading=_finite_number(last_movement.get("selected_action_heading")),
        )
        if final_image_path is None:
            missing_frames.append(
                {"frame_index": final_index, "pano_id": final_pano_id, "reason": "missing_final_render"}
            )
        else:
            frames.append(
                _panorama_frame_payload(
                    frame_index=final_index,
                    pano_id=final_pano_id,
                    room_id=_string_or_none(last_movement.get("next_room_id")),
                    next_room_id=None,
                    subgoal_room_id=_string_or_none(last_movement.get("subgoal_room_id"))
                    or _string_or_none(episode.get("target_room_id")),
                    active_target_room_id=_string_or_none(last_movement.get("active_target_room_id"))
                    or _string_or_none(episode.get("target_room_id")),
                    heading=_finite_number(last_movement.get("selected_action_heading")),
                    image_path=final_image_path,
                )
            )

    return {
        "schema_version": 1,
        "frame_count": len(frames),
        "missing_count": len(missing_frames),
        "frames": frames,
        "missing_frames": missing_frames,
    }


def _episode_payload(payload: dict[str, Any]) -> dict[str, Any]:
    episode = payload.get("episode")
    return episode if isinstance(episode, dict) else payload


def _trajectory_movements(payload: dict[str, Any]) -> list[dict[str, Any]]:
    movements: list[dict[str, Any]] = []
    for round_payload in payload.get("rounds") or []:
        if not isinstance(round_payload, dict):
            continue
        for step in round_payload.get("movement_steps") or []:
            if isinstance(step, dict):
                movements.append(step)
    return movements


def _image_path_for_movement(movement: dict[str, Any]) -> Path | None:
    pano_id = _string_or_none(movement.get("current_pano_id"))
    if not pano_id:
        return None

    action_heading = _finite_number(movement.get("selected_action_heading"))
    if action_heading is not None:
        action_image = find_render_capture(pano_id, heading=action_heading)
        if action_image is not None:
            return action_image

    selected_capture_index = _integer_or_none(movement.get("selected_capture_index"))
    for score in _candidate_view_scores(movement, selected_capture_index):
        raw_path = _string_or_none(score.get("path"))
        if not raw_path:
            continue
        try:
            image_path = resolve_project_image_path(raw_path)
        except ValueError:
            continue
        if image_path.exists():
            return image_path

    return find_render_capture(
        pano_id,
        capture_index=selected_capture_index,
        heading=_finite_number(movement.get("selected_view_heading")),
    )


def _candidate_view_scores(movement: dict[str, Any], selected_capture_index: int | None) -> list[dict[str, Any]]:
    scores = [score for score in movement.get("view_scores") or [] if isinstance(score, dict)]
    selected = [score for score in scores if score.get("selected") is True]
    if selected:
        return selected + [score for score in scores if score not in selected]
    if selected_capture_index is not None:
        matching = [
            score for score in scores if _integer_or_none(score.get("capture_index")) == selected_capture_index
        ]
        if matching:
            return matching + [score for score in scores if score not in matching]
    return scores


def _panorama_frame_payload(
    *,
    frame_index: int,
    pano_id: str,
    room_id: str | None,
    next_room_id: str | None,
    subgoal_room_id: str | None,
    active_target_room_id: str | None,
    heading: float | None,
    image_path: Path,
) -> dict[str, Any]:
    return {
        "frame_index": frame_index,
        "pano_id": pano_id,
        "room_id": room_id,
        "next_room_id": next_room_id,
        "subgoal_room_id": subgoal_room_id,
        "active_target_room_id": active_target_room_id,
        "heading": heading,
        "duration_ms": 900,
        "image_url": "/api/render-image?path=" + quote(str(image_path), safe=""),
    }


def _final_pano_id(payload: dict[str, Any], movements: list[dict[str, Any]]) -> str | None:
    final_pano_id = _string_or_none(payload.get("final_pano_id"))
    if final_pano_id:
        return final_pano_id
    pano_path = payload.get("pano_path")
    if isinstance(pano_path, list) and pano_path:
        return _string_or_none(pano_path[-1])
    if movements:
        return _string_or_none(movements[-1].get("next_pano_id"))
    return None


def resolve_project_image_path(raw_path: str) -> Path:
    if not raw_path:
        raise ValueError("Empty image path.")
    candidate = Path(raw_path)
    if candidate.is_absolute():
        resolved = candidate.resolve()
        if not _is_allowed_image_path(resolved):
            marker = f"{PROJECT_ROOT.name}/"
            if marker in raw_path:
                resolved = (PROJECT_ROOT / raw_path.split(marker, 1)[1]).resolve()
    else:
        resolved = (PROJECT_ROOT / candidate).resolve()
    if not _is_allowed_image_path(resolved):
        raise ValueError("Image path is outside the allowed project image directories.")
    if resolved.suffix.lower() not in IMAGE_SUFFIXES:
        raise ValueError("Unsupported image type.")
    return resolved


def find_render_capture(pano_id: str, capture_index: int | None = None, heading: float | None = None) -> Path | None:
    pano_dir = (PROJECT_ROOT / "renders" / "room_grounding_fov90" / pano_id).resolve()
    if not _path_is_relative_to(pano_dir, PROJECT_ROOT / "renders") or not pano_dir.is_dir():
        return None
    candidates = _render_candidates(pano_dir, pano_id, capture_index)
    if not candidates:
        candidates = _render_candidates(pano_dir, pano_id, None)
    if not candidates:
        return None
    if heading is None:
        return candidates[0]
    return min(candidates, key=lambda path: _heading_distance(_capture_heading(path), heading))


def _render_candidates(pano_dir: Path, pano_id: str, capture_index: int | None) -> list[Path]:
    capture_prefix = None if capture_index is None else f"{pano_id}_{capture_index:02d}_"
    candidates = []
    for path in sorted(pano_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if capture_prefix is not None and not path.name.startswith(capture_prefix):
            continue
        if capture_prefix is None and not path.name.startswith(f"{pano_id}_"):
            continue
        candidates.append(path)
    return candidates


def _capture_heading(path: Path) -> float | None:
    match = re.search(r"_([0-9]{3})deg\.[^.]+$", path.name)
    return float(match.group(1)) if match else None


def _heading_distance(left: float | None, right: float) -> float:
    if left is None:
        return 360.0
    return abs(((left - right + 180.0) % 360.0) - 180.0)


def _is_allowed_image_path(path: Path) -> bool:
    return any(_path_is_relative_to(path, root) for root in (PROJECT_ROOT / "renders", PROJECT_ROOT / "outputs"))


def _path_is_relative_to(path: Path, root: Path) -> bool:
    resolved_root = root.resolve()
    return path == resolved_root or resolved_root in path.parents


def _string_or_none(value: object) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _integer_or_none(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve the exported panorama viewer with trajectory image APIs.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir = output_dir.resolve()
    if not (output_dir / "index.html").exists() or not (output_dir / "viewer_data.json").exists():
        raise SystemExit(
            f"Viewer export is missing under {output_dir}. Run `python3 tools/pano_viewer/export.py` first."
        )
    PanoViewerRequestHandler.output_dir = output_dir
    server = ThreadingHTTPServer((args.host, args.port), PanoViewerRequestHandler)
    print(f"Serving pano viewer at http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping pano viewer.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
