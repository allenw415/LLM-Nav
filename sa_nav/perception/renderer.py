from __future__ import annotations

import json
import math
import os
import random
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Callable

CARDINAL_HEADINGS = (0.0, 90.0, 180.0, 270.0)
GROUNDING_HEADINGS = (330.0, 60.0, 150.0, 240.0)
MUSEUM_HEADINGS = (330.0, 60.0, 150.0, 240.0)
CARDINAL_LABELS = ("north", "east", "south", "west")
CARDINAL4_CAPTURE_LABELS = ("view_0", "view_1", "view_2", "view_3")
MUSEUM_INTERSTITIAL_LABELS = ("north_to_east", "east_to_south", "south_to_west", "west_to_north")
MUSEUM_CAPTURE_LABELS = tuple(
    label
    for pair in zip(CARDINAL_LABELS, MUSEUM_INTERSTITIAL_LABELS)
    for label in pair
)


def normalize_heading(heading: float) -> float:
    return float(heading) % 360.0


def sanitize_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value) or "pano"


def circular_mean_headings(headings: list[float]) -> float:
    if not headings:
        return 0.0

    sin_sum = sum(math.sin(math.radians(heading)) for heading in headings)
    cos_sum = sum(math.cos(math.radians(heading)) for heading in headings)
    if abs(sin_sum) < 1e-9 and abs(cos_sum) < 1e-9:
        return normalize_heading(headings[0])
    return normalize_heading(math.degrees(math.atan2(sin_sum, cos_sum)))


def build_streetview_url(
    *,
    api_key: str,
    pano_id: str,
    heading: float,
    pitch: float,
    fov: int,
    width: int,
    height: int,
) -> str:
    query = urllib.parse.urlencode(
        {
            "size": f"{width}x{height}",
            "pano": pano_id,
            "heading": f"{normalize_heading(heading):.6f}",
            "pitch": f"{pitch:.6f}",
            "fov": str(fov),
            "key": api_key,
        }
    )
    return f"https://maps.googleapis.com/maps/api/streetview?{query}"


def _download_image(url: str, output_path: Path, *, timeout: float = 60.0) -> None:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        content_type = response.headers.get("Content-Type", "")
        if "image" not in content_type:
            body = response.read(200).decode("utf-8", errors="replace")
            raise RuntimeError(f"Street View API did not return an image: {content_type} {body}")
        output_path.write_bytes(response.read())


def streetview_metadata_available(
    *,
    api_key: str,
    pano_id: str,
    timeout: float = 60.0,
) -> bool:
    """Check panorama availability through Google's non-image metadata endpoint."""
    query = urllib.parse.urlencode({"pano": pano_id, "key": api_key})
    url = f"https://maps.googleapis.com/maps/api/streetview/metadata?{query}"
    with urllib.request.urlopen(url, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    status = payload.get("status") if isinstance(payload, dict) else None
    if status == "OK":
        return True
    if status in {"ZERO_RESULTS", "NOT_FOUND"}:
        return False
    raise RuntimeError(
        f"Street View metadata request failed for pano {pano_id}: {status or 'invalid_response'}"
    )


def resolve_manifest_capture_path(manifest_path: Path, capture_path: str) -> Path:
    candidate = Path(capture_path)
    if candidate.is_absolute():
        return candidate
    return manifest_path.parent / candidate


def expected_capture_labels(heading_mode: str) -> tuple[str, ...] | None:
    if heading_mode == "museum":
        return MUSEUM_CAPTURE_LABELS
    if heading_mode in {"grounding", "cardinal"}:
        return CARDINAL_LABELS
    if heading_mode == "cardinal4":
        return CARDINAL4_CAPTURE_LABELS
    return None


def expected_capture_headings(heading_mode: str) -> tuple[float, ...] | None:
    if heading_mode == "grounding":
        return GROUNDING_HEADINGS
    if heading_mode in {"cardinal", "cardinal4"}:
        return CARDINAL_HEADINGS
    return None


def manifest_cache_miss_reason(
    manifest_path: str | Path | None,
    *,
    expected_pano_id: str | None = None,
    expected_heading_mode: str | None = None,
    expected_pitch: float | None = None,
    expected_fov: int | None = None,
    expected_width: int | None = None,
    expected_height: int | None = None,
    expected_labels: tuple[str, ...] | list[str] | None = None,
    expected_headings: tuple[float, ...] | list[float] | None = None,
) -> str | None:
    """Return None only for a complete manifest matching the requested render spec."""
    if manifest_path is None:
        return "not_mapped"
    path = Path(manifest_path)
    if not path.is_file():
        return "manifest_missing"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "manifest_invalid"
    if not isinstance(payload, dict):
        return "manifest_invalid"
    if expected_pano_id is not None and payload.get("pano_id") != expected_pano_id:
        return "pano_id_mismatch"
    if expected_heading_mode is not None and payload.get("heading_mode") != expected_heading_mode:
        return "heading_mode_mismatch"
    if expected_pitch is not None:
        try:
            pitch_matches = abs(float(payload.get("pitch")) - float(expected_pitch)) <= 1e-6
        except (TypeError, ValueError):
            pitch_matches = False
        if not pitch_matches:
            return "pitch_mismatch"
    if expected_fov is not None:
        try:
            fov_matches = int(payload.get("fov")) == int(expected_fov)
        except (TypeError, ValueError):
            fov_matches = False
        if not fov_matches:
            return "fov_mismatch"
    size = payload.get("size")
    if expected_width is not None or expected_height is not None:
        if not isinstance(size, dict):
            return "render_size_missing"
        if expected_width is not None:
            try:
                width_matches = int(size.get("width")) == int(expected_width)
            except (TypeError, ValueError):
                width_matches = False
            if not width_matches:
                return "render_width_mismatch"
        if expected_height is not None:
            try:
                height_matches = int(size.get("height")) == int(expected_height)
            except (TypeError, ValueError):
                height_matches = False
            if not height_matches:
                return "render_height_mismatch"
    captures = payload.get("captures")
    if not isinstance(captures, list) or not captures:
        return "manifest_has_no_captures"
    labels = tuple(expected_labels or ())
    headings = tuple(expected_headings or ())
    expected_capture_count = len(labels) or len(headings)
    if expected_capture_count and len(captures) != expected_capture_count:
        return "capture_count_mismatch"
    for index, capture in enumerate(captures):
        if not isinstance(capture, dict):
            return "capture_invalid"
        if labels and capture.get("label") != labels[index]:
            return "capture_label_mismatch"
        if headings:
            try:
                heading_matches = (
                    abs(normalize_heading(float(capture.get("heading"))) - headings[index])
                    <= 1e-6
                )
            except (TypeError, ValueError):
                heading_matches = False
            if not heading_matches:
                return "capture_heading_mismatch"
        image_path = capture.get("path")
        if not isinstance(image_path, str) or not image_path:
            return "capture_path_missing"
        if not resolve_manifest_capture_path(path, image_path).is_file():
            return "capture_image_missing"
    return None


class PanoramaRenderer:
    """
    Perception-side renderer that turns a pano node into multi-view images and a manifest.
    """

    def __init__(
        self,
        pano_graph: dict[str, dict],
        *,
        image_downloader: Callable[[str, Path], None] | None = None,
        image_timeout: float = 60.0,
        rng: random.Random | None = None,
        metadata_checker: Callable[..., bool] | None = None,
    ):
        self.pano_graph = pano_graph
        self.image_timeout = float(image_timeout)
        self.image_downloader = image_downloader or (
            lambda url, output_path: _download_image(url, output_path, timeout=self.image_timeout)
        )
        self.rng = rng or random.Random()
        self.metadata_checker = metadata_checker

    def render(
        self,
        *,
        pano_id: str,
        api_key: str,
        output_dir: str | Path,
        heading_mode: str = "cardinal",
        custom_captures: list[tuple[str, float]] | None = None,
        pitch: float = 0.0,
        fov: int = 45,
        width: int = 512,
        height: int = 512,
        graph_path: str | Path | None = None,
        progress_callback: Callable[[dict], None] | None = None,
    ) -> dict:
        record = self._get_pano_record(pano_id, required=(heading_mode == "graph"))
        captures_to_render = (
            [(str(label), normalize_heading(float(heading))) for label, heading in custom_captures]
            if custom_captures is not None
            else self._resolve_captures(record, heading_mode)
        )

        output_dir = Path(output_dir).resolve()
        pano_slug = sanitize_name(pano_id)
        pano_output_dir = output_dir / pano_slug
        pano_output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = pano_output_dir / f"{pano_slug}_manifest.json"

        cached_manifest = self._load_cached_manifest(
            manifest_path,
            pano_id=pano_id,
            heading_mode=heading_mode,
            pitch=pitch,
            fov=fov,
            width=width,
            height=height,
            graph_path=graph_path,
            captures_to_render=captures_to_render,
        )
        if cached_manifest is not None:
            cached_manifest["manifest_path"] = str(manifest_path)
            if progress_callback is not None:
                progress_callback(
                    {
                        "event": "render_cached",
                        "pano_id": pano_id,
                        "capture_count": len(captures_to_render),
                        "manifest_path": str(manifest_path),
                    }
                )
            return cached_manifest

        if self.metadata_checker is not None:
            if progress_callback is not None:
                progress_callback({"event": "render_metadata_start", "pano_id": pano_id})
            if not self.metadata_checker(api_key=api_key, pano_id=pano_id):
                raise RuntimeError(f"Street View metadata reports no imagery for pano {pano_id}.")
            if progress_callback is not None:
                progress_callback({"event": "render_metadata_done", "pano_id": pano_id})

        captures = []
        with tempfile.TemporaryDirectory(prefix=f".{pano_slug}.render-", dir=output_dir) as tmpdir:
            temporary_dir = Path(tmpdir)
            staged_images: list[tuple[Path, Path]] = []
            for index, (label, heading) in enumerate(captures_to_render):
                filename = f"{pano_slug}_{index:02d}_{label}_{int(round(heading)):03d}deg.png"
                image_path = pano_output_dir / filename
                staged_path = temporary_dir / filename
                image_url = build_streetview_url(
                    api_key=api_key,
                    pano_id=pano_id,
                    heading=heading,
                    pitch=pitch,
                    fov=fov,
                    width=width,
                    height=height,
                )
                if progress_callback is not None:
                    progress_callback(
                        {
                            "event": "render_capture_start",
                            "pano_id": pano_id,
                            "capture_index": index + 1,
                            "capture_count": len(captures_to_render),
                            "label": label,
                            "heading": heading,
                            "path": str(image_path),
                        }
                    )
                self.image_downloader(image_url, staged_path)
                if progress_callback is not None:
                    progress_callback(
                        {
                            "event": "render_capture_done",
                            "pano_id": pano_id,
                            "capture_index": index + 1,
                            "capture_count": len(captures_to_render),
                            "label": label,
                            "heading": heading,
                            "path": str(image_path),
                        }
                    )
                staged_images.append((staged_path, image_path))
                captures.append(
                    {
                        "label": label,
                        "heading": heading,
                        "path": str(image_path),
                        "url": None,
                    }
                )

            manifest = {
                "graph_path": str(graph_path) if graph_path is not None else None,
                "pano_id": pano_id,
                "floor": self._record_floor(record),
                "lat": record.get("lat"),
                "lng": record.get("lng"),
                "heading_mode": heading_mode,
                "pitch": pitch,
                "fov": fov,
                "size": {"width": width, "height": height},
                "captures": captures,
            }
            staged_manifest = temporary_dir / manifest_path.name
            staged_manifest.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            for staged_path, image_path in staged_images:
                os.replace(staged_path, image_path)
            for sidecar_pattern in ("*_detections.json", "*_candidate_themes.json"):
                for sidecar_path in pano_output_dir.glob(sidecar_pattern):
                    sidecar_path.unlink()
            os.replace(staged_manifest, manifest_path)
        manifest["manifest_path"] = str(manifest_path)
        if progress_callback is not None:
            progress_callback(
                {
                    "event": "render_done",
                    "pano_id": pano_id,
                    "capture_count": len(captures_to_render),
                    "manifest_path": str(manifest_path),
                }
            )
        return manifest

    @staticmethod
    def _load_cached_manifest(
        manifest_path: Path,
        *,
        pano_id: str,
        heading_mode: str,
        pitch: float,
        fov: int,
        width: int,
        height: int,
        graph_path: str | Path | None,
        captures_to_render: list[tuple[str, float]],
    ) -> dict | None:
        if manifest_cache_miss_reason(
            manifest_path,
            expected_pano_id=pano_id,
            expected_heading_mode=heading_mode,
            expected_pitch=pitch,
            expected_fov=fov,
            expected_width=width,
            expected_height=height,
            expected_labels=tuple(label for label, _ in captures_to_render),
            expected_headings=(
                tuple(heading for _, heading in captures_to_render)
                if heading_mode != "museum"
                else None
            ),
        ) is not None:
            return None

        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if manifest.get("graph_path") != (str(graph_path) if graph_path is not None else None):
            return None
        return manifest

    def _get_pano_record(self, pano_id: str, *, required: bool) -> dict:
        record = self.pano_graph.get(pano_id)
        if not isinstance(record, dict):
            if not required:
                return {}
            raise KeyError(f"pano_id not found in graph: {pano_id}")
        return record

    def _resolve_captures(self, record: dict, heading_mode: str) -> list[tuple[str, float]]:
        if heading_mode == "museum":
            return self._museum_captures()
        if heading_mode == "grounding":
            return list(zip(CARDINAL_LABELS, GROUNDING_HEADINGS))
        if heading_mode == "cardinal":
            return list(zip(CARDINAL_LABELS, CARDINAL_HEADINGS))
        if heading_mode == "cardinal4":
            return list(zip(CARDINAL4_CAPTURE_LABELS, CARDINAL_HEADINGS))
        return [(f"view{index}", heading) for index, heading in enumerate(self._graph_aligned_headings(record))]

    def _graph_aligned_headings(self, record: dict) -> list[float]:
        graph_headings = self._neighbor_headings(record)
        if not graph_headings:
            return list(CARDINAL_HEADINGS)

        base_heading = circular_mean_headings(graph_headings)
        return [normalize_heading(base_heading + index * 90.0) for index in range(4)]

    def _museum_captures(self) -> list[tuple[str, float]]:
        captures: list[tuple[str, float]] = []
        for index, heading in enumerate(MUSEUM_HEADINGS):
            captures.append((CARDINAL_LABELS[index], heading))
            next_heading = MUSEUM_HEADINGS[(index + 1) % len(MUSEUM_HEADINGS)]
            captures.append((MUSEUM_INTERSTITIAL_LABELS[index], self._sample_heading_between(heading, next_heading)))
        return captures

    def _sample_heading_between(self, start_heading: float, end_heading: float) -> float:
        span = (normalize_heading(end_heading) - normalize_heading(start_heading)) % 360.0
        if span == 0.0:
            span = 360.0
        fraction = min(max(self.rng.random(), 1e-6), 1.0 - 1e-6)
        return normalize_heading(start_heading + span * fraction)

    @staticmethod
    def _record_floor(record: dict) -> str | None:
        floor = record.get("floor")
        if floor is None:
            return None
        return str(floor)

    @staticmethod
    def _neighbor_headings(record: dict) -> list[float]:
        headings: list[float] = []
        if isinstance(record.get("neighbors"), list):
            for neighbor in record["neighbors"]:
                if isinstance(neighbor, dict) and isinstance(neighbor.get("geocentric_heading_deg"), (int, float)):
                    headings.append(normalize_heading(float(neighbor["geocentric_heading_deg"])))
        if headings:
            return headings

        if isinstance(record.get("links"), list):
            for link in record["links"]:
                if isinstance(link, dict) and isinstance(link.get("heading"), (int, float)):
                    headings.append(normalize_heading(float(link["heading"])))
        return headings
