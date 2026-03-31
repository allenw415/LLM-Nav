from __future__ import annotations

import json
import math
import mimetypes
import os
import urllib.parse
import urllib.request
from base64 import b64encode
from pathlib import Path
from typing import Callable

from .models import EntityDetection, Observation, RenderedView, ViewDetection
from .prompts import build_view_detection_input, build_view_detection_instructions, build_view_detection_schema

CARDINAL_HEADINGS = (0.0, 90.0, 180.0, 270.0)
MUSEUM_HEADINGS = (330.0, 60.0, 150.0, 240.0)
CARDINAL_LABELS = ("north", "east", "south", "west")


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


def _download_image(url: str, output_path: Path) -> None:
    with urllib.request.urlopen(url) as response:
        content_type = response.headers.get("Content-Type", "")
        if "image" not in content_type:
            body = response.read(200).decode("utf-8", errors="replace")
            raise RuntimeError(f"Street View API did not return an image: {content_type} {body}")
        output_path.write_bytes(response.read())


class PanoramaRenderer:
    """
    Perception-side renderer that turns a pano node into multi-view images and a manifest.

    This keeps `pano -> rendered views` inside the perception subsystem, while CLI scripts
    remain thin wrappers.
    """

    def __init__(
        self,
        pano_graph: dict[str, dict],
        *,
        image_downloader: Callable[[str, Path], None] | None = None,
    ):
        self.pano_graph = pano_graph
        self.image_downloader = image_downloader or _download_image

    def render(
        self,
        *,
        pano_id: str,
        api_key: str,
        output_dir: str | Path,
        heading_mode: str = "cardinal",
        pitch: float = 0.0,
        fov: int = 45,
        width: int = 640,
        height: int = 640,
        graph_path: str | Path | None = None,
    ) -> dict:
        record = self._get_pano_record(pano_id, required=(heading_mode == "graph"))
        headings = self._resolve_headings(record, heading_mode)
        labels = self._labels_for_heading_mode(heading_mode)

        output_dir = Path(output_dir)
        pano_slug = sanitize_name(pano_id)
        pano_output_dir = output_dir / pano_slug
        pano_output_dir.mkdir(parents=True, exist_ok=True)

        captures = []
        for index, heading in enumerate(headings):
            label = labels[index]
            filename = f"{pano_slug}_{index:02d}_{label}_{int(round(heading)):03d}deg.png"
            image_path = pano_output_dir / filename
            image_url = build_streetview_url(
                api_key=api_key,
                pano_id=pano_id,
                heading=heading,
                pitch=pitch,
                fov=fov,
                width=width,
                height=height,
            )
            self.image_downloader(image_url, image_path)
            captures.append(
                {
                    "label": label,
                    "heading": heading,
                    "path": str(image_path),
                    "url": image_url,
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
        manifest_path = pano_output_dir / f"{pano_slug}_manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        manifest["manifest_path"] = str(manifest_path)
        return manifest

    def _get_pano_record(self, pano_id: str, *, required: bool) -> dict:
        record = self.pano_graph.get(pano_id)
        if not isinstance(record, dict):
            if not required:
                return {}
            raise KeyError(f"pano_id not found in graph: {pano_id}")
        return record

    def _resolve_headings(self, record: dict, heading_mode: str) -> list[float]:
        if heading_mode == "museum":
            return list(MUSEUM_HEADINGS)
        if heading_mode == "cardinal":
            return list(CARDINAL_HEADINGS)
        return self._graph_aligned_headings(record)

    @staticmethod
    def _labels_for_heading_mode(heading_mode: str) -> list[str]:
        if heading_mode in {"museum", "cardinal"}:
            return list(CARDINAL_LABELS)
        return ["view0", "view1", "view2", "view3"]

    def _graph_aligned_headings(self, record: dict) -> list[float]:
        graph_headings = self._neighbor_headings(record)
        if not graph_headings:
            return list(CARDINAL_HEADINGS)

        base_heading = circular_mean_headings(graph_headings)
        return [normalize_heading(base_heading + index * 90.0) for index in range(4)]

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


class ViewDetector:
    """
    View-level visual recognition stage.

    Detection priority:
    1. Optional sibling `*_detections.json` file for offline/manual testing
    2. OpenAI Responses API with image input for real VLM detection
    """

    def __init__(
        self,
        *,
        model: str = "gpt-5-mini",
        api_key: str | None = None,
        api_base: str = "https://api.openai.com/v1",
        request_timeout: float = 60.0,
        response_client: Callable[[dict], dict] | None = None,
        use_detection_files: bool = True,
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.api_base = api_base.rstrip("/")
        self.request_timeout = request_timeout
        self.response_client = response_client
        self.use_detection_files = use_detection_files
        self.last_traces: list[dict] = []

    def detect(self, manifest_path: str | Path) -> list[ViewDetection]:
        manifest_path = Path(manifest_path)
        self.last_traces = []
        detection_path = manifest_path.with_name(f"{manifest_path.stem}_detections.json")
        if self.use_detection_files and detection_path.exists():
            return self._load_detection_file(detection_path)

        if not self.api_key and self.response_client is None:
            return []

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        view_detections: list[ViewDetection] = []
        for capture in manifest.get("captures", []):
            if not isinstance(capture, dict):
                continue
            label = str(capture.get("label", "unknown"))
            image_path = capture.get("path")
            if not isinstance(image_path, str) or not image_path:
                continue
            entities = self._detect_view(Path(image_path), capture_label=label)
            view_detections.append(ViewDetection(capture_label=label, entities=entities))
        return view_detections

    def _load_detection_file(self, detection_path: Path) -> list[ViewDetection]:
        payload = json.loads(detection_path.read_text(encoding="utf-8"))
        grouped: dict[str, ViewDetection] = {}
        for record in payload.get("entities", []):
            if not isinstance(record, dict):
                continue
            name = record.get("name")
            if not isinstance(name, str) or not name:
                continue
            capture_label = str(record.get("capture_label", "unknown"))
            detection = grouped.setdefault(capture_label, ViewDetection(capture_label=capture_label))
            detection.entities.append(
                EntityDetection(
                    name=name,
                    confidence=float(record.get("confidence", 0.0)),
                    kind=str(record.get("kind", "entity")),
                    source_view=capture_label,
                    metadata={
                        key: value
                        for key, value in record.items()
                        if key not in {"name", "confidence", "kind", "capture_label"}
                    },
                )
            )
        return list(grouped.values())

    def _detect_view(self, image_path: Path, *, capture_label: str) -> list[EntityDetection]:
        request_body = self._build_request_body(image_path, capture_label=capture_label)
        payload = self._create_response(request_body)
        self.last_traces.append(
            {
                "capture_label": capture_label,
                "request": self._redact_request_body(request_body),
                "response": self._clone_json(payload),
            }
        )
        parsed = self._parse_output_payload(payload)
        entities: list[EntityDetection] = []
        for record in parsed.get("entities", []):
            if not isinstance(record, dict):
                continue
            name = record.get("name")
            kind = record.get("kind")
            confidence = record.get("confidence")
            if not isinstance(name, str) or not name:
                continue
            if not isinstance(kind, str) or not kind:
                kind = "other"
            if not isinstance(confidence, (int, float)):
                confidence = 0.0
            entities.append(
                EntityDetection(
                    name=name,
                    confidence=float(confidence),
                    kind=kind,
                    source_view=capture_label,
                )
            )
        return entities

    def _build_request_body(self, image_path: Path, *, capture_label: str) -> dict:
        return {
            "model": self.model,
            "instructions": build_view_detection_instructions(),
            "input": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": build_view_detection_input(capture_label),
                        },
                        {
                            "type": "input_image",
                            "image_url": self._image_to_data_url(image_path),
                            "detail": "high",
                        },
                    ],
                }
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "view_detection",
                    "strict": True,
                    "schema": build_view_detection_schema(),
                }
            },
        }

    def _create_response(self, request_body: dict) -> dict:
        if self.response_client is not None:
            return self.response_client(request_body)

        request = urllib.request.Request(
            f"{self.api_base}/responses",
            data=json.dumps(request_body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.request_timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    @staticmethod
    def _parse_output_payload(payload: dict) -> dict:
        output_text = payload.get("output_text")
        if not isinstance(output_text, str) or not output_text.strip():
            fragments: list[str] = []
            for item in payload.get("output", []):
                if not isinstance(item, dict) or item.get("type") != "message":
                    continue
                for content in item.get("content", []):
                    if not isinstance(content, dict):
                        continue
                    text = content.get("text")
                    if isinstance(text, str):
                        fragments.append(text)
            output_text = "".join(fragments)
        if not isinstance(output_text, str) or not output_text.strip():
            return {"entities": []}
        return json.loads(output_text)

    @staticmethod
    def _clone_json(payload: dict) -> dict:
        return json.loads(json.dumps(payload))

    @classmethod
    def _redact_request_body(cls, request_body: dict) -> dict:
        cloned = cls._clone_json(request_body)
        for item in cloned.get("input", []):
            if not isinstance(item, dict):
                continue
            for content in item.get("content", []):
                if not isinstance(content, dict):
                    continue
                if content.get("type") == "input_image" and isinstance(content.get("image_url"), str):
                    content["image_url"] = "<IMAGE_DATA_URL_OMITTED>"
        return cloned

    @staticmethod
    def _image_to_data_url(image_path: Path) -> str:
        mime_type, _ = mimetypes.guess_type(str(image_path))
        if not mime_type:
            mime_type = "image/png"
        return f"data:{mime_type};base64,{b64encode(image_path.read_bytes()).decode('ascii')}"


class MultiViewAggregator:
    """
    Aggregate rendered views and multi-view detections into a single observation.
    """

    def __init__(self, pano_graph: dict[str, dict]):
        self.pano_graph = pano_graph

    def aggregate(
        self,
        manifest_path: str | Path,
        *,
        current_heading: float,
        view_detections: list[ViewDetection] | None = None,
    ) -> Observation:
        manifest_path = Path(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        pano_id = str(manifest["pano_id"])
        views = [
            RenderedView(
                label=str(capture["label"]),
                heading=float(capture["heading"]),
                path=str(capture["path"]),
                url=capture.get("url"),
            )
            for capture in manifest.get("captures", [])
            if isinstance(capture, dict)
        ]

        entities = self._flatten_entities(view_detections or [])
        heading_estimate = normalize_heading(current_heading)
        return Observation(
            pano_id=pano_id,
            views=views,
            entities=entities,
            heading_estimate=heading_estimate,
            metadata={
                "manifest_path": str(manifest_path),
                "heading_mode": manifest.get("heading_mode"),
            },
        )

    @staticmethod
    def _flatten_entities(view_detections: list[ViewDetection]) -> list[EntityDetection]:
        grouped: dict[tuple[str, str], EntityDetection] = {}
        for view_detection in view_detections:
            for entity in view_detection.entities:
                key = (entity.name.strip().lower(), entity.kind)
                existing = grouped.get(key)
                if existing is None:
                    grouped[key] = EntityDetection(
                        name=entity.name,
                        confidence=entity.confidence,
                        kind=entity.kind,
                        source_view=entity.source_view,
                        metadata={
                            "source_views": [entity.source_view],
                            "view_count": 1,
                        },
                    )
                    continue
                existing.confidence = max(existing.confidence, entity.confidence)
                source_views = existing.metadata.setdefault("source_views", [])
                if entity.source_view not in source_views:
                    source_views.append(entity.source_view)
                    existing.metadata["view_count"] = len(source_views)
        return sorted(grouped.values(), key=lambda item: (-item.confidence, item.name.lower()))


class PerceptionPipeline:
    """
    Explicit perception flow: render -> detect -> aggregate.
    """

    def __init__(
        self,
        *,
        pano_graph: dict[str, dict],
        renderer: PanoramaRenderer | None = None,
        detector: ViewDetector | None = None,
        aggregator: MultiViewAggregator | None = None,
    ):
        self.pano_graph = pano_graph
        self.renderer = renderer or PanoramaRenderer(pano_graph)
        self.detector = detector or ViewDetector()
        self.aggregator = aggregator or MultiViewAggregator(pano_graph)

    def render_views(self, **kwargs) -> dict:
        return self.renderer.render(**kwargs)

    def detect_views(self, manifest_path: str | Path) -> list[ViewDetection]:
        return self.detector.detect(manifest_path)

    def aggregate_observation(
        self,
        manifest_path: str | Path,
        *,
        current_heading: float,
        view_detections: list[ViewDetection] | None = None,
    ) -> Observation:
        return self.aggregator.aggregate(
            manifest_path,
            current_heading=current_heading,
            view_detections=view_detections,
        )

    def observe_from_manifest(self, manifest_path: str | Path, *, current_heading: float) -> Observation:
        detections = self.detect_views(manifest_path)
        return self.aggregate_observation(
            manifest_path,
            current_heading=current_heading,
            view_detections=detections,
        )


class ManifestPerceptionProvider:
    """
    Backward-compatible wrapper over the explicit perception pipeline.
    """

    def __init__(self, pano_graph: dict[str, dict]):
        self.pipeline = PerceptionPipeline(pano_graph=pano_graph)

    def observe(self, manifest_path: str | Path, *, current_heading: float) -> Observation:
        return self.pipeline.observe_from_manifest(manifest_path, current_heading=current_heading)
