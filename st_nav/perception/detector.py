from __future__ import annotations

import json
import mimetypes
import os
import socket
import urllib.request
from base64 import b64encode
from pathlib import Path
from typing import Callable

from ..common.prompts import (
    build_view_detection_input,
    build_view_detection_instructions,
    build_view_detection_schema,
)
from ..common.types import EntityDetection, Observation, RenderedView, ViewDetection
from .renderer import PanoramaRenderer, normalize_heading


class ViewDetector:
    """
    Multi-view visual recognition stage.

    Detection priority:
    1. Optional sibling `*_detections.json` file for offline/manual testing
    2. OpenAI Responses API with multi-image input for real VLM detection
    """

    def __init__(
        self,
        *,
        model: str = "gpt-5-mini",
        api_key: str | None = None,
        api_base: str = "https://api.openai.com/v1",
        request_timeout: float = 180.0,
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
        trace_path = manifest_path.with_name(f"{manifest_path.stem}_detections_trace.json")
        if self.use_detection_files and detection_path.exists():
            self.last_traces = self._load_trace_file(trace_path)
            return self._load_detection_file(detection_path)

        if not self.api_key and self.response_client is None:
            return []

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        detections = self._detect_manifest(manifest)
        self._write_detection_file(detection_path, detections)
        self._write_trace_file(trace_path)
        return detections

    def _load_detection_file(self, detection_path: Path) -> list[ViewDetection]:
        payload = json.loads(detection_path.read_text(encoding="utf-8"))
        grouped: dict[str, ViewDetection] = {}
        for record in payload.get("entities", []):
            if not isinstance(record, dict):
                continue
            capture_label = record.get("capture_label")
            if isinstance(capture_label, str) and capture_label:
                detection = grouped.setdefault(capture_label, ViewDetection(capture_label=capture_label))
                entity = self._entity_from_record(record, default_source_view=capture_label)
                if entity is not None:
                    detection.entities.append(entity)
                continue

            detection = grouped.setdefault("multiview", ViewDetection(capture_label="multiview"))
            entity = self._entity_from_record(record, default_source_view="multiview")
            if entity is not None:
                detection.entities.append(entity)
        return list(grouped.values())

    def _write_detection_file(self, detection_path: Path, view_detections: list[ViewDetection]) -> None:
        records: list[dict] = []
        for view_detection in view_detections:
            for entity in view_detection.entities:
                record = {
                    "name": entity.name,
                    "confidence": entity.confidence,
                    "kind": entity.kind,
                }
                source_views = entity.metadata.get("source_views")
                if isinstance(source_views, list) and source_views:
                    record["source_views"] = list(source_views)
                elif entity.source_view:
                    record["capture_label"] = entity.source_view

                for key, value in entity.metadata.items():
                    if key not in {"source_views"}:
                        record[key] = value
                records.append(record)

        detection_path.write_text(
            json.dumps({"entities": records}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _write_trace_file(self, trace_path: Path) -> None:
        trace_path.write_text(
            json.dumps({"requests_and_responses": self.last_traces}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def _load_trace_file(trace_path: Path) -> list[dict]:
        if not trace_path.exists():
            return []
        try:
            payload = json.loads(trace_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        traces = payload.get("requests_and_responses")
        if not isinstance(traces, list):
            return []
        return [trace for trace in traces if isinstance(trace, dict)]

    def _detect_manifest(self, manifest: dict) -> list[ViewDetection]:
        captures = [
            capture
            for capture in manifest.get("captures", [])
            if isinstance(capture, dict) and isinstance(capture.get("path"), str) and capture.get("path")
        ]
        if not captures:
            return []

        request_body = self._build_request_body(captures)
        payload = self._create_response(request_body)
        self.last_traces.append(
            {
                "capture_label": "multiview",
                "capture_labels": [str(capture.get("label", "unknown")) for capture in captures],
                "request": self._redact_request_body(request_body),
                "response": self._clone_json(payload),
            }
        )
        parsed = self._parse_output_payload(payload)
        entities: list[EntityDetection] = []
        for record in parsed.get("entities", []):
            if not isinstance(record, dict):
                continue
            entity = self._entity_from_record(record, default_source_view="multiview")
            if entity is not None:
                entities.append(entity)
        return [ViewDetection(capture_label="multiview", entities=entities)] if entities else []

    def _build_request_body(self, captures: list[dict]) -> dict:
        content = [{"type": "input_text", "text": build_view_detection_input(captures)}]
        for capture in captures:
            label = str(capture.get("label", "unknown"))
            heading = capture.get("heading")
            heading_text = f"{float(heading):.1f} deg" if isinstance(heading, (int, float)) else "unknown heading"
            content.extend(
                [
                    {
                        "type": "input_text",
                        "text": f"View label: {label}. Heading: {heading_text}.",
                    },
                    {
                        "type": "input_image",
                        "image_url": self._image_to_data_url(Path(str(capture["path"]))),
                        "detail": "high",
                    },
                ]
            )
        return {
            "model": self.model,
            "instructions": build_view_detection_instructions(),
            "input": [
                {
                    "role": "user",
                    "content": content,
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

    @staticmethod
    def _entity_from_record(record: dict, *, default_source_view: str) -> EntityDetection | None:
        name = record.get("name")
        kind = record.get("kind")
        confidence = record.get("confidence")
        source_views = record.get("source_views")

        if not isinstance(name, str) or not name:
            return None
        if not isinstance(kind, str) or not kind:
            kind = "other"
        if not isinstance(confidence, (int, float)):
            confidence = 0.0

        normalized_source_views: list[str] = []
        if isinstance(source_views, list):
            for value in source_views:
                if isinstance(value, str) and value and value not in normalized_source_views:
                    normalized_source_views.append(value)

        capture_label = record.get("capture_label")
        if not normalized_source_views and isinstance(capture_label, str) and capture_label:
            normalized_source_views.append(capture_label)
        if not normalized_source_views and default_source_view:
            normalized_source_views.append(default_source_view)

        metadata = {
            key: value
            for key, value in record.items()
            if key not in {"name", "confidence", "kind", "capture_label", "source_views"}
        }
        metadata["source_views"] = normalized_source_views
        metadata["view_count"] = len(normalized_source_views)
        return EntityDetection(
            name=name,
            confidence=float(confidence),
            kind=kind,
            source_view=normalized_source_views[0] if len(normalized_source_views) == 1 else "multiview",
            metadata=metadata,
        )

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
        try:
            with urllib.request.urlopen(request, timeout=self.request_timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except TimeoutError as exc:
            raise TimeoutError(
                "OpenAI Responses API timed out after "
                f"{self.request_timeout:.0f}s while processing the multi-view panorama request. "
                "Try a larger request timeout."
            ) from exc
        except socket.timeout as exc:
            raise TimeoutError(
                "OpenAI Responses API timed out after "
                f"{self.request_timeout:.0f}s while processing the multi-view panorama request. "
                "Try a larger request timeout."
            ) from exc

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
                "floor": manifest.get("floor"),
                "lat": manifest.get("lat"),
                "lng": manifest.get("lng"),
            },
        )

    @staticmethod
    def _flatten_entities(view_detections: list[ViewDetection]) -> list[EntityDetection]:
        grouped: dict[tuple[str, str], EntityDetection] = {}
        for view_detection in view_detections:
            for entity in view_detection.entities:
                source_views = MultiViewAggregator._source_views_for_entity(entity)
                key = (entity.name.strip().lower(), entity.kind)
                existing = grouped.get(key)
                if existing is None:
                    metadata = dict(entity.metadata)
                    metadata["source_views"] = source_views
                    metadata["view_count"] = len(source_views)
                    grouped[key] = EntityDetection(
                        name=entity.name,
                        confidence=entity.confidence,
                        kind=entity.kind,
                        source_view=source_views[0] if len(source_views) == 1 else "multiview",
                        metadata=metadata,
                    )
                    continue
                existing.confidence = max(existing.confidence, entity.confidence)
                existing_source_views = existing.metadata.setdefault("source_views", [])
                for source_view in source_views:
                    if source_view not in existing_source_views:
                        existing_source_views.append(source_view)
                existing.metadata["view_count"] = len(existing_source_views)
                existing.source_view = existing_source_views[0] if len(existing_source_views) == 1 else "multiview"
        return sorted(grouped.values(), key=lambda item: (-item.confidence, item.name.lower()))

    @staticmethod
    def _source_views_for_entity(entity: EntityDetection) -> list[str]:
        source_views = entity.metadata.get("source_views")
        if isinstance(source_views, list):
            normalized = []
            for value in source_views:
                if isinstance(value, str) and value and value not in normalized:
                    normalized.append(value)
            if normalized:
                return normalized
        if entity.source_view:
            return [entity.source_view]
        return []


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
