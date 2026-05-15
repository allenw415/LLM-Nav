from __future__ import annotations

import json
import mimetypes
from base64 import b64encode
from pathlib import Path
from typing import Callable

from ..common.env import resolve_model_environment
from ..common.model_client import DEFAULT_OPENAI_API_BASE, ModelResponseClient, parse_json_output, resolve_api_kind
from ..common.prompts import (
    ENTITY_LOCATION_SCOPES,
    build_view_detection_input,
    build_view_detection_instructions,
    build_view_detection_schema,
    build_view_theme_extraction_input,
    build_view_theme_extraction_instructions,
    build_view_theme_extraction_schema,
    build_visual_detection_localization_input,
    build_visual_detection_localization_instructions,
    build_visual_detection_localization_schema,
)
from ..common.scoring import evidence_scores_to_distribution, normalize_positive_scores
from ..common.types import EntityDetection, Observation, RenderedView, ViewDetection
from .renderer import PanoramaRenderer, normalize_heading


class ViewDetector:
    """
    Multi-view visual recognition stage.

    Detection priority:
    1. Optional sibling `*_detections.json` file for offline/manual testing
    2. Configurable model API with multi-image input for real VLM detection
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
        api_kind: str | None = None,
        request_timeout: float | None = None,
        response_client: Callable[[dict], dict] | None = None,
        use_detection_files: bool = True,
        enable_view_themes: bool = False,
        room_graph: dict[str, dict] | None = None,
        grounding_index=None,
    ):
        settings = resolve_model_environment(
            default_model="gpt-5-mini",
            default_api_base=DEFAULT_OPENAI_API_BASE,
            default_api_kind="responses",
        )
        self.model = model or settings.model_name or "gpt-5-mini"
        self.api_key = api_key or settings.api_key
        self.api_base = (api_base or settings.api_base or DEFAULT_OPENAI_API_BASE).rstrip("/")
        self.api_kind = resolve_api_kind(api_kind or settings.api_kind)
        self.request_timeout = float(request_timeout if request_timeout is not None else (settings.request_timeout or 180.0))
        self.response_client = response_client
        self.use_detection_files = use_detection_files
        self.enable_view_themes = enable_view_themes
        self.room_graph = room_graph
        self.grounding_index = grounding_index
        self.last_traces: list[dict] = []
        self.last_visual_localization: dict | None = None
        self.last_candidate_room_ids: list[str] = []
        self.last_view_theme_observations: list[dict] = []
        self.model_client = ModelResponseClient(
            provider=settings.provider,
            api_key=self.api_key,
            api_base=self.api_base,
            api_kind=self.api_kind,
            request_timeout=self.request_timeout,
            num_ctx=settings.num_ctx,
            temperature=settings.temperature,
            response_client=self.response_client,
        )

    def detect(self, manifest_path: str | Path) -> list[ViewDetection]:
        manifest_path = Path(manifest_path)
        self.last_traces = []
        self.last_visual_localization = None
        self.last_candidate_room_ids = []
        self.last_view_theme_observations = []
        detection_path = manifest_path.with_name(f"{manifest_path.stem}_detections.json")
        trace_path = manifest_path.with_name(f"{manifest_path.stem}_detections_trace.json")
        if self.use_detection_files and detection_path.exists():
            manifest = self._load_manifest_for_cache_check(manifest_path)
            if (
                self._detection_cache_matches_current_settings(detection_path, manifest)
                or (not self.model_client.is_configured() and not self.enable_view_themes)
            ):
                self.last_traces = self._load_trace_file(trace_path)
                return self._load_detection_file(detection_path)

        if not self.model_client.is_configured():
            return []

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        detections = self._detect_manifest(manifest)
        self._write_detection_file(detection_path, detections)
        self._write_trace_file(trace_path)
        return detections

    def _load_detection_file(self, detection_path: Path) -> list[ViewDetection]:
        payload = json.loads(detection_path.read_text(encoding="utf-8"))
        metadata = self._metadata_from_detection_payload(payload)
        self.last_visual_localization = metadata.get("visual_localization")
        self.last_candidate_room_ids = list(metadata.get("candidate_room_ids", []))
        self.last_view_theme_observations = list(metadata.get("view_theme_observations", []))
        grouped: dict[str, ViewDetection] = {}
        for record in payload.get("entities", []):
            if not isinstance(record, dict):
                continue
            capture_label = record.get("capture_label")
            if isinstance(capture_label, str) and capture_label:
                detection = grouped.setdefault(capture_label, ViewDetection(capture_label=capture_label, metadata=dict(metadata)))
                entity = self._entity_from_record(record, default_source_view=capture_label)
                if entity is not None:
                    detection.entities.append(entity)
                continue

            detection = grouped.setdefault("multiview", ViewDetection(capture_label="multiview", metadata=dict(metadata)))
            entity = self._entity_from_record(record, default_source_view="multiview")
            if entity is not None:
                detection.entities.append(entity)
        if not grouped and metadata:
            grouped["multiview"] = ViewDetection(capture_label="multiview", metadata=dict(metadata))
        return list(grouped.values())

    def _detection_cache_matches_current_settings(self, detection_path: Path, manifest: dict | None) -> bool:
        try:
            payload = json.loads(detection_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        if self.enable_view_themes:
            if payload.get("cache_version") != 3:
                return False
            if not isinstance(payload.get("view_theme_observations"), list):
                return False
        if manifest is None:
            return True
        expected_candidate_room_ids = self._candidate_room_ids(manifest)
        if not expected_candidate_room_ids:
            return True
        cached_candidate_room_ids = payload.get("candidate_room_ids")
        return cached_candidate_room_ids == expected_candidate_room_ids

    @staticmethod
    def _load_manifest_for_cache_check(manifest_path: Path) -> dict | None:
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return payload if isinstance(payload, dict) else None

    def _write_detection_file(self, detection_path: Path, view_detections: list[ViewDetection]) -> None:
        records: list[dict] = []
        metadata = self._merged_detection_metadata(view_detections)
        for view_detection in view_detections:
            for entity in view_detection.entities:
                record = {
                    "name": entity.name,
                    "confidence": entity.confidence,
                    "kind": entity.kind,
                    "location_scope": entity.location_scope,
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

        payload = {"cache_version": 3, "entities": records}
        if metadata.get("candidate_room_ids"):
            payload["candidate_room_ids"] = list(metadata["candidate_room_ids"])
        if isinstance(metadata.get("visual_localization"), dict):
            payload["visual_localization"] = dict(metadata["visual_localization"])
        if isinstance(metadata.get("view_theme_observations"), list):
            payload["view_theme_observations"] = list(metadata["view_theme_observations"])
        detection_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

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

        candidate_room_ids = self._candidate_room_ids(manifest)
        candidates = self._candidate_records(candidate_room_ids)
        request_body = (
            self._build_visual_request_body(captures, candidates=candidates)
            if candidates
            else self._build_request_body(captures)
        )
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
        metadata: dict = {}
        visual_localization = parsed.get("visual_localization")
        if isinstance(visual_localization, dict):
            metadata["visual_localization"] = self._normalize_visual_localization(
                visual_localization,
                candidate_room_ids=candidate_room_ids,
            )
            self.last_visual_localization = dict(metadata["visual_localization"])
        if self.enable_view_themes:
            view_theme_payload = self._extract_view_themes(captures)
            observations = view_theme_payload.get("view_theme_observations")
            if isinstance(observations, list):
                metadata["view_theme_observations"] = self._normalize_view_theme_observations(observations)
                self.last_view_theme_observations = list(metadata["view_theme_observations"])
        if candidate_room_ids:
            metadata["candidate_room_ids"] = list(candidate_room_ids)
            self.last_candidate_room_ids = list(candidate_room_ids)
        return [ViewDetection(capture_label="multiview", entities=entities, metadata=metadata)] if entities or metadata else []

    def _extract_view_themes(self, captures: list[dict]) -> dict:
        request_body = self._build_view_theme_request_body(captures)
        payload = self._create_response(request_body)
        self.last_traces.append(
            {
                "capture_label": "view_themes",
                "capture_labels": [f"view_{index}" for index, _ in enumerate(captures)],
                "request": self._redact_request_body(request_body),
                "response": self._clone_json(payload),
            }
        )
        parsed = self._parse_output_payload(payload)
        return parsed if isinstance(parsed, dict) else {}

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

    def _build_visual_request_body(self, captures: list[dict], *, candidates: list[dict]) -> dict:
        content = [
            {
                "type": "input_text",
                "text": build_visual_detection_localization_input(captures=captures, candidates=candidates),
            }
        ]
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
        room_ids = [str(candidate["room_id"]) for candidate in candidates]
        return {
            "model": self.model,
            "instructions": build_visual_detection_localization_instructions(),
            "input": [
                {
                    "role": "user",
                    "content": content,
                }
            ],
            "reasoning": {"effort": "low"},
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "visual_detection_localization",
                    "strict": True,
                    "schema": build_visual_detection_localization_schema(room_ids),
                }
            },
        }

    def _build_view_theme_request_body(self, captures: list[dict]) -> dict:
        content = [{"type": "input_text", "text": build_view_theme_extraction_input(captures)}]
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
                        "image_url": self._image_to_data_url(Path(str(capture["path"]))),
                        "detail": "high",
                    },
                ]
            )
        view_ids = [f"view_{index}" for index, _ in enumerate(captures)]
        return {
            "model": self.model,
            "instructions": build_view_theme_extraction_instructions(),
            "input": [
                {
                    "role": "user",
                    "content": content,
                }
            ],
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

    @staticmethod
    def _entity_from_record(record: dict, *, default_source_view: str) -> EntityDetection | None:
        name = record.get("name")
        kind = record.get("kind")
        confidence = record.get("confidence")
        source_views = record.get("source_views")
        location_scope = record.get("location_scope")

        if not isinstance(name, str) or not name:
            return None
        if not isinstance(kind, str) or not kind:
            kind = "other"
        if not isinstance(confidence, (int, float)):
            confidence = 0.0
        if location_scope not in ENTITY_LOCATION_SCOPES:
            location_scope = "inside"

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
        metadata["location_scope"] = location_scope
        return EntityDetection(
            name=name,
            confidence=float(confidence),
            kind=kind,
            source_view=normalized_source_views[0] if len(normalized_source_views) == 1 else "multiview",
            location_scope=location_scope,
            metadata=metadata,
        )

    def _candidate_room_ids(self, manifest: dict) -> list[str]:
        if not isinstance(self.room_graph, dict) or not self.room_graph:
            return []
        floor = manifest.get("floor")
        if floor is None:
            return sorted(self.room_graph.keys())
        floor_text = str(floor)
        return [
            room_id
            for room_id in sorted(self.room_graph.keys())
            if str(self.room_graph.get(room_id, {}).get("floor")) == floor_text
        ]

    def _candidate_records(self, candidate_room_ids: list[str]) -> list[dict]:
        if not candidate_room_ids or not isinstance(self.room_graph, dict):
            return []
        candidates = []
        for room_id in candidate_room_ids:
            node = self.room_graph.get(room_id, {})
            entry = self.grounding_index.room_entry(room_id) if self.grounding_index is not None else None
            entry = entry or {}
            candidates.append(
                {
                    "room_id": room_id,
                    "title": node.get("title"),
                    "category": node.get("category"),
                    "aliases": list(node.get("aliases") or []) + list(entry.get("aliases") or []),
                    "anchor_entities": list(entry.get("anchor_entities") or []),
                }
            )
        return candidates

    @staticmethod
    def _metadata_from_detection_payload(payload: dict) -> dict:
        metadata: dict = {}
        candidate_room_ids = payload.get("candidate_room_ids")
        if isinstance(candidate_room_ids, list):
            metadata["candidate_room_ids"] = [
                value for value in candidate_room_ids if isinstance(value, str) and value
            ]
        visual_localization = payload.get("visual_localization")
        if isinstance(visual_localization, dict):
            metadata["visual_localization"] = dict(visual_localization)
        view_theme_observations = payload.get("view_theme_observations")
        if isinstance(view_theme_observations, list):
            metadata["view_theme_observations"] = ViewDetector._normalize_view_theme_observations(view_theme_observations)
        return metadata

    @staticmethod
    def _merged_detection_metadata(view_detections: list[ViewDetection]) -> dict:
        merged: dict = {}
        for view_detection in view_detections:
            if not isinstance(view_detection.metadata, dict):
                continue
            if "visual_localization" not in merged and isinstance(view_detection.metadata.get("visual_localization"), dict):
                merged["visual_localization"] = dict(view_detection.metadata["visual_localization"])
            if "candidate_room_ids" not in merged and isinstance(view_detection.metadata.get("candidate_room_ids"), list):
                merged["candidate_room_ids"] = list(view_detection.metadata["candidate_room_ids"])
            if "view_theme_observations" not in merged and isinstance(view_detection.metadata.get("view_theme_observations"), list):
                merged["view_theme_observations"] = list(view_detection.metadata["view_theme_observations"])
        return merged

    @staticmethod
    def _normalize_view_theme_observations(observations: object) -> list[dict]:
        if not isinstance(observations, list):
            return []
        normalized = []
        for record in observations:
            if not isinstance(record, dict):
                continue
            view_id = record.get("view_id")
            if not isinstance(view_id, str) or not view_id:
                continue
            observed_theme = record.get("observed_theme")
            confidence = record.get("confidence")
            visible_room_label = record.get("visible_room_label")
            evidence = record.get("evidence")
            visual_evidence = record.get("visual_evidence")
            theme_matches = record.get("theme_matches")
            current_or_adjacent = record.get("current_or_adjacent")
            spatial_boundary_evidence = record.get("spatial_boundary_evidence")
            reason = record.get("reason")
            normalized.append(
                {
                    "view_id": view_id,
                    "observed_theme": observed_theme if isinstance(observed_theme, str) else "",
                    "confidence": max(0.0, min(1.0, float(confidence))) if isinstance(confidence, (int, float)) else 0.0,
                    "visible_room_label": visible_room_label if isinstance(visible_room_label, str) and visible_room_label else None,
                    "evidence": [value for value in evidence if isinstance(value, str) and value] if isinstance(evidence, list) else [],
                    "visual_evidence": [value for value in visual_evidence if isinstance(value, str) and value]
                    if isinstance(visual_evidence, list)
                    else [],
                    "theme_matches": ViewDetector._normalize_theme_matches(theme_matches),
                    "current_or_adjacent": current_or_adjacent
                    if current_or_adjacent in {"current", "adjacent", "both", "ambiguous"}
                    else "ambiguous",
                    "spatial_boundary_evidence": [
                        value for value in spatial_boundary_evidence if isinstance(value, str) and value
                    ]
                    if isinstance(spatial_boundary_evidence, list)
                    else [],
                    "reason": reason if isinstance(reason, str) else "",
                }
            )
        return normalized

    @staticmethod
    def _normalize_theme_matches(theme_matches: object) -> list[dict]:
        if not isinstance(theme_matches, list):
            return []
        normalized: list[dict] = []
        for match in theme_matches:
            if not isinstance(match, dict):
                continue
            room_ids = match.get("room_ids")
            canonical_theme = match.get("canonical_theme")
            confidence = match.get("confidence")
            reason = match.get("reason")
            normalized.append(
                {
                    "room_ids": [value for value in room_ids if isinstance(value, str) and value]
                    if isinstance(room_ids, list)
                    else [],
                    "canonical_theme": canonical_theme if isinstance(canonical_theme, str) else "",
                    "confidence": max(0.0, min(1.0, float(confidence)))
                    if isinstance(confidence, (int, float))
                    else 0.0,
                    "reason": reason if isinstance(reason, str) else "",
                }
            )
        return normalized

    @staticmethod
    def _normalize_visual_localization(visual_localization: dict, *, candidate_room_ids: list[str]) -> dict:
        candidate_set = set(candidate_room_ids)
        raw_scores_by_room = {room_id: 0.0 for room_id in candidate_room_ids}
        room_scores = []
        raw_scores = visual_localization.get("room_scores")
        if isinstance(raw_scores, list):
            for record in raw_scores:
                if not isinstance(record, dict):
                    continue
                room_id = record.get("room_id")
                score = record.get("score")
                if room_id in candidate_set and isinstance(score, (int, float)):
                    evidence_score = max(0.0, min(10.0, float(score)))
                    raw_scores_by_room[room_id] = evidence_score
                    normalized_record = {"room_id": room_id, "score": evidence_score}
                    evidence_type = record.get("evidence_type")
                    if isinstance(evidence_type, str) and evidence_type:
                        normalized_record["evidence_type"] = evidence_type
                    reason = record.get("reason")
                    if isinstance(reason, str):
                        normalized_record["reason"] = reason
                    room_scores.append(normalized_record)
        room_distribution_by_room = (
            evidence_scores_to_distribution(raw_scores_by_room)
            if room_scores
            else {room_id: 0.0 for room_id in candidate_room_ids}
        )
        raw_distribution = visual_localization.get("room_distribution")
        if not room_scores and isinstance(raw_distribution, list):
            probability_scores = {room_id: 0.0 for room_id in candidate_room_ids}
            for record in raw_distribution:
                if not isinstance(record, dict):
                    continue
                room_id = record.get("room_id")
                score = record.get("score")
                if room_id in candidate_set and isinstance(score, (int, float)):
                    probability_scores[room_id] = max(0.0, float(score))
            normalized_probability_scores = normalize_positive_scores(probability_scores)
            if any(value > 0.0 for value in normalized_probability_scores.values()):
                room_distribution_by_room = normalized_probability_scores
        room_distribution = [
            {"room_id": room_id, "score": float(room_distribution_by_room.get(room_id, 0.0))}
            for room_id in candidate_room_ids
        ]
        evidence_entities = visual_localization.get("evidence_entities")
        if not isinstance(evidence_entities, list):
            evidence_entities = []
        predicted_room_id = max(room_distribution_by_room, key=room_distribution_by_room.get) if room_distribution_by_room else None
        if predicted_room_id and room_distribution_by_room.get(predicted_room_id, 0.0) <= 0.0:
            predicted_room_id = None
        summary = visual_localization.get("summary")
        return {
            "predicted_room_id": predicted_room_id,
            "confidence": room_distribution_by_room.get(predicted_room_id, 0.0) if predicted_room_id else 0.0,
            "room_scores": room_scores,
            "room_distribution": room_distribution,
            "evidence_entities": [value for value in evidence_entities if isinstance(value, str) and value],
            "summary": summary if isinstance(summary, str) else "",
        }

    def _create_response(self, request_body: dict) -> dict:
        try:
            return self.model_client.create(request_body)
        except TimeoutError as exc:
            raise TimeoutError(
                "Model API timed out after "
                f"{self.request_timeout:.0f}s while processing the multi-view panorama request. "
                "Try a larger request timeout."
            ) from exc

    @staticmethod
    def _parse_output_payload(payload: dict) -> dict:
        try:
            return parse_json_output(payload)
        except ValueError:
            return {"entities": []}

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
        detection_metadata = self._observation_detection_metadata(view_detections or [], entities)
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
                **detection_metadata,
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
                        location_scope=MultiViewAggregator._normalized_location_scope(entity.location_scope),
                        metadata=metadata,
                    )
                    continue
                existing.confidence = max(existing.confidence, entity.confidence)
                existing.location_scope = MultiViewAggregator._merge_location_scopes(
                    existing.location_scope,
                    entity.location_scope,
                )
                existing.metadata["location_scope"] = existing.location_scope
                existing_source_views = existing.metadata.setdefault("source_views", [])
                for source_view in source_views:
                    if source_view not in existing_source_views:
                        existing_source_views.append(source_view)
                existing.metadata["view_count"] = len(existing_source_views)
                existing.source_view = existing_source_views[0] if len(existing_source_views) == 1 else "multiview"
        return sorted(grouped.values(), key=lambda item: (-item.confidence, item.name.lower()))

    @staticmethod
    def _observation_detection_metadata(view_detections: list[ViewDetection], entities: list[EntityDetection]) -> dict:
        metadata: dict = {}
        for view_detection in view_detections:
            if not isinstance(view_detection.metadata, dict):
                continue
            if "visual_localization" not in metadata and isinstance(view_detection.metadata.get("visual_localization"), dict):
                metadata["visual_localization"] = dict(view_detection.metadata["visual_localization"])
            if "candidate_room_ids" not in metadata and isinstance(view_detection.metadata.get("candidate_room_ids"), list):
                metadata["candidate_room_ids"] = list(view_detection.metadata["candidate_room_ids"])
            if "view_theme_observations" not in metadata and isinstance(view_detection.metadata.get("view_theme_observations"), list):
                metadata["view_theme_observations"] = list(view_detection.metadata["view_theme_observations"])
        metadata["inside_entities"] = [
            MultiViewAggregator._entity_to_dict(entity)
            for entity in entities
            if MultiViewAggregator._normalized_location_scope(entity.location_scope) == "inside"
        ]
        metadata["outside_entities"] = [
            MultiViewAggregator._entity_to_dict(entity)
            for entity in entities
            if MultiViewAggregator._normalized_location_scope(entity.location_scope) == "outside"
        ]
        return metadata

    @staticmethod
    def _entity_to_dict(entity: EntityDetection) -> dict:
        return {
            "name": entity.name,
            "kind": entity.kind,
            "confidence": float(entity.confidence),
            "source_view": entity.source_view,
            "source_views": list(entity.metadata.get("source_views", [])) or [entity.source_view],
            "location_scope": MultiViewAggregator._normalized_location_scope(entity.location_scope),
        }

    @staticmethod
    def _normalized_location_scope(scope: str) -> str:
        if scope in ENTITY_LOCATION_SCOPES:
            return scope
        return "inside"

    @staticmethod
    def _merge_location_scopes(existing_scope: str, next_scope: str) -> str:
        scopes = {
            MultiViewAggregator._normalized_location_scope(existing_scope),
            MultiViewAggregator._normalized_location_scope(next_scope),
        }
        if "inside" in scopes:
            return "inside"
        if "unknown" in scopes:
            return "unknown"
        return "outside"

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
        room_graph: dict[str, dict] | None = None,
        grounding_index=None,
        renderer: PanoramaRenderer | None = None,
        detector: ViewDetector | None = None,
        aggregator: MultiViewAggregator | None = None,
    ):
        self.pano_graph = pano_graph
        self.renderer = renderer or PanoramaRenderer(pano_graph)
        self.detector = detector or ViewDetector(room_graph=room_graph, grounding_index=grounding_index)
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
