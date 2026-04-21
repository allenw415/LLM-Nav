from __future__ import annotations

import json
import mimetypes
import os
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from base64 import b64encode
from collections import deque
from pathlib import Path
from typing import Callable

from st_nav.common.env import resolve_model_environment
from st_nav.common.model_client import DEFAULT_OPENAI_API_BASE, ModelResponseClient, parse_json_output, resolve_api_kind

CARDINAL_CAPTURE_LABELS = ("north", "east", "south", "west")


def invert_room_grounding(room_grounding: dict[str, dict]) -> dict[str, list[str]]:
    pano_to_rooms: dict[str, list[str]] = {}
    for room_id, entry in room_grounding.items():
        if not isinstance(entry, dict):
            continue
        pano_ids = entry.get("pano_ids", [])
        if not isinstance(pano_ids, list):
            continue
        for pano_id in pano_ids:
            if not isinstance(pano_id, str) or not pano_id:
                continue
            room_ids = pano_to_rooms.setdefault(pano_id, [])
            if room_id not in room_ids:
                room_ids.append(room_id)
    return {pano_id: sorted(room_ids) for pano_id, room_ids in pano_to_rooms.items()}


def build_room_candidates(
    room_graph: dict[str, dict],
    room_grounding: dict[str, dict],
    *,
    floor: str | None = None,
    same_floor_only: bool = True,
) -> list[dict]:
    candidates: list[dict] = []
    normalized_floor = str(floor) if floor is not None else None
    for room_id in sorted(room_graph.keys()):
        node = room_graph.get(room_id)
        if not isinstance(node, dict):
            continue
        room_floor = node.get("floor")
        room_floor_text = str(room_floor) if room_floor is not None else None
        if same_floor_only and normalized_floor is not None and room_floor_text != normalized_floor:
            continue

        grounding_entry = room_grounding.get(room_id)
        aliases: list[str] = []
        anchor_entities: list[str] = []
        if isinstance(node.get("aliases"), list):
            aliases.extend(str(value) for value in node["aliases"] if isinstance(value, str) and value)
        if isinstance(grounding_entry, dict):
            if isinstance(grounding_entry.get("aliases"), list):
                aliases.extend(str(value) for value in grounding_entry["aliases"] if isinstance(value, str) and value)
            if isinstance(grounding_entry.get("anchor_entities"), list):
                anchor_entities.extend(
                    str(value) for value in grounding_entry["anchor_entities"] if isinstance(value, str) and value
                )

        deduped_aliases = _dedupe_preserve_order(aliases)
        deduped_anchors = _dedupe_preserve_order(anchor_entities)
        candidates.append(
            {
                "room_id": room_id,
                "floor": room_floor_text,
                "display_name": _string_or_none(node.get("display_name")) or room_id,
                "title": _string_or_none(node.get("title")),
                "category": _string_or_none(node.get("category")),
                "aliases": deduped_aliases,
                "anchor_entities": deduped_anchors,
            }
        )
    return candidates


def build_room_grounding_schema(room_ids: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {
            "predicted_room_id": {
                "type": ["string", "null"],
                "enum": room_ids + [None],
                "description": "The single best-matching room id, or null if the panorama is too ambiguous.",
            },
            "confidence": {
                "type": "number",
                "description": "A score from 0 to 1 reflecting visual confidence for the predicted room.",
            },
            "evidence": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Short visual cues supporting the prediction, grounded in the provided images.",
            },
            "alternative_room_ids": {
                "type": "array",
                "items": {"type": "string", "enum": room_ids},
                "description": "Optional fallback candidates ranked after the top prediction.",
            },
            "summary": {
                "type": "string",
                "description": "One short sentence describing why the room was selected.",
            },
        },
        "required": [
            "predicted_room_id",
            "confidence",
            "evidence",
            "alternative_room_ids",
            "summary",
        ],
        "additionalProperties": False,
    }


def build_room_grounding_prompt(*, manifest: dict, candidates: list[dict]) -> str:
    pano_id = _string_or_none(manifest.get("pano_id")) or "unknown"
    floor = _string_or_none(manifest.get("floor")) or "unknown"
    heading_mode = _string_or_none(manifest.get("heading_mode")) or "unknown"

    candidate_lines = []
    for candidate in candidates:
        parts = [
            f"room_id={candidate['room_id']}",
            f"floor={candidate.get('floor') or 'unknown'}",
        ]
        title = candidate.get("title")
        if isinstance(title, str) and title:
            parts.append(f"title={title}")
        category = candidate.get("category")
        if isinstance(category, str) and category:
            parts.append(f"category={category}")
        aliases = candidate.get("aliases")
        if isinstance(aliases, list) and aliases:
            parts.append("aliases=" + ", ".join(str(value) for value in aliases))
        anchor_entities = candidate.get("anchor_entities")
        if isinstance(anchor_entities, list) and anchor_entities:
            parts.append("anchors=" + ", ".join(str(value) for value in anchor_entities))
        candidate_lines.append("- " + " | ".join(parts))

    lines = [
        "You are doing room grounding for British Museum Street View panoramas.",
        "You will receive multiple overlapping rendered views from the same pano plus a closed list of candidate rooms.",
        "Pick the best matching room only from the candidate list.",
        "Use only visible evidence such as room signage, iconic artifacts, sculptures, reliefs, gallery text, architectural layout, and doorway structure.",
        "Do not use hidden metadata, outside knowledge, or guesses based only on museum-wide themes.",
        "If the room is too ambiguous, return predicted_room_id as null.",
        "Keep evidence short and visual.",
        f"Panorama id: {pano_id}.",
        f"Panorama floor: {floor}.",
        f"Heading mode: {heading_mode}.",
        f"Candidate room count: {len(candidates)}.",
        "Candidate rooms:",
        *candidate_lines,
        "Return JSON only.",
    ]
    return "\n".join(lines)


def collect_seed_panos_for_rooms(
    room_grounding: dict[str, dict],
    room_ids: list[str],
) -> tuple[dict[str, list[str]], list[str]]:
    seed_panos_by_room: dict[str, list[str]] = {}
    missing_rooms: list[str] = []
    for room_id in room_ids:
        entry = room_grounding.get(room_id)
        pano_ids: list[str] = []
        if isinstance(entry, dict) and isinstance(entry.get("pano_ids"), list):
            pano_ids = [str(pano_id) for pano_id in entry["pano_ids"] if isinstance(pano_id, str) and pano_id]
        if pano_ids:
            seed_panos_by_room[room_id] = pano_ids
        else:
            missing_rooms.append(room_id)
    return seed_panos_by_room, missing_rooms


def expand_seed_panos_by_hops(
    pano_graph: dict[str, dict],
    seed_panos_by_room: dict[str, list[str]],
    *,
    max_hops: int = 1,
    floor: str | None = None,
) -> dict[str, dict]:
    normalized_floor = str(floor) if floor is not None else None
    queue: deque[tuple[str, str, int]] = deque()
    pano_records: dict[str, dict] = {}

    for room_id, seed_pano_ids in seed_panos_by_room.items():
        for pano_id in seed_pano_ids:
            node = pano_graph.get(pano_id)
            if not isinstance(node, dict):
                continue
            node_floor = _string_or_none(node.get("floor"))
            if normalized_floor is not None and node_floor != normalized_floor:
                continue
            queue.append((room_id, pano_id, 0))

    while queue:
        seed_room_id, pano_id, hops = queue.popleft()
        node = pano_graph.get(pano_id)
        if not isinstance(node, dict):
            continue
        node_floor = _string_or_none(node.get("floor"))
        if normalized_floor is not None and node_floor != normalized_floor:
            continue

        record = pano_records.get(pano_id)
        if record is None:
            pano_records[pano_id] = {
                "pano_id": pano_id,
                "floor": node_floor,
                "seed_distance_hops": hops,
                "nearest_seed_room_ids": [seed_room_id],
            }
        else:
            best_hops = int(record["seed_distance_hops"])
            if hops < best_hops:
                record["seed_distance_hops"] = hops
                record["nearest_seed_room_ids"] = [seed_room_id]
            elif hops == best_hops and seed_room_id not in record["nearest_seed_room_ids"]:
                record["nearest_seed_room_ids"].append(seed_room_id)
            if hops > best_hops:
                continue

        if hops >= max_hops:
            continue

        for neighbor in node.get("neighbors", []):
            if not isinstance(neighbor, dict):
                continue
            target_pano_id = neighbor.get("target_pano_id")
            if not isinstance(target_pano_id, str) or not target_pano_id:
                continue
            target_node = pano_graph.get(target_pano_id)
            if not isinstance(target_node, dict):
                continue
            target_floor = _string_or_none(target_node.get("floor"))
            if normalized_floor is not None and target_floor != normalized_floor:
                continue
            target_record = pano_records.get(target_pano_id)
            if target_record is not None and int(target_record["seed_distance_hops"]) < hops + 1:
                continue
            queue.append((seed_room_id, target_pano_id, hops + 1))

    for record in pano_records.values():
        record["nearest_seed_room_ids"] = sorted(record["nearest_seed_room_ids"])
    return {pano_id: pano_records[pano_id] for pano_id in sorted(pano_records.keys())}


def expand_seed_panos_by_region_growing(
    pano_graph: dict[str, dict],
    seed_panos_by_room: dict[str, list[str]],
    *,
    classify_pano: Callable[[str], dict],
    max_depth: int = 1,
    floor: str | None = None,
    min_confidence: float = 0.75,
    limit: int | None = None,
) -> dict[str, dict]:
    normalized_floor = str(floor) if floor is not None else None
    queue: deque[tuple[str, str, int]] = deque()
    visited_depth_by_room: dict[tuple[str, str], int] = {}
    pano_records: dict[str, dict] = {}

    for room_id, seed_pano_ids in seed_panos_by_room.items():
        for pano_id in seed_pano_ids:
            node = pano_graph.get(pano_id)
            if not isinstance(node, dict):
                continue
            node_floor = _string_or_none(node.get("floor"))
            if normalized_floor is not None and node_floor != normalized_floor:
                continue
            queue.append((room_id, pano_id, 0))

    while queue:
        frontier_room_id, pano_id, depth = queue.popleft()
        state_key = (frontier_room_id, pano_id)
        previous_depth = visited_depth_by_room.get(state_key)
        if previous_depth is not None and previous_depth <= depth:
            continue
        visited_depth_by_room[state_key] = depth

        node = pano_graph.get(pano_id)
        if not isinstance(node, dict):
            continue
        node_floor = _string_or_none(node.get("floor"))
        if normalized_floor is not None and node_floor != normalized_floor:
            continue

        record = pano_records.get(pano_id)
        if record is None:
            if limit is not None and len(pano_records) >= max(limit, 0):
                continue
            classification = classify_pano(pano_id)
            record = {
                "pano_id": pano_id,
                "floor": node_floor,
                "region_depth": depth,
                "frontier_room_ids": [frontier_room_id],
                "expansion_room_ids": [],
                "classification": classification,
            }
            pano_records[pano_id] = record
        else:
            best_depth = int(record["region_depth"])
            if depth < best_depth:
                record["region_depth"] = depth
                record["frontier_room_ids"] = [frontier_room_id]
            elif depth == best_depth and frontier_room_id not in record["frontier_room_ids"]:
                record["frontier_room_ids"].append(frontier_room_id)

        classification = record["classification"]
        predicted_room_id = classification.get("predicted_room_id")
        confidence = classification.get("confidence")
        can_expand = (
            isinstance(predicted_room_id, str)
            and predicted_room_id == frontier_room_id
            and isinstance(confidence, (int, float))
            and float(confidence) >= min_confidence
        )
        if can_expand and frontier_room_id not in record["expansion_room_ids"]:
            record["expansion_room_ids"].append(frontier_room_id)

        if not can_expand or depth >= max_depth:
            continue

        for neighbor in node.get("neighbors", []):
            if not isinstance(neighbor, dict):
                continue
            target_pano_id = neighbor.get("target_pano_id")
            if not isinstance(target_pano_id, str) or not target_pano_id:
                continue
            target_node = pano_graph.get(target_pano_id)
            if not isinstance(target_node, dict):
                continue
            target_floor = _string_or_none(target_node.get("floor"))
            if normalized_floor is not None and target_floor != normalized_floor:
                continue
            queue.append((frontier_room_id, target_pano_id, depth + 1))

    for record in pano_records.values():
        record["frontier_room_ids"] = sorted(record["frontier_room_ids"])
        record["expansion_room_ids"] = sorted(record["expansion_room_ids"])
    return {pano_id: pano_records[pano_id] for pano_id in sorted(pano_records.keys())}


class GeminiRoomGrounder:
    """
    Ground a rendered multi-view pano to a candidate room using the Gemini API.

    Results are cached next to the manifest so repeated evaluation runs can reuse
    previous outputs.
    """

    def __init__(
        self,
        *,
        model: str = "gemini-2.5-flash",
        api_key: str | None = None,
        api_base: str = "https://generativelanguage.googleapis.com/v1beta",
        request_timeout: float = 180.0,
        response_client: Callable[[dict], dict] | None = None,
        use_grounding_files: bool = True,
        same_floor_only: bool = True,
        max_captures: int = 4,
        max_retries: int = 5,
        retry_backoff_seconds: float = 10.0,
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self.api_base = api_base.rstrip("/")
        self.request_timeout = request_timeout
        self.response_client = response_client
        self.use_grounding_files = use_grounding_files
        self.same_floor_only = same_floor_only
        self.max_captures = max_captures
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self.last_traces: list[dict] = []

    def ground(
        self,
        manifest_path: str | Path,
        *,
        room_graph: dict[str, dict],
        room_grounding: dict[str, dict],
    ) -> dict:
        manifest_path = Path(manifest_path)
        self.last_traces = []
        output_path = manifest_path.with_name(f"{manifest_path.stem}_room_grounding.json")
        trace_path = manifest_path.with_name(f"{manifest_path.stem}_room_grounding_trace.json")
        if self.use_grounding_files and output_path.exists():
            result = json.loads(output_path.read_text(encoding="utf-8"))
            if trace_path.exists():
                self.last_traces = self._load_trace_file(trace_path)
            return result

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        candidates = build_room_candidates(
            room_graph,
            room_grounding,
            floor=_string_or_none(manifest.get("floor")),
            same_floor_only=self.same_floor_only,
        )
        if not candidates:
            raise RuntimeError("No candidate rooms available for room grounding.")

        request_body = self._build_request_body(manifest, candidates)
        payload = self._create_response(request_body)
        usage = extract_gemini_usage_metadata(payload)
        self.last_traces.append(
            {
                "request": self._redact_request_body(request_body),
                "response": self._clone_json(payload),
                "usage": usage,
            }
        )
        parsed = self._parse_output_payload(payload)
        result = self._normalize_result(
            parsed,
            manifest=manifest,
            manifest_path=manifest_path,
            candidates=candidates,
        )
        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        trace_path.write_text(
            json.dumps({"requests_and_responses": self.last_traces}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return result

    def _build_request_body(self, manifest: dict, candidates: list[dict]) -> dict:
        captures = [
            capture
            for capture in manifest.get("captures", [])
            if isinstance(capture, dict) and isinstance(capture.get("path"), str) and capture.get("path")
        ]
        if not captures:
            raise RuntimeError("Manifest does not contain any render captures.")
        selected_captures = select_grounding_captures(captures, max_captures=self.max_captures)

        parts: list[dict] = [{"text": build_room_grounding_prompt(manifest=manifest, candidates=candidates)}]
        for capture in selected_captures:
            label = _string_or_none(capture.get("label")) or "unknown"
            heading = capture.get("heading")
            heading_text = f"{float(heading):.1f} deg" if isinstance(heading, (int, float)) else "unknown"
            image_path = Path(str(capture["path"]))
            mime_type, _ = mimetypes.guess_type(str(image_path))
            if not mime_type:
                mime_type = "image/png"
            parts.append({"text": f"View label: {label}. Heading: {heading_text}."})
            parts.append(
                {
                    "inline_data": {
                        "mime_type": mime_type,
                        "data": b64encode(image_path.read_bytes()).decode("ascii"),
                    }
                }
            )

        candidate_room_ids = [str(candidate["room_id"]) for candidate in candidates]
        return {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseJsonSchema": build_room_grounding_schema(candidate_room_ids),
            },
        }

    def _create_response(self, request_body: dict) -> dict:
        if self.response_client is not None:
            return self.response_client(request_body)
        if not self.api_key:
            raise RuntimeError("Missing GEMINI_API_KEY or GOOGLE_API_KEY.")

        encoded_model = urllib.parse.quote(self.model, safe="")
        request = urllib.request.Request(
            f"{self.api_base}/models/{encoded_model}:generateContent",
            data=json.dumps(request_body).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": self.api_key,
            },
            method="POST",
        )
        attempt = 0
        while True:
            try:
                with urllib.request.urlopen(request, timeout=self.request_timeout) as response:
                    return json.loads(response.read().decode("utf-8"))
            except TimeoutError as exc:
                raise TimeoutError(
                    f"Gemini API timed out after {self.request_timeout:.0f}s while grounding the panorama."
                ) from exc
            except socket.timeout as exc:
                raise TimeoutError(
                    f"Gemini API timed out after {self.request_timeout:.0f}s while grounding the panorama."
                ) from exc
            except urllib.error.HTTPError as exc:
                if exc.code != 429 or attempt >= self.max_retries:
                    raise
                retry_after = exc.headers.get("Retry-After") if exc.headers is not None else None
                sleep_seconds = _retry_delay_seconds(
                    attempt=attempt,
                    retry_after=retry_after,
                    base_seconds=self.retry_backoff_seconds,
                )
                time.sleep(sleep_seconds)
                attempt += 1

    @staticmethod
    def _parse_output_payload(payload: dict) -> dict:
        fragments: list[str] = []
        for candidate in payload.get("candidates", []):
            if not isinstance(candidate, dict):
                continue
            content = candidate.get("content")
            if not isinstance(content, dict):
                continue
            for part in content.get("parts", []):
                if not isinstance(part, dict):
                    continue
                text = part.get("text")
                if isinstance(text, str):
                    fragments.append(text)

        output_text = "".join(fragments).strip()
        if not output_text:
            raise RuntimeError(f"Gemini response did not include text output: {json.dumps(payload, ensure_ascii=False)}")
        return json.loads(output_text)

    @staticmethod
    def _normalize_result(parsed: dict, *, manifest: dict, manifest_path: Path, candidates: list[dict]) -> dict:
        candidate_room_ids = [str(candidate["room_id"]) for candidate in candidates]
        candidate_room_id_set = set(candidate_room_ids)

        predicted_room_id = parsed.get("predicted_room_id")
        if not isinstance(predicted_room_id, str) or predicted_room_id not in candidate_room_id_set:
            predicted_room_id = None

        confidence = parsed.get("confidence")
        if not isinstance(confidence, (int, float)):
            confidence = 0.0

        evidence = []
        raw_evidence = parsed.get("evidence")
        if isinstance(raw_evidence, list):
            for value in raw_evidence:
                if isinstance(value, str) and value:
                    evidence.append(value)

        alternative_room_ids: list[str] = []
        raw_alternatives = parsed.get("alternative_room_ids")
        if isinstance(raw_alternatives, list):
            for room_id in raw_alternatives:
                if (
                    isinstance(room_id, str)
                    and room_id in candidate_room_id_set
                    and room_id != predicted_room_id
                    and room_id not in alternative_room_ids
                ):
                    alternative_room_ids.append(room_id)

        summary = parsed.get("summary")
        if not isinstance(summary, str):
            summary = ""

        return {
            "pano_id": _string_or_none(manifest.get("pano_id")),
            "floor": _string_or_none(manifest.get("floor")),
            "manifest_path": str(manifest_path),
            "candidate_room_ids": candidate_room_ids,
            "predicted_room_id": predicted_room_id,
            "confidence": float(confidence),
            "evidence": evidence,
            "alternative_room_ids": alternative_room_ids,
            "summary": summary,
        }

    @staticmethod
    def _clone_json(payload: dict) -> dict:
        return json.loads(json.dumps(payload))

    @classmethod
    def _redact_request_body(cls, request_body: dict) -> dict:
        cloned = cls._clone_json(request_body)
        for content in cloned.get("contents", []):
            if not isinstance(content, dict):
                continue
            for part in content.get("parts", []):
                if not isinstance(part, dict):
                    continue
                inline_data = part.get("inline_data")
                if isinstance(inline_data, dict) and "data" in inline_data:
                    inline_data["data"] = "<IMAGE_BYTES_OMITTED>"
        return cloned

    @staticmethod
    def _load_trace_file(trace_path: Path) -> list[dict]:
        try:
            payload = json.loads(trace_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        traces = payload.get("requests_and_responses")
        if not isinstance(traces, list):
            return []
        return [trace for trace in traces if isinstance(trace, dict)]


class ModelRoomGrounder:
    """
    Ground a rendered multi-view pano to a candidate room using the shared model
    client so the same workflow can run on Gemini, OpenAI-compatible servers,
    or Ollama-backed Gemma models.
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        provider: str | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
        api_kind: str | None = None,
        request_timeout: float | None = None,
        profile: str | None = None,
        response_client: Callable[[dict], dict] | None = None,
        use_grounding_files: bool = True,
        same_floor_only: bool = True,
        max_captures: int = 4,
        max_retries: int = 5,
        retry_backoff_seconds: float = 10.0,
    ):
        settings = resolve_model_environment(
            default_model="gpt-5-mini",
            default_api_base=DEFAULT_OPENAI_API_BASE,
            default_api_kind="responses",
            profile=profile,
        )
        self.provider = (provider or settings.provider or "").strip().lower() or None
        self.model = model or settings.model_name or "gpt-5-mini"
        self.api_key = api_key or settings.api_key
        self.api_base = (api_base or settings.api_base or DEFAULT_OPENAI_API_BASE).rstrip("/")
        self.api_kind = resolve_api_kind(api_kind or settings.api_kind)
        self.request_timeout = float(request_timeout if request_timeout is not None else (settings.request_timeout or 180.0))
        self.profile = profile or settings.active_profile
        self.response_client = response_client
        self.use_grounding_files = use_grounding_files
        self.same_floor_only = same_floor_only
        self.max_captures = max_captures
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self.last_traces: list[dict] = []
        self.model_client = ModelResponseClient(
            provider=self.provider,
            api_key=self.api_key,
            api_base=self.api_base,
            api_kind=self.api_kind,
            request_timeout=self.request_timeout,
            num_ctx=settings.num_ctx,
            temperature=settings.temperature,
            response_client=self.response_client,
        )

    def ground(
        self,
        manifest_path: str | Path,
        *,
        room_graph: dict[str, dict],
        room_grounding: dict[str, dict],
    ) -> dict:
        manifest_path = Path(manifest_path)
        self.last_traces = []
        output_path = manifest_path.with_name(f"{manifest_path.stem}_room_grounding.json")
        trace_path = manifest_path.with_name(f"{manifest_path.stem}_room_grounding_trace.json")
        if self.use_grounding_files and output_path.exists():
            result = json.loads(output_path.read_text(encoding="utf-8"))
            if trace_path.exists():
                self.last_traces = GeminiRoomGrounder._load_trace_file(trace_path)
            return result

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        candidates = build_room_candidates(
            room_graph,
            room_grounding,
            floor=_string_or_none(manifest.get("floor")),
            same_floor_only=self.same_floor_only,
        )
        if not candidates:
            raise RuntimeError("No candidate rooms available for room grounding.")
        if not self.model_client.is_configured():
            raise RuntimeError("Missing model API configuration for room grounding.")

        request_body = self._build_request_body(manifest, candidates)
        payload = self._create_response(request_body)
        usage = extract_model_usage_metadata(payload)
        self.last_traces.append(
            {
                "request": self._redact_request_body(request_body),
                "response": self._clone_json(payload),
                "usage": usage,
            }
        )
        parsed = parse_json_output(payload)
        result = GeminiRoomGrounder._normalize_result(
            parsed,
            manifest=manifest,
            manifest_path=manifest_path,
            candidates=candidates,
        )
        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        trace_path.write_text(
            json.dumps({"requests_and_responses": self.last_traces}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return result

    def _create_response(self, request_body: dict) -> dict:
        attempt = 0
        while True:
            try:
                return self.model_client.create(request_body)
            except RuntimeError as exc:
                if not _is_transient_model_runtime_error(exc) or attempt >= self.max_retries:
                    raise
                sleep_seconds = _retry_delay_seconds(
                    attempt=attempt,
                    retry_after=None,
                    base_seconds=self.retry_backoff_seconds,
                )
                time.sleep(sleep_seconds)
                attempt += 1

    def _build_request_body(self, manifest: dict, candidates: list[dict]) -> dict:
        captures = [
            capture
            for capture in manifest.get("captures", [])
            if isinstance(capture, dict) and isinstance(capture.get("path"), str) and capture.get("path")
        ]
        if not captures:
            raise RuntimeError("Manifest does not contain any render captures.")
        selected_captures = select_grounding_captures(captures, max_captures=self.max_captures)
        candidate_room_ids = [str(candidate["room_id"]) for candidate in candidates]

        content: list[dict] = [
            {
                "type": "input_text",
                "text": build_room_grounding_prompt(manifest=manifest, candidates=candidates),
            }
        ]
        for capture in selected_captures:
            label = _string_or_none(capture.get("label")) or "unknown"
            heading = capture.get("heading")
            heading_text = f"{float(heading):.1f} deg" if isinstance(heading, (int, float)) else "unknown"
            content.append({"type": "input_text", "text": f"View label: {label}. Heading: {heading_text}."})
            content.append(
                {
                    "type": "input_image",
                    "image_url": _image_to_data_url(Path(str(capture["path"]))),
                    "detail": "high",
                }
            )

        return {
            "model": self.model,
            "input": [{"role": "user", "content": content}],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "room_grounding",
                    "strict": True,
                    "schema": build_room_grounding_schema(candidate_room_ids),
                }
            },
        }

    @staticmethod
    def _clone_json(payload: dict) -> dict:
        return json.loads(json.dumps(payload))

    @classmethod
    def _redact_request_body(cls, request_body: dict) -> dict:
        cloned = cls._clone_json(request_body)
        for item in cloned.get("input", []):
            if not isinstance(item, dict):
                continue
            for block in item.get("content", []):
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "input_image" and "image_url" in block:
                    block["image_url"] = "<IMAGE_BYTES_OMITTED>"
        return cloned


def _string_or_none(value: object) -> str | None:
    if isinstance(value, str) and value:
        return value
    return None


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _retry_delay_seconds(*, attempt: int, retry_after: str | None, base_seconds: float) -> float:
    if isinstance(retry_after, str):
        try:
            parsed = float(retry_after.strip())
        except ValueError:
            parsed = None
        if parsed is not None and parsed > 0:
            return parsed
    return max(base_seconds * (2**attempt), 1.0)


def _is_transient_model_runtime_error(exc: RuntimeError) -> bool:
    message = str(exc).upper()
    transient_markers = (
        "HTTP 429",
        "HTTP 500",
        "HTTP 502",
        "HTTP 503",
        "HTTP 504",
        "UNAVAILABLE",
        "SERVICE UNAVAILABLE",
        "RATE LIMIT",
    )
    return any(marker in message for marker in transient_markers)


def _image_to_data_url(image_path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(str(image_path))
    if not mime_type:
        mime_type = "image/png"
    return f"data:{mime_type};base64,{b64encode(image_path.read_bytes()).decode('ascii')}"


def select_grounding_captures(captures: list[dict], *, max_captures: int = 4) -> list[dict]:
    if max_captures <= 0 or len(captures) <= max_captures:
        return list(captures)

    capture_by_label: dict[str, dict] = {}
    for capture in captures:
        label = capture.get("label")
        if isinstance(label, str) and label and label not in capture_by_label:
            capture_by_label[label] = capture

    preferred = [capture_by_label[label] for label in CARDINAL_CAPTURE_LABELS if label in capture_by_label]
    if len(preferred) >= max_captures:
        return preferred[:max_captures]

    selected = list(preferred)
    for capture in captures:
        if capture in selected:
            continue
        selected.append(capture)
        if len(selected) >= max_captures:
            break
    return selected


def merge_records_by_pano_id(existing_records: list[dict], new_records: list[dict]) -> list[dict]:
    merged: dict[str, dict] = {}

    for record in existing_records:
        pano_id = record.get("pano_id") if isinstance(record, dict) else None
        if not isinstance(pano_id, str) or not pano_id:
            continue
        merged[pano_id] = dict(record)

    for record in new_records:
        pano_id = record.get("pano_id") if isinstance(record, dict) else None
        if not isinstance(pano_id, str) or not pano_id:
            continue
        merged[pano_id] = {**merged.get(pano_id, {}), **record}

    return [merged[pano_id] for pano_id in sorted(merged.keys())]


def build_manual_annotation_records(
    grounding_records: list[dict],
    *,
    existing_manual_records: list[dict] | None = None,
    min_confidence: float = 0.75,
    prefill_manual_room_id_from_prediction: bool = False,
) -> list[dict]:
    manual_by_pano: dict[str, dict] = {}
    for record in existing_manual_records or []:
        pano_id = record.get("pano_id") if isinstance(record, dict) else None
        if not isinstance(pano_id, str) or not pano_id:
            continue
        manual_by_pano[pano_id] = dict(record)

    annotation_records: list[dict] = []
    seen_pano_ids: set[str] = set()
    for grounding_record in grounding_records:
        pano_id = grounding_record.get("pano_id") if isinstance(grounding_record, dict) else None
        if not isinstance(pano_id, str) or not pano_id:
            continue
        seen_pano_ids.add(pano_id)
        existing = manual_by_pano.get(pano_id, {})
        predicted_room_id = grounding_record.get("predicted_room_id")
        default_manual_room_id = (
            predicted_room_id
            if prefill_manual_room_id_from_prediction and isinstance(predicted_room_id, str) and predicted_room_id
            else None
        )
        needs_review = _record_requires_review(grounding_record, min_confidence=min_confidence)
        annotation_records.append(
            {
                "pano_id": pano_id,
                "manifest_path": grounding_record.get("manifest_path"),
                "region_depth": grounding_record.get("region_depth"),
                "frontier_room_ids": list(grounding_record.get("frontier_room_ids", [])),
                "expansion_room_ids": list(grounding_record.get("expansion_room_ids", [])),
                "gemini_predicted_room_id": predicted_room_id,
                "gemini_confidence": grounding_record.get("confidence"),
                "gemini_alternative_room_ids": list(grounding_record.get("alternative_room_ids", [])),
                "gemini_summary": grounding_record.get("summary"),
                "needs_review": needs_review,
                "manual_status": existing.get("manual_status", "pending" if needs_review else "accepted"),
                "manual_room_id": existing.get("manual_room_id", default_manual_room_id),
                "notes": existing.get("notes", ""),
            }
        )

    for pano_id in sorted(manual_by_pano.keys()):
        if pano_id in seen_pano_ids:
            continue
        annotation_records.append(manual_by_pano[pano_id])

    return annotation_records


def _record_requires_review(record: dict, *, min_confidence: float) -> bool:
    predicted_room_id = record.get("predicted_room_id")
    confidence = record.get("confidence")
    if not isinstance(predicted_room_id, str) or not predicted_room_id:
        return True
    if not isinstance(confidence, (int, float)) or float(confidence) < min_confidence:
        return True
    alternatives = record.get("alternative_room_ids")
    return isinstance(alternatives, list) and len(alternatives) > 0 and float(confidence) < 0.95


def collect_manual_seed_panos(
    manual_records: list[dict],
    *,
    room_ids: list[str] | None = None,
    accepted_statuses: set[str] | None = None,
) -> dict[str, list[str]]:
    allowed_room_ids = set(room_ids or [])
    accepted_statuses = accepted_statuses or {"accepted"}
    seed_panos_by_room: dict[str, list[str]] = {}

    for record in manual_records:
        if not isinstance(record, dict):
            continue
        status = record.get("manual_status")
        room_id = record.get("manual_room_id")
        pano_id = record.get("pano_id")
        if not isinstance(status, str) or status not in accepted_statuses:
            continue
        if not isinstance(room_id, str) or not room_id:
            continue
        if allowed_room_ids and room_id not in allowed_room_ids:
            continue
        if not isinstance(pano_id, str) or not pano_id:
            continue
        room_seed_panos = seed_panos_by_room.setdefault(room_id, [])
        if pano_id not in room_seed_panos:
            room_seed_panos.append(pano_id)

    return {room_id: sorted(pano_ids) for room_id, pano_ids in seed_panos_by_room.items()}


def merge_seed_panos_by_room(*seed_maps: dict[str, list[str]]) -> dict[str, list[str]]:
    merged: dict[str, list[str]] = {}
    for seed_map in seed_maps:
        for room_id, pano_ids in seed_map.items():
            room_seed_panos = merged.setdefault(room_id, [])
            for pano_id in pano_ids:
                if pano_id not in room_seed_panos:
                    room_seed_panos.append(pano_id)
    return {room_id: sorted(pano_ids) for room_id, pano_ids in merged.items()}


def extract_gemini_usage_metadata(payload: dict) -> dict | None:
    usage = payload.get("usageMetadata")
    if not isinstance(usage, dict):
        return None

    mapped = {
        "prompt_token_count": usage.get("promptTokenCount"),
        "candidates_token_count": usage.get("candidatesTokenCount"),
        "total_token_count": usage.get("totalTokenCount"),
        "thoughts_token_count": usage.get("thoughtsTokenCount"),
        "cached_content_token_count": usage.get("cachedContentTokenCount"),
    }
    normalized = {
        key: int(value)
        for key, value in mapped.items()
        if isinstance(value, int) or (isinstance(value, float) and float(value).is_integer())
    }
    return normalized or None


def extract_model_usage_metadata(payload: dict) -> dict | None:
    gemini_usage = extract_gemini_usage_metadata(payload)
    if gemini_usage is not None:
        return gemini_usage

    usage = payload.get("usage")
    if isinstance(usage, dict):
        mapped = {
            "prompt_token_count": usage.get("prompt_tokens", usage.get("input_tokens")),
            "candidates_token_count": usage.get("completion_tokens", usage.get("output_tokens")),
            "total_token_count": usage.get("total_tokens"),
            "thoughts_token_count": usage.get("reasoning_tokens"),
            "cached_content_token_count": usage.get("cached_tokens"),
        }
        normalized = {
            key: int(value)
            for key, value in mapped.items()
            if isinstance(value, int) or (isinstance(value, float) and float(value).is_integer())
        }
        if normalized:
            return normalized

    mapped = {
        "prompt_token_count": payload.get("prompt_eval_count"),
        "candidates_token_count": payload.get("eval_count"),
    }
    normalized = {
        key: int(value)
        for key, value in mapped.items()
        if isinstance(value, int) or (isinstance(value, float) and float(value).is_integer())
    }
    if normalized:
        normalized["total_token_count"] = (
            normalized.get("prompt_token_count", 0) + normalized.get("candidates_token_count", 0)
        )
        return normalized
    return None


def aggregate_gemini_usage_from_traces(traces: list[dict]) -> dict:
    totals = {
        "request_count": 0,
        "prompt_token_count": 0,
        "candidates_token_count": 0,
        "total_token_count": 0,
        "thoughts_token_count": 0,
        "cached_content_token_count": 0,
    }
    found_usage = False

    for trace in traces:
        if not isinstance(trace, dict):
            continue
        usage = trace.get("usage")
        if not isinstance(usage, dict):
            response = trace.get("response")
            if isinstance(response, dict):
                usage = extract_gemini_usage_metadata(response)
        if not isinstance(usage, dict):
            continue
        found_usage = True
        totals["request_count"] += 1
        for key in (
            "prompt_token_count",
            "candidates_token_count",
            "total_token_count",
            "thoughts_token_count",
            "cached_content_token_count",
        ):
            value = usage.get(key)
            if isinstance(value, int):
                totals[key] += value

    return totals if found_usage else {}


def aggregate_model_usage_from_traces(traces: list[dict]) -> dict:
    totals = {
        "request_count": 0,
        "prompt_token_count": 0,
        "candidates_token_count": 0,
        "total_token_count": 0,
        "thoughts_token_count": 0,
        "cached_content_token_count": 0,
    }
    found_usage = False

    for trace in traces:
        if not isinstance(trace, dict):
            continue
        usage = trace.get("usage")
        if not isinstance(usage, dict):
            response = trace.get("response")
            if isinstance(response, dict):
                usage = extract_model_usage_metadata(response)
        if not isinstance(usage, dict):
            continue
        found_usage = True
        totals["request_count"] += 1
        for key in (
            "prompt_token_count",
            "candidates_token_count",
            "total_token_count",
            "thoughts_token_count",
            "cached_content_token_count",
        ):
            value = usage.get(key)
            if isinstance(value, int):
                totals[key] += value

    return totals if found_usage else {}


def build_compact_pano_room_mapping(
    grounding_records: list[dict],
    *,
    manual_records: list[dict] | None = None,
) -> dict:
    manual_by_pano: dict[str, dict] = {}
    for record in manual_records or []:
        pano_id = record.get("pano_id") if isinstance(record, dict) else None
        if not isinstance(pano_id, str) or not pano_id:
            continue
        manual_by_pano[pano_id] = record

    mappings: dict[str, str] = {}
    sources: dict[str, str] = {}
    for record in grounding_records:
        pano_id = record.get("pano_id") if isinstance(record, dict) else None
        if not isinstance(pano_id, str) or not pano_id:
            continue

        manual_record = manual_by_pano.get(pano_id, {})
        manual_status = manual_record.get("manual_status")
        manual_room_id = manual_record.get("manual_room_id")
        if (
            isinstance(manual_status, str)
            and manual_status in {"accepted", "boundary", "ambiguous"}
            and isinstance(manual_room_id, str)
            and manual_room_id
        ):
            mappings[pano_id] = manual_room_id
            sources[pano_id] = f"manual:{manual_status}"
            continue

        predicted_room_id = record.get("predicted_room_id")
        if isinstance(predicted_room_id, str) and predicted_room_id:
            mappings[pano_id] = predicted_room_id
            sources[pano_id] = "gemini"

    return {
        "mappings": {pano_id: mappings[pano_id] for pano_id in sorted(mappings.keys())},
        "sources": {pano_id: sources[pano_id] for pano_id in sorted(sources.keys())},
    }
