from __future__ import annotations

from copy import deepcopy
from typing import Any


JsonDict = dict[str, Any]


def normalize_raw_link(link: Any) -> JsonDict | None:
    if not isinstance(link, dict):
        return None
    pano_id = link.get("pano") or link.get("panoID") or link.get("pano_id")
    if not isinstance(pano_id, str) or not pano_id:
        return None
    return {
        "pano": pano_id,
        "heading": link.get("heading"),
        "description": link.get("description"),
    }


def normalize_raw_pano_record(pano_id: str, record: JsonDict) -> JsonDict:
    normalized = deepcopy(record)
    normalized["pano"] = normalized.get("pano") or normalized.get("panoID") or normalized.get("pano_id") or pano_id
    normalized["links"] = [
        link
        for link in (normalize_raw_link(link) for link in normalized.get("links", []))
        if link is not None
    ]
    return normalized


def merge_raw_links(existing_links: list[Any], incoming_links: list[Any]) -> tuple[list[JsonDict], int]:
    merged: list[JsonDict] = []
    by_target: dict[str, JsonDict] = {}
    added = 0

    for link in existing_links:
        normalized = normalize_raw_link(link)
        if normalized is None:
            continue
        target = normalized["pano"]
        if target not in by_target:
            by_target[target] = normalized
            merged.append(normalized)

    for link in incoming_links:
        normalized = normalize_raw_link(link)
        if normalized is None:
            continue
        target = normalized["pano"]
        if target not in by_target:
            by_target[target] = normalized
            merged.append(normalized)
            added += 1
            continue

        current = by_target[target]
        if current.get("heading") is None and normalized.get("heading") is not None:
            current["heading"] = normalized["heading"]
        if current.get("description") is None and normalized.get("description") is not None:
            current["description"] = normalized["description"]

    return merged, added


def merge_raw_pano_record(existing: JsonDict, incoming: JsonDict) -> tuple[JsonDict, int]:
    merged = deepcopy(existing)
    for key in ("pano", "lat", "lng", "imageDate", "inside_polygon"):
        if merged.get(key) is None and incoming.get(key) is not None:
            merged[key] = incoming[key]

    existing_distance = merged.get("dist_m_from_seed")
    incoming_distance = incoming.get("dist_m_from_seed")
    if existing_distance is None:
        merged["dist_m_from_seed"] = incoming_distance
    elif incoming_distance is not None:
        try:
            merged["dist_m_from_seed"] = min(float(existing_distance), float(incoming_distance))
        except (TypeError, ValueError):
            pass

    merged_links, added_links = merge_raw_links(merged.get("links", []), incoming.get("links", []))
    merged["links"] = merged_links
    return merged, added_links


def merge_raw_crawl_payloads(base_payload: JsonDict, incoming_payloads: list[JsonDict]) -> tuple[JsonDict, JsonDict]:
    merged = deepcopy(base_payload)
    base_panos = merged.setdefault("panos", {})
    if not isinstance(base_panos, dict):
        raise ValueError("base payload must contain a panos dict")

    stats = {
        "base_panos": len(base_panos),
        "incoming_files": len(incoming_payloads),
        "incoming_panos": 0,
        "added_panos": 0,
        "merged_existing_panos": 0,
        "added_links": 0,
        "skipped_invalid_records": 0,
    }

    seeds = []
    if isinstance(merged.get("seed"), dict):
        seeds.append(deepcopy(merged["seed"]))
    for payload in incoming_payloads:
        if isinstance(payload.get("seed"), dict):
            seeds.append(deepcopy(payload["seed"]))

        incoming_panos = payload.get("panos", {})
        if not isinstance(incoming_panos, dict):
            raise ValueError("incoming payload must contain a panos dict")

        stats["incoming_panos"] += len(incoming_panos)
        for pano_key, record in incoming_panos.items():
            if not isinstance(record, dict):
                stats["skipped_invalid_records"] += 1
                continue

            pano_id = record.get("pano") or record.get("panoID") or record.get("pano_id") or pano_key
            if not isinstance(pano_id, str) or not pano_id:
                stats["skipped_invalid_records"] += 1
                continue

            normalized = normalize_raw_pano_record(pano_id, record)
            if pano_id not in base_panos:
                base_panos[pano_id] = normalized
                stats["added_panos"] += 1
                continue

            base_panos[pano_id], added_links = merge_raw_pano_record(base_panos[pano_id], normalized)
            stats["merged_existing_panos"] += 1
            stats["added_links"] += added_links

    merged["merged_seeds"] = seeds
    merged["merge_summary"] = stats | {"output_panos": len(base_panos)}
    return merged, merged["merge_summary"]
