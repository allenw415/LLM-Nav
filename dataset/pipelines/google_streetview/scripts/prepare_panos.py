import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple


def normalize_heading(heading: Optional[float]) -> Optional[float]:
    if heading is None:
        return None
    try:
        return float(heading) % 360.0
    except Exception:
        return None


def reverse_heading(heading: Optional[float]) -> Optional[float]:
    normalized = normalize_heading(heading)
    if normalized is None:
        return None
    return (normalized + 180.0) % 360.0


def normalize_grouped_panos(data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    將依樓層分組的 pano 資料標準化成：
      {
        "<floor>": {
          "<panoID>": {
            "panoID": "...",
            "lat": ...,
            "lng": ...,
            "links": [{"panoID": "...", "heading": ..., "description": ...}]
          }
        }
      }
    """
    normalized: Dict[str, Dict[str, Any]] = {}

    for floor, pano_dict in data.items():
        if not isinstance(pano_dict, dict):
            continue

        floor_nodes: Dict[str, Any] = {}
        for pano_key, record in pano_dict.items():
            if not isinstance(record, dict):
                continue

            pano_id = record.get("panoID") or record.get("pano") or record.get("pano_id") or pano_key
            floor_nodes[pano_key] = {
                "panoID": pano_id,
                "lat": record.get("lat"),
                "lng": record.get("lng"),
                "links": normalize_links(record.get("links", [])),
            }

        normalized[str(floor)] = floor_nodes

    return normalized


def normalize_links(links: Any) -> list[Dict[str, Any]]:
    if not isinstance(links, list):
        return []

    normalized_links = []
    for link in links:
        if not isinstance(link, dict):
            continue
        pano_id = link.get("panoID") or link.get("pano") or link.get("pano_id")
        if not pano_id:
            continue
        normalized_links.append(
            {
                "panoID": pano_id,
                "heading": link.get("heading"),
                "description": link.get("description"),
            }
        )
    return normalized_links


def flatten_grouped_panos(grouped_panos: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    flat: Dict[str, Any] = {}

    for floor, pano_dict in grouped_panos.items():
        for pano_key, record in pano_dict.items():
            if not isinstance(record, dict):
                continue

            pano_id = record.get("panoID") or pano_key
            flat[pano_key] = {
                "panoID": pano_id,
                "lat": record.get("lat"),
                "lng": record.get("lng"),
                "links": deepcopy(record.get("links", [])),
                "floor": str(floor),
            }

    return flat


def add_missing_reverse_links(graph: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, int]]:
    outgoing: Dict[str, Set[str]] = {}
    for pano_id, record in graph.items():
        outgoing[pano_id] = {
            link.get("panoID")
            for link in record.get("links", [])
            if isinstance(link, dict) and isinstance(link.get("panoID"), str) and link.get("panoID")
        }

    stats = {
        "original_edges": 0,
        "added_reverse_edges": 0,
        "skipped_already_bidirectional": 0,
        "skipped_target_missing": 0,
        "skipped_self_loop": 0,
    }

    for source_id, record in graph.items():
        links = record.get("links", [])
        if not isinstance(links, list):
            continue

        for link in links:
            if not isinstance(link, dict):
                continue
            target_id = link.get("panoID")
            if not isinstance(target_id, str) or not target_id:
                continue

            stats["original_edges"] += 1

            if target_id == source_id:
                stats["skipped_self_loop"] += 1
                continue

            if target_id not in graph:
                stats["skipped_target_missing"] += 1
                continue

            if source_id in outgoing.get(target_id, set()):
                stats["skipped_already_bidirectional"] += 1
                continue

            graph[target_id].setdefault("links", [])
            if not isinstance(graph[target_id]["links"], list):
                graph[target_id]["links"] = []

            graph[target_id]["links"].append(
                {
                    "panoID": source_id,
                    "heading": reverse_heading(link.get("heading")),
                    "description": None,
                }
            )
            outgoing.setdefault(target_id, set()).add(source_id)
            stats["added_reverse_edges"] += 1

    return graph, stats


def build_navigation_graph(grouped_panos: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    normalized = normalize_grouped_panos(grouped_panos)
    flat = flatten_grouped_panos(normalized)
    graph, bidirectional_stats = add_missing_reverse_links(flat)
    summary = {
        "floors": len(normalized),
        "nodes": len(graph),
        "bidirectional_stats": bidirectional_stats,
    }
    return graph, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a final navigation graph from floor-grouped Street View pano data."
    )
    parser.add_argument("--input-path", default="panos_by_floor.json")
    parser.add_argument("--output-path", default="panos.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)

    grouped_panos = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(grouped_panos, dict):
        raise ValueError("輸入 JSON 必須是依樓層分組的 dict。")

    graph, summary = build_navigation_graph(grouped_panos)
    output_path.write_text(
        json.dumps(graph, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Saved: {output_path} (nodes={summary['nodes']})")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
