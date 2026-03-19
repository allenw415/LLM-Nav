import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict


FILENAME_PATTERN = re.compile(r"^streetview_panos_(?P<floor>[a-zA-Z0-9]+)_[0-9]+\.json$")


def floor_token_to_key(token: str) -> str:
    lowered = token.lower()
    if lowered == "05":
        return "0.5"
    if lowered.startswith("b") and lowered[1:].isdigit():
        return str(-int(lowered[1:]))
    return lowered


def normalize_links(links: Any) -> list[Dict[str, Any]]:
    if not isinstance(links, list):
        return []

    normalized = []
    for link in links:
        if not isinstance(link, dict):
            continue
        pano_id = link.get("panoID") or link.get("pano") or link.get("pano_id")
        if not pano_id:
            continue
        normalized.append(
            {
                "panoID": pano_id,
                "heading": link.get("heading"),
                "description": link.get("description"),
            }
        )
    return normalized


def normalize_pano_records(panos: Dict[str, Any]) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {}
    for pano_key, record in panos.items():
        if not isinstance(record, dict):
            continue
        pano_id = record.get("panoID") or record.get("pano") or record.get("pano_id") or pano_key
        normalized[pano_key] = {
            "panoID": pano_id,
            "lat": record.get("lat"),
            "lng": record.get("lng"),
            "imageDate": record.get("imageDate"),
            "links": normalize_links(record.get("links", [])),
            "inside_polygon": record.get("inside_polygon"),
            "dist_m_from_seed": record.get("dist_m_from_seed"),
        }
    return normalized


def infer_floor_key_from_path(path: Path) -> str:
    match = FILENAME_PATTERN.match(path.name)
    if match is None:
        raise ValueError(
            f"無法從檔名推斷樓層: {path.name}。預期格式例如 streetview_panos_0_1.json"
        )
    return floor_token_to_key(match.group("floor"))


def load_crawled_floor_file(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    panos = data.get("panos")
    if not isinstance(panos, dict):
        raise ValueError(f"{path} 缺少 panos dict。")
    return normalize_pano_records(panos)


def build_panos_by_floor(crawled_data_dir: str | Path) -> Dict[str, Dict[str, Any]]:
    crawled_data_dir = Path(crawled_data_dir)
    grouped: Dict[str, Dict[str, Any]] = {}

    for path in sorted(crawled_data_dir.glob("streetview_panos_*.json")):
        floor_key = infer_floor_key_from_path(path)
        floor_nodes = load_crawled_floor_file(path)

        if floor_key in grouped:
            raise ValueError(f"重複的樓層 key: {floor_key}，來源檔案包含 {path.name}")

        grouped[floor_key] = floor_nodes

    if not grouped:
        raise ValueError(f"{crawled_data_dir} 裡找不到 streetview_panos_*.json")

    return grouped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build panos_by_floor.json from crawled_data/*.json files."
    )
    parser.add_argument("--input-dir", default="crawled_data")
    parser.add_argument("--output-path", default="panos_by_floor.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    grouped = build_panos_by_floor(args.input_dir)
    output_path = Path(args.output_path)
    output_path.write_text(
        json.dumps(grouped, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary = {floor: len(nodes) for floor, nodes in grouped.items()}
    print(f"Saved: {output_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
