from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional


LatLng = Tuple[float, float]  # (lat, lng)


def _parse_kml_coordinates(coord_text: str) -> List[LatLng]:
    """
    KML coordinates 格式通常是：
      lng,lat,alt lng,lat,alt ...
    altitude 可能缺省。
    """
    pts: List[LatLng] = []
    # 可能有換行/空白
    for token in coord_text.strip().replace("\n", " ").replace("\t", " ").split():
        parts = token.split(",")
        if len(parts) < 2:
            continue
        lng = float(parts[0])
        lat = float(parts[1])
        pts.append((lat, lng))
    return pts


def _strip_namespace(tag: str) -> str:
    # '{http://www.opengis.net/kml/2.2}Placemark' -> 'Placemark'
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def parse_kml_polygons(kml_path: str) -> List[Dict[str, Any]]:
    """
    回傳：
      [
        {
          "name": "...(Placemark name)...",
          "outer": [(lat,lng), ...],
          "inners": [[(lat,lng), ...], ...]  # 可能為空
        },
        ...
      ]
    """
    xml_text = Path(kml_path).read_text(encoding="utf-8")
    root = ET.fromstring(xml_text)

    # 找 namespace（如果有）
    ns = {}
    if root.tag.startswith("{"):
        uri = root.tag.split("}")[0].strip("{")
        ns = {"kml": uri}

    def findall(el, path):
        return el.findall(path, ns) if ns else el.findall(path)

    def findtext(el, path) -> Optional[str]:
        found = el.find(path, ns) if ns else el.find(path)
        return found.text if (found is not None and found.text) else None

    polygons: List[Dict[str, Any]] = []

    # Placemark 可能包含 Polygon 或 MultiGeometry(Polygon...)
    placemarks = findall(root, ".//kml:Placemark" if ns else ".//Placemark")
    for pm in placemarks:
        name = findtext(pm, "kml:name" if ns else "name") or "unnamed"

        # 1) 直接 Polygon
        poly_nodes = findall(pm, ".//kml:Polygon" if ns else ".//Polygon")

        for poly in poly_nodes:
            outer_text = findtext(
                poly,
                ".//kml:outerBoundaryIs/kml:LinearRing/kml:coordinates"
                if ns else
                ".//outerBoundaryIs/LinearRing/coordinates"
            )
            if not outer_text:
                continue

            outer = _parse_kml_coordinates(outer_text)

            inner_coords = findall(
                poly,
                ".//kml:innerBoundaryIs/kml:LinearRing/kml:coordinates"
                if ns else
                ".//innerBoundaryIs/LinearRing/coordinates"
            )
            inners: List[List[LatLng]] = []
            for c in inner_coords:
                if c is not None and c.text:
                    inners.append(_parse_kml_coordinates(c.text))

            polygons.append(
                {
                    "name": name,
                    "outer": outer,
                    "inners": inners,
                }
            )

    return polygons


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse polygons from a Google Earth KML file.")
    parser.add_argument("--input-path", default="google_earth.kml")
    parser.add_argument("--output-path", default="polygons.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)

    polys = parse_kml_polygons(str(input_path))

    out = {
        "source_kml": str(input_path),
        "polygon_count": len(polys),
        "polygons": polys,
    }
    output_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {output_path} (polygons={len(polys)})")

    if polys:
        print("\nFirst polygon outer points (lat,lng):")
        for lat, lng in polys[0]["outer"]:
            print(f"{lat:.7f}, {lng:.7f}")


if __name__ == "__main__":
    main()
