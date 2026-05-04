from __future__ import annotations

import json
import unittest
import xml.etree.ElementTree as ET

from st_nav_data.pano_visualization import (
    build_room_color_map,
    build_dot,
    build_geojson,
    build_gexf,
    build_graphml,
    build_visualization_payload,
    extract_grounding_mapping,
    shortest_pano_path,
)


class PanoVisualizationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pano_graph = {
            "pano-a": {
                "pano_id": "pano-a",
                "floor": "0",
                "lat": 51.0,
                "lng": -0.1,
                "neighbors": [
                    {"target_pano_id": "pano-b", "geocentric_heading_deg": 90.0},
                    {"target_pano_id": "missing", "geocentric_heading_deg": 270.0},
                ],
            },
            "pano-b": {
                "pano_id": "pano-b",
                "floor": "0",
                "lat": 51.001,
                "lng": -0.099,
                "neighbors": [{"target_pano_id": "pano-c", "geocentric_heading_deg": 180.0}],
            },
            "pano-c": {
                "pano_id": "pano-c",
                "floor": "1",
                "lat": 51.002,
                "lng": -0.098,
                "neighbors": [],
            },
        }
        self.room_graph = {
            "Room 8": {"title": "Assyria: Nimrud", "category": "Middle East"},
        }
        self.grounding = {
            "mappings": {
                "pano-a": "Room 8",
                "pano-b": "null",
            },
            "sources": {
                "pano-a": "manual:accepted",
            },
        }

    def test_extract_grounding_mapping_normalizes_null_values(self) -> None:
        mappings, sources = extract_grounding_mapping(self.grounding)

        self.assertEqual(mappings["pano-a"], "Room 8")
        self.assertIsNone(mappings["pano-b"])
        self.assertEqual(sources["pano-a"], "manual:accepted")

    def test_extract_grounding_mapping_strips_room_id_whitespace(self) -> None:
        mappings, _ = extract_grounding_mapping({"mappings": {"pano-a": " Room 8 "}})

        self.assertEqual(mappings["pano-a"], "Room 8")

    def test_build_visualization_payload_enriches_nodes_and_marks_dangling_edges(self) -> None:
        payload = build_visualization_payload(
            self.pano_graph,
            room_graph=self.room_graph,
            grounding_payload=self.grounding,
        )

        self.assertEqual(payload["summary"]["node_count"], 3)
        self.assertEqual(payload["summary"]["edge_count"], 3)
        self.assertEqual(payload["summary"]["dangling_edge_count"], 1)
        nodes = {node["id"]: node for node in payload["nodes"]}
        self.assertEqual(nodes["pano-a"]["room_id"], "Room 8")
        self.assertEqual(nodes["pano-a"]["room_title"], "Assyria: Nimrud")
        self.assertEqual(nodes["pano-b"]["grounding_status"], "null")
        self.assertEqual(nodes["pano-c"]["grounding_status"], "unknown")
        self.assertEqual(nodes["pano-b"]["degree_in"], 1)

    def test_build_room_color_map_assigns_unique_colors_for_current_room_scale(self) -> None:
        room_ids = [f"Room {index}" for index in range(1, 31)]

        color_map = build_room_color_map(room_ids)

        self.assertEqual(len(color_map), 30)
        self.assertEqual(len(set(color_map.values())), 30)

    def test_geojson_exports_point_nodes_and_linestring_edges(self) -> None:
        payload = build_visualization_payload(self.pano_graph, grounding_payload=self.grounding)

        node_geojson = build_geojson(payload, feature_type="nodes")
        edge_geojson = build_geojson(payload, feature_type="edges")

        self.assertEqual(node_geojson["type"], "FeatureCollection")
        self.assertEqual(len(node_geojson["features"]), 3)
        self.assertEqual(node_geojson["features"][0]["geometry"]["type"], "Point")
        self.assertEqual(len(edge_geojson["features"]), 2)
        self.assertEqual(edge_geojson["features"][0]["geometry"]["type"], "LineString")
        json.dumps(node_geojson)
        json.dumps(edge_geojson)

    def test_gexf_and_graphml_exports_are_parseable_xml(self) -> None:
        payload = build_visualization_payload(self.pano_graph, grounding_payload=self.grounding)

        gexf = build_gexf(payload)
        graphml = build_graphml(payload)

        self.assertEqual(ET.fromstring(gexf).tag, "{http://www.gexf.net/1.3}gexf")
        self.assertEqual(ET.fromstring(graphml).tag, "{http://graphml.graphdrawing.org/xmlns}graphml")

    def test_shortest_path_and_dot_can_highlight_route(self) -> None:
        payload = build_visualization_payload(self.pano_graph, grounding_payload=self.grounding)

        route = shortest_pano_path(payload, "pano-a", "pano-c")
        dot = build_dot(payload, floor="0", route_pano_ids=route)

        self.assertEqual(route, ["pano-a", "pano-b", "pano-c"])
        self.assertIn('"pano-a" -> "pano-b"', dot)
        self.assertIn('color="#ef4444"', dot)
        self.assertNotIn('"pano-b" -> "missing"', dot)


if __name__ == "__main__":
    unittest.main()
