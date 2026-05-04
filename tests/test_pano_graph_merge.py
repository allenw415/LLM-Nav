from __future__ import annotations

import unittest

from st_nav_data.pano_graph_merge import merge_raw_crawl_payloads


class PanoGraphMergeTests(unittest.TestCase):
    def test_merge_raw_crawl_payloads_adds_new_panos_and_links(self) -> None:
        base = {
            "seed": {"lat": 1.0, "lng": 2.0},
            "panos": {
                "pano-a": {
                    "pano": "pano-a",
                    "lat": 1.0,
                    "lng": 2.0,
                    "links": [{"pano": "pano-b", "heading": 90.0}],
                }
            },
        }
        incoming = {
            "seed": {"lat": 1.1, "lng": 2.1},
            "panos": {
                "pano-a": {
                    "pano": "pano-a",
                    "lat": 1.0,
                    "lng": 2.0,
                    "links": [
                        {"pano": "pano-b", "heading": 90.0},
                        {"pano": "pano-c", "heading": 180.0},
                    ],
                },
                "pano-c": {
                    "pano": "pano-c",
                    "lat": 1.1,
                    "lng": 2.1,
                    "links": [{"pano": "pano-a", "heading": 0.0}],
                },
            },
        }

        merged, summary = merge_raw_crawl_payloads(base, [incoming])

        self.assertEqual(summary["base_panos"], 1)
        self.assertEqual(summary["incoming_panos"], 2)
        self.assertEqual(summary["added_panos"], 1)
        self.assertEqual(summary["merged_existing_panos"], 1)
        self.assertEqual(summary["added_links"], 1)
        self.assertEqual(summary["output_panos"], 2)
        self.assertEqual(
            [link["pano"] for link in merged["panos"]["pano-a"]["links"]],
            ["pano-b", "pano-c"],
        )

    def test_merge_raw_crawl_payloads_accepts_pano_id_link_variants(self) -> None:
        base = {"panos": {}}
        incoming = {
            "panos": {
                "key-a": {
                    "panoID": "pano-a",
                    "lat": 1,
                    "lng": 2,
                    "links": [{"panoID": "pano-b", "heading": 45}],
                }
            }
        }

        merged, summary = merge_raw_crawl_payloads(base, [incoming])

        self.assertEqual(summary["added_panos"], 1)
        self.assertIn("pano-a", merged["panos"])
        self.assertEqual(merged["panos"]["pano-a"]["links"], [{"pano": "pano-b", "heading": 45, "description": None}])


if __name__ == "__main__":
    unittest.main()
