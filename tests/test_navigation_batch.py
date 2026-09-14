from __future__ import annotations

import json
from pathlib import Path
import random
import tempfile
import unittest

from st_nav.evaluation.navigation_batch import (
    build_cache_inventory,
    compute_navigation_spl_metrics,
    compute_step_budget,
    filter_floor_artifacts,
    pano_path_distance_m,
    summarize_batch_results,
    sync_floor0_assets,
)
from st_nav.perception.renderer import MUSEUM_CAPTURE_LABELS, PanoramaRenderer


def write_manifest(
    root: Path,
    *,
    pano_id: str,
    floor: str,
    graph_path: Path,
    content: bytes = b"image",
) -> Path:
    pano_dir = root / pano_id
    pano_dir.mkdir(parents=True, exist_ok=True)
    captures = []
    for index, label in enumerate(MUSEUM_CAPTURE_LABELS):
        image_path = pano_dir / f"{pano_id}-{index}.png"
        image_path.write_bytes(content + str(index).encode("ascii"))
        captures.append(
            {
                "label": label,
                "heading": float(index * 45),
                "path": str(image_path),
                "url": "https://example.invalid/?key=secret",
            }
        )
    manifest_path = pano_dir / f"{pano_id}_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "graph_path": str(graph_path),
                "pano_id": pano_id,
                "floor": floor,
                "heading_mode": "museum",
                "pitch": 0.0,
                "fov": 90,
                "size": {"width": 512, "height": 512},
                "captures": captures,
            }
        ),
        encoding="utf-8",
    )
    return manifest_path


class NavigationBatchTests(unittest.TestCase):
    def test_step_budget_matches_baseline_formula(self) -> None:
        self.assertEqual(compute_step_budget(2), 50)
        self.assertEqual(compute_step_budget(20), 60)
        self.assertEqual(compute_step_budget(70), 210)
        self.assertEqual(compute_step_budget(200), 300)

    def test_spl_uses_metres_and_retains_step_approximation(self) -> None:
        metrics = compute_navigation_spl_metrics(
            success=True,
            shortest_path_distance_m=100.0,
            actual_path_distance_m=150.0,
            shortest_path_pano_steps=5,
            movement_steps=10,
        )

        self.assertAlmostEqual(metrics["spl"], 2.0 / 3.0)
        self.assertEqual(metrics["spl_metric"], "distance_m")
        self.assertEqual(metrics["step_spl_legacy"], 0.5)
        self.assertEqual(metrics["actual_over_shortest_ratio"], 1.5)

    def test_failed_navigation_has_zero_spl(self) -> None:
        metrics = compute_navigation_spl_metrics(
            success=False,
            shortest_path_distance_m=100.0,
            actual_path_distance_m=25.0,
            shortest_path_pano_steps=5,
            movement_steps=1,
        )

        self.assertEqual(metrics["spl"], 0.0)
        self.assertEqual(metrics["step_spl_legacy"], 0.0)

    def test_pano_path_distance_sums_haversine_segments(self) -> None:
        pano_graph = {
            "a": {"lat": 0.0, "lng": 0.0},
            "b": {"lat": 0.0, "lng": 0.001},
            "c": {"lat": 0.0, "lng": 0.002},
        }

        distance = pano_path_distance_m(["a", "b", "c"], pano_graph)

        self.assertAlmostEqual(distance, 222.389853, places=5)

    def test_filter_floor_artifacts_removes_cross_floor_edges(self) -> None:
        pano_graph = {
            "p0": {
                "pano_id": "p0",
                "floor": "0",
                "neighbors": [
                    {"target_pano_id": "p0b"},
                    {"target_pano_id": "p1"},
                ],
            },
            "p0b": {"pano_id": "p0b", "floor": "0", "neighbors": []},
            "p1": {"pano_id": "p1", "floor": "1", "neighbors": []},
        }
        room_graph = {
            "r0": {
                "room_id": "r0",
                "floor": "0",
                "neighbors": [
                    {"target_room_id": "r0b"},
                    {"target_room_id": "r1"},
                ],
            },
            "r0b": {"room_id": "r0b", "floor": "0", "neighbors": []},
            "r1": {"room_id": "r1", "floor": "1", "neighbors": []},
        }
        grounding = {"mappings": {"p0": "r0", "p0b": None, "p1": "r1"}}

        panos, rooms, filtered_grounding = filter_floor_artifacts(
            pano_graph=pano_graph,
            room_graph=room_graph,
            pano_room_grounding=grounding,
            floor="0",
        )

        self.assertEqual(set(panos), {"p0", "p0b"})
        self.assertEqual(panos["p0"]["neighbors"], [{"target_pano_id": "p0b"}])
        self.assertEqual(set(rooms), {"r0", "r0b"})
        self.assertEqual(rooms["r0"]["neighbors"], [{"target_room_id": "r0b"}])
        self.assertEqual(filtered_grounding["mappings"], {"p0": "r0", "p0b": None})

    def test_sync_copies_only_floor_zero_and_rewrites_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_source = root / "dataset-source"
            dataset_source.mkdir()
            (dataset_source / "test_set.jsonl").write_text('{"test_id":"T1"}\n', encoding="utf-8")
            dataset_destination = root / "dataset-destination"
            render_source = root / "render-source"
            render_destination = root / "render-destination"
            render_source.mkdir()
            render_destination.mkdir()
            graph_path = root / "pano_graph.json"
            pano_graph = {
                "p0": {"pano_id": "p0", "floor": "0", "neighbors": []},
                "p0-missing": {"pano_id": "p0-missing", "floor": "0", "neighbors": []},
                "p1": {"pano_id": "p1", "floor": "1", "neighbors": []},
            }
            graph_path.write_text(json.dumps(pano_graph), encoding="utf-8")
            write_manifest(render_source, pano_id="p0", floor="0", graph_path=graph_path)
            write_manifest(render_source, pano_id="p1", floor="1", graph_path=graph_path)
            destination_only = render_destination / "destination-only"
            destination_only.mkdir()
            (destination_only / "keep.txt").write_text("keep", encoding="utf-8")
            stale_dir = render_destination / "p0"
            stale_dir.mkdir()
            (stale_dir / "p0_manifest_detections.json").write_text("{}", encoding="utf-8")

            dry_run = sync_floor0_assets(
                dataset_source=dataset_source,
                dataset_destination=dataset_destination,
                render_source=render_source,
                render_destination=render_destination,
                pano_graph=pano_graph,
                graph_path=graph_path,
                floor="0",
                apply=False,
            )
            self.assertEqual(dry_run["renders"]["selected_floor_manifest_count"], 1)
            self.assertEqual(dry_run["renders"]["skipped_non_floor_manifest_count"], 1)
            self.assertEqual(dry_run["renders"]["floor_missing_after_source_sync_count"], 1)
            self.assertFalse(dataset_destination.exists())

            applied = sync_floor0_assets(
                dataset_source=dataset_source,
                dataset_destination=dataset_destination,
                render_source=render_source,
                render_destination=render_destination,
                pano_graph=pano_graph,
                graph_path=graph_path,
                floor="0",
                apply=True,
            )
            self.assertTrue((dataset_destination / "test_set.jsonl").is_file())
            self.assertTrue((destination_only / "keep.txt").is_file())
            self.assertFalse((stale_dir / "p0_manifest_detections.json").exists())
            self.assertEqual(applied["renders"]["sidecars_invalidated"], 1)
            self.assertFalse((render_destination / "p1").exists())
            copied_manifest_path = render_destination / "p0" / "p0_manifest.json"
            copied = json.loads(copied_manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(copied["graph_path"], str(graph_path.resolve()))
            self.assertTrue(all("url" not in capture for capture in copied["captures"]))
            self.assertTrue(
                all(
                    Path(capture["path"]).parent == (render_destination / "p0").resolve()
                    for capture in copied["captures"]
                )
            )

    def test_cache_inventory_requires_all_eight_images(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            graph_path = root / "graph.json"
            pano_graph = {
                "p0": {"pano_id": "p0", "floor": "0", "neighbors": []},
                "p1": {"pano_id": "p1", "floor": "0", "neighbors": []},
            }
            graph_path.write_text(json.dumps(pano_graph), encoding="utf-8")
            manifest_path = write_manifest(root, pano_id="p0", floor="0", graph_path=graph_path)
            inventory = build_cache_inventory(
                pano_graph=pano_graph,
                render_root=root,
                floor="0",
                reference_pano_ids={"p0", "p1"},
                start_pano_ids={"p0"},
            )
            self.assertEqual(inventory["floor_cached_count"], 1)
            self.assertEqual(inventory["floor_missing_pano_ids"], ["p1"])
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            Path(payload["captures"][0]["path"]).unlink()
            invalid = build_cache_inventory(
                pano_graph=pano_graph,
                render_root=root,
                floor="0",
            )
            self.assertEqual(invalid["floor_cached_count"], 0)
            self.assertEqual(invalid["invalid_manifest_count"], 1)

    def test_cardinal4_inventory_requires_exact_four_headings(self) -> None:
        downloads: list[Path] = []

        def downloader(url: str, path: Path) -> None:
            downloads.append(path)
            path.write_bytes(b"image")

        pano_graph = {"p0": {"pano_id": "p0", "floor": "0", "neighbors": []}}
        renderer = PanoramaRenderer(
            pano_graph,
            image_downloader=downloader,
            metadata_checker=lambda **kwargs: True,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = renderer.render(
                pano_id="p0",
                api_key="key",
                output_dir=tmpdir,
                heading_mode="cardinal4",
                pitch=0.0,
                fov=90,
                width=512,
                height=512,
            )
            payload = json.loads(Path(manifest["manifest_path"]).read_text(encoding="utf-8"))
            self.assertEqual(len(downloads), 4)
            self.assertEqual(
                [capture["label"] for capture in payload["captures"]],
                ["view_0", "view_1", "view_2", "view_3"],
            )
            self.assertEqual(
                [capture["heading"] for capture in payload["captures"]],
                [0.0, 90.0, 180.0, 270.0],
            )
            inventory = build_cache_inventory(
                pano_graph=pano_graph,
                render_root=tmpdir,
                floor="0",
                heading_mode="cardinal4",
            )
            self.assertEqual(inventory["floor_cached_count"], 1)

            payload["captures"][0]["heading"] = 330.0
            Path(manifest["manifest_path"]).write_text(json.dumps(payload), encoding="utf-8")
            invalid = build_cache_inventory(
                pano_graph=pano_graph,
                render_root=tmpdir,
                floor="0",
                heading_mode="cardinal4",
            )
            self.assertEqual(invalid["floor_cached_count"], 0)
            self.assertEqual(invalid["invalid"][0]["reason"], "capture_heading_mismatch")

    def test_renderer_renders_once_then_reuses_cache(self) -> None:
        downloads: list[Path] = []
        metadata_calls: list[str] = []

        def downloader(url: str, path: Path) -> None:
            downloads.append(path)
            path.write_bytes(b"image")

        def metadata_checker(*, api_key: str, pano_id: str) -> bool:
            metadata_calls.append(pano_id)
            return True

        renderer = PanoramaRenderer(
            {"p0": {"pano_id": "p0", "floor": "0", "neighbors": []}},
            image_downloader=downloader,
            metadata_checker=metadata_checker,
            rng=random.Random(0),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            first = renderer.render(
                pano_id="p0",
                api_key="key",
                output_dir=tmpdir,
                heading_mode="museum",
                pitch=0.0,
                fov=90,
                width=512,
                height=512,
                graph_path="graph.json",
            )
            second = renderer.render(
                pano_id="p0",
                api_key="key",
                output_dir=tmpdir,
                heading_mode="museum",
                pitch=0.0,
                fov=90,
                width=512,
                height=512,
                graph_path="graph.json",
            )
            self.assertEqual(len(downloads), 8)
            self.assertEqual(metadata_calls, ["p0"])
            self.assertEqual(first["manifest_path"], second["manifest_path"])

    def test_renderer_does_not_publish_partial_manifest(self) -> None:
        attempts = 0

        def downloader(url: str, path: Path) -> None:
            nonlocal attempts
            attempts += 1
            if attempts == 3:
                raise RuntimeError("download failed")
            path.write_bytes(b"image")

        renderer = PanoramaRenderer(
            {"p0": {"pano_id": "p0", "floor": "0", "neighbors": []}},
            image_downloader=downloader,
            metadata_checker=lambda **kwargs: True,
            rng=random.Random(0),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(RuntimeError, "download failed"):
                renderer.render(
                    pano_id="p0",
                    api_key="key",
                    output_dir=tmpdir,
                    heading_mode="museum",
                    pitch=0.0,
                    fov=90,
                    width=512,
                    height=512,
                )
            self.assertFalse((Path(tmpdir) / "p0" / "p0_manifest.json").exists())
            self.assertEqual(list((Path(tmpdir) / "p0").glob("*.png")), [])

    def test_renderer_stops_when_metadata_has_no_imagery(self) -> None:
        downloads = []

        renderer = PanoramaRenderer(
            {"p0": {"pano_id": "p0", "floor": "0", "neighbors": []}},
            image_downloader=lambda url, path: downloads.append(path),
            metadata_checker=lambda **kwargs: False,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(RuntimeError, "no imagery"):
                renderer.render(
                    pano_id="p0",
                    api_key="key",
                    output_dir=tmpdir,
                    heading_mode="museum",
                    fov=90,
                )
        self.assertEqual(downloads, [])

    def test_batch_summary_aggregates_render_costs(self) -> None:
        summary = summarize_batch_results(
            [
                {
                    "success": True,
                    "spl": 0.5,
                    "movement_steps": 4,
                    "render": {
                        "cache_hit_events": 3,
                        "cache_miss_events": 1,
                        "new_render_pano_count": 1,
                        "new_image_requests": 8,
                    },
                    "model_calls": {
                        "stages": {
                            "query_parser": {
                                "logical_call_count": 1,
                                "text_logical_call_count": 1,
                                "http_attempt_count": 1,
                                "successful_http_attempt_count": 1,
                                "text_http_attempt_count": 1,
                            },
                            "detection": {
                                "logical_call_count": 2,
                                "image_logical_call_count": 2,
                                "http_attempt_count": 3,
                                "successful_http_attempt_count": 2,
                                "failed_http_attempt_count": 1,
                                "image_http_attempt_count": 3,
                            },
                        }
                    },
                },
                {
                    "success": False,
                    "spl": 0.0,
                    "movement_steps": 2,
                    "error": {"type": "RuntimeError"},
                    "render": {
                        "cache_hit_events": 1,
                        "cache_miss_events": 0,
                        "new_render_pano_count": 0,
                        "new_image_requests": 0,
                    },
                },
            ]
        )
        self.assertEqual(summary["success_count"], 1)
        self.assertEqual(summary["average_spl"], 0.25)
        self.assertEqual(summary["render"]["new_image_requests"], 8)
        self.assertEqual(summary["error_count"], 1)
        self.assertEqual(summary["model_calls"]["coverage_case_count"], 1)
        self.assertEqual(summary["model_calls"]["missing_telemetry_case_count"], 1)
        self.assertEqual(summary["model_calls"]["total"]["logical_call_count"], 3)
        self.assertEqual(summary["model_calls"]["total"]["image_logical_call_count"], 2)
        self.assertEqual(summary["model_calls"]["total"]["http_attempt_count"], 4)


if __name__ == "__main__":
    unittest.main()
