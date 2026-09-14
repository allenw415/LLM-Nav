from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.pano_viewer import server


class PanoViewerServerTests(unittest.TestCase):
    def test_manifest_resolves_frames_and_nearest_circular_heading(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_root = Path(tmp)
            pano_a = project_root / "renders/room_grounding_fov90/A"
            pano_b = project_root / "renders/room_grounding_fov90/B"
            pano_a.mkdir(parents=True)
            pano_b.mkdir(parents=True)
            (pano_a / "A_00_north_010deg.png").write_bytes(b"north")
            nearest = pano_a / "A_01_west_to_north_350deg.png"
            nearest.write_bytes(b"nearest")
            (pano_b / "B_00_north_005deg.png").write_bytes(b"final")

            payload = {
                "pano_path": ["A", "B"],
                "final_pano_id": "B",
                "target_room_id": "Room 2",
                "rounds": [
                    {
                        "movement_steps": [
                            {
                                "current_pano_id": "A",
                                "next_pano_id": "B",
                                "current_room_id": "Room 1",
                                "next_room_id": "Room 2",
                                "selected_action_heading": 355,
                            }
                        ]
                    }
                ],
            }

            with patch.object(server, "PROJECT_ROOT", project_root):
                manifest = server.build_panorama_frame_manifest(payload)

            self.assertEqual(manifest["frame_count"], 2)
            self.assertEqual(manifest["missing_count"], 0)
            self.assertIn("A_01_west_to_north_350deg.png", manifest["frames"][0]["image_url"])
            self.assertEqual(manifest["frames"][1]["pano_id"], "B")

    def test_manifest_accepts_evaluator_wrapper(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_root = Path(tmp)
            pano_dir = project_root / "renders/room_grounding_fov90/A"
            pano_dir.mkdir(parents=True)
            (pano_dir / "A_00_north_000deg.png").write_bytes(b"frame")
            payload = {
                "episode": {
                    "pano_path": ["A"],
                    "final_pano_id": "A",
                    "rounds": [],
                }
            }

            with patch.object(server, "PROJECT_ROOT", project_root):
                manifest = server.build_panorama_frame_manifest(payload)

            self.assertEqual(manifest["frame_count"], 1)
            self.assertEqual(manifest["frames"][0]["pano_id"], "A")

    def test_image_paths_are_limited_to_render_and_output_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_root = Path(tmp)
            allowed = project_root / "outputs/frame.png"
            allowed.parent.mkdir(parents=True)
            allowed.write_bytes(b"frame")

            with patch.object(server, "PROJECT_ROOT", project_root):
                self.assertEqual(server.resolve_project_image_path("outputs/frame.png"), allowed)
                with self.assertRaisesRegex(ValueError, "outside the allowed"):
                    server.resolve_project_image_path("../frame.png")
                with self.assertRaisesRegex(ValueError, "Unsupported image type"):
                    server.resolve_project_image_path("outputs/frame.txt")


if __name__ == "__main__":
    unittest.main()
