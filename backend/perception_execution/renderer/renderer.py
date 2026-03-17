from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .streetview_client import StreetViewClient


@dataclass
class RendererConfig:
    image_size: Tuple[int, int] = (640, 640)
    output_dir: str = "runtime_views"
    current_filename: str = "current_view.jpg"


class StreetViewRenderer:
    """
    將 state 轉成圖片，並存到本地給 viewer / detector 使用。
    """

    def __init__(
        self,
        api_key: str,
        config: Optional[RendererConfig] = None,
        client: Optional[StreetViewClient] = None,
    ) -> None:
        self.config = config or RendererConfig()
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.client = client or StreetViewClient(
            api_key=api_key,
            image_size=self.config.image_size,
        )

    def zoom_to_fov(self, zoom: int) -> int:
        """
        你的系統內部用 zoom，Static API 實際送 fov。
        可自行調整 mapping。
        """
        mapping = {
            0: 120,
            1: 90,
            2: 60,
            3: 40,
            4: 25,
            5: 15,
        }
        return mapping.get(int(zoom), 90)

    def get_current_output_path(self) -> Path:
        return self.output_dir / self.config.current_filename

    def render_view(self, state: Dict[str, Any]) -> Dict[str, Any]:
        pano_id = state["panoID"]
        heading = float(state.get("heading", 0.0))
        pitch = float(state.get("pitch", 0.0))
        zoom = int(state.get("zoom", 1))
        fov = self.zoom_to_fov(zoom)

        output_path = self.get_current_output_path()
        self.client.save_image(
            output_path=output_path,
            pano_id=pano_id,
            heading=heading,
            pitch=pitch,
            fov=fov,
        )

        return {
            "image_path": str(output_path),
            "meta": {
                "panoID": pano_id,
                "heading": heading,
                "pitch": pitch,
                "zoom": zoom,
                "fov": fov,
                "floor": state.get("floor"),
            },
        }

    def render_multi_view(
        self,
        state: Dict[str, Any],
        relative_headings: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        if relative_headings is None:
            relative_headings = [0.0, 90.0, 180.0, 270.0]

        pano_id = state["panoID"]
        base_heading = float(state.get("heading", 0.0))
        pitch = float(state.get("pitch", 0.0))
        zoom = int(state.get("zoom", 1))
        fov = self.zoom_to_fov(zoom)

        views = []
        for i, rel_h in enumerate(relative_headings):
            heading = (base_heading + rel_h) % 360.0
            output_path = self.output_dir / f"view_{i}.jpg"
            self.client.save_image(
                output_path=output_path,
                pano_id=pano_id,
                heading=heading,
                pitch=pitch,
                fov=fov,
            )
            views.append(
                {
                    "image_path": str(output_path),
                    "meta": {
                        "panoID": pano_id,
                        "heading": heading,
                        "pitch": pitch,
                        "zoom": zoom,
                        "fov": fov,
                        "relative_heading": rel_h,
                        "floor": state.get("floor"),
                    },
                }
            )

        return {
            "views": views,
            "meta": {
                "panoID": pano_id,
                "base_heading": base_heading,
                "pitch": pitch,
                "zoom": zoom,
                "fov": fov,
                "floor": state.get("floor"),
            },
        }

    def render_for_detection(self, state: Dict[str, Any], mode: str = "single") -> Dict[str, Any]:
        if mode == "single":
            return self.render_view(state)
        if mode == "four":
            return self.render_multi_view(state)
        raise ValueError(f"Unsupported render mode: {mode}")