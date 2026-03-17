from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlencode

import requests
from PIL import Image


class StreetViewClient:
    """
    Google Street View Static API client.

    說明：
    - 使用 panoID + heading + pitch + fov 取得靜態影像
    - zoom 不是 Static API 原生參數，因此由上層自行轉成 fov
    """

    BASE_URL = "https://maps.googleapis.com/maps/api/streetview"
    METADATA_URL = "https://maps.googleapis.com/maps/api/streetview/metadata"

    def __init__(
        self,
        api_key: str,
        image_size: Tuple[int, int] = (640, 640),
        timeout: int = 20,
    ) -> None:
        self.api_key = api_key
        self.image_size = image_size
        self.timeout = timeout

    def build_image_url(
        self,
        pano_id: str,
        heading: float = 0.0,
        pitch: float = 0.0,
        fov: int = 90,
    ) -> str:
        params = {
            "size": f"{self.image_size[0]}x{self.image_size[1]}",
            "pano": pano_id,
            "heading": heading,
            "pitch": pitch,
            "fov": fov,
            "key": self.api_key,
        }
        return f"{self.BASE_URL}?{urlencode(params)}"

    def build_metadata_url(self, pano_id: str) -> str:
        params = {
            "pano": pano_id,
            "key": self.api_key,
        }
        return f"{self.METADATA_URL}?{urlencode(params)}"

    def get_metadata(self, pano_id: str) -> Dict[str, Any]:
        url = self.build_metadata_url(pano_id)
        resp = requests.get(url, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def get_image(
        self,
        pano_id: str,
        heading: float = 0.0,
        pitch: float = 0.0,
        fov: int = 90,
    ) -> Image.Image:
        url = self.build_image_url(
            pano_id=pano_id,
            heading=heading,
            pitch=pitch,
            fov=fov,
        )
        resp = requests.get(url, timeout=self.timeout)
        resp.raise_for_status()
        return Image.open(BytesIO(resp.content)).convert("RGB")

    def save_image(
        self,
        output_path: str | Path,
        pano_id: str,
        heading: float = 0.0,
        pitch: float = 0.0,
        fov: int = 90,
    ) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        image = self.get_image(
            pano_id=pano_id,
            heading=heading,
            pitch=pitch,
            fov=fov,
        )
        image.save(output_path)
        return output_path