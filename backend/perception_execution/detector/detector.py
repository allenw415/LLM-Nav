from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class DetectorConfig:
    use_vlm: bool = True
    use_ocr: bool = True


class PerceptionDetector:
    """
    目前先保留 VLM API 與 OCR 的接口。
    你之後只要把 _call_vlm() 改成真正的 API 呼叫即可。
    """

    def __init__(self, config: Optional[DetectorConfig] = None) -> None:
        self.config = config or DetectorConfig()

    def build_observation(self, rendered: Dict[str, Any]) -> Dict[str, Any]:
        if "image_path" in rendered:
            return self._detect_single_view(rendered)
        if "views" in rendered:
            return self._detect_multi_view(rendered)
        raise ValueError("Invalid rendered input")

    def _detect_single_view(self, rendered: Dict[str, Any]) -> Dict[str, Any]:
        image_path = rendered["image_path"]
        meta = rendered["meta"]

        vlm_result = self._call_vlm(image_path=image_path, meta=meta) if self.config.use_vlm else {}
        ocr_result = self._call_ocr(image_path=image_path) if self.config.use_ocr else []

        return {
            "type": "single_view_observation",
            "panoID": meta.get("panoID"),
            "heading": meta.get("heading"),
            "pitch": meta.get("pitch"),
            "zoom": meta.get("zoom"),
            "floor": meta.get("floor"),
            "scene_desc": vlm_result.get("scene_desc", ""),
            "landmarks": vlm_result.get("landmarks", []),
            "artifacts": vlm_result.get("artifacts", []),
            "ocr_texts": ocr_result or vlm_result.get("ocr_texts", []),
            "confidence": float(vlm_result.get("confidence", 0.0)),
            "image_path": image_path,
        }

    def _detect_multi_view(self, rendered: Dict[str, Any]) -> Dict[str, Any]:
        per_view = []
        for item in rendered["views"]:
            per_view.append(self._detect_single_view(item))

        return {
            "type": "multi_view_observation",
            "panoID": rendered["meta"].get("panoID"),
            "floor": rendered["meta"].get("floor"),
            "views": per_view,
            "landmarks": [x for v in per_view for x in v.get("landmarks", [])],
            "artifacts": [x for v in per_view for x in v.get("artifacts", [])],
            "ocr_texts": [x for v in per_view for x in v.get("ocr_texts", [])],
            "scene_desc": " | ".join([v.get("scene_desc", "") for v in per_view if v.get("scene_desc")]),
            "confidence": sum(v.get("confidence", 0.0) for v in per_view) / max(len(per_view), 1),
        }

    def _call_vlm(self, image_path: str, meta: Dict[str, Any]) -> Dict[str, Any]:
        """
        TODO:
        之後改成真正 VLM API 呼叫。
        建議回傳固定 JSON 格式。
        """
        return {
            "scene_desc": f"Mock VLM result for {image_path}",
            "landmarks": [],
            "artifacts": [],
            "ocr_texts": [],
            "confidence": 0.3,
        }

    def _call_ocr(self, image_path: str) -> List[Dict[str, Any]]:
        """
        TODO:
        OCR 為輔助，現在先留空。
        """
        return []