class VLMClient:
    def analyze_image(self, image, prompt: str, metadata=None):
        return {
            "scene_desc": "",
            "landmarks": [],
            "artifacts": [],
            "ocr_texts": [],
            "confidence": 0.0,
        }