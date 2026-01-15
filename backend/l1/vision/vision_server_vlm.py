# vision_server_vlm.py
from __future__ import annotations

import base64
import io
import json
import re
import time
from typing import Any, Dict, List, Literal, Optional, Tuple

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from PIL import Image

from config import VISION  # <-- 只改 config.py 就能切換


# -------------------------
# Request / Response schema
# -------------------------

class AnalyzeRequest(BaseModel):
    image_b64: str
    prompt: str = ""
    hints: Dict[str, Any] = Field(default_factory=dict)

    # 執行期可覆蓋（你也可以完全不用傳，統一走 config.py）
    backend: Optional[Literal["openai", "gemini", "hf_local"]] = None
    model: Optional[str] = None

    temperature: Optional[float] = None
    max_output_tokens: Optional[int] = None


class LocationHypothesis(BaseModel):
    place_id: str
    confidence: float
    evidence: List[str] = []


class AnalyzeResponse(BaseModel):
    status: str = "ok"
    location_hypotheses: List[LocationHypothesis] = []
    landmarks: List[str] = []
    descriptions: List[str] = []

    raw_text: str = ""
    debug: Dict[str, Any] = Field(default_factory=dict)


# -------------------------
# Utilities
# -------------------------

def b64_to_bytes(b64: str) -> bytes:
    return base64.b64decode(b64.encode("utf-8"))


def normalize_to_png(img_bytes: bytes) -> bytes:
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def extract_json_block(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    if not text:
        return {}

    m = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        text = m.group(1).strip()

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    m2 = re.search(r"(\{.*\})", text, flags=re.DOTALL)
    if m2:
        try:
            obj = json.loads(m2.group(1))
            if isinstance(obj, dict):
                return obj
        except Exception:
            return {}

    return {}


def make_instruction_for_json(prompt: str) -> str:
    return (
        "Return ONLY valid JSON (no markdown, no extra text).\n"
        "JSON schema:\n"
        "{\n"
        '  "status": "ok"|"fail",\n'
        '  "location_hypotheses": [{"place_id": string, "confidence": number, "evidence": [string]}],\n'
        '  "landmarks": [string],\n'
        '  "descriptions": [string]\n'
        "}\n"
        "If uncertain, return empty location_hypotheses and status=\"ok\".\n\n"
        "Task:\n"
        f"{prompt}\n"
    )


def normalize_output(obj: Dict[str, Any], raw_text: str, debug: Dict[str, Any]) -> AnalyzeResponse:
    status = str(obj.get("status", "ok"))

    hyps = obj.get("location_hypotheses", []) or []
    lms = obj.get("landmarks", []) or []
    descriptions = obj.get("descriptions", []) or []

    out_hyps: List[LocationHypothesis] = []
    if isinstance(hyps, list):
        for h in hyps:
            if not isinstance(h, dict):
                continue
            pid = str(h.get("place_id", "")).strip()
            if not pid:
                continue
            try:
                conf = float(h.get("confidence", 0.0))
            except Exception:
                conf = 0.0
            ev = h.get("evidence", [])
            if not isinstance(ev, list):
                ev = [str(ev)]
            out_hyps.append(LocationHypothesis(place_id=pid, confidence=conf, evidence=[str(x) for x in ev]))

    if not isinstance(lms, list):
        lms = [str(lms)]
    if not isinstance(descriptions, list):
        descriptions = [str(descriptions)]

    return AnalyzeResponse(
        status=status,
        location_hypotheses=out_hyps,
        landmarks=[str(x) for x in lms],
        descriptions=[str(x) for x in descriptions],
        raw_text=raw_text,
        debug=debug,
    )


# -------------------------
# Backends
# -------------------------

def run_openai(png_bytes: bytes, prompt: str, temperature: float, max_output_tokens: int, model_override: Optional[str]) -> Tuple[str, Dict[str, Any]]:
    try:
        from openai import OpenAI
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"openai sdk not installed: {e}")

    if not VISION.openai_api_key:
        raise HTTPException(status_code=400, detail="config.py: VISION.openai_api_key is empty")

    model = model_override or VISION.openai_model
    client = OpenAI(api_key=VISION.openai_api_key)

    b64 = base64.b64encode(png_bytes).decode("utf-8")
    data_url = f"data:image/png;base64,{b64}"
    instructions = make_instruction_for_json(prompt)

    t0 = time.time()
    resp = client.responses.create(
        model=model,
        instructions=instructions,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Analyze the image and follow the JSON schema."},
                    {"type": "input_image", "image_url": data_url},
                ],
            }
        ],
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )
    dt = time.time() - t0

    raw_text = getattr(resp, "output_text", "") or ""
    return raw_text, {"backend": "openai", "model": model, "latency_s": dt}


def run_gemini(png_bytes: bytes, prompt: str, model_override: Optional[str]) -> Tuple[str, Dict[str, Any]]:
    try:
        from google import genai
        from google.genai import types
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"google-genai not installed: {e}")

    if not VISION.gemini_api_key:
        raise HTTPException(status_code=400, detail="config.py: VISION.gemini_api_key is empty")

    model = model_override or VISION.gemini_model
    client = genai.Client(api_key=VISION.gemini_api_key)

    instruction = make_instruction_for_json(prompt)

    t0 = time.time()
    response = client.models.generate_content(
        model=model,
        contents=[
            types.Part.from_bytes(data=png_bytes, mime_type="image/png"),
            instruction,
        ],
    )
    dt = time.time() - t0

    raw_text = getattr(response, "text", "") or ""
    return raw_text, {"backend": "gemini", "model": model, "latency_s": dt}


_HF_CACHE: Dict[str, Any] = {}


def run_hf_local(png_bytes: bytes, prompt: str, max_output_tokens: int, model_override: Optional[str]) -> Tuple[str, Dict[str, Any]]:
    try:
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"HF local deps missing: {e}")

    model_id = model_override or VISION.hf_model

    if model_id not in _HF_CACHE:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if VISION.hf_device in ("cpu", "cuda"):
            device = VISION.hf_device

        dtype = torch.bfloat16 if device == "cuda" else torch.float32

        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForImageTextToText.from_pretrained(model_id, torch_dtype=dtype).to(device)

        _HF_CACHE[model_id] = {"processor": processor, "model": model, "device": device}

    pack = _HF_CACHE[model_id]
    processor = pack["processor"]
    model = pack["model"]
    device = pack["device"]

    img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    instruction = make_instruction_for_json(prompt)

    t0 = time.time()
    inputs = processor(text=instruction, images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        out_ids = model.generate(**inputs, max_new_tokens=max_output_tokens)
    dt = time.time() - t0

    raw_text = processor.batch_decode(out_ids, skip_special_tokens=True)[0].strip()
    return raw_text, {"backend": "hf_local", "model": model_id, "device": device, "latency_s": dt}


# -------------------------
# FastAPI
# -------------------------

app = FastAPI(title="Multi-backend VLM Vision Server (config.py driven)")


@app.get("/health")
def health():
    return {"ok": True, "backend": VISION.backend}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest) -> AnalyzeResponse:
    t0 = time.time()

    # defaults from config.py (can be overridden by request)
    backend = req.backend or VISION.backend
    temperature = req.temperature if req.temperature is not None else VISION.temperature
    max_output_tokens = req.max_output_tokens if req.max_output_tokens is not None else VISION.max_output_tokens

    try:
        img_bytes = b64_to_bytes(req.image_b64)
        png_bytes = normalize_to_png(img_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"invalid image_b64: {e}")

    if backend == "openai":
        raw_text, dbg = run_openai(png_bytes, req.prompt, temperature, max_output_tokens, req.model)
    elif backend == "gemini":
        raw_text, dbg = run_gemini(png_bytes, req.prompt, req.model)
    elif backend == "hf_local":
        raw_text, dbg = run_hf_local(png_bytes, req.prompt, max_output_tokens, req.model)
    else:
        raise HTTPException(status_code=400, detail=f"unknown backend: {backend}")

    obj = extract_json_block(raw_text)

    debug = {
        "backend_effective": backend,
        "model_override": req.model,
        "total_s": time.time() - t0,
        "hints": req.hints,
        **dbg,
    }

    if not obj:
        return AnalyzeResponse(
            status="fail",
            location_hypotheses=[],
            landmarks=[],
            descriptions=[],
            raw_text=raw_text,
            debug={**debug, "parse_error": "invalid_json"},
        )

    return normalize_output(obj, raw_text=raw_text, debug=debug)
