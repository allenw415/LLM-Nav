# l1_layer.py
from __future__ import annotations

import base64
import io
import json
import time
import uuid
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Literal, Optional, Tuple

import requests

from control import StreetViewController  # ONLY L1 imports control.py


Action = Literal[
    "front",
    "turn_left",
    "turn_right",
    "pitch_up",
    "pitch_level",
    "pitch_down",
    "zoom_in",
    "zoom_out",
]
PitchMode = Literal[60, 90, 120]


# -----------------------------
# Vision client (pluggable)
# -----------------------------

class VisionClient:
    def analyze(self, png_bytes: bytes, prompt: str, hints: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError


@dataclass
class HTTPVisionClientConfig:
    endpoint: str
    timeout_s: int = 30
    headers: Optional[Dict[str, str]] = None


class HTTPVisionClient(VisionClient):
    """
    POST {image_b64, prompt, hints} -> JSON:
    {
      "status": "ok"|"fail",
      "location_hypotheses": [{"place_id": "...", "confidence": 0.7, "evidence": [...]}, ...],
      "landmarks": [...],
      "ocr_text": [...]
    }
    """
    def __init__(self, cfg: HTTPVisionClientConfig):
        self.cfg = cfg

    def analyze(self, png_bytes: bytes, prompt: str, hints: Dict[str, Any]) -> Dict[str, Any]:
        payload = {
            "image_b64": base64.b64encode(png_bytes).decode("utf-8"),
            "prompt": prompt,
            "hints": hints,
        }
        r = requests.post(
            self.cfg.endpoint,
            json=payload,
            headers=self.cfg.headers,
            timeout=self.cfg.timeout_s,
        )
        r.raise_for_status()
        out = r.json()
        if not isinstance(out, dict):
            return {"status": "fail", "error": "vision_return_not_dict", "raw": str(out)}
        out.setdefault("status", "ok")
        out.setdefault("location_hypotheses", [])
        out.setdefault("landmarks", [])
        out.setdefault("ocr_text", [])
        return out


# -----------------------------
# L1 configs
# -----------------------------

@dataclass
class ActivePerceptionConfig:
    scan_turns: int = 4
    pitch_sequence: Tuple[PitchMode, ...] = (90, 60, 120)
    zoom_sequence: Tuple[int, ...] = (0, +1, -1)  # 0=none, +1=zoom_in, -1=zoom_out
    allow_move_when_fail: bool = True
    max_move_trials: int = 1
    settle_ms: int = 120
    max_images: int = 20


@dataclass
class L1Config:
    web_root: str
    entry_html: str = "index.html"
    headless: bool = True
    viewport: Tuple[int, int] = (1280, 720)
    slow_mo_ms: int = 0

    museum_name: str = "British Museum"
    target_sign_types: Tuple[str, ...] = ("room number sign", "gallery sign", "direction board")

    success_conf_threshold: float = 0.65
    active: ActivePerceptionConfig = ActivePerceptionConfig()


# -----------------------------
# Packet schema
# -----------------------------

@dataclass
class ImageRecord:
    view: str
    rel_heading_steps: int
    pitch: PitchMode
    zoom_level: int
    png_b64: str


@dataclass
class ActionFeedback:
    last_action: Optional[str]
    success: Optional[bool]
    note: str = ""


@dataclass
class ObservationPacket:
    t: float
    obs_id: str
    images: List[ImageRecord]
    vision: Dict[str, Any]
    action_feedback: ActionFeedback
    console: List[Dict[str, Any]]


# -----------------------------
# L1 runtime
# -----------------------------

class L1Runtime:
    """
    - ONLY allowed: screenshot_pano, action_*, key_*, get_console
    - DO NOT call goto_node/get_state/get_status_text here.
    """

    def __init__(self, cfg: L1Config, vision: VisionClient):
        self.cfg = cfg
        self.vision = vision
        self.ctrl = StreetViewController(
            web_root=cfg.web_root,
            entry_html=cfg.entry_html,
            headless=cfg.headless,
            viewport=cfg.viewport,
            slow_mo_ms=cfg.slow_mo_ms,
        )

        # L1 local relative state (NOT reading page state)
        self._rel_heading_steps = 0
        self._pitch: PitchMode = 90
        self._zoom_level = 0

    def start(self) -> None:
        self.ctrl.start()

    def close(self) -> None:
        self.ctrl.close()

    def __enter__(self) -> "L1Runtime":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # -------- allowed wrappers --------

    def _capture_png(self) -> bytes:
        return self.ctrl.screenshot_pano()

    def _read_console(self, clear: bool = True) -> List[Dict[str, Any]]:
        entries = self.ctrl.get_console(clear=clear, kinds=["log", "error", "pageerror"])
        return [{"ts": e.ts, "kind": e.kind, "text": e.text} for e in entries]

    def act(self, action: Action) -> ActionFeedback:
        if action == "turn_left":
            self.ctrl.action_turn_left(settle_ms=self.cfg.active.settle_ms)
            self._rel_heading_steps = (self._rel_heading_steps - 1) % 4
            return ActionFeedback(last_action=action, success=None)

        if action == "turn_right":
            self.ctrl.action_turn_right(settle_ms=self.cfg.active.settle_ms)
            self._rel_heading_steps = (self._rel_heading_steps + 1) % 4
            return ActionFeedback(last_action=action, success=None)

        if action == "pitch_up":
            self.ctrl.action_pitch_up(settle_ms=self.cfg.active.settle_ms)
            self._pitch = 60
            return ActionFeedback(last_action=action, success=None)

        if action == "pitch_level":
            self.ctrl.action_pitch_level(settle_ms=self.cfg.active.settle_ms)
            self._pitch = 90
            return ActionFeedback(last_action=action, success=None)

        if action == "pitch_down":
            self.ctrl.action_pitch_down(settle_ms=self.cfg.active.settle_ms)
            self._pitch = 120
            return ActionFeedback(last_action=action, success=None)

        if action == "zoom_in":
            self.ctrl.action_zoom_in(settle_ms=self.cfg.active.settle_ms)
            self._zoom_level += 1
            return ActionFeedback(last_action=action, success=None)

        if action == "zoom_out":
            self.ctrl.action_zoom_out(settle_ms=self.cfg.active.settle_ms)
            self._zoom_level -= 1
            return ActionFeedback(last_action=action, success=None)

        if action == "front":
            ok = bool(self.ctrl.action_front(settle_ms=self.cfg.active.settle_ms))
            return ActionFeedback(last_action=action, success=ok, note="moved" if ok else "blocked_or_no_transition")

        raise ValueError(f"Action not allowed: {action}")

    def execute_action_sequence(self, actions: List[Action]) -> List[ActionFeedback]:
        out: List[ActionFeedback] = []
        for a in actions:
            out.append(self.act(a))
        return out

    # -------- prompts --------

    def build_vision_prompt(self, task_hint: str) -> str:
        sign_list = ", ".join(self.cfg.target_sign_types)
        return (
            f"You are a visual localization assistant in {self.cfg.museum_name}.\n"
            f"Identify the CURRENT room/gallery from visible cues.\n"
            f"Focus on: {sign_list}.\n"
            f"Task hint: {task_hint}\n"
            f"Return JSON with: location_hypotheses(place_id, confidence, evidence), landmarks, ocr_text.\n"
        )

    def _imgrec(self, view: str, png_bytes: bytes) -> ImageRecord:
        return ImageRecord(
            view=view,
            rel_heading_steps=self._rel_heading_steps,
            pitch=self._pitch,
            zoom_level=self._zoom_level,
            png_b64=base64.b64encode(png_bytes).decode("utf-8"),
        )

    def _best_conf(self, vision: Dict[str, Any]) -> float:
        hyps = vision.get("location_hypotheses") or []
        if not hyps:
            return 0.0
        return max((float(h.get("confidence", 0.0)) for h in hyps), default=0.0)

    def _safe_vision_call(self, png: bytes, prompt: str, hints: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return self.vision.analyze(png, prompt=prompt, hints=hints)
        except Exception as e:
            return {
                "status": "fail",
                "error": f"vision_call_failed: {type(e).__name__}: {e}",
                "location_hypotheses": [],
                "landmarks": [],
                "ocr_text": [],
            }

    # -------- observe --------

    def observe_once(self, task_hint: str, extra_hints: Optional[Dict[str, Any]] = None) -> ObservationPacket:
        obs_id = str(uuid.uuid4())
        t0 = time.time()

        png = self._capture_png()
        img = self._imgrec("main", png)

        prompt = self.build_vision_prompt(task_hint)
        hints = dict(extra_hints or {})
        hints.update({"rel_heading_steps": self._rel_heading_steps, "pitch": self._pitch, "zoom_level": self._zoom_level})

        vision = self._safe_vision_call(png, prompt, hints)
        console = self._read_console(clear=True)

        return ObservationPacket(
            t=t0,
            obs_id=obs_id,
            images=[img],
            vision=vision,
            action_feedback=ActionFeedback(last_action=None, success=None),
            console=console,
        )

    def active_observe(self, task_hint: str, extra_hints: Optional[Dict[str, Any]] = None) -> ObservationPacket:
        obs_id = str(uuid.uuid4())
        t0 = time.time()

        images: List[ImageRecord] = []
        console_all: List[Dict[str, Any]] = []

        prompt = self.build_vision_prompt(task_hint)
        hints = dict(extra_hints or {})

        def capture(view: str) -> bytes:
            png_bytes = self._capture_png()
            images.append(self._imgrec(view, png_bytes))
            console_all.extend(self._read_console(clear=True))
            return png_bytes

        # 0) main
        png = capture("main")
        hints.update({"rel_heading_steps": self._rel_heading_steps, "pitch": self._pitch, "zoom_level": self._zoom_level})
        vision = self._safe_vision_call(png, prompt, hints)

        if self._best_conf(vision) >= self.cfg.success_conf_threshold:
            return ObservationPacket(t=t0, obs_id=obs_id, images=images, vision=vision,
                                     action_feedback=ActionFeedback(last_action=None, success=None),
                                     console=console_all)

        # 1) scan
        for _ in range(1, self.cfg.active.scan_turns):
            if len(images) >= self.cfg.active.max_images:
                break
            self.act("turn_right")
            png = capture("scan")
            hints.update({"rel_heading_steps": self._rel_heading_steps, "pitch": self._pitch, "zoom_level": self._zoom_level})
            vision = self._safe_vision_call(png, prompt, hints)
            if self._best_conf(vision) >= self.cfg.success_conf_threshold:
                return ObservationPacket(t=t0, obs_id=obs_id, images=images, vision=vision,
                                         action_feedback=ActionFeedback(last_action="turn_right", success=None),
                                         console=console_all)

        # 2) pitch sweep
        for p in self.cfg.active.pitch_sequence:
            if len(images) >= self.cfg.active.max_images:
                break
            if p == 60:
                self.act("pitch_up")
            elif p == 90:
                self.act("pitch_level")
            else:
                self.act("pitch_down")
            png = capture("pitch")
            hints.update({"rel_heading_steps": self._rel_heading_steps, "pitch": self._pitch, "zoom_level": self._zoom_level})
            vision = self._safe_vision_call(png, prompt, hints)
            if self._best_conf(vision) >= self.cfg.success_conf_threshold:
                return ObservationPacket(t=t0, obs_id=obs_id, images=images, vision=vision,
                                         action_feedback=ActionFeedback(last_action=None, success=None),
                                         console=console_all)

        # 3) zoom sweep
        for z in self.cfg.active.zoom_sequence:
            if len(images) >= self.cfg.active.max_images:
                break
            if z == +1:
                self.act("zoom_in")
            elif z == -1:
                self.act("zoom_out")
            png = capture("zoom")
            hints.update({"rel_heading_steps": self._rel_heading_steps, "pitch": self._pitch, "zoom_level": self._zoom_level})
            vision = self._safe_vision_call(png, prompt, hints)
            if self._best_conf(vision) >= self.cfg.success_conf_threshold:
                return ObservationPacket(t=t0, obs_id=obs_id, images=images, vision=vision,
                                         action_feedback=ActionFeedback(last_action=None, success=None),
                                         console=console_all)

        # 4) optional move + re-observe
        last_feedback = ActionFeedback(last_action=None, success=None, note="still_uncertain_after_active_perception")
        if self.cfg.active.allow_move_when_fail:
            for _ in range(self.cfg.active.max_move_trials):
                fb = self.act("front")
                last_feedback = fb
                png = capture("after_move")
                hints.update({"rel_heading_steps": self._rel_heading_steps, "pitch": self._pitch, "zoom_level": self._zoom_level})
                vision = self._safe_vision_call(png, prompt, hints)
                if self._best_conf(vision) >= self.cfg.success_conf_threshold:
                    return ObservationPacket(t=t0, obs_id=obs_id, images=images, vision=vision,
                                             action_feedback=fb, console=console_all)

        return ObservationPacket(
            t=t0,
            obs_id=obs_id,
            images=images,
            vision=vision,
            action_feedback=last_feedback,
            console=console_all,
        )
