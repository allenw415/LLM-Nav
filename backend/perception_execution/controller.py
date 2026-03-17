from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

from detector.detector import PerceptionDetector
from executor.executor import ActionExecutor
from renderer.renderer import StreetViewRenderer
from renderer.viewer import ImageViewer


PolicyFn = Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any]]


@dataclass
class ControllerConfig:
    render_mode: str = "single"   # "single" or "four"
    low_confidence_threshold: float = 0.5
    default_turn_angle: float = 30.0
    default_zoom_step: int = 1
    auto_start_viewer: bool = True


class PerceptionExecutionController:
    def __init__(
        self,
        renderer: StreetViewRenderer,
        detector: PerceptionDetector,
        executor: ActionExecutor,
        viewer: Optional[ImageViewer] = None,
        config: Optional[ControllerConfig] = None,
    ) -> None:
        self.renderer = renderer
        self.detector = detector
        self.executor = executor
        self.viewer = viewer
        self.config = config or ControllerConfig()

        if self.viewer is not None and self.config.auto_start_viewer:
            self.viewer.start()

    def observe(self, state: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        rendered = self.renderer.render_for_detection(
            state=state,
            mode=self.config.render_mode,
        )
        observation = self.detector.build_observation(rendered)
        return rendered, observation

    def act(self, state: Dict[str, Any], action: Dict[str, Any]) -> Dict[str, Any]:
        return self.executor.execute(state, action)

    def step(
        self,
        state: Dict[str, Any],
        policy_fn: Optional[PolicyFn] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        rendered, observation = self.observe(state)

        if self.viewer is not None:
            self.viewer.update(
                state=state,
                observation=observation,
                action={"type": "OBSERVE", "value": None},
            )

        if policy_fn is None:
            action = self.default_policy(state, observation)
        else:
            action = policy_fn(state, observation)

        new_state = self.act(state, action)

        rendered_after, observation_after = self.observe(new_state)

        if self.viewer is not None:
            self.viewer.update(
                state=new_state,
                observation=observation_after,
                action=action,
            )

        return new_state, observation_after, action

    def default_policy(self, state: Dict[str, Any], observation: Dict[str, Any]) -> Dict[str, Any]:
        confidence = float(observation.get("confidence", 0.0))
        landmarks = observation.get("landmarks", [])
        ocr_texts = observation.get("ocr_texts", [])

        if confidence < self.config.low_confidence_threshold:
            zoom = int(state.get("zoom", 1))
            if zoom < self.executor.max_zoom:
                return {"type": "ZOOM_IN", "value": self.config.default_zoom_step}
            return {"type": "TURN_RIGHT", "value": self.config.default_turn_angle}

        if not landmarks and not ocr_texts:
            return {"type": "TURN_RIGHT", "value": self.config.default_turn_angle}

        current_pano = state["panoID"]
        links = self.executor.graph.get(current_pano, {}).get("links", [])
        if links:
            return {"type": "MOVE_TO_LINK", "value": 0}

        return {"type": "NO_OP", "value": None}