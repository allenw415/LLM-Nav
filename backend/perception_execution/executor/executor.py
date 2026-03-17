from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional

from .action_utils import normalize_heading, clamp_pitch, clamp_zoom


class ActionExecutor:
    """
    負責執行兩種動作：
    1. 同一 pano 內的觀察動作：turn / pitch / zoom
    2. graph 上的離散移動：move_to_link / move_to_pano
    """

    def __init__(
        self,
        graph: Dict[str, Any],
        min_pitch: float = -90.0,
        max_pitch: float = 90.0,
        min_zoom: int = 0,
        max_zoom: int = 5,
    ) -> None:
        self.graph = graph
        self.min_pitch = min_pitch
        self.max_pitch = max_pitch
        self.min_zoom = min_zoom
        self.max_zoom = max_zoom

    def execute(self, state: Dict[str, Any], action: Dict[str, Any]) -> Dict[str, Any]:
        action_type = action["type"]
        value = action.get("value")

        if action_type == "TURN_LEFT":
            return self.turn_left(state, float(value))
        elif action_type == "TURN_RIGHT":
            return self.turn_right(state, float(value))
        elif action_type == "SET_HEADING":
            return self.set_heading(state, float(value))
        elif action_type == "LOOK_UP":
            return self.look_up(state, float(value))
        elif action_type == "LOOK_DOWN":
            return self.look_down(state, float(value))
        elif action_type == "ZOOM_IN":
            return self.zoom_in(state, int(value or 1))
        elif action_type == "ZOOM_OUT":
            return self.zoom_out(state, int(value or 1))
        elif action_type == "MOVE_TO_LINK":
            return self.move_to_link(state, int(value))
        elif action_type == "MOVE_TO_PANO":
            return self.move_to_pano(state, str(value))
        elif action_type == "NO_OP":
            return self.no_op(state)
        else:
            raise ValueError(f"Unsupported action type: {action_type}")

    def turn_left(self, state: Dict[str, Any], angle: float) -> Dict[str, Any]:
        new_state = copy.deepcopy(state)
        new_state["heading"] = normalize_heading(float(state.get("heading", 0.0)) - angle)
        new_state["last_action"] = {"type": "TURN_LEFT", "value": angle}
        return new_state

    def turn_right(self, state: Dict[str, Any], angle: float) -> Dict[str, Any]:
        new_state = copy.deepcopy(state)
        new_state["heading"] = normalize_heading(float(state.get("heading", 0.0)) + angle)
        new_state["last_action"] = {"type": "TURN_RIGHT", "value": angle}
        return new_state

    def set_heading(self, state: Dict[str, Any], heading: float) -> Dict[str, Any]:
        new_state = copy.deepcopy(state)
        new_state["heading"] = normalize_heading(heading)
        new_state["last_action"] = {"type": "SET_HEADING", "value": heading}
        return new_state

    def look_up(self, state: Dict[str, Any], angle: float) -> Dict[str, Any]:
        new_state = copy.deepcopy(state)
        pitch = float(state.get("pitch", 0.0)) + angle
        new_state["pitch"] = clamp_pitch(pitch, self.min_pitch, self.max_pitch)
        new_state["last_action"] = {"type": "LOOK_UP", "value": angle}
        return new_state

    def look_down(self, state: Dict[str, Any], angle: float) -> Dict[str, Any]:
        new_state = copy.deepcopy(state)
        pitch = float(state.get("pitch", 0.0)) - angle
        new_state["pitch"] = clamp_pitch(pitch, self.min_pitch, self.max_pitch)
        new_state["last_action"] = {"type": "LOOK_DOWN", "value": angle}
        return new_state

    def zoom_in(self, state: Dict[str, Any], step: int = 1) -> Dict[str, Any]:
        new_state = copy.deepcopy(state)
        zoom = int(state.get("zoom", 1)) + step
        new_state["zoom"] = clamp_zoom(zoom, self.min_zoom, self.max_zoom)
        new_state["last_action"] = {"type": "ZOOM_IN", "value": step}
        return new_state

    def zoom_out(self, state: Dict[str, Any], step: int = 1) -> Dict[str, Any]:
        new_state = copy.deepcopy(state)
        zoom = int(state.get("zoom", 1)) - step
        new_state["zoom"] = clamp_zoom(zoom, self.min_zoom, self.max_zoom)
        new_state["last_action"] = {"type": "ZOOM_OUT", "value": step}
        return new_state

    def move_to_link(self, state: Dict[str, Any], link_idx: int) -> Dict[str, Any]:
        current_pano = state["panoID"]
        node = self.graph.get(current_pano)
        if node is None:
            raise KeyError(f"Current panoID not found in graph: {current_pano}")

        links: List[Dict[str, Any]] = node.get("links", [])
        if not (0 <= link_idx < len(links)):
            raise IndexError(f"Invalid link_idx={link_idx} for panoID={current_pano}")

        next_link = links[link_idx]
        next_pano = next_link["panoID"]
        next_floor = self.graph.get(next_pano, {}).get("floor", state.get("floor"))
        next_heading = float(next_link.get("heading", state.get("heading", 0.0)))

        new_state = copy.deepcopy(state)
        new_state["panoID"] = next_pano
        new_state["floor"] = next_floor
        new_state["heading"] = normalize_heading(next_heading)
        new_state["pitch"] = 0.0
        new_state["zoom"] = 1
        new_state["last_action"] = {"type": "MOVE_TO_LINK", "value": link_idx}
        return new_state

    def move_to_pano(self, state: Dict[str, Any], next_pano_id: str) -> Dict[str, Any]:
        current_pano = state["panoID"]
        node = self.graph.get(current_pano)
        if node is None:
            raise KeyError(f"Current panoID not found in graph: {current_pano}")

        links: List[Dict[str, Any]] = node.get("links", [])
        matched_link = None
        for link in links:
            if link["panoID"] == next_pano_id:
                matched_link = link
                break

        if matched_link is None:
            raise ValueError(f"next_pano_id={next_pano_id} is not a neighbor of current panoID={current_pano}")

        next_floor = self.graph.get(next_pano_id, {}).get("floor", state.get("floor"))
        next_heading = float(matched_link.get("heading", state.get("heading", 0.0)))

        new_state = copy.deepcopy(state)
        new_state["panoID"] = next_pano_id
        new_state["floor"] = next_floor
        new_state["heading"] = normalize_heading(next_heading)
        new_state["pitch"] = 0.0
        new_state["zoom"] = 1
        new_state["last_action"] = {"type": "MOVE_TO_PANO", "value": next_pano_id}
        return new_state

    def no_op(self, state: Dict[str, Any]) -> Dict[str, Any]:
        new_state = copy.deepcopy(state)
        new_state["last_action"] = {"type": "NO_OP", "value": None}
        return new_state