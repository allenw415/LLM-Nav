# l3_layer.py
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field
from openai import OpenAI


EdgeType = Literal["corridor", "stairs", "elevator", "door", "other"]
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


# -----------------------------
# Constraints schema (aligned with L2)
# -----------------------------

class ConstraintsModel(BaseModel):
    avoid_edge_types: List[EdgeType] = Field(default_factory=list)
    prefer_edge_types: List[EdgeType] = Field(default_factory=list)
    blacklist_nodes: List[str] = Field(default_factory=list)
    blacklist_tags: List[str] = Field(default_factory=list)
    max_candidates: int = 3
    prefer_bonus: float = 0.2
    crowded_penalty: float = 0.0


# -----------------------------
# User intent
# -----------------------------

class ParsedUserIntent(BaseModel):
    goal_place_id: Optional[str] = None
    goal_description: Optional[str] = None
    constraints: ConstraintsModel = Field(default_factory=ConstraintsModel)
    additional_notes: List[str] = Field(default_factory=list)


# -----------------------------
# L3 outputs
# -----------------------------

class RouteChoice(BaseModel):
    selected_candidate_id: Optional[str] = None
    reason: str


class L1DirectiveStep(BaseModel):
    kind: Literal["act", "observe"]
    actions: List[Action] = Field(default_factory=list)
    task_hint: str = ""
    expected_cues: List[str] = Field(default_factory=list)
    stop_when_matched: bool = True


class L3Decision(BaseModel):
    route_choice: RouteChoice
    long_horizon_instructions: List[str]
    short_horizon_instructions: List[str]
    l1_plan: List[L1DirectiveStep]
    updated_constraints: ConstraintsModel
    need_more_perception: bool = False
    clarification_questions: List[str] = Field(default_factory=list)


# -----------------------------
# L3 layer
# -----------------------------

class L3ReasoningLayer:
    def __init__(
        self,
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_output_tokens: int = 900,
        client: Optional[OpenAI] = None,
    ):
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.client = client or OpenAI()

    # ---- parse user intent (constraints) ----

    def parse_user_intent(self, user_text: str, known_place_ids: Optional[List[str]] = None) -> ParsedUserIntent:
        known_place_ids = known_place_ids or []

        sys = (
            "You are a navigation constraint parser for a museum navigation system.\n"
            "Extract constraints from the user request.\n"
            "Rules:\n"
            "- Avoid stairs => constraints.avoid_edge_types=['stairs']\n"
            "- Prefer elevator => constraints.prefer_edge_types=['elevator']\n"
            "- Do not invent place_ids.\n"
            "If the user mentions an exact place_id that appears in the known list, put it in goal_place_id.\n"
            "Otherwise keep goal_place_id=null and fill goal_description.\n"
        )

        user = {
            "user_text": user_text,
            "known_place_ids_sample": known_place_ids[:200],
        }

        resp = self.client.responses.parse(
            model=self.model,
            input=[
                {"role": "system", "content": sys},
                {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
            ],
            text_format=ParsedUserIntent,
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
        )
        return resp.output_parsed

    # ---- decide (cached plan packet, no file read) ----

    def decide_from_plan_packet(
        self,
        plan_packet: Dict[str, Any],
        user_text: str,
        progress: Optional[Dict[str, Any]] = None,
        lock_candidate_id: Optional[str] = None,
        allow_change_route: bool = False,
    ) -> L3Decision:
        """
        用「已快取」的 plan_packet 直接產生下一段 L1 指令（不讀 plan.json）。
        若 lock_candidate_id 存在且 allow_change_route=False，要求模型維持同一條 route。
        """
        progress = progress or {}

        sys = (
            "You are the LLM Reasoning Layer of a 3-layer museum navigation agent.\n"
            "You must:\n"
            "1) Convert route candidates + waypoints into human-readable instructions.\n"
            "2) Produce an L1 executable plan using ONLY these actions: "
            "front, turn_left, turn_right, pitch_up, pitch_level, pitch_down, zoom_in, zoom_out.\n"
            "3) L1 can also 'observe' (take screenshot + vision).\n"
            "Do NOT assume exact heading. Use verification by expected cues (signs).\n"
            "If uncertain: observe -> scan -> zoom_in -> front -> observe.\n"
        )

        payload = {
            "user_text": user_text,
            "plan_packet": plan_packet,
            "progress": progress,
            "lock_candidate_id": lock_candidate_id,
            "allow_change_route": allow_change_route,
            "routing_rule": (
                "If lock_candidate_id is provided and allow_change_route=false, "
                "you MUST keep route_choice.selected_candidate_id=lock_candidate_id."
            ),
        }

        resp = self.client.responses.parse(
            model=self.model,
            input=[
                {"role": "system", "content": sys},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            text_format=L3Decision,
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
        )
        out: L3Decision = resp.output_parsed
        return out

    # ---- decide (replan mode: read plan.json) ----

    def decide_from_plan_file(
        self,
        plan_json_path: str,
        user_text: str,
        progress: Optional[Dict[str, Any]] = None,
    ) -> L3Decision:
        """
        只在 replan 時使用：明確「讀 plan.json」再決策。
        """
        plan_packet = json.loads(Path(plan_json_path).read_text(encoding="utf-8"))

        # 先解析 intent，讓 updated_constraints 更穩定
        known_ids: List[str] = []
        for c in plan_packet.get("route_candidates", []) or []:
            for pid in c.get("path", []) or []:
                if isinstance(pid, str):
                    known_ids.append(pid)
        known_ids = list(dict.fromkeys(known_ids))

        intent = self.parse_user_intent(user_text, known_place_ids=known_ids)

        out = self.decide_from_plan_packet(
            plan_packet=plan_packet,
            user_text=user_text,
            progress=progress,
            lock_candidate_id=None,
            allow_change_route=True,
        )

        # 強制把 constraints 以 parser 解析的為準（避免模型亂改）
        out.updated_constraints = intent.constraints
        return out
