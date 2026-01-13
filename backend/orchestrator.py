# orchestrator.py
from __future__ import annotations

import base64
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import multiprocessing as mp


# -------------------------
# L1 Worker (isolated process)
# -------------------------

def _l1_worker_main(cmd_q: mp.Queue, resp_q: mp.Queue, l1_cfg: Dict[str, Any], vision_cfg: Dict[str, Any]) -> None:
    """
    子程序：唯一會 import l1_layer（進而 import control.py）
    """
    try:
        from l1_layer import L1Runtime, L1Config, HTTPVisionClient, HTTPVisionClientConfig  # noqa

        cfg = L1Config(**l1_cfg)
        vision = HTTPVisionClient(HTTPVisionClientConfig(**vision_cfg))

        with L1Runtime(cfg, vision) as l1:
            resp_q.put({"ok": True, "event": "started"})

            while True:
                msg = cmd_q.get()
                cmd = msg.get("cmd")

                if cmd == "close":
                    resp_q.put({"ok": True, "event": "closed"})
                    return

                if cmd == "observe":
                    mode = msg.get("mode", "active")  # "active" | "once"
                    task_hint = msg.get("task_hint", "")
                    extra_hints = msg.get("extra_hints", None)

                    if mode == "once":
                        pkt = l1.observe_once(task_hint=task_hint, extra_hints=extra_hints)
                    else:
                        pkt = l1.active_observe(task_hint=task_hint, extra_hints=extra_hints)

                    resp_q.put({"ok": True, "event": "observe", "packet": asdict(pkt)})
                    continue

                if cmd == "act":
                    actions = msg.get("actions", [])
                    fbs = l1.execute_action_sequence(actions)
                    resp_q.put({"ok": True, "event": "act", "feedback": [asdict(x) for x in fbs]})
                    continue

                resp_q.put({"ok": False, "error": f"unknown_cmd: {cmd}"})

    except Exception as e:
        resp_q.put({"ok": False, "error": f"l1_worker_crash: {type(e).__name__}: {e}"})


# -------------------------
# Helpers
# -------------------------

def _save_obs_images(obs_packet: Dict[str, Any], out_dir: str) -> None:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    obs_id = obs_packet.get("obs_id", "no_obs_id")
    for i, im in enumerate(obs_packet.get("images", []) or []):
        b64 = im.get("png_b64")
        if not b64:
            continue
        png = base64.b64decode(b64.encode("utf-8"))
        view = im.get("view", "view")
        h = im.get("rel_heading_steps", 0)
        p = im.get("pitch", 90)
        z = im.get("zoom_level", 0)
        fn = f"{obs_id}_{i:02d}_{view}_h{h}_p{p}_z{z}.png"
        (Path(out_dir) / fn).write_bytes(png)


def _goal_reached(obs_packet: Dict[str, Any], goal_place_id: str, conf_th: float = 0.80) -> bool:
    vision = obs_packet.get("vision", {}) or {}
    hyps = vision.get("location_hypotheses", []) or []
    for h in hyps:
        if h.get("place_id") == goal_place_id and float(h.get("confidence", 0.0)) >= conf_th:
            return True
    return False


def _constraints_changed(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    return json.dumps(a, sort_keys=True, ensure_ascii=False) != json.dumps(b, sort_keys=True, ensure_ascii=False)


# -------------------------
# Orchestrator
# -------------------------

class Orchestrator:
    """
    規則（依你要求）：
    - 未偏離原計畫：L2 不需產 plan.json；L3 不需再讀 plan.json
    - 偏離/不確定/constraints 變更：才 replan -> 產 plan.json -> L3 重新讀 plan.json
    """

    def __init__(
        self,
        *,
        graph_path: str,
        goal_place_id: str,
        user_text: str,

        # L1
        web_root: str,
        entry_html: str,
        vision_endpoint: str,
        headless: bool = True,

        # outputs
        plan_json_path: str = "plan.json",
        decision_json_path: str = "decision.json",
        debug_obs_dir: str = "",

        # loop controls
        max_rounds: int = 200,
        on_track_mass_threshold: float = 0.60,
        goal_conf_threshold: float = 0.80,

        # L3 model
        model_name: str = "",
    ):
        self.graph_path = graph_path
        self.goal_place_id = goal_place_id
        self.user_text = user_text

        self.plan_json_path = plan_json_path
        self.decision_json_path = decision_json_path
        self.debug_obs_dir = debug_obs_dir

        self.max_rounds = max_rounds
        self.on_track_mass_threshold = on_track_mass_threshold
        self.goal_conf_threshold = goal_conf_threshold

        # L2 / L3 in main process (no control.py)
        from l2_layer import MuseumGraph, SpatialMapMemoryLayer, Constraints  # noqa
        from l3_layer import L3ReasoningLayer  # noqa

        self.Constraints = Constraints

        g = MuseumGraph.from_json(graph_path)
        self.l2 = SpatialMapMemoryLayer(g)

        self.l3 = L3ReasoningLayer(model=(model_name or os.getenv("OPENAI_MODEL", "gpt-4o-mini")))

        # cached plan & decision
        self.current_plan_packet: Optional[Dict[str, Any]] = None
        self.current_decision: Optional[Dict[str, Any]] = None
        self.current_constraints: Dict[str, Any] = {}

        self.selected_candidate_id: Optional[str] = None
        self.selected_path: List[str] = []

        # decision execution cursor
        self.step_idx: int = 0

        # L1 worker config
        self.l1_cfg = {
            "web_root": web_root,
            "entry_html": entry_html,
            "headless": headless,
        }
        self.vision_cfg = {
            "endpoint": vision_endpoint,
            "timeout_s": 45,
            "headers": None,
        }

        self.cmd_q: mp.Queue = mp.Queue()
        self.resp_q: mp.Queue = mp.Queue()
        self.proc: Optional[mp.Process] = None

        # last observation for progress summary
        self.last_obs: Optional[Dict[str, Any]] = None

    # ----------------- L1 IPC -----------------

    def _l1_start(self) -> None:
        if self.proc and self.proc.is_alive():
            return
        self.proc = mp.Process(
            target=_l1_worker_main,
            args=(self.cmd_q, self.resp_q, self.l1_cfg, self.vision_cfg),
            daemon=True,
        )
        self.proc.start()
        msg = self.resp_q.get(timeout=60)
        if not msg.get("ok"):
            raise RuntimeError(f"L1 start failed: {msg}")

    def _l1_close(self) -> None:
        if not self.proc:
            return
        try:
            self.cmd_q.put({"cmd": "close"})
            _ = self.resp_q.get(timeout=10)
        except Exception:
            pass
        if self.proc.is_alive():
            self.proc.terminate()
        self.proc = None

    def _l1_observe(self, task_hint: str, mode: str = "active") -> Dict[str, Any]:
        self.cmd_q.put({"cmd": "observe", "mode": mode, "task_hint": task_hint})
        msg = self.resp_q.get(timeout=240)
        if not msg.get("ok"):
            raise RuntimeError(f"L1 observe failed: {msg}")
        obs = msg["packet"]
        self.last_obs = obs
        if self.debug_obs_dir:
            _save_obs_images(obs, self.debug_obs_dir)
        return obs

    def _l1_act(self, actions: List[str]) -> List[Dict[str, Any]]:
        self.cmd_q.put({"cmd": "act", "actions": actions})
        msg = self.resp_q.get(timeout=120)
        if not msg.get("ok"):
            raise RuntimeError(f"L1 act failed: {msg}")
        return msg.get("feedback", []) or []

    # ----------------- on-track detection -----------------

    def _prob_mass_on_path(self, path: List[str]) -> float:
        if not path:
            return 0.0
        return sum(float(self.l2.belief.get(pid, 0.0)) for pid in path)

    def _is_on_track(self) -> bool:
        if not self.selected_path:
            return False
        top1 = max(self.l2.belief.items(), key=lambda kv: kv[1])[0]
        if top1 in self.selected_path:
            return True
        return self._prob_mass_on_path(self.selected_path) >= self.on_track_mass_threshold

    def _progress_summary(self) -> Dict[str, Any]:
        """
        給 L3 的簡短進度資訊（不需要很精準）。
        """
        topk = self.l2.belief_topk(3)
        ocr = ((self.last_obs or {}).get("vision", {}) or {}).get("ocr_text", []) or []
        return {
            "belief_topk": topk,
            "last_ocr": ocr[:8],
            "goal": self.goal_place_id,
            "selected_candidate_id": self.selected_candidate_id,
        }

    # ----------------- initial constraints -----------------

    def _initial_constraints_from_user(self) -> None:
        known_ids = list(self.l2.g.nodes.keys())
        intent = self.l3.parse_user_intent(self.user_text, known_place_ids=known_ids)
        cons_dict = intent.constraints.model_dump()

        self.l2.set_task(self.goal_place_id, self.Constraints(**cons_dict))
        self.current_constraints = cons_dict

    # ----------------- (re)plan & decision -----------------

    def _update_selected_path_from_plan(self, plan_packet: Dict[str, Any], candidate_id: Optional[str]) -> None:
        selected_path: List[str] = []
        if candidate_id:
            for c in (plan_packet.get("route_candidates", []) or []):
                if c.get("id") == candidate_id:
                    selected_path = list(c.get("path", []) or [])
                    break

        if not selected_path and (plan_packet.get("route_candidates") or []):
            selected_path = list(plan_packet["route_candidates"][0].get("path", []) or [])
            candidate_id = plan_packet["route_candidates"][0].get("id")

        self.selected_candidate_id = candidate_id
        self.selected_path = selected_path

    def _replan(self) -> None:
        """
        真正的 replan：產 plan.json，並要求 L3 重新讀 plan.json
        """
        plan_packet = asdict(self.l2.plan())
        Path(self.plan_json_path).write_text(json.dumps(plan_packet, ensure_ascii=False, indent=2), encoding="utf-8")

        decision = self.l3.decide_from_plan_file(
            plan_json_path=self.plan_json_path,
            user_text=self.user_text,
            progress=self._progress_summary(),
        ).model_dump()

        Path(self.decision_json_path).write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")

        self.current_plan_packet = plan_packet
        self.current_decision = decision
        self.step_idx = 0

        # constraints sync (if changed)
        upd = decision.get("updated_constraints", {}) or {}
        if upd and _constraints_changed(upd, self.current_constraints):
            self.l2.set_task(self.goal_place_id, self.Constraints(**upd))
            self.current_constraints = upd

        # chosen route
        sel_id = (decision.get("route_choice", {}) or {}).get("selected_candidate_id")
        self._update_selected_path_from_plan(plan_packet, sel_id)

    def _refresh_decision_without_replan(self) -> None:
        """
        未偏離時：不寫/不讀 plan.json，只用 cached plan_packet 產下一段 L1 指令
        """
        if not self.current_plan_packet:
            # 沒 plan cache 就只能 replan
            self._replan()
            return

        decision = self.l3.decide_from_plan_packet(
            plan_packet=self.current_plan_packet,
            user_text=self.user_text,
            progress=self._progress_summary(),
            lock_candidate_id=self.selected_candidate_id,
            allow_change_route=False,
        ).model_dump()

        # 不寫 plan.json；decision.json 也不一定要寫（你要 debug 才寫）
        # 這裡我選擇「不寫」以符合你對 I/O 的要求；若要寫可自行打開。
        self.current_decision = decision
        self.step_idx = 0

    # ----------------- main loop -----------------

    def run(self) -> Dict[str, Any]:
        self._l1_start()

        # 1) 初始 constraints
        self._initial_constraints_from_user()

        # 2) 初始觀察
        obs = self._l1_observe(
            task_hint=f"Localize current position. Goal is {self.goal_place_id}. Focus on room/gallery number signs.",
            mode="active",
        )
        self.l2.ingest_observation(obs)

        if _goal_reached(obs, self.goal_place_id, self.goal_conf_threshold):
            self._l1_close()
            return {"status": "success", "reason": "goal_reached_at_start", "final_obs_id": obs.get("obs_id")}

        # 3) 初始一定 replan（會產 plan.json，L3 讀 plan.json）
        self._replan()

        rounds = 0
        while rounds < self.max_rounds:
            rounds += 1

            # 若 L2 覺得不確定 或 belief off-route -> replan
            if self.l2.need_more_perception() or (not self._is_on_track()):
                self._replan()
                continue

            if not self.current_decision:
                self._replan()
                continue

            l1_plan = self.current_decision.get("l1_plan", []) or []

            # 如果這段指令用完了，但還沒到：用 cached plan 讓 L3 再給下一段（不讀 plan.json）
            if self.step_idx >= len(l1_plan):
                self._refresh_decision_without_replan()
                continue

            step = l1_plan[self.step_idx]
            self.step_idx += 1

            kind = step.get("kind")
            if kind == "observe":
                task_hint = step.get("task_hint") or "Localize current position; focus on room number signs."
                obs = self._l1_observe(task_hint=task_hint, mode="active")
                self.l2.ingest_observation(obs)

                if _goal_reached(obs, self.goal_place_id, self.goal_conf_threshold):
                    self._l1_close()
                    return {"status": "success", "reason": "goal_reached", "rounds": rounds, "final_obs_id": obs.get("obs_id")}

                # observe 後立即檢查是否偏離；偏離才 replan
                if self.l2.need_more_perception() or (not self._is_on_track()):
                    self._replan()
                continue

            if kind == "act":
                actions = step.get("actions", []) or []
                if actions:
                    _ = self._l1_act(actions)

                # 若包含前進，強制插入一次 observe 來更新 belief/驗證
                if "front" in actions:
                    obs = self._l1_observe(
                        task_hint="After moving, re-localize. Look for nearest room/gallery number signs or corridor signs.",
                        mode="active",
                    )
                    self.l2.ingest_observation(obs)

                    if _goal_reached(obs, self.goal_place_id, self.goal_conf_threshold):
                        self._l1_close()
                        return {"status": "success", "reason": "goal_reached", "rounds": rounds, "final_obs_id": obs.get("obs_id")}

                    if self.l2.need_more_perception() or (not self._is_on_track()):
                        self._replan()
                continue

            # unknown kind -> force observe + replan
            obs = self._l1_observe(task_hint="Re-localize current position. Focus on room number signs.", mode="active")
            self.l2.ingest_observation(obs)
            self._replan()

        self._l1_close()
        return {"status": "failed", "reason": "max_rounds_exceeded", "rounds": rounds}


# -------------------------
# CLI
# -------------------------

def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", required=True, help="museum_graph.json")
    ap.add_argument("--goal", required=True, help="goal place_id (must exist in graph)")
    ap.add_argument("--user", required=True, help="user preference text")

    ap.add_argument("--web_root", required=True, help="web root folder (contains index.html)")
    ap.add_argument("--entry_html", default="index.html")
    ap.add_argument("--vision_endpoint", required=True, help="HTTP endpoint for VLM/OCR server")
    ap.add_argument("--headless", action="store_true")

    ap.add_argument("--plan_json", default="plan.json")
    ap.add_argument("--decision_json", default="decision.json")
    ap.add_argument("--debug_obs_dir", default="", help="save observed screenshots here")

    ap.add_argument("--max_rounds", type=int, default=200)
    ap.add_argument("--on_track_mass_th", type=float, default=0.60)
    ap.add_argument("--goal_conf_th", type=float, default=0.80)
    ap.add_argument("--model", default="", help="OpenAI model name (or use OPENAI_MODEL env)")

    args = ap.parse_args()

    orch = Orchestrator(
        graph_path=args.graph,
        goal_place_id=args.goal,
        user_text=args.user,
        web_root=args.web_root,
        entry_html=args.entry_html,
        vision_endpoint=args.vision_endpoint,
        headless=args.headless,
        plan_json_path=args.plan_json,
        decision_json_path=args.decision_json,
        debug_obs_dir=args.debug_obs_dir,
        max_rounds=args.max_rounds,
        on_track_mass_threshold=args.on_track_mass_th,
        goal_conf_threshold=args.goal_conf_th,
        model_name=args.model,
    )

    result = orch.run()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass
    main()
