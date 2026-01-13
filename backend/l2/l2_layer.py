# l2_layer.py
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Literal
from heapq import heappush, heappop
from collections import deque, defaultdict


EdgeType = Literal["corridor", "stairs", "elevator", "door", "other"]


# -----------------------------
# Graph schema
# -----------------------------

@dataclass
class Node:
    place_id: str
    name: str = ""
    floor: Optional[int] = None
    tags: List[str] = field(default_factory=list)
    cues: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Edge:
    u: str
    v: str
    cost: float = 1.0
    edge_type: EdgeType = "corridor"
    bidirectional: bool = True
    tags: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)


class MuseumGraph:
    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self.adj: Dict[str, List[Tuple[str, Edge]]] = defaultdict(list)

    @staticmethod
    def from_json(path: str | Path) -> "MuseumGraph":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        g = MuseumGraph()

        for nd in data.get("nodes", []):
            n = Node(
                place_id=nd["place_id"],
                name=nd.get("name", ""),
                floor=nd.get("floor"),
                tags=list(nd.get("tags", [])),
                cues=list(nd.get("cues", [])),
                meta=dict(nd.get("meta", {})),
            )
            g.nodes[n.place_id] = n

        for ed in data.get("edges", []):
            e = Edge(
                u=ed["u"],
                v=ed["v"],
                cost=float(ed.get("cost", 1.0)),
                edge_type=ed.get("edge_type", "corridor"),
                bidirectional=bool(ed.get("bidirectional", True)),
                tags=list(ed.get("tags", [])),
                meta=dict(ed.get("meta", {})),
            )
            g.add_edge(e)

        return g

    def add_edge(self, e: Edge) -> None:
        if e.u not in self.nodes or e.v not in self.nodes:
            raise KeyError(f"Edge references unknown node: {e.u} -> {e.v}")
        self.adj[e.u].append((e.v, e))
        if e.bidirectional:
            rev = Edge(u=e.v, v=e.u, cost=e.cost, edge_type=e.edge_type,
                       bidirectional=e.bidirectional, tags=list(e.tags), meta=dict(e.meta))
            self.adj[e.v].append((e.u, rev))

    def neighbors(self, u: str) -> List[Tuple[str, Edge]]:
        return self.adj.get(u, [])


# -----------------------------
# Constraints
# -----------------------------

@dataclass
class Constraints:
    avoid_edge_types: List[EdgeType] = field(default_factory=list)
    prefer_edge_types: List[EdgeType] = field(default_factory=list)
    blacklist_nodes: List[str] = field(default_factory=list)
    blacklist_tags: List[str] = field(default_factory=list)
    max_candidates: int = 3
    prefer_bonus: float = 0.2
    crowded_penalty: float = 0.0

    def edge_allowed(self, e: Edge) -> bool:
        return e.edge_type not in self.avoid_edge_types

    def node_allowed(self, n: Node) -> bool:
        if n.place_id in self.blacklist_nodes:
            return False
        if any(t in self.blacklist_tags for t in n.tags):
            return False
        return True

    def edge_effective_cost(self, e: Edge) -> float:
        c = float(e.cost)
        if e.edge_type in self.prefer_edge_types:
            c = max(0.001, c * (1.0 - self.prefer_bonus))
        if self.crowded_penalty > 0 and any(t == "crowded" for t in e.tags):
            c += self.crowded_penalty
        return c


# -----------------------------
# Configs
# -----------------------------

@dataclass
class BeliefConfig:
    topk_starts: int = 3
    confident_threshold: float = 0.70
    epsilon: float = 1e-6


@dataclass
class MemoryConfig:
    episodic_maxlen: int = 50


@dataclass
class L2Config:
    belief: BeliefConfig = BeliefConfig()
    memory: MemoryConfig = MemoryConfig()
    waypoint_lookahead: int = 3
    waypoint_min_cues: int = 1
    need_more_perception_threshold: float = 0.45


# -----------------------------
# Memory records
# -----------------------------

@dataclass
class EpisodeRecord:
    t: float
    obs_id: str
    location_hypotheses: List[Dict[str, Any]]
    ocr_text: List[str]
    landmarks: List[str]
    action_feedback: Dict[str, Any]
    console: List[Dict[str, Any]]


@dataclass
class PlanCandidate:
    id: str
    start: str
    goal: str
    path: List[str]
    cost: float
    notes: List[str] = field(default_factory=list)


@dataclass
class Waypoint:
    place_id: str
    expected_cues: List[str] = field(default_factory=list)


@dataclass
class PlanPacket:
    t: float
    goal: str
    constraints: Dict[str, Any]
    belief_topk: List[Dict[str, Any]]
    route_candidates: List[Dict[str, Any]]
    next_verifiable_waypoints: List[Dict[str, Any]]
    need_more_perception: bool
    exploration_hints: List[str] = field(default_factory=list)


class SpatialMapMemoryLayer:
    def __init__(self, graph: MuseumGraph, cfg: Optional[L2Config] = None):
        self.g = graph
        self.cfg = cfg or L2Config()

        n = max(1, len(self.g.nodes))
        self.belief: Dict[str, float] = {pid: 1.0 / n for pid in self.g.nodes}

        self.episodic: deque[EpisodeRecord] = deque(maxlen=self.cfg.memory.episodic_maxlen)

        self.goal: Optional[str] = None
        self.constraints: Constraints = Constraints()

    # -----------------------------
    # Public APIs
    # -----------------------------

    def set_task(self, goal_place_id: str, constraints: Optional[Constraints] = None) -> None:
        if goal_place_id not in self.g.nodes:
            raise KeyError(f"Unknown goal place_id: {goal_place_id}")
        self.goal = goal_place_id
        if constraints is not None:
            self.constraints = constraints

    def ingest_observation(self, obs: Dict[str, Any]) -> None:
        rec = EpisodeRecord(
            t=float(obs.get("t", time.time())),
            obs_id=str(obs.get("obs_id", "")),
            location_hypotheses=list((obs.get("vision", {}) or {}).get("location_hypotheses", []) or []),
            ocr_text=list((obs.get("vision", {}) or {}).get("ocr_text", []) or []),
            landmarks=list((obs.get("vision", {}) or {}).get("landmarks", []) or []),
            action_feedback=dict(obs.get("action_feedback", {}) or {}),
            console=list(obs.get("console", []) or []),
        )
        self.episodic.append(rec)

        self._apply_motion_model(rec.action_feedback)
        self._apply_observation_update(rec.location_hypotheses)

    def belief_topk(self, k: Optional[int] = None) -> List[Dict[str, Any]]:
        k = k or self.cfg.belief.topk_starts
        items = sorted(self.belief.items(), key=lambda kv: kv[1], reverse=True)[: max(1, k)]
        return [{"place_id": pid, "prob": float(p)} for pid, p in items]

    def need_more_perception(self) -> bool:
        top1 = max(self.belief.values()) if self.belief else 0.0
        return top1 < self.cfg.need_more_perception_threshold

    def plan(self) -> PlanPacket:
        if not self.goal:
            raise RuntimeError("Goal not set. Call set_task() first.")

        belief_topk = self.belief_topk(self.cfg.belief.topk_starts)
        top1_prob = belief_topk[0]["prob"] if belief_topk else 0.0
        need_more = top1_prob < self.cfg.need_more_perception_threshold

        if top1_prob >= self.cfg.belief.confident_threshold:
            starts = [belief_topk[0]["place_id"]]
        else:
            starts = [x["place_id"] for x in belief_topk]

        candidates: List[PlanCandidate] = []
        for s in starts:
            if s not in self.g.nodes:
                continue
            if not self.constraints.node_allowed(self.g.nodes[s]):
                continue
            if not self.constraints.node_allowed(self.g.nodes[self.goal]):
                continue

            path, cost, notes = self._dijkstra(s, self.goal, self.constraints)
            if path:
                candidates.append(PlanCandidate(
                    id=f"{s}->{self.goal}",
                    start=s,
                    goal=self.goal,
                    path=path,
                    cost=cost,
                    notes=notes,
                ))

        start_prob = {x["place_id"]: x["prob"] for x in belief_topk}
        candidates.sort(key=lambda c: (c.cost, -start_prob.get(c.start, 0.0)))
        candidates = candidates[: max(1, self.constraints.max_candidates)]

        waypoints: List[Waypoint] = []
        if candidates:
            waypoints = self._select_verifiable_waypoints(candidates[0].path)

        exploration_hints = self._make_exploration_hints() if need_more else []

        return PlanPacket(
            t=time.time(),
            goal=self.goal,
            constraints=asdict(self.constraints),
            belief_topk=belief_topk,
            route_candidates=[asdict(c) for c in candidates],
            next_verifiable_waypoints=[asdict(w) for w in waypoints],
            need_more_perception=need_more,
            exploration_hints=exploration_hints,
        )

    # -----------------------------
    # Internals: belief update
    # -----------------------------

    def _normalize(self, dist: Dict[str, float]) -> Dict[str, float]:
        s = sum(dist.values())
        if s <= 0:
            n = max(1, len(self.g.nodes))
            return {pid: 1.0 / n for pid in self.g.nodes}
        return {k: v / s for k, v in dist.items()}

    def _apply_motion_model(self, action_feedback: Dict[str, Any]) -> None:
        last_action = action_feedback.get("last_action")
        success = action_feedback.get("success")
        if last_action != "front" or success is not True:
            return

        new_belief = {pid: 0.0 for pid in self.g.nodes}
        for u, p_u in self.belief.items():
            if p_u <= 0:
                continue
            nbrs = [v for v, e in self.g.neighbors(u) if self.constraints.edge_allowed(e)]
            if not nbrs:
                new_belief[u] += p_u
                continue
            w = 1.0 / len(nbrs)
            for v in nbrs:
                new_belief[v] += p_u * w

        self.belief = self._normalize(new_belief)

    def _apply_observation_update(self, hyps: List[Dict[str, Any]]) -> None:
        eps = float(self.cfg.belief.epsilon)
        likelihood = {pid: eps for pid in self.g.nodes}

        for h in hyps or []:
            pid = h.get("place_id")
            if pid in likelihood:
                conf = float(h.get("confidence", 0.0))
                likelihood[pid] += max(0.0, conf) ** 2

        post = {pid: self.belief.get(pid, 0.0) * likelihood[pid] for pid in self.g.nodes}
        self.belief = self._normalize(post)

    # -----------------------------
    # Internals: planning
    # -----------------------------

    def _dijkstra(self, start: str, goal: str, cons: Constraints) -> Tuple[List[str], float, List[str]]:
        if start == goal:
            return [start], 0.0, ["already_at_goal"]

        if start not in self.g.nodes or goal not in self.g.nodes:
            return [], math.inf, ["unknown_node"]
        if not cons.node_allowed(self.g.nodes[start]) or not cons.node_allowed(self.g.nodes[goal]):
            return [], math.inf, ["node_blocked_by_constraints"]

        dist: Dict[str, float] = {start: 0.0}
        prev: Dict[str, Optional[str]] = {start: None}
        prev_edge: Dict[str, Optional[Edge]] = {start: None}
        pq: List[Tuple[float, str]] = []
        heappush(pq, (0.0, start))

        while pq:
            d, u = heappop(pq)
            if d != dist.get(u, math.inf):
                continue
            if u == goal:
                break

            for v, e in self.g.neighbors(u):
                if v not in self.g.nodes:
                    continue
                if not cons.edge_allowed(e):
                    continue
                if not cons.node_allowed(self.g.nodes[v]):
                    continue

                nd = d + cons.edge_effective_cost(e)
                if nd < dist.get(v, math.inf):
                    dist[v] = nd
                    prev[v] = u
                    prev_edge[v] = e
                    heappush(pq, (nd, v))

        if goal not in dist:
            return [], math.inf, ["no_path_under_constraints"]

        path: List[str] = []
        notes: List[str] = []
        cur: Optional[str] = goal
        while cur is not None:
            path.append(cur)
            pe = prev_edge.get(cur)
            if pe:
                if pe.edge_type == "stairs":
                    notes.append("includes stairs")
                if pe.edge_type == "elevator":
                    notes.append("uses elevator")
            cur = prev.get(cur)
        path.reverse()
        notes = sorted(set(notes))
        return path, float(dist[goal]), notes

    def _select_verifiable_waypoints(self, path: List[str]) -> List[Waypoint]:
        out: List[Waypoint] = []
        look = min(len(path), max(1, self.cfg.waypoint_lookahead))
        for pid in path[:look]:
            n = self.g.nodes.get(pid)
            if not n:
                continue
            if len(n.cues) >= self.cfg.waypoint_min_cues:
                out.append(Waypoint(place_id=pid, expected_cues=list(n.cues)))

        if not out and len(path) >= 2:
            n2 = self.g.nodes.get(path[1])
            if n2:
                out.append(Waypoint(place_id=n2.place_id, expected_cues=list(n2.cues)))
        return out

    def _make_exploration_hints(self) -> List[str]:
        hints: List[str] = []
        if self.episodic:
            last = self.episodic[-1]
            if last.ocr_text:
                hints.append(f"Last OCR snippets: {' | '.join(last.ocr_text[:6])}")
            if last.landmarks:
                hints.append(f"Last landmarks: {' | '.join(last.landmarks[:6])}")

        hints.extend([
            "Try a 4-direction scan (turn right 3 times), capture each view.",
            "Try pitch sweep (60/90/120) to catch hanging signs or low plaques.",
            "Try zoom in once to read small room number signs.",
            "If still uncertain, move forward one step then re-observe.",
        ])
        return hints
