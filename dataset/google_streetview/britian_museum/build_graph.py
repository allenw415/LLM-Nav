import json
import math
import networkx as nx
from pathlib import Path

MAP_JSON = Path("map.json")          # 你的檔案
MANUAL_EDGES = Path("manual_edges.json")  # 你自己定義邊（可不存在）

def make_node_id(name: str, level: str, local_id: str, idx: int) -> str:
    local = local_id.strip() if local_id else f"idx{idx:03d}"
    level = level.strip() if level else "UNKNOWN"
    return f"{name}__{level}__{local}"

def rough_dir_by_latlng(lat1, lng1, lat2, lng2, dominance_ratio=1.2) -> str:
    """
    回傳 N/E/S/W 之一（很粗略）
    dominance_ratio: 越大越偏向只選擇差距更明顯的軸
    """
    dlat = lat2 - lat1
    dlng = lng2 - lng1
    if abs(dlat) >= abs(dlng) * dominance_ratio:
        return "N" if dlat > 0 else "S"
    if abs(dlng) >= abs(dlat) * dominance_ratio:
        return "E" if dlng > 0 else "W"
    # 差不多 → 先用最大的軸決定（或回傳空字串讓你手動）
    if abs(dlat) >= abs(dlng):
        return "N" if dlat > 0 else "S"
    return "E" if dlng > 0 else "W"

def haversine_m(lat1, lon1, lat2, lon2) -> float:
    # 大地距離（公尺）— 若室內 lat/lng 不準，這個當參考即可
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dlmb/2)**2
    return 2*R*math.asin(math.sqrt(a))

# 1) 讀入並攤平節點
data = json.loads(MAP_JSON.read_text(encoding="utf-8"))
G = nx.DiGraph()

# 用來查詢：name -> list[node_id]
name_index = {}

for obj in data["objects"]:
    name = obj["name"]
    for idx, rec in enumerate(obj["records"]):
        node_id = make_node_id(name, rec.get("level",""), rec.get("id",""), idx)
        G.add_node(
            node_id,
            name=name,
            level=rec.get("level",""),
            description=rec.get("description",""),
            lat=rec.get("lat"),
            lng=rec.get("lng"),
            panoId=rec.get("panoId",""),
            heading=rec.get("heading",""),
            zoom=rec.get("zoom",""),
        )
        name_index.setdefault(name, []).append(node_id)

# 輔助：在同樓層、同 name 中找「最接近某點」的 node
def pick_best_target_by_name_level(target_name: str, level: str, lat: float, lng: float):
    candidates = [nid for nid in name_index.get(target_name, []) if G.nodes[nid].get("level","") == level]
    if not candidates:
        candidates = name_index.get(target_name, [])
    if not candidates:
        return None

    best = None
    best_d = float("inf")
    for nid in candidates:
        n = G.nodes[nid]
        if n.get("lat") is None or n.get("lng") is None:
            continue
        d = haversine_m(lat, lng, n["lat"], n["lng"])
        if d < best_d:
            best_d = d
            best = nid
    return best

# 2) 自動建邊：用 nearest_other_place
AUTO_DIR = False  # 想先自動給方向就改 True
for u in list(G.nodes):
    u_attr = G.nodes[u]
    # 找回原始 record 的 nearest_other_place：我們沒直接存，這裡用簡單方式重建（推薦你在 node attr 也存一份）
    # 這段示範：改成直接在轉 json 時就把 nearest_other_place 留在 node attr 最好
    pass

# 如果你想立刻用 nearest_other_place 建邊：建議你在 map.json 的每個 record 保留 nearest_other_place
# 下面改成從原 data 直接走一次：
for obj in data["objects"]:
    name = obj["name"]
    for idx, rec in enumerate(obj["records"]):
        u = make_node_id(name, rec.get("level",""), rec.get("id",""), idx)
        lat, lng = rec.get("lat"), rec.get("lng")
        level = rec.get("level","")
        for target_name in rec.get("nearest_other_place", []):
            v = pick_best_target_by_name_level(target_name, level, lat, lng)
            if v is None:
                continue
            weight = haversine_m(lat, lng, G.nodes[v]["lat"], G.nodes[v]["lng"]) if (lat and lng and G.nodes[v]["lat"] and G.nodes[v]["lng"]) else 1.0
            direction = rough_dir_by_latlng(lat, lng, G.nodes[v]["lat"], G.nodes[v]["lng"]) if AUTO_DIR and lat is not None else ""
            G.add_edge(u, v, dir=direction, weight=weight, source="nearest_other_place")

# 3) 手動覆寫/新增邊（你自己定義東南西北）
if MANUAL_EDGES.exists():
    manual = json.loads(MANUAL_EDGES.read_text(encoding="utf-8"))
    for e in manual:
        u, v = e["u"], e["v"]
        if u not in G.nodes or v not in G.nodes:
            raise ValueError(f"manual edge node not found: {u} -> {v}")
        attrs = {"dir": e.get("dir",""), "source": "manual_override"}
        if "weight" in e: attrs["weight"] = e["weight"]
        if "type" in e: attrs["type"] = e["type"]
        G.add_edge(u, v, **attrs)

# 4) 輸出（GraphML / JSON 都可）
nx.write_graphml(G, "museum_graph.graphml")  # 用 Gephi / Cytoscape 開很方便
print("nodes:", G.number_of_nodes(), "edges:", G.number_of_edges())
