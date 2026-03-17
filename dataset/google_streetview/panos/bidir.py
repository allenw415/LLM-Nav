import json
from pathlib import Path
from typing import Dict, Any, Tuple, Set, Optional


def norm_heading(h: Optional[float]) -> Optional[float]:
    if h is None:
        return None
    try:
        return float(h) % 360.0
    except Exception:
        return None


def reverse_heading(h: Optional[float]) -> Optional[float]:
    h2 = norm_heading(h)
    if h2 is None:
        return None
    return (h2 + 180.0) % 360.0


def link_key(pano: str) -> str:
    # 以目標 pano 當作去重 key（同一對 pano 最多一條 link）
    return pano


def make_bidirectional_links(panos: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, int]]:
    """
    panos: { node_id: { panoID, lat, lng, links:[{panoID, heading, description}], floor } }

    回傳：
      - 更新後的 panos
      - stats: 補了幾條、跳過幾條、找不到目標幾條...
    """
    # 建立快速查詢：每個 node 已有的 outgoing 目標集合
    outgoing: Dict[str, Set[str]] = {}
    for u, urec in panos.items():
        lkset: Set[str] = set()
        for lk in (urec.get("links") or []):
            v = lk.get("panoID")
            if isinstance(v, str) and v:
                lkset.add(link_key(v))
        outgoing[u] = lkset

    stats = {
        "original_edges": 0,
        "added_reverse_edges": 0,
        "skipped_already_bidirectional": 0,
        "skipped_target_missing": 0,
        "skipped_self_loop": 0,
    }

    # 逐邊補反向
    for u, urec in panos.items():
        links = urec.get("links") or []
        if not isinstance(links, list):
            continue

        for lk in links:
            if not isinstance(lk, dict):
                continue
            v = lk.get("panoID")
            if not isinstance(v, str) or not v:
                continue

            stats["original_edges"] += 1

            if v == u:
                stats["skipped_self_loop"] += 1
                continue

            if v not in panos:
                # 你的資料集沒有這個 pano 節點，沒辦法補
                stats["skipped_target_missing"] += 1
                continue

            # 檢查 v 是否已經指回 u
            if link_key(u) in outgoing.get(v, set()):
                stats["skipped_already_bidirectional"] += 1
                continue

            # 補反向 link：v -> u
            rev = {
                "panoID": u,
                "heading": reverse_heading(lk.get("heading")),
                "description": None,
            }
            panos[v].setdefault("links", [])
            if not isinstance(panos[v]["links"], list):
                panos[v]["links"] = []

            panos[v]["links"].append(rev)
            outgoing.setdefault(v, set()).add(link_key(u))
            stats["added_reverse_edges"] += 1

    return panos, stats


if __name__ == "__main__":
    in_path = Path("flatten_panos.json")
    out_path = Path("bidir_panos.json")

    panos = json.loads(in_path.read_text(encoding="utf-8"))
    if not isinstance(panos, dict):
        raise ValueError("輸入 JSON 不是 dict 格式（預期 {panoId: {...}}）")

    panos2, stats = make_bidirectional_links(panos)

    out_path.write_text(json.dumps(panos2, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {out_path} ✅")
    print("Stats:", json.dumps(stats, ensure_ascii=False, indent=2))
