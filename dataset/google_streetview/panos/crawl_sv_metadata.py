import json
import math
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

from playwright.sync_api import sync_playwright

LatLng = Tuple[float, float]  # (lat, lng)


# ----------------------------
# Key loaders
# ----------------------------
def load_gmaps_api_key_from_env_js(env_js_path: str = ".env.js") -> str:
    text = Path(env_js_path).read_text(encoding="utf-8")
    m = re.search(r'window\.GMAPS_API_KEY\s*=\s*["\']([^"\']+)["\']\s*;?', text)
    if not m:
        raise ValueError(
            f'找不到 window.GMAPS_API_KEY 於 {env_js_path}。\n'
            '請確保格式像：window.GMAPS_API_KEY = "YOUR_KEY";'
        )
    return m.group(1).strip()


def load_polygon_from_polygons_json(
    polygons_json_path: str = "polygons.json",
    placemark_name: str = "british museum",
) -> List[LatLng]:
    """
    讀取你 parse_kml_polygon.py 產生的 polygons.json，並依 placemark_name 取出 outer polygon。
    polygons.json 結構預期為：
      {"polygons":[{"name":"...", "outer":[[lat,lng],...], "inners":[...]}]}
    """
    data = json.loads(Path(polygons_json_path).read_text(encoding="utf-8"))
    polys = data.get("polygons", [])
    if not polys:
        raise ValueError(f"{polygons_json_path} 裡找不到 polygons。")

    # 依 name 找
    target = None
    for p in polys:
        if str(p.get("name", "")).strip().lower() == placemark_name.strip().lower():
            target = p
            break

    if target is None:
        # 如果只有一個 polygon，就直接用它（比較貼心）
        if len(polys) == 1:
            target = polys[0]
        else:
            names = [p.get("name") for p in polys]
            raise ValueError(
                f'找不到 name="{placemark_name}" 的 polygon。可用的 names: {names}'
            )

    outer = target.get("outer")
    if not outer or len(outer) < 3:
        raise ValueError(f'polygon "{target.get("name")}" 的 outer 不合法或點數不足。')

    # 轉成 List[Tuple[lat,lng]]
    polygon_outer: List[LatLng] = [(float(lat), float(lng)) for lat, lng in outer]
    return polygon_outer


# ----------------------------
# Geometry helpers
# ----------------------------
def haversine_m(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    R = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lng2 - lng1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlmb / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def point_in_polygon(lat: float, lng: float, polygon: List[LatLng]) -> bool:
    """
    Ray casting (odd-even rule).
    polygon: [(lat, lng), ...] 允許首尾相同/不相同
    """
    if len(polygon) < 3:
        return False

    # 移除重複最後點（若首尾相同）
    poly = polygon[:-1] if polygon[0] == polygon[-1] else polygon
    n = len(poly)

    x = lng
    y = lat
    inside = False

    for i in range(n):
        y1, x1 = poly[i][0], poly[i][1]
        y2, x2 = poly[(i + 1) % n][0], poly[(i + 1) % n][1]

        # y 是否在 (y1,y2] 的跨越區間
        if (y1 > y) != (y2 > y):
            # 計算水平射線與邊的交點 x_intersect
            x_intersect = (x2 - x1) * (y - y1) / (y2 - y1 + 1e-15) + x1
            if x < x_intersect:
                inside = not inside

    return inside


# ----------------------------
# Google Maps JS (StreetViewService)
# ----------------------------
HTML_TEMPLATE = """<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <title>SV Crawler</title>
    <script>
      window.__SV_READY__ = false;
      window.__SV_ERROR__ = null;
      window.__SV_SERVICE__ = null;

      function __initSV__() {
        try {
          window.__SV_SERVICE__ = new google.maps.StreetViewService();
          window.__SV_READY__ = true;
        } catch (e) {
          window.__SV_ERROR__ = String(e);
        }
      }
    </script>
    <script async defer src="https://maps.googleapis.com/maps/api/js?key=__API_KEY__&callback=__initSV__"></script>
  </head>
  <body></body>
</html>
"""


def build_page_html(api_key: str) -> str:
    return HTML_TEMPLATE.replace("__API_KEY__", api_key)


def js_get_panorama_by_location(page, lat: float, lng: float, radius_m: int) -> Dict[str, Any]:
    return page.evaluate(
        """async ({lat, lng, radius}) => {
          if (!window.__SV_READY__) throw new Error("StreetViewService not ready");
          const sv = window.__SV_SERVICE__;

          const res = await new Promise((resolve) => {
            sv.getPanorama({ location: {lat, lng}, radius }, (data, status) => {
              resolve({ data, status });
            });
          });

          const d = res.data;
          return {
            status: res.status || null,
            data: d ? {
              pano: d.location?.pano ?? null,
              lat: d.location?.latLng ? d.location.latLng.lat() : null,
              lng: d.location?.latLng ? d.location.latLng.lng() : null,
              imageDate: d.imageDate || null,
              links: (d.links || []).map(x => ({
                pano: x.pano || null,
                heading: (typeof x.heading === "number") ? x.heading : null,
                description: x.description || null
              }))
            } : null
          };
        }""",
        {"lat": lat, "lng": lng, "radius": radius_m},
    )


def js_get_panorama_by_pano(page, pano_id: str) -> Dict[str, Any]:
    return page.evaluate(
        """async ({pano}) => {
          if (!window.__SV_READY__) throw new Error("StreetViewService not ready");
          const sv = window.__SV_SERVICE__;

          const res = await new Promise((resolve) => {
            sv.getPanorama({ pano }, (data, status) => {
              resolve({ data, status });
            });
          });

          const d = res.data;
          return {
            status: res.status || null,
            data: d ? {
              pano: d.location?.pano ?? null,
              lat: d.location?.latLng ? d.location.latLng.lat() : null,
              lng: d.location?.latLng ? d.location.latLng.lng() : null,
              imageDate: d.imageDate || null,
              links: (d.links || []).map(x => ({
                pano: x.pano || null,
                heading: (typeof x.heading === "number") ? x.heading : null,
                description: x.description || null
              }))
            } : null
          };
        }""",
        {"pano": pano_id},
    )


# ----------------------------
# Crawler (polygon boundary)
# ----------------------------
def crawl_streetview_panos_in_polygon(
    seed_lat: float,
    seed_lng: float,
    *,
    polygon_outer: List[LatLng],
    search_radius_m: int = 120,
    max_nodes: int = 400,
    env_js_path: str = ".env.js",
    headless: bool = True,
) -> Dict[str, Any]:
    api_key = load_gmaps_api_key_from_env_js(env_js_path)
    html = build_page_html(api_key)

    results: Dict[str, Any] = {}
    queue: List[str] = []
    visited: set[str] = set()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        page = browser.new_page()
        page.set_content(html, wait_until="domcontentloaded")

        page.wait_for_function("window.__SV_READY__ === true || window.__SV_ERROR__ !== null", timeout=30000)
        err = page.evaluate("window.__SV_ERROR__")
        if err:
            browser.close()
            raise RuntimeError(f"StreetViewService 初始化失敗：{err}")

        # 找 seed pano
        seed = js_get_panorama_by_location(page, seed_lat, seed_lng, search_radius_m)
        if seed.get("status") != "OK" or not seed.get("data") or not seed["data"].get("pano"):
            browser.close()
            return {"seed": {"lat": seed_lat, "lng": seed_lng}, "status": seed.get("status"), "panos": {}}

        queue.append(seed["data"]["pano"])

        while queue and len(results) < max_nodes:
            pano = queue.pop(0)
            if not pano or pano in visited:
                continue
            visited.add(pano)

            res = js_get_panorama_by_pano(page, pano)
            if res.get("status") != "OK" or not res.get("data"):
                continue

            d = res["data"]
            lat, lng = d.get("lat"), d.get("lng")
            if lat is None or lng is None:
                continue

            if not point_in_polygon(lat, lng, polygon_outer):
                # 不在 polygon 內：不收、不展開
                continue

            d["inside_polygon"] = True
            d["dist_m_from_seed"] = haversine_m(seed_lat, seed_lng, lat, lng)
            results[pano] = d

            # 只對 polygon 內的節點展開 links
            for lk in d.get("links", []):
                nxt = lk.get("pano")
                if nxt and nxt not in visited:
                    queue.append(nxt)

        browser.close()

    return {
        "seed": {"lat": seed_lat, "lng": seed_lng},
        "search_radius_m": search_radius_m,
        "max_nodes": max_nodes,
        "polygon_outer": polygon_outer,
        "panos": results,
    }


if __name__ == "__main__":
    # 直接讀 polygons.json
    polygon_outer = load_polygon_from_polygons_json(
        polygons_json_path="polygons.json",
        placemark_name="british museum",
    )

    # seed 建議放在館區內/附近；這裡用你 polygon 大概中心附近
    out = crawl_streetview_panos_in_polygon(
        seed_lat=51.5192548,
        seed_lng=-0.1280553,
        polygon_outer=polygon_outer,
        search_radius_m=200,
        max_nodes=1000000000000,  # 不限制數量（實際上會受限於 polygon 範圍和 Google API）
        env_js_path=".env.js",
        headless=True,
    )

    Path("streetview_panos.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(f"Saved: streetview_panos.json (panos={len(out['panos'])})")
