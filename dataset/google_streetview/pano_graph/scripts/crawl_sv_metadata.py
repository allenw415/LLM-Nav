import argparse
import json
import math
import re
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

LatLng = Tuple[float, float]  # (lat, lng)

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

GET_PANORAMA_JS = """async ({ request }) => {
  if (!window.__SV_READY__) throw new Error("StreetViewService not ready");
  const sv = window.__SV_SERVICE__;

  const res = await new Promise((resolve) => {
    sv.getPanorama(request, (data, status) => {
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
}"""


@dataclass(frozen=True)
class CrawlConfig:
    seed_lat: float
    seed_lng: float
    polygon_outer: List[LatLng]
    search_radius_m: int = 120
    max_nodes: int = 400
    env_js_path: str = ".env.js"
    headless: bool = True
    ready_timeout_ms: int = 30000


@dataclass(frozen=True)
class RunConfig:
    polygons_json_path: str = "polygons.json"
    placemark_name: str = "british museum"
    output_path: str = "streetview_panos.json"


def load_gmaps_api_key_from_env_js(env_js_path: str = ".env.js") -> str:
    text = Path(env_js_path).read_text(encoding="utf-8")
    match = re.search(r'window\.GMAPS_API_KEY\s*=\s*["\']([^"\']+)["\']\s*;?', text)
    if not match:
        raise ValueError(
            f'找不到 window.GMAPS_API_KEY 於 {env_js_path}。\n'
            '請確保格式像：window.GMAPS_API_KEY = "YOUR_KEY";'
        )
    return match.group(1).strip()


def load_polygon_from_polygons_json(
    polygons_json_path: str = "polygons.json",
    placemark_name: str = "british museum",
) -> List[LatLng]:
    """
    讀取 parse_kml_polygon.py 產生的 polygons.json，並依 placemark_name 取出 outer polygon。
    polygons.json 結構預期為：
      {"polygons":[{"name":"...", "outer":[[lat,lng],...], "inners":[...]}]}
    """
    data = json.loads(Path(polygons_json_path).read_text(encoding="utf-8"))
    polygons = data.get("polygons", [])
    if not polygons:
        raise ValueError(f"{polygons_json_path} 裡找不到 polygons。")

    target = None
    for polygon in polygons:
        if str(polygon.get("name", "")).strip().lower() == placemark_name.strip().lower():
            target = polygon
            break

    if target is None:
        if len(polygons) == 1:
            target = polygons[0]
        else:
            names = [polygon.get("name") for polygon in polygons]
            raise ValueError(
                f'找不到 name="{placemark_name}" 的 polygon。可用的 names: {names}'
            )

    outer = target.get("outer")
    if not outer or len(outer) < 3:
        raise ValueError(f'polygon "{target.get("name")}" 的 outer 不合法或點數不足。')

    return [(float(lat), float(lng)) for lat, lng in outer]


def haversine_m(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    radius_m = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlng / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return radius_m * c


def point_in_polygon(lat: float, lng: float, polygon: List[LatLng]) -> bool:
    """
    Ray casting (odd-even rule).
    polygon: [(lat, lng), ...] 允許首尾相同/不相同
    """
    if len(polygon) < 3:
        return False

    normalized_polygon = polygon[:-1] if polygon[0] == polygon[-1] else polygon
    inside = False
    x = lng
    y = lat

    for index, (y1, x1) in enumerate(normalized_polygon):
        y2, x2 = normalized_polygon[(index + 1) % len(normalized_polygon)]
        if (y1 > y) != (y2 > y):
            x_intersect = (x2 - x1) * (y - y1) / (y2 - y1 + 1e-15) + x1
            if x < x_intersect:
                inside = not inside

    return inside


def build_page_html(api_key: str) -> str:
    return HTML_TEMPLATE.replace("__API_KEY__", api_key)


class StreetViewServiceSession:
    def __init__(self, api_key: str, *, headless: bool = True, ready_timeout_ms: int = 30000):
        self.api_key = api_key
        self.headless = headless
        self.ready_timeout_ms = ready_timeout_ms
        self._playwright = None
        self._browser: Optional[Any] = None
        self._page: Optional[Any] = None

    def __enter__(self) -> "StreetViewServiceSession":
        try:
            from playwright.sync_api import sync_playwright
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "需要先安裝 playwright 才能執行爬蟲。請先執行: pip install playwright"
            ) from exc

        try:
            self._playwright = sync_playwright().start()
            self._browser = self._playwright.chromium.launch(headless=self.headless)
            self._page = self._browser.new_page()
            self._page.set_content(build_page_html(self.api_key), wait_until="domcontentloaded")
            self._page.wait_for_function(
                "window.__SV_READY__ === true || window.__SV_ERROR__ !== null",
                timeout=self.ready_timeout_ms,
            )
            error = self._page.evaluate("window.__SV_ERROR__")
            if error:
                raise RuntimeError(f"StreetViewService 初始化失敗：{error}")
            return self
        except Exception:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._browser is not None:
            self._browser.close()
        if self._playwright is not None:
            self._playwright.stop()
        self._browser = None
        self._page = None
        self._playwright = None

    @property
    def page(self) -> Any:
        if self._page is None:
            raise RuntimeError("StreetViewServiceSession 尚未初始化。")
        return self._page

    def get_panorama_by_location(self, lat: float, lng: float, radius_m: int) -> Dict[str, Any]:
        return self._get_panorama({"location": {"lat": lat, "lng": lng}, "radius": radius_m})

    def get_panorama_by_pano(self, pano_id: str) -> Dict[str, Any]:
        return self._get_panorama({"pano": pano_id})

    def _get_panorama(self, request: Dict[str, Any]) -> Dict[str, Any]:
        return self.page.evaluate(GET_PANORAMA_JS, {"request": request})


def crawl_streetview_panos_in_polygon(
    seed_lat: float,
    seed_lng: float,
    *,
    polygon_outer: List[LatLng],
    search_radius_m: int = 120,
    max_nodes: int = 400,
    env_js_path: str = ".env.js",
    headless: bool = True,
    ready_timeout_ms: int = 30000,
) -> Dict[str, Any]:
    config = CrawlConfig(
        seed_lat=seed_lat,
        seed_lng=seed_lng,
        polygon_outer=polygon_outer,
        search_radius_m=search_radius_m,
        max_nodes=max_nodes,
        env_js_path=env_js_path,
        headless=headless,
        ready_timeout_ms=ready_timeout_ms,
    )
    return crawl_with_config(config)


def crawl_with_config(config: CrawlConfig) -> Dict[str, Any]:
    api_key = load_gmaps_api_key_from_env_js(config.env_js_path)
    results: Dict[str, Any] = {}
    visited: set[str] = set()
    queue: Deque[str] = deque()

    with StreetViewServiceSession(
        api_key,
        headless=config.headless,
        ready_timeout_ms=config.ready_timeout_ms,
    ) as session:
        seed = session.get_panorama_by_location(
            config.seed_lat,
            config.seed_lng,
            config.search_radius_m,
        )
        seed_pano_id = _get_seed_pano_id(seed)
        if seed_pano_id is None:
            return {
                "seed": {"lat": config.seed_lat, "lng": config.seed_lng},
                "status": seed.get("status"),
                "panos": {},
            }

        queue.append(seed_pano_id)

        while queue and len(results) < config.max_nodes:
            pano_id = queue.popleft()
            if not pano_id or pano_id in visited:
                continue
            visited.add(pano_id)

            panorama = session.get_panorama_by_pano(pano_id)
            record = panorama.get("data")
            if panorama.get("status") != "OK" or not record:
                continue

            lat = record.get("lat")
            lng = record.get("lng")
            if lat is None or lng is None:
                continue

            if not point_in_polygon(lat, lng, config.polygon_outer):
                continue

            record["inside_polygon"] = True
            record["dist_m_from_seed"] = haversine_m(config.seed_lat, config.seed_lng, lat, lng)
            results[pano_id] = record

            for link in record.get("links", []):
                next_pano_id = link.get("pano")
                if next_pano_id and next_pano_id not in visited:
                    queue.append(next_pano_id)

    return {
        "seed": {"lat": config.seed_lat, "lng": config.seed_lng},
        "search_radius_m": config.search_radius_m,
        "max_nodes": config.max_nodes,
        "polygon_outer": config.polygon_outer,
        "panos": results,
    }


def _get_seed_pano_id(seed: Dict[str, Any]) -> Optional[str]:
    data = seed.get("data")
    if seed.get("status") != "OK" or not data:
        return None
    pano_id = data.get("pano")
    return pano_id if isinstance(pano_id, str) and pano_id else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crawl Street View panoramas inside a polygon.")
    parser.add_argument("--polygons-json-path", default="polygons.json")
    parser.add_argument("--placemark-name", default="british museum")
    parser.add_argument("--seed-lat", type=float, default=51.5192548)
    parser.add_argument("--seed-lng", type=float, default=-0.1280553)
    parser.add_argument("--search-radius-m", type=int, default=200)
    parser.add_argument("--max-nodes", type=int, default=1_000_000_000_000)
    parser.add_argument("--env-js-path", default=".env.js")
    parser.add_argument("--output-path", default="streetview_panos.json")
    parser.add_argument("--ready-timeout-ms", type=int, default=30000)
    parser.add_argument(
        "--show-browser",
        action="store_true",
        help="以非 headless 模式啟動瀏覽器，方便除錯。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_config = RunConfig(
        polygons_json_path=args.polygons_json_path,
        placemark_name=args.placemark_name,
        output_path=args.output_path,
    )
    polygon_outer = load_polygon_from_polygons_json(
        polygons_json_path=run_config.polygons_json_path,
        placemark_name=run_config.placemark_name,
    )
    crawl_config = CrawlConfig(
        seed_lat=args.seed_lat,
        seed_lng=args.seed_lng,
        polygon_outer=polygon_outer,
        search_radius_m=args.search_radius_m,
        max_nodes=args.max_nodes,
        env_js_path=args.env_js_path,
        headless=not args.show_browser,
        ready_timeout_ms=args.ready_timeout_ms,
    )

    output = crawl_with_config(crawl_config)
    output_path = Path(run_config.output_path)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved: {output_path} (panos={len(output.get('panos', {}))})")


if __name__ == "__main__":
    main()
