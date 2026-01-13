# control.py
from __future__ import annotations

import contextlib
import io
import threading
import time
from dataclasses import dataclass
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import List, Optional, Literal, Dict, Any, Tuple

from playwright.sync_api import sync_playwright, Browser, Page, TimeoutError as PlaywrightTimeoutError

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


ConsoleKind = Literal["log", "info", "warning", "error", "debug", "pageerror"]


@dataclass
class ConsoleEntry:
    ts: float
    kind: str
    text: str


class _QuietHandler(SimpleHTTPRequestHandler):
    # 安靜一點，避免 http.server 一直印 log
    def log_message(self, format: str, *args) -> None:
        return


def _start_static_server(directory: Path, host: str = "127.0.0.1", port: int = 0):
    """
    用 http.server 將 web_root serve 起來。
    port=0 會由 OS 自動分配可用埠，避免 race condition。
    """
    directory = directory.resolve()

    def handler(*args, **kwargs):
        return _QuietHandler(*args, directory=str(directory), **kwargs)

    httpd = ThreadingHTTPServer((host, port), handler)
    real_port = int(httpd.server_address[1])

    th = threading.Thread(target=httpd.serve_forever, daemon=True)
    th.start()
    return httpd, th, real_port


class StreetViewController:
    """
    用 Playwright 控制你的 Street View Graph Navigator 網頁。

    功能：
    1) screenshot_pano(): 截圖 #pano 並回傳 PNG bytes
    2) action_*(): 點按/鍵盤操作前進、轉向、俯仰、縮放
    3) get_console(): 取回 console / pageerror 訊息
    4) get_state(): 獲取當前導航狀態
    5) goto_node(): 跳轉到指定節點
    """

    def __init__(
        self,
        web_root: str | Path,
        entry_html: str = "index.html",
        headless: bool = True,
        viewport: Tuple[int, int] = (1280, 720),
        slow_mo_ms: int = 0,
    ):
        self.web_root = Path(web_root)
        self.entry_html = entry_html
        self.headless = headless
        self.viewport = viewport
        self.slow_mo_ms = slow_mo_ms

        self._httpd = None
        self._server_thread = None
        self._port = None

        self._pw = None
        self._browser: Optional[Browser] = None
        self._page: Optional[Page] = None

        self._console: List[ConsoleEntry] = []
        self._console_lock = threading.Lock()

    # ---------- lifecycle ----------

    def start(self) -> None:
        """啟動本機 server + 開瀏覽器 + 進入頁面。"""
        if not self.web_root.exists():
            raise FileNotFoundError(f"web_root not found: {self.web_root}")

        try:
            # 1) start local server
            self._httpd, self._server_thread, self._port = _start_static_server(self.web_root)

            url = f"http://127.0.0.1:{self._port}/{self.entry_html}"

            # 2) start playwright
            self._pw = sync_playwright().start()
            self._browser = self._pw.chromium.launch(headless=self.headless, slow_mo=self.slow_mo_ms)
            ctx = self._browser.new_context(viewport={"width": self.viewport[0], "height": self.viewport[1]})
            self._page = ctx.new_page()

            # 3) hook console
            self._page.on("console", self._on_console)
            self._page.on("pageerror", self._on_page_error)

            # 4) goto
            self._page.goto(url, wait_until="domcontentloaded")

            # 5) wait ready
            self._wait_ready()

        except Exception:
            # 任何一步失敗就確保資源釋放乾淨
            self.close()
            raise

    def close(self) -> None:
        """關閉瀏覽器與本機 server。"""
        with contextlib.suppress(Exception):
            if self._page:
                self._page.context.close()
        with contextlib.suppress(Exception):
            if self._browser:
                self._browser.close()
        with contextlib.suppress(Exception):
            if self._pw:
                self._pw.stop()

        self._page = None
        self._browser = None
        self._pw = None

        with contextlib.suppress(Exception):
            if self._httpd:
                self._httpd.shutdown()
                self._httpd.server_close()

        self._httpd = None
        self._server_thread = None
        self._port = None

    def __enter__(self) -> "StreetViewController":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def is_alive(self) -> bool:
        """粗略檢查 controller 是否仍可用。"""
        return self._page is not None and self._browser is not None

    # ---------- internal ----------

    def _on_console(self, msg) -> None:
        with self._console_lock:
            self._console.append(ConsoleEntry(ts=time.time(), kind=str(msg.type), text=str(msg.text)))

    def _on_page_error(self, err) -> None:
        with self._console_lock:
            self._console.append(ConsoleEntry(ts=time.time(), kind="pageerror", text=str(err)))

    def _wait_ready(self, timeout_ms: int = 30000) -> None:
        if not self._page:
            raise RuntimeError("Controller not started")

        # 等 DOM 元素存在
        self._page.wait_for_selector("#pano", timeout=timeout_ms)

        # 等 google maps loaded
        self._page.wait_for_function("() => !!(window.google && window.google.maps)", timeout=timeout_ms)

        # 可選：如果你想要更嚴格，確保按鈕也存在
        self._page.wait_for_selector("#moveFront", timeout=timeout_ms)

    def _click(self, selector: str) -> None:
        if not self._page:
            raise RuntimeError("Controller not started")
        self._page.locator(selector).click()

    def _press(self, key: str) -> None:
        if not self._page:
            raise RuntimeError("Controller not started")
        self._page.keyboard.press(key)

    def _wait(self, ms: int) -> None:
        if ms > 0 and self._page:
            self._page.wait_for_timeout(ms)

    def _wait_for_function(self, expression: str, arg: Any = None, timeout_ms: int = 3000) -> bool:
        """
        等待某個 JS 條件成立。成功回 True，逾時回 False。
        expression 建議寫成 (arg) => boolean 形式。
        """
        if not self._page:
            raise RuntimeError("Controller not started")
        try:
            self._page.wait_for_function(expression, arg=arg, timeout=timeout_ms)
            return True
        except PlaywrightTimeoutError:
            return False

    # ---------- public: screenshot ----------

    def screenshot_pano(self) -> bytes:
        """截圖 #pano，回傳 PNG bytes。"""
        if not self._page:
            raise RuntimeError("Controller not started")
        return self._page.locator("#pano").screenshot(type="png")

    def screenshot_pano_image(self) -> "Image.Image":
        """
        截圖並回傳 PIL Image 物件。
        Raises:
            ImportError: 如果沒有安裝 Pillow
        """
        if not HAS_PIL:
            raise ImportError("需要安裝 Pillow: pip install Pillow")

        png_bytes = self.screenshot_pano()
        img = Image.open(io.BytesIO(png_bytes))
        return img.copy()

    # ---------- optional helpers ----------

    def eval(self, expression: str, arg: Any = None) -> Any:
        """
        直接執行頁面內 JS。
        盡量用 arg 傳參，避免用 f-string 拼接造成注入或字元問題。
        """
        if not self._page:
            raise RuntimeError("Controller not started")
        return self._page.evaluate(expression, arg)

    # ---------- public: state ----------

    def get_state(self) -> Dict[str, Any]:
        """
        獲取當前導航狀態（盡量不依賴猜測的全域變數）。
        若你之後在 app.js 暴露 window.SV_API.getState()，這裡會自動優先使用。
        """
        return self.eval(
            """
            () => {
                // 1) Prefer a stable API if exists
                if (window.SV_API && typeof window.SV_API.getState === 'function') {
                    const s = window.SV_API.getState();
                    return s ?? {};
                }

                // 2) Fallback to existing globals (best-effort)
                const st = window.state || {};
                const headingIdx = (typeof st.headingIdx === 'number') ? st.headingIdx : 0;

                const headingDegArr = window.HEADING_DEGS;
                const headingDeg = Array.isArray(headingDegArr) ? (headingDegArr[headingIdx] ?? null) : null;

                return {
                    nodeId: st.nodeId ?? null,
                    headingIdx,
                    headingDeg,
                    pitchMode: st.pitchMode ?? 90,
                    zoom: st.zoom ?? 1
                };
            }
            """
        )

    def get_status_text(self) -> str:
        """獲取狀態列文字"""
        txt = self.eval("() => document.getElementById('status')?.textContent || ''")
        return str(txt).strip()

    def get_available_nodes(self) -> List[Dict[str, Any]]:
        """
        取得可選節點列表：優先讀 window.nodes（若存在），否則讀 DOM 的 #nodeSelect。
        """
        return self.eval(
            """
            () => {
                if (Array.isArray(window.nodes) && window.nodes.length) {
                    return window.nodes.map(n => ({
                        nodeId: n.nodeId ?? n.id ?? n.name ?? null,
                        name: n.name ?? null,
                        description: n.description ?? null,
                        floor: n.floor ?? null,
                        lat: n.lat ?? null,
                        lng: n.lng ?? null
                    }));
                }

                const sel = document.getElementById('nodeSelect');
                if (!sel) return [];

                return Array.from(sel.options).map(o => ({
                    nodeId: o.value,
                    text: o.textContent || ''
                }));
            }
            """
        )

    def can_move_front(self) -> bool:
        """
        檢查當前朝向前方是否可移動（依賴你頁面內 window.adj / window.state 的資料結構）。
        """
        result = self.eval(
            """
            () => {
                const state = window.state;
                const adj = window.adj;

                if (!state?.nodeId || !adj) return false;

                const m = adj.get ? adj.get(state.nodeId) : null; // if adj is a Map
                if (!m) return false;

                const absDirIdx = (state.headingIdx ?? 0) % 4;
                return m.has ? m.has(absDirIdx) : false;
            }
            """
        )
        return bool(result)

    # ---------- public: navigation ----------

    def goto_node(
        self,
        node_id: str,
        arrive_heading_idx: Optional[int] = None,
        timeout_ms: int = 8000,
        settle_ms: int = 200,
    ) -> Dict[str, Any]:
        """
        直接跳轉到指定節點（安全傳參，不用字串拼接）。
        會等待 nodeId 變成指定節點（若 app.js 有同步更新 state）。
        """
        if not self._page:
            raise RuntimeError("Controller not started")

        payload = {"nodeId": node_id, "idx": arrive_heading_idx}

        # 1) call goToNode
        self._page.evaluate(
            """
            ({nodeId, idx}) => {
                const opts = (idx === null || idx === undefined) ? {} : { arriveHeadingIdx: idx };

                const fn = (window.SV_API && typeof window.SV_API.goToNode === 'function')
                    ? window.SV_API.goToNode
                    : (typeof window.goToNode === 'function' ? window.goToNode : (typeof goToNode === 'function' ? goToNode : null));

                if (!fn) throw new Error("goToNode is not found on page (window.SV_API.goToNode / window.goToNode / global goToNode).");
                return fn(nodeId, opts);
            }
            """,
            payload,
        )

        # 2) wait until nodeId equals target
        self._wait_for_function(
            "(target) => (window.state?.nodeId ?? null) === target",
            arg=node_id,
            timeout_ms=timeout_ms,
        )

        # 3) small settle for rendering
        self._wait(settle_ms)
        return self.get_state()

    # ---------- public: actions (buttons) ----------

    def action_front(self, timeout_ms: int = 8000, settle_ms: int = 200) -> bool:
        """
        前進：點擊按鈕後等待 nodeId 改變。
        Returns: 是否成功移動（節點改變）
        """
        prev_node = self.get_state().get("nodeId")
        self._click("#moveFront")

        ok = self._wait_for_function(
            "(prev) => (window.state?.nodeId ?? null) !== prev",
            arg=prev_node,
            timeout_ms=timeout_ms,
        )
        self._wait(settle_ms)
        return bool(ok)

    def action_turn_left(self, settle_ms: int = 80) -> None:
        self._click("#turnLeft")
        self._wait(settle_ms)

    def action_turn_right(self, settle_ms: int = 80) -> None:
        self._click("#turnRight")
        self._wait(settle_ms)

    def action_pitch_up(self, settle_ms: int = 80) -> None:
        self._click("#pitchUp")
        self._wait(settle_ms)

    def action_pitch_level(self, settle_ms: int = 80) -> None:
        self._click("#pitchLevel")
        self._wait(settle_ms)

    def action_pitch_down(self, settle_ms: int = 80) -> None:
        self._click("#pitchDown")
        self._wait(settle_ms)

    def action_zoom_in(self, settle_ms: int = 80) -> None:
        self._click("#zoomIn")
        self._wait(settle_ms)

    def action_zoom_out(self, settle_ms: int = 80) -> None:
        self._click("#zoomOut")
        self._wait(settle_ms)

    # ---------- public: actions (keyboard shortcuts) ----------
    # hint: ↑移動，A/D 轉向，W/S 俯仰(60/90/120)，+/- 縮放
    # 注意：'+' 在 keyboard event 常是 '='（Shift+'='），因此用 'Equal' 更穩；'-' 用 'Minus'

    def key_front(self, timeout_ms: int = 8000, settle_ms: int = 200) -> bool:
        prev_node = self.get_state().get("nodeId")
        self._press("ArrowUp")

        ok = self._wait_for_function(
            "(prev) => (window.state?.nodeId ?? null) !== prev",
            arg=prev_node,
            timeout_ms=timeout_ms,
        )
        self._wait(settle_ms)
        return bool(ok)

    def key_turn_left(self, settle_ms: int = 80) -> None:
        self._press("KeyA")
        self._wait(settle_ms)

    def key_turn_right(self, settle_ms: int = 80) -> None:
        self._press("KeyD")
        self._wait(settle_ms)

    def key_pitch_up(self, settle_ms: int = 80) -> None:
        self._press("KeyW")
        self._wait(settle_ms)

    def key_pitch_down(self, settle_ms: int = 80) -> None:
        self._press("KeyS")
        self._wait(settle_ms)

    def key_zoom_in(self, settle_ms: int = 80) -> None:
        self._press("Equal")
        self._wait(settle_ms)

    def key_zoom_out(self, settle_ms: int = 80) -> None:
        self._press("Minus")
        self._wait(settle_ms)

    # ---------- public: console ----------

    def get_console(
        self,
        clear: bool = True,
        kinds: Optional[List[ConsoleKind]] = None,
    ) -> List[ConsoleEntry]:
        """
        取回累積的 console/pageerror 訊息

        clear 行為：
        - kinds is None: clear=True 會清空全部
        - kinds not None: clear=True 只清掉那些 kinds，其他保留
        """
        with self._console_lock:
            if kinds is None:
                out = list(self._console)
                if clear:
                    self._console.clear()
                return out

            out = [e for e in self._console if e.kind in kinds]
            if clear:
                self._console = [e for e in self._console if e.kind not in kinds]
            return out

    def get_errors(self, clear: bool = True) -> List[ConsoleEntry]:
        """只獲取錯誤訊息"""
        return self.get_console(clear=clear, kinds=["error", "pageerror"])
