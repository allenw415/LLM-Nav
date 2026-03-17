from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Dict, Optional

from flask import Flask, jsonify, render_template_string, send_from_directory


HTML_TEMPLATE = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Perception Execution Viewer</title>
  <meta http-equiv="refresh" content="1">
  <style>
    body { font-family: Arial, sans-serif; margin: 24px; }
    .wrap { display: flex; gap: 24px; align-items: flex-start; }
    img { border: 1px solid #ccc; max-width: 700px; }
    pre { background: #f5f5f5; padding: 12px; overflow-x: auto; }
  </style>
</head>
<body>
  <h2>Perception Execution Viewer</h2>
  <div class="wrap">
    <div>
      <img src="/image/current_view.jpg?ts={{ ts }}" alt="current view">
    </div>
    <div>
      <h3>State</h3>
      <pre>{{ state }}</pre>
      <h3>Observation</h3>
      <pre>{{ observation }}</pre>
      <h3>Action</h3>
      <pre>{{ action }}</pre>
    </div>
  </div>
</body>
</html>
"""


class ImageViewer:
    """
    本地 Flask viewer。
    功能：
    - 顯示 current_view.jpg
    - 顯示 state / observation / action
    """

    def __init__(
        self,
        image_dir: str = "runtime_views",
        host: str = "127.0.0.1",
        port: int = 5000,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.image_dir.mkdir(parents=True, exist_ok=True)

        self.host = host
        self.port = port

        self._state: Dict[str, Any] = {}
        self._observation: Dict[str, Any] = {}
        self._action: Dict[str, Any] = {}

        self.app = Flask(__name__)
        self._setup_routes()
        self._thread: Optional[threading.Thread] = None

    def _setup_routes(self) -> None:
        @self.app.route("/")
        def index():
            import time
            return render_template_string(
                HTML_TEMPLATE,
                ts=int(time.time()),
                state=self._pretty(self._state),
                observation=self._pretty(self._observation),
                action=self._pretty(self._action),
            )

        @self.app.route("/image/<path:filename>")
        def serve_image(filename: str):
            return send_from_directory(self.image_dir, filename)

        @self.app.route("/state")
        def state_json():
            return jsonify({
                "state": self._state,
                "observation": self._observation,
                "action": self._action,
            })

    def _pretty(self, obj: Any) -> str:
        import json
        try:
            return json.dumps(obj, ensure_ascii=False, indent=2)
        except Exception:
            return str(obj)

    def update(
        self,
        state: Optional[Dict[str, Any]] = None,
        observation: Optional[Dict[str, Any]] = None,
        action: Optional[Dict[str, Any]] = None,
    ) -> None:
        if state is not None:
            self._state = state
        if observation is not None:
            self._observation = observation
        if action is not None:
            self._action = action

    def start(self, debug: bool = False) -> None:
        if self._thread is not None:
            return

        def _run():
            self.app.run(host=self.host, port=self.port, debug=debug, use_reloader=False)

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

    def url(self) -> str:
        return f"http://{self.host}:{self.port}"