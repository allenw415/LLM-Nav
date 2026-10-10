from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
from typing import Type

from .anthropic_batch import PersistentAnthropicBatchBroker


class AnthropicBatchGatewayHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    # ``socketserver.TCPServer`` defaults to a backlog of 5.  The campaign
    # starts 32 workers together, so that default can reset otherwise valid
    # loopback connections before a request-handler thread is created.
    request_queue_size = 256

    def __init__(
        self,
        server_address: tuple[str, int],
        handler_class: Type[BaseHTTPRequestHandler],
        *,
        broker: PersistentAnthropicBatchBroker,
        local_api_key: str,
    ) -> None:
        super().__init__(server_address, handler_class)
        self.broker = broker
        self.local_api_key = local_api_key


class AnthropicBatchGatewayRequestHandler(BaseHTTPRequestHandler):
    server: AnthropicBatchGatewayHTTPServer
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:  # noqa: N802
        if self.path.rstrip("/") not in {"/health", "/status"}:
            self._write_json(404, {"error": {"type": "not_found"}})
            return
        payload = self.server.broker.snapshot()
        payload["status"] = "ok"
        self._write_json(200, payload)

    def do_POST(self) -> None:  # noqa: N802
        if self.path.rstrip("/") != "/v1/messages":
            self._write_json(404, {"error": {"type": "not_found"}})
            return
        if self.headers.get("x-api-key") != self.server.local_api_key:
            self._write_json(
                401,
                {
                    "type": "error",
                    "error": {
                        "type": "authentication_error",
                        "message": "Invalid local batch gateway key.",
                    },
                },
            )
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            content_length = 0
        if content_length <= 0:
            self._write_json(
                400,
                {
                    "type": "error",
                    "error": {
                        "type": "invalid_request_error",
                        "message": "Missing request body.",
                    },
                },
            )
            return
        try:
            payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("Request body must be a JSON object.")
            response = self.server.broker.submit(payload)
        except Exception as error:
            self._write_json(
                502,
                {
                    "type": "error",
                    "error": {
                        "type": "batch_gateway_error",
                        "message": f"{type(error).__name__}: {str(error)[:1000]}",
                    },
                },
            )
            return
        self._write_json(200, response)

    def log_message(self, format: str, *args) -> None:
        # The campaign writes aggregate status separately. Avoid logging paths,
        # request timing, or headers from thousands of long-lived connections.
        return

    def _write_json(self, status_code: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        try:
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            return


def start_anthropic_batch_gateway_server(
    *,
    broker: PersistentAnthropicBatchBroker,
    host: str = "127.0.0.1",
    port: int = 8765,
    local_api_key: str,
) -> tuple[AnthropicBatchGatewayHTTPServer, threading.Thread]:
    server = AnthropicBatchGatewayHTTPServer(
        (host, int(port)),
        AnthropicBatchGatewayRequestHandler,
        broker=broker,
        local_api_key=local_api_key,
    )
    thread = threading.Thread(
        target=server.serve_forever,
        name="anthropic-batch-gateway-http",
        daemon=True,
    )
    thread.start()
    return server, thread
