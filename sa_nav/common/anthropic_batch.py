from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Iterable, Iterator

from .model_client import (
    ANTHROPIC_VERSION,
    DEFAULT_ANTHROPIC_API_BASE,
    DEFAULT_ANTHROPIC_MAX_TOKENS,
    ModelResponseClient,
)


MAX_BATCH_REQUESTS = 100_000
MAX_BATCH_BYTES = 256 * 1024 * 1024
DEFAULT_BATCH_TARGET_BYTES = 200 * 1024 * 1024
CUSTOM_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class AnthropicBatchRequestError(RuntimeError):
    """Raised when an individual request in a Message Batch does not succeed."""


class AnthropicBillingPauseError(RuntimeError):
    """Raised when billing or a spend limit requires operator action."""


def anthropic_billing_pause_reason(error: BaseException) -> dict | None:
    """Classify non-transient Anthropic billing/spend-limit failures."""

    message = str(error)
    normalized = " ".join(message.lower().split())
    markers = (
        "http 402",
        '"type": "billing_error"',
        '"type":"billing_error"',
        "enforced_spend_limit_reached",
        "reached your specified api usage limits",
        "reached your specified workspace api usage limits",
        "crossed its monthly api usage threshold",
        "credit balance is too low",
        "insufficient credit",
    )
    if not any(marker in normalized for marker in markers):
        return None
    return {
        "error_type": type(error).__name__,
        "error_message": message[:1000],
        "detected_at_unix": time.time(),
    }


@dataclass
class _BatchWaiter:
    event: threading.Event = field(default_factory=threading.Event)
    response: dict | None = None
    error: BaseException | None = None


@dataclass
class _QueuedBatchRequest:
    fingerprint: str
    params: dict | None
    enqueued_at: float
    waiters: list[_BatchWaiter] = field(default_factory=list)


def prepare_message_batch_request(
    *,
    custom_id: str,
    request_body: dict,
    default_max_tokens: int = DEFAULT_ANTHROPIC_MAX_TOKENS,
) -> dict:
    if not CUSTOM_ID_PATTERN.fullmatch(custom_id):
        raise ValueError(
            "Anthropic batch custom_id must match ^[A-Za-z0-9_-]{1,64}$."
        )
    params = ModelResponseClient._responses_to_anthropic_messages_payload(
        request_body,
        default_max_tokens=default_max_tokens,
    )
    return {"custom_id": custom_id, "params": params}


def serialized_batch_bytes(requests: Iterable[dict]) -> int:
    payload = {"requests": list(requests)}
    return len(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def split_message_batch_requests(
    requests: Iterable[dict],
    *,
    max_requests: int = MAX_BATCH_REQUESTS,
    target_bytes: int = DEFAULT_BATCH_TARGET_BYTES,
) -> list[list[dict]]:
    if max_requests < 1 or max_requests > MAX_BATCH_REQUESTS:
        raise ValueError(f"max_requests must be between 1 and {MAX_BATCH_REQUESTS}.")
    if target_bytes < 1 or target_bytes > MAX_BATCH_BYTES:
        raise ValueError(f"target_bytes must be between 1 and {MAX_BATCH_BYTES}.")

    batches: list[list[dict]] = []
    current: list[dict] = []
    # Compact JSON for {"requests":[]} is 15 bytes. Each additional request
    # costs its serialized bytes plus one comma after the first request.
    current_bytes = len(b'{"requests":[]}')
    seen_custom_ids: set[str] = set()
    for request in requests:
        custom_id = request.get("custom_id") if isinstance(request, dict) else None
        if not isinstance(custom_id, str) or not CUSTOM_ID_PATTERN.fullmatch(custom_id):
            raise ValueError("Every batch request must have a valid custom_id.")
        if custom_id in seen_custom_ids:
            raise ValueError(f"Duplicate Anthropic batch custom_id: {custom_id}")
        seen_custom_ids.add(custom_id)

        encoded_bytes = len(
            json.dumps(request, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        )
        single_request_bytes = len(b'{"requests":[') + encoded_bytes + len(b"]}")
        if single_request_bytes > target_bytes:
            raise ValueError(
                f"Batch request {custom_id} is {single_request_bytes} bytes, "
                f"larger than target_bytes={target_bytes}."
            )

        separator_bytes = 1 if current else 0
        would_exceed_size = current_bytes + separator_bytes + encoded_bytes > target_bytes
        would_exceed_count = len(current) >= max_requests
        if current and (would_exceed_size or would_exceed_count):
            batches.append(current)
            current = []
            current_bytes = len(b'{"requests":[]}')
            separator_bytes = 0

        current.append(request)
        current_bytes += separator_bytes + encoded_bytes

    if current:
        batches.append(current)
    return batches


def normalize_message_batch_result(record: dict) -> dict:
    normalized = json.loads(json.dumps(record))
    result = normalized.get("result")
    if not isinstance(result, dict) or result.get("type") != "succeeded":
        return normalized
    message = result.get("message")
    if isinstance(message, dict):
        result["message"] = ModelResponseClient._normalize_payload(message)
    return normalized


def atomic_write_batch_checkpoint(path: str | Path, payload: dict) -> None:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=checkpoint_path.parent,
        prefix=f".{checkpoint_path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
        temporary_path = Path(stream.name)
    os.replace(temporary_path, checkpoint_path)


class AnthropicMessageBatchClient:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        api_base: str = DEFAULT_ANTHROPIC_API_BASE,
        workspace_id: str | None = None,
        request_timeout: float = 300.0,
    ) -> None:
        self.api_key = (
            api_key
            or os.environ.get("ANTHROPIC_API_KEY")
            or os.environ.get("SA_NAV_API_KEY")
        )
        self.api_base = api_base.rstrip("/")
        self.workspace_id = workspace_id or os.environ.get("ANTHROPIC_WORKSPACE_ID")
        self.request_timeout = float(request_timeout)

    def create(self, requests: list[dict]) -> dict:
        if not requests:
            raise ValueError("An Anthropic Message Batch must contain at least one request.")
        if len(requests) > MAX_BATCH_REQUESTS:
            raise ValueError(f"An Anthropic Message Batch supports at most {MAX_BATCH_REQUESTS} requests.")
        custom_ids: set[str] = set()
        for request in requests:
            custom_id = request.get("custom_id") if isinstance(request, dict) else None
            if not isinstance(custom_id, str) or not CUSTOM_ID_PATTERN.fullmatch(custom_id):
                raise ValueError("Every batch request must have a valid custom_id.")
            if custom_id in custom_ids:
                raise ValueError(f"Duplicate Anthropic batch custom_id: {custom_id}")
            custom_ids.add(custom_id)
        payload_bytes = serialized_batch_bytes(requests)
        if payload_bytes > MAX_BATCH_BYTES:
            raise ValueError(
                f"Anthropic Message Batch payload is {payload_bytes} bytes; "
                f"the hard limit is {MAX_BATCH_BYTES}."
            )
        return self._request_json("POST", "/messages/batches", {"requests": requests})

    def retrieve(self, batch_id: str) -> dict:
        return self._request_json("GET", f"/messages/batches/{batch_id}")

    def cancel(self, batch_id: str) -> dict:
        return self._request_json("POST", f"/messages/batches/{batch_id}/cancel", {})

    def delete(self, batch_id: str) -> dict:
        return self._request_json("DELETE", f"/messages/batches/{batch_id}")

    def wait_until_ended(
        self,
        batch_id: str,
        *,
        poll_interval_seconds: float = 30.0,
        max_wait_seconds: float | None = None,
    ) -> dict:
        started = time.monotonic()
        while True:
            batch = self.retrieve(batch_id)
            if batch.get("processing_status") == "ended":
                return batch
            if max_wait_seconds is not None and time.monotonic() - started >= max_wait_seconds:
                raise TimeoutError(
                    f"Anthropic Message Batch {batch_id} did not end within {max_wait_seconds:.0f}s."
                )
            time.sleep(max(0.0, float(poll_interval_seconds)))

    def results(self, batch_id: str) -> Iterator[dict]:
        request = urllib.request.Request(
            f"{self.api_base}/messages/batches/{batch_id}/results",
            headers=self._headers(),
            method="GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.request_timeout) as response:
                for raw_line in response:
                    if not raw_line.strip():
                        continue
                    yield normalize_message_batch_result(
                        json.loads(raw_line.decode("utf-8"))
                    )
        except urllib.error.HTTPError as exc:
            self._raise_http_error(exc, request.full_url)

    def _request_json(
        self,
        method: str,
        path: str,
        payload: dict | None = None,
    ) -> dict:
        if not self.api_key:
            raise RuntimeError("Missing ANTHROPIC_API_KEY or SA_NAV_API_KEY.")
        url = f"{self.api_base}{path}"
        request = urllib.request.Request(
            url,
            data=(
                json.dumps(payload, ensure_ascii=False).encode("utf-8")
                if payload is not None
                else None
            ),
            headers=self._headers(),
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=self.request_timeout) as response:
                body = response.read()
        except urllib.error.HTTPError as exc:
            self._raise_http_error(exc, url)
        return json.loads(body.decode("utf-8"))

    def _headers(self) -> dict[str, str]:
        if not self.api_key:
            raise RuntimeError("Missing ANTHROPIC_API_KEY or SA_NAV_API_KEY.")
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": ANTHROPIC_VERSION,
        }
        if self.workspace_id and self.workspace_id.strip():
            headers["anthropic-workspace-id"] = self.workspace_id.strip()
        return headers

    @staticmethod
    def _raise_http_error(exc: urllib.error.HTTPError, url: str) -> None:
        try:
            body = exc.read().decode("utf-8", errors="replace")
        except Exception:
            body = ""
        compact = " ".join(body.split())
        if len(compact) > 500:
            compact = compact[:500].rstrip() + "..."
        detail = f" Response body: {compact}" if compact else ""
        raise RuntimeError(
            f"Anthropic Message Batch API request failed with HTTP {exc.code} "
            f"from {url}: {exc.reason}.{detail}"
        ) from exc


class PersistentAnthropicBatchBroker:
    """
    Combine concurrent native Messages requests into recoverable Message Batches.

    Callers submit already-converted Anthropic ``/v1/messages`` parameters and
    block until their individual result is available. Successful responses are
    cached by a full SHA-256 fingerprint. Batch checkpoints contain no API key
    and no image/request payload, so an interrupted run can retrieve submitted
    batch results and serve them when the same navigation request is replayed.
    """

    def __init__(
        self,
        *,
        client: AnthropicMessageBatchClient,
        state_dir: str | Path,
        flush_interval_seconds: float = 2.0,
        poll_interval_seconds: float = 30.0,
        max_wait_seconds: float = 24 * 60 * 60,
        max_requests_per_batch: int = MAX_BATCH_REQUESTS,
        target_batch_bytes: int = DEFAULT_BATCH_TARGET_BYTES,
        max_concurrent_batches: int = 4,
    ) -> None:
        if flush_interval_seconds < 0:
            raise ValueError("flush_interval_seconds must be non-negative.")
        if poll_interval_seconds < 0:
            raise ValueError("poll_interval_seconds must be non-negative.")
        if max_wait_seconds <= 0:
            raise ValueError("max_wait_seconds must be positive.")
        if max_concurrent_batches < 1:
            raise ValueError("max_concurrent_batches must be at least 1.")
        if max_requests_per_batch < 1 or max_requests_per_batch > MAX_BATCH_REQUESTS:
            raise ValueError(
                f"max_requests_per_batch must be between 1 and {MAX_BATCH_REQUESTS}."
            )
        if target_batch_bytes < 1 or target_batch_bytes > MAX_BATCH_BYTES:
            raise ValueError(
                f"target_batch_bytes must be between 1 and {MAX_BATCH_BYTES}."
            )

        self.client = client
        self.state_dir = Path(state_dir).resolve()
        self.response_dir = self.state_dir / "responses"
        self.batch_dir = self.state_dir / "batches"
        self.flush_interval_seconds = float(flush_interval_seconds)
        self.poll_interval_seconds = float(poll_interval_seconds)
        self.max_wait_seconds = float(max_wait_seconds)
        self.max_requests_per_batch = int(max_requests_per_batch)
        self.target_batch_bytes = int(target_batch_bytes)
        self.max_concurrent_batches = int(max_concurrent_batches)

        self._condition = threading.Condition()
        self._entries: dict[str, _QueuedBatchRequest] = {}
        self._pending_fingerprints: list[str] = []
        self._stopping = False
        self._started = False
        self._scheduler_thread: threading.Thread | None = None
        self._executor: ThreadPoolExecutor | None = None
        self._lock_stream = None
        self._billing_pause: dict | None = None
        self._stats = {
            "submitted_batch_count": 0,
            "submitted_request_count": 0,
            "recovered_batch_count": 0,
            "response_cache_hit_count": 0,
            "joined_inflight_request_count": 0,
            "succeeded_request_count": 0,
            "failed_request_count": 0,
        }

    @staticmethod
    def request_fingerprint(params: dict) -> str:
        canonical = json.dumps(
            {"schema": "anthropic_native_messages_request", "params": params},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

    def start(self) -> None:
        with self._condition:
            if self._started:
                return
            self.state_dir.mkdir(parents=True, exist_ok=True)
            self.response_dir.mkdir(parents=True, exist_ok=True)
            self.batch_dir.mkdir(parents=True, exist_ok=True)
            lock_path = self.state_dir / "gateway.lock"
            self._lock_stream = lock_path.open("a+", encoding="utf-8")
            try:
                fcntl.flock(
                    self._lock_stream.fileno(),
                    fcntl.LOCK_EX | fcntl.LOCK_NB,
                )
            except BlockingIOError as error:
                self._lock_stream.close()
                self._lock_stream = None
                raise RuntimeError(
                    f"Another Anthropic batch gateway is using {self.state_dir}."
                ) from error
            self._executor = ThreadPoolExecutor(
                max_workers=self.max_concurrent_batches,
                thread_name_prefix="anthropic-message-batch",
            )
            self._started = True
            self._stopping = False

        self._recover_incomplete_batches()
        self._scheduler_thread = threading.Thread(
            target=self._scheduler_loop,
            name="anthropic-batch-scheduler",
            daemon=True,
        )
        self._scheduler_thread.start()

    def submit(self, params: dict, *, timeout_seconds: float | None = None) -> dict:
        if not isinstance(params, dict):
            raise TypeError("Anthropic batch request params must be a JSON object.")
        if not self._started:
            raise RuntimeError("PersistentAnthropicBatchBroker.start() must be called first.")
        fingerprint = self.request_fingerprint(params)
        cached = self._read_cached_response(fingerprint)
        if cached is not None:
            with self._condition:
                self._stats["response_cache_hit_count"] += 1
            return cached

        waiter = _BatchWaiter()
        with self._condition:
            if self._stopping:
                raise RuntimeError("Anthropic batch broker is stopping.")
            if self._billing_pause is not None:
                raise AnthropicBillingPauseError(
                    "Anthropic Batch campaign is paused for billing or spend-limit "
                    "action. Add credit or raise the applicable limit, then restart "
                    "the same campaign command."
                )
            # A result may have been persisted between the first cache read and
            # taking the queue lock.
            cached = self._read_cached_response(fingerprint)
            if cached is not None:
                self._stats["response_cache_hit_count"] += 1
                return cached
            entry = self._entries.get(fingerprint)
            if entry is None:
                entry = _QueuedBatchRequest(
                    fingerprint=fingerprint,
                    params=json.loads(json.dumps(params)),
                    enqueued_at=time.monotonic(),
                )
                self._entries[fingerprint] = entry
                self._pending_fingerprints.append(fingerprint)
                self._condition.notify_all()
            else:
                self._stats["joined_inflight_request_count"] += 1
                if entry.params is None:
                    entry.params = json.loads(json.dumps(params))
            entry.waiters.append(waiter)

        completed = waiter.event.wait(timeout=timeout_seconds)
        if not completed:
            with self._condition:
                entry = self._entries.get(fingerprint)
                if entry is not None and waiter in entry.waiters:
                    entry.waiters.remove(waiter)
            raise TimeoutError(
                f"Anthropic batch request {fingerprint[:12]} did not complete "
                f"within {float(timeout_seconds):.0f}s."
            )
        if waiter.error is not None:
            raise waiter.error
        if not isinstance(waiter.response, dict):
            raise RuntimeError(
                f"Anthropic batch request {fingerprint[:12]} completed without a response."
            )
        return waiter.response

    def snapshot(self) -> dict:
        with self._condition:
            return {
                "schema": "anthropic_batch_broker_status",
                "schema_version": 1,
                "started": self._started,
                "stopping": self._stopping,
                "queued_request_count": len(self._pending_fingerprints),
                "inflight_request_count": len(self._entries),
                "billing_paused": self._billing_pause is not None,
                "billing_pause": (
                    json.loads(json.dumps(self._billing_pause))
                    if self._billing_pause is not None
                    else None
                ),
                **dict(self._stats),
            }

    def close(self, *, wait: bool = True) -> None:
        with self._condition:
            if not self._started:
                return
            self._stopping = True
            self._condition.notify_all()
        if self._scheduler_thread is not None:
            self._scheduler_thread.join(timeout=None if wait else 1.0)
        if self._executor is not None:
            self._executor.shutdown(wait=wait, cancel_futures=not wait)
        with self._condition:
            self._started = False
        if self._lock_stream is not None:
            try:
                fcntl.flock(self._lock_stream.fileno(), fcntl.LOCK_UN)
            finally:
                self._lock_stream.close()
                self._lock_stream = None

    def _scheduler_loop(self) -> None:
        while True:
            with self._condition:
                while not self._pending_fingerprints and not self._stopping:
                    self._condition.wait()
                if self._stopping and not self._pending_fingerprints:
                    return
                first_enqueued = min(
                    self._entries[fingerprint].enqueued_at
                    for fingerprint in self._pending_fingerprints
                    if fingerprint in self._entries
                )
                deadline = first_enqueued + self.flush_interval_seconds
                while not self._stopping:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    self._condition.wait(timeout=remaining)

                fingerprints = list(dict.fromkeys(self._pending_fingerprints))
                self._pending_fingerprints.clear()
                records: list[dict] = []
                for fingerprint in fingerprints:
                    entry = self._entries.get(fingerprint)
                    if entry is None or entry.params is None:
                        continue
                    records.append(
                        {"custom_id": fingerprint, "params": entry.params}
                    )

            if not records:
                continue
            try:
                chunks = split_message_batch_requests(
                    records,
                    max_requests=self.max_requests_per_batch,
                    target_bytes=self.target_batch_bytes,
                )
            except Exception as error:
                for record in records:
                    self._complete_error(str(record["custom_id"]), error)
                continue
            executor = self._executor
            if executor is None:
                error = RuntimeError("Anthropic batch executor is not available.")
                for record in records:
                    self._complete_error(str(record["custom_id"]), error)
                continue
            for chunk in chunks:
                executor.submit(self._submit_and_collect_batch, chunk)

    def _submit_and_collect_batch(self, requests: list[dict]) -> None:
        fingerprints = [str(request["custom_id"]) for request in requests]
        checkpoint_path = self.batch_dir / f"submission_{time.time_ns()}.json"
        checkpoint = {
            "schema": "anthropic_navigation_message_batch",
            "schema_version": 1,
            "status": "submitting",
            "created_at_unix": time.time(),
            "custom_ids": fingerprints,
            "request_count": len(requests),
            "payload_bytes": serialized_batch_bytes(requests),
        }
        atomic_write_batch_checkpoint(checkpoint_path, checkpoint)
        try:
            batch = self.client.create(requests)
            checkpoint.update(
                {
                    "status": "submitted",
                    "batch_id": batch.get("id"),
                    "batch": self._safe_batch_metadata(batch),
                }
            )
            atomic_write_batch_checkpoint(checkpoint_path, checkpoint)
            with self._condition:
                self._stats["submitted_batch_count"] += 1
                self._stats["submitted_request_count"] += len(requests)
            self._wait_and_collect(
                checkpoint_path=checkpoint_path,
                checkpoint=checkpoint,
                fingerprints=fingerprints,
            )
        except Exception as error:
            self._record_billing_pause(error)
            checkpoint.update(
                {
                    "status": "gateway_error",
                    "error_type": type(error).__name__,
                    "error_message": str(error)[:1000],
                    "updated_at_unix": time.time(),
                }
            )
            atomic_write_batch_checkpoint(checkpoint_path, checkpoint)
            for fingerprint in fingerprints:
                self._complete_error(fingerprint, error)

    def _wait_and_collect(
        self,
        *,
        checkpoint_path: Path,
        checkpoint: dict,
        fingerprints: list[str],
    ) -> None:
        batch_id = checkpoint.get("batch_id")
        if not isinstance(batch_id, str) or not batch_id:
            raise RuntimeError("Anthropic batch checkpoint is missing batch_id.")
        ended = self.client.wait_until_ended(
            batch_id,
            poll_interval_seconds=self.poll_interval_seconds,
            max_wait_seconds=self.max_wait_seconds,
        )
        checkpoint.update(
            {
                "status": "collecting_results",
                "batch": self._safe_batch_metadata(ended),
                "updated_at_unix": time.time(),
            }
        )
        atomic_write_batch_checkpoint(checkpoint_path, checkpoint)
        results = list(self.client.results(batch_id))
        result_by_id = {
            str(record.get("custom_id")): record
            for record in results
            if isinstance(record, dict) and isinstance(record.get("custom_id"), str)
        }
        succeeded = 0
        failed = 0
        for fingerprint in fingerprints:
            record = result_by_id.get(fingerprint)
            result = record.get("result") if isinstance(record, dict) else None
            if isinstance(result, dict) and result.get("type") == "succeeded":
                message = result.get("message")
                if isinstance(message, dict):
                    self._complete_success(fingerprint, message)
                    succeeded += 1
                    continue
            error = AnthropicBatchRequestError(
                self._batch_result_error_message(
                    batch_id=batch_id,
                    fingerprint=fingerprint,
                    record=record,
                )
            )
            self._record_billing_pause(error)
            self._complete_error(fingerprint, error)
            failed += 1
        checkpoint.update(
            {
                "status": "complete",
                "succeeded_request_count": succeeded,
                "failed_request_count": failed,
                "result_count": len(results),
                "completed_at_unix": time.time(),
            }
        )
        atomic_write_batch_checkpoint(checkpoint_path, checkpoint)

    def _recover_incomplete_batches(self) -> None:
        executor = self._executor
        if executor is None:
            return
        for checkpoint_path in sorted(self.batch_dir.glob("*.json")):
            try:
                checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if checkpoint.get("status") == "complete":
                continue
            batch_id = checkpoint.get("batch_id")
            fingerprints = checkpoint.get("custom_ids")
            if not isinstance(batch_id, str) or not batch_id:
                # No durable batch id means there is nothing safe to recover.
                continue
            if not isinstance(fingerprints, list) or not all(
                isinstance(value, str) and len(value) == 64
                for value in fingerprints
            ):
                continue
            missing = [
                fingerprint
                for fingerprint in fingerprints
                if self._read_cached_response(fingerprint) is None
            ]
            if not missing:
                checkpoint["status"] = "complete"
                checkpoint["recovered_from_response_cache"] = True
                atomic_write_batch_checkpoint(checkpoint_path, checkpoint)
                continue
            with self._condition:
                for fingerprint in missing:
                    self._entries.setdefault(
                        fingerprint,
                        _QueuedBatchRequest(
                            fingerprint=fingerprint,
                            params=None,
                            enqueued_at=time.monotonic(),
                        ),
                    )
                self._stats["recovered_batch_count"] += 1
            executor.submit(
                self._recover_batch,
                checkpoint_path,
                checkpoint,
                list(fingerprints),
            )

    def _recover_batch(
        self,
        checkpoint_path: Path,
        checkpoint: dict,
        fingerprints: list[str],
    ) -> None:
        try:
            checkpoint["status"] = "recovering"
            checkpoint["recovery_started_at_unix"] = time.time()
            atomic_write_batch_checkpoint(checkpoint_path, checkpoint)
            self._wait_and_collect(
                checkpoint_path=checkpoint_path,
                checkpoint=checkpoint,
                fingerprints=fingerprints,
            )
        except Exception as error:
            checkpoint.update(
                {
                    "status": "recovery_error",
                    "error_type": type(error).__name__,
                    "error_message": str(error)[:1000],
                    "updated_at_unix": time.time(),
                }
            )
            atomic_write_batch_checkpoint(checkpoint_path, checkpoint)
            for fingerprint in fingerprints:
                if self._read_cached_response(fingerprint) is None:
                    self._complete_error(fingerprint, error)

    def _complete_success(self, fingerprint: str, response: dict) -> None:
        self._write_cached_response(fingerprint, response)
        with self._condition:
            entry = self._entries.pop(fingerprint, None)
            self._stats["succeeded_request_count"] += 1
        if entry is None:
            return
        for waiter in entry.waiters:
            waiter.response = json.loads(json.dumps(response))
            waiter.event.set()

    def _complete_error(self, fingerprint: str, error: BaseException) -> None:
        with self._condition:
            entry = self._entries.pop(fingerprint, None)
            self._stats["failed_request_count"] += 1
        if entry is None:
            return
        for waiter in entry.waiters:
            waiter.error = error
            waiter.event.set()

    def _record_billing_pause(self, error: BaseException) -> None:
        reason = anthropic_billing_pause_reason(error)
        if reason is None:
            return
        with self._condition:
            if self._billing_pause is None:
                self._billing_pause = reason
            self._condition.notify_all()

    def _response_path(self, fingerprint: str) -> Path:
        return self.response_dir / fingerprint[:2] / f"{fingerprint}.json"

    def _read_cached_response(self, fingerprint: str) -> dict | None:
        path = self._response_path(fingerprint)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if payload.get("fingerprint") != fingerprint:
            return None
        response = payload.get("response")
        return json.loads(json.dumps(response)) if isinstance(response, dict) else None

    def _write_cached_response(self, fingerprint: str, response: dict) -> None:
        atomic_write_batch_checkpoint(
            self._response_path(fingerprint),
            {
                "schema": "anthropic_batch_response_cache",
                "schema_version": 1,
                "fingerprint": fingerprint,
                "cached_at_unix": time.time(),
                "response": response,
            },
        )

    @staticmethod
    def _safe_batch_metadata(batch: dict) -> dict:
        allowed = {
            "id",
            "type",
            "processing_status",
            "request_counts",
            "ended_at",
            "created_at",
            "expires_at",
            "cancel_initiated_at",
            "results_url",
        }
        return {
            key: json.loads(json.dumps(value))
            for key, value in batch.items()
            if key in allowed
        }

    @staticmethod
    def _batch_result_error_message(
        *,
        batch_id: str,
        fingerprint: str,
        record: dict | None,
    ) -> str:
        if record is None:
            detail = "result record was missing"
        else:
            result = record.get("result")
            detail = json.dumps(result, ensure_ascii=False)[:1000]
        return (
            f"Anthropic Message Batch {batch_id} request "
            f"{fingerprint[:12]} failed: {detail}"
        )
