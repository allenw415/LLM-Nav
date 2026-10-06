from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Iterator


REQUEST_CACHE_SCHEMA_VERSION = 1


def canonical_json_sha256(payload: object) -> str:
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def model_transport_identity(*, model: str, model_client) -> dict[str, object]:
    """Return the deterministic, secret-free settings that affect a response."""
    return {
        "provider": getattr(model_client, "provider", None),
        "api_base": getattr(model_client, "api_base", None),
        "api_kind": getattr(model_client, "api_kind", None),
        "model": model,
        "num_ctx": getattr(model_client, "num_ctx", None),
        "temperature": getattr(model_client, "temperature", None),
        "think": getattr(model_client, "think", None),
    }


def request_fingerprint(
    *,
    namespace: str,
    cache_version: int,
    request_body: object,
    transport_identity: dict[str, object],
) -> str:
    canonical = json.dumps(
        {
            "request_cache_schema_version": REQUEST_CACHE_SCHEMA_VERSION,
            "namespace": namespace,
            "cache_version": cache_version,
            "transport": transport_identity,
            "request": request_body,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def cache_entry_path(
    cache_dir: str | Path,
    fingerprint: str,
    *,
    group: str | None = None,
) -> Path:
    root = Path(cache_dir)
    if group:
        root = root / safe_path_component(group)
    return root / fingerprint[:2] / f"{fingerprint}.json"


def safe_path_component(value: object) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "unknown")).strip("._")
    return cleaned or "unknown"


def read_json_object(path: str | Path) -> dict | None:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def atomic_write_json(path: str | Path, payload: object) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
            temporary_path = Path(stream.name)
        os.replace(temporary_path, destination)
        temporary_path = None
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass


@contextmanager
def exclusive_cache_lock(cache_path: str | Path) -> Iterator[None]:
    """Serialize cache fill for one key; the OS releases the lock on crashes."""
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(f"{path.name}.lock")
    with lock_path.open("a+", encoding="utf-8") as lock_stream:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)
