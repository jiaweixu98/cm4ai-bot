"""Append-only usage events for the shared bridge_usage_events ledger.

Writes go through a bounded queue and one background thread so a chat or
search request never waits on Postgres. A full queue or a database error
drops the batch and does not raise into the request.
"""

import hashlib
import json
import logging
import os
import queue
import re
import threading
import time
import uuid
from typing import Any

import psycopg
from psycopg.types.json import Jsonb

logger = logging.getLogger(__name__)

_QUEUE: queue.Queue = queue.Queue(maxsize=2000)
_LOCK = threading.Lock()
_STARTED = False
_STOP = threading.Event()
_DROPPED = 0
_CONN = None

_UUID = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
    re.IGNORECASE,
)
_SESSION_REF = re.compile(r"^[a-f0-9]{8,64}$", re.IGNORECASE)
_QUERY_CAP = 2000
_RESPONSE_CAP = 60_000

_INSERT = """
INSERT INTO bridge_usage_events (
    id, account_id, event_type, metadata, query_sha256, query_length, query_text,
    source, status, duration_ms, session_ref, chat_session_id, turn_id,
    response_text, response_summary
) VALUES (
    %s, %s, %s, %s, %s, %s, %s,
    %s, %s, %s, %s, %s, %s,
    %s, %s
)
"""


def dropped_count() -> int:
    return _DROPPED


def reset_for_tests() -> None:
    global _DROPPED
    with _LOCK:
        _DROPPED = 0


def _clip(value: Any, limit: int) -> str:
    return str(value or "").strip()[:limit]


def _uuid(value: Any) -> str | None:
    text = str(value or "").strip()
    return text if _UUID.match(text) else None


def _json_value(value: Any, limit: int) -> Jsonb:
    if not isinstance(value, dict):
        return Jsonb({})
    try:
        encoded = json.dumps(value)
    except (TypeError, ValueError):
        return Jsonb({"truncated": True})
    if len(encoded) > limit:
        return Jsonb({"truncated": True})
    return Jsonb(value)


def _row(fields: dict[str, Any]) -> tuple | None:
    account_id = _uuid(fields.get("account_id"))
    if not account_id:
        return None
    query = str(fields.get("query_text") or "").strip()
    response = _clip(fields.get("response_text"), _RESPONSE_CAP)
    source = str(fields.get("source") or "matrix")
    if source not in {"auth", "graph", "matrix"}:
        source = "matrix"
    status = "error" if fields.get("status") == "error" else "ok"
    duration = fields.get("duration_ms")
    duration_ms = None
    if isinstance(duration, (int, float)) and duration >= 0:
        duration_ms = min(int(duration), 86_400_000)
    session_ref = str(fields.get("session_ref") or "").strip()
    if not _SESSION_REF.match(session_ref):
        session_ref = ""
    event_type = _clip(fields.get("event_type"), 80) or "unknown"
    return (
        str(uuid.uuid4()),
        account_id,
        event_type,
        _json_value(fields.get("metadata"), 8000),
        hashlib.sha256(query.encode("utf-8")).hexdigest() if query else None,
        len(query) if query else None,
        query[:_QUERY_CAP] if query else None,
        source,
        status,
        duration_ms,
        session_ref or None,
        _uuid(fields.get("chat_session_id")),
        _uuid(fields.get("turn_id")),
        response or None,
        _json_value(fields.get("response_summary"), 24000) if fields.get("response_summary") else None,
    )


def _write_batch(rows: list[tuple]) -> None:
    global _CONN
    url = os.environ.get("DATABASE_URL", "").strip()
    if not url:
        raise RuntimeError("DATABASE_URL is not configured")
    if _CONN is None or _CONN.closed:
        _CONN = psycopg.connect(url)
    with _CONN.cursor() as cur:
        cur.executemany(_INSERT, rows)
    _CONN.commit()


def _close_conn() -> None:
    global _CONN
    conn = _CONN
    _CONN = None
    if conn is None:
        return
    try:
        conn.close()
    except Exception:
        pass


def _worker() -> None:
    while not _STOP.is_set():
        try:
            first = _QUEUE.get(timeout=1)
        except queue.Empty:
            continue
        batch = [first]
        deadline = time.monotonic() + 0.05
        while len(batch) < 50 and time.monotonic() < deadline:
            try:
                batch.append(_QUEUE.get(timeout=0.05))
            except queue.Empty:
                break
        flushes = []
        rows = []
        for item in batch:
            if isinstance(item, dict) and item.get("__flush__") is not None:
                flushes.append(item["__flush__"])
            else:
                rows.append(item)
        if rows:
            try:
                _write_batch(rows)
            except Exception as exc:
                logger.warning("usage log write failed: %s", type(exc).__name__)
                try:
                    if _CONN is not None:
                        _CONN.rollback()
                except Exception:
                    pass
                _close_conn()
                time.sleep(0.2)
        for done in flushes:
            done.set()


def _ensure_worker() -> None:
    global _STARTED
    with _LOCK:
        if _STARTED:
            return
        threading.Thread(target=_worker, name="usage-log", daemon=True).start()
        _STARTED = True


def log_event(**fields: Any) -> None:
    global _DROPPED
    try:
        if not str(fields.get("account_id") or "").strip():
            return
        row = _row(fields)
        if row is None:
            return
        _ensure_worker()
        _QUEUE.put_nowait(row)
    except queue.Full:
        with _LOCK:
            _DROPPED += 1
    except Exception as exc:
        logger.warning("usage log enqueue failed: %s", type(exc).__name__)


def flush(timeout: float = 2.0) -> None:
    _ensure_worker()
    done = threading.Event()
    try:
        _QUEUE.put({"__flush__": done}, timeout=max(0.1, timeout))
    except queue.Full:
        return
    done.wait(timeout)
