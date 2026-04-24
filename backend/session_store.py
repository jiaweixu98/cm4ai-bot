import base64
import hashlib
import hmac
import json
import os
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

_SESSION_STORE_LOCK = threading.Lock()
_SESSION_STORE_READY = False


def _get_database_url() -> str:
    url = os.environ.get("DATABASE_URL", "").strip()
    if not url:
        raise RuntimeError("DATABASE_URL is not configured")
    return url


@contextmanager
def get_db_conn():
    conn = psycopg.connect(_get_database_url(), row_factory=dict_row)
    try:
        yield conn
    finally:
        conn.close()


def ensure_session_store() -> None:
    global _SESSION_STORE_READY
    if _SESSION_STORE_READY:
        return

    with _SESSION_STORE_LOCK:
        if _SESSION_STORE_READY:
            return

        with get_db_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS matrix_chat_sessions (
                        id UUID PRIMARY KEY,
                        owner_orcid TEXT NOT NULL,
                        owner_name TEXT NOT NULL DEFAULT '',
                        focal_author_id TEXT NOT NULL,
                        focal_author_name TEXT NOT NULL DEFAULT '',
                        title TEXT NOT NULL DEFAULT '',
                        last_message_preview TEXT NOT NULL DEFAULT '',
                        messages JSONB NOT NULL DEFAULT '[]'::jsonb,
                        state JSONB NOT NULL DEFAULT '{}'::jsonb,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        last_message_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_matrix_chat_sessions_owner_last
                    ON matrix_chat_sessions (owner_orcid, last_message_at DESC)
                    """
                )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_matrix_chat_sessions_owner_focal_last
                    ON matrix_chat_sessions (owner_orcid, focal_author_id, last_message_at DESC)
                    """
                )
            conn.commit()

        _SESSION_STORE_READY = True


def _to_base64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("utf-8")


def _from_base64url(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(f"{value}{padding}")


def validate_matrix_user_token(token: str) -> dict[str, str] | None:
    secret = os.environ.get("BRIDGE_INTERNAL_API_TOKEN", "").strip()
    if not token or not secret:
        return None

    try:
        encoded_payload, provided_signature = token.split(".", 1)
    except ValueError:
        return None

    expected_signature = _to_base64url(
        hmac.new(secret.encode("utf-8"), encoded_payload.encode("utf-8"), hashlib.sha256).digest()
    )
    if not hmac.compare_digest(expected_signature, provided_signature):
        return None

    try:
        payload = json.loads(_from_base64url(encoded_payload).decode("utf-8"))
    except Exception:
        return None

    if payload.get("v") != 1:
        return None

    exp = int(payload.get("exp") or 0)
    now_seconds = int(datetime.now(tz=timezone.utc).timestamp())
    if exp <= now_seconds:
        return None

    orcid = str(payload.get("orcid") or "").strip()
    name = str(payload.get("name") or orcid).strip()
    if not orcid:
        return None

    return {
        "orcid": orcid,
        "name": name,
    }


def _truncate(value: str, max_len: int) -> str:
    normalized = " ".join(str(value or "").split()).strip()
    if len(normalized) <= max_len:
        return normalized
    return normalized[: max_len - 1].rstrip() + "…"


def _derive_session_title(messages: list[dict[str, Any]], state: dict[str, Any], focal_author_name: str) -> str:
    query = str((state or {}).get("currentQuery") or "").strip()
    if query:
        return _truncate(query, 72)

    for message in messages:
        if str(message.get("role") or "") != "user":
            continue
        content = str(message.get("content") or "").strip()
        if content:
            return _truncate(content, 48)

    if focal_author_name:
        return _truncate(f"Draft session for {focal_author_name}", 72)

    return "Draft session"


def _last_message_preview(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        content = str(message.get("content") or "").strip()
        if content:
            return _truncate(content, 160)
    return ""


def _serialize_session_row(row: dict[str, Any], include_full: bool = False) -> dict[str, Any]:
    session = {
        "id": str(row["id"]),
        "owner_orcid": row["owner_orcid"],
        "owner_name": row["owner_name"],
        "focal_author_id": row["focal_author_id"],
        "focal_author_name": row["focal_author_name"],
        "title": row["title"],
        "last_message_preview": row["last_message_preview"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "last_message_at": row["last_message_at"],
    }
    if include_full:
        session["messages"] = row.get("messages") or []
        session["state"] = row.get("state") or {}
    return session


def list_chat_sessions(owner_orcid: str, focal_author_id: str, limit: int = 20) -> list[dict[str, Any]]:
    ensure_session_store()
    with get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    id,
                    owner_orcid,
                    owner_name,
                    focal_author_id,
                    focal_author_name,
                    title,
                    last_message_preview,
                    created_at,
                    updated_at,
                    last_message_at
                FROM matrix_chat_sessions
                WHERE owner_orcid = %s AND focal_author_id = %s
                ORDER BY last_message_at DESC, created_at DESC
                LIMIT %s
                """,
                (owner_orcid, str(focal_author_id), max(1, min(limit, 100))),
            )
            rows = cur.fetchall()
    return [_serialize_session_row(row) for row in rows]


def get_chat_session(session_id: str, owner_orcid: str) -> dict[str, Any] | None:
    ensure_session_store()
    with get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    id,
                    owner_orcid,
                    owner_name,
                    focal_author_id,
                    focal_author_name,
                    title,
                    last_message_preview,
                    messages,
                    state,
                    created_at,
                    updated_at,
                    last_message_at
                FROM matrix_chat_sessions
                WHERE id = %s AND owner_orcid = %s
                """,
                (session_id, owner_orcid),
            )
            row = cur.fetchone()
    if not row:
        return None
    return _serialize_session_row(row, include_full=True)


def create_chat_session(
    owner_orcid: str,
    owner_name: str,
    focal_author_id: str,
    focal_author_name: str,
    messages: list[dict[str, Any]] | None = None,
    state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ensure_session_store()
    session_id = str(uuid.uuid4())
    normalized_messages = list(messages or [])
    normalized_state = dict(state or {})
    title = _derive_session_title(normalized_messages, normalized_state, focal_author_name)
    preview = _last_message_preview(normalized_messages)

    with get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO matrix_chat_sessions (
                    id,
                    owner_orcid,
                    owner_name,
                    focal_author_id,
                    focal_author_name,
                    title,
                    last_message_preview,
                    messages,
                    state
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING
                    id,
                    owner_orcid,
                    owner_name,
                    focal_author_id,
                    focal_author_name,
                    title,
                    last_message_preview,
                    messages,
                    state,
                    created_at,
                    updated_at,
                    last_message_at
                """,
                (
                    session_id,
                    owner_orcid,
                    owner_name,
                    str(focal_author_id),
                    str(focal_author_name or ""),
                    title,
                    preview,
                    Jsonb(normalized_messages),
                    Jsonb(normalized_state),
                ),
            )
            row = cur.fetchone()
        conn.commit()
    return _serialize_session_row(row, include_full=True)


def save_chat_session(
    session_id: str,
    owner_orcid: str,
    owner_name: str,
    focal_author_id: str,
    focal_author_name: str,
    messages: list[dict[str, Any]],
    state: dict[str, Any],
) -> dict[str, Any]:
    ensure_session_store()
    normalized_messages = list(messages or [])
    normalized_state = dict(state or {})
    title = _derive_session_title(normalized_messages, normalized_state, focal_author_name)
    preview = _last_message_preview(normalized_messages)

    with get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO matrix_chat_sessions (
                    id,
                    owner_orcid,
                    owner_name,
                    focal_author_id,
                    focal_author_name,
                    title,
                    last_message_preview,
                    messages,
                    state
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE
                SET
                    owner_name = EXCLUDED.owner_name,
                    focal_author_id = EXCLUDED.focal_author_id,
                    focal_author_name = EXCLUDED.focal_author_name,
                    title = EXCLUDED.title,
                    last_message_preview = EXCLUDED.last_message_preview,
                    messages = EXCLUDED.messages,
                    state = EXCLUDED.state,
                    updated_at = NOW(),
                    last_message_at = NOW()
                WHERE matrix_chat_sessions.owner_orcid = EXCLUDED.owner_orcid
                RETURNING
                    id,
                    owner_orcid,
                    owner_name,
                    focal_author_id,
                    focal_author_name,
                    title,
                    last_message_preview,
                    messages,
                    state,
                    created_at,
                    updated_at,
                    last_message_at
                """,
                (
                    session_id,
                    owner_orcid,
                    owner_name,
                    str(focal_author_id),
                    str(focal_author_name or ""),
                    title,
                    preview,
                    Jsonb(normalized_messages),
                    Jsonb(normalized_state),
                ),
            )
            row = cur.fetchone()
        conn.commit()

    if not row:
        raise PermissionError("Session does not belong to the authenticated user")
    return _serialize_session_row(row, include_full=True)
