import hashlib
import hmac
import json
import os
import time
import unittest
from contextlib import contextmanager
from unittest.mock import patch

import session_store


class FakeCursor:
    def __init__(self, rows):
        self.rows = rows
        self.query = ""
        self.params = ()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def execute(self, query, params=()):
        self.query = str(query)
        self.params = params

    def fetchall(self):
        return self.rows

    def fetchone(self):
        return self.rows[0] if self.rows else None


class FakeConnection:
    def __init__(self, rows):
        self.cursor_instance = FakeCursor(rows)
        self.committed = False

    def cursor(self):
        return self.cursor_instance

    def commit(self):
        self.committed = True


@contextmanager
def fake_connection(connection):
    yield connection


def session_row():
    return {
        "id": "00000000-0000-0000-0000-000000000001",
        "owner_orcid": "0000-0001",
        "owner_name": "Researcher",
        "focal_author_id": "42",
        "focal_author_name": "Researcher",
        "title": "A real question",
        "intent": "mentor",
        "last_message_preview": "A real question",
        "messages": [{"role": "user", "content": "A real question"}],
        "state": {"intent": "mentor"},
        "created_at": "created",
        "updated_at": "updated",
        "last_message_at": "last-message",
    }


class SessionStoreQueryTests(unittest.TestCase):
    def test_stale_save_returns_current_history_without_truncating_it(self):
        connection = FakeConnection([])
        current = session_row()
        with patch.object(session_store, "ensure_session_store"), patch.object(
            session_store, "get_db_conn", return_value=fake_connection(connection)
        ), patch.object(session_store, "get_chat_session", return_value=current) as lookup:
            saved = session_store.save_chat_session(
                session_id=current["id"], owner_orcid="0000-0001", owner_name="Researcher",
                focal_author_id="42", focal_author_name="Researcher", messages=[], state={},
            )
        self.assertIn("jsonb_array_length(EXCLUDED.messages) >= jsonb_array_length(matrix_chat_sessions.messages)", connection.cursor_instance.query)
        lookup.assert_called_once_with(current["id"], "0000-0001")
        self.assertEqual(saved["messages"], current["messages"])

    def test_owner_wide_history_omits_empty_chats_and_defaults_to_100(self):
        connection = FakeConnection([session_row()])
        with patch.object(session_store, "ensure_session_store"), patch.object(
            session_store, "get_db_conn", return_value=fake_connection(connection)
        ):
            sessions = session_store.list_chat_sessions("0000-0001")

        query = connection.cursor_instance.query
        self.assertIn("owner_orcid = %s", query)
        self.assertIn("jsonb_array_length(messages) > 0", query)
        self.assertNotIn("focal_author_id = %s", query)
        self.assertEqual(connection.cursor_instance.params, ("0000-0001", 100))
        self.assertEqual([session["id"] for session in sessions], [session_row()["id"]])

    def test_unchanged_messages_keep_activity_timestamps(self):
        connection = FakeConnection([session_row()])
        with patch.object(session_store, "ensure_session_store"), patch.object(
            session_store, "get_db_conn", return_value=fake_connection(connection)
        ):
            saved = session_store.save_chat_session(
                session_id=session_row()["id"],
                owner_orcid="0000-0001",
                owner_name="Researcher",
                focal_author_id="42",
                focal_author_name="Researcher",
                messages=session_row()["messages"],
                state=session_row()["state"],
            )

        query = connection.cursor_instance.query
        self.assertEqual(query.count("messages IS DISTINCT FROM EXCLUDED.messages"), 2)
        self.assertIn("ELSE matrix_chat_sessions.updated_at", query)
        self.assertIn("ELSE matrix_chat_sessions.last_message_at", query)
        self.assertTrue(connection.committed)
        self.assertEqual(saved["last_message_at"], "last-message")

    def test_tokens_without_account_still_validate(self):
        secret = "test-matrix-secret"
        previous = os.environ.get("BRIDGE_INTERNAL_API_TOKEN")
        os.environ["BRIDGE_INTERNAL_API_TOKEN"] = secret
        try:
            payload = {"v": 1, "orcid": "0000-0001", "name": "Researcher", "exp": int(time.time()) + 60}
            encoded = session_store._to_base64url(json.dumps(payload).encode("utf-8"))
            signature = session_store._to_base64url(
                hmac.new(secret.encode("utf-8"), encoded.encode("utf-8"), hashlib.sha256).digest()
            )
            identity = session_store.validate_matrix_user_token(f"{encoded}.{signature}")
        finally:
            if previous is None:
                os.environ.pop("BRIDGE_INTERNAL_API_TOKEN", None)
            else:
                os.environ["BRIDGE_INTERNAL_API_TOKEN"] = previous
        self.assertEqual(identity["orcid"], "0000-0001")
        self.assertEqual(identity["account_id"], "")
        self.assertEqual(identity["session_ref"], "")


if __name__ == "__main__":
    unittest.main()
