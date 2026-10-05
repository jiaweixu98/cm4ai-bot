import queue
import unittest
from unittest.mock import patch

import usage_log


ACCOUNT = "11111111-1111-4111-8111-111111111111"


class UsageLogTests(unittest.TestCase):
    def setUp(self):
        usage_log.reset_for_tests()
        self.original_write = usage_log._write_batch

    def tearDown(self):
        usage_log._write_batch = self.original_write

    def test_truncates_text_and_skips_invalid_accounts(self):
        row = usage_log._row({
            "account_id": ACCOUNT,
            "event_type": "matrix_chat_turn",
            "query_text": "q" * 2500,
            "response_text": "a" * 65000,
            "session_ref": "0123456789abcdef",
            "chat_session_id": "not-a-uuid",
        })
        self.assertIsNotNone(row)
        self.assertEqual(row[6], "q" * 2000)
        self.assertEqual(row[5], 2500)
        self.assertEqual(len(row[13]), 60000)
        self.assertIsNone(row[11])
        self.assertIsNone(usage_log._row({"account_id": "guest", "event_type": "matrix_chat_turn"}))

    def test_guests_are_not_queued(self):
        with patch.object(usage_log._QUEUE, "put_nowait") as put:
            usage_log.log_event(event_type="matrix_chat_turn", query_text="hello")
        put.assert_not_called()

    def test_full_queue_drops_without_raising(self):
        before = usage_log.dropped_count()
        with patch.object(usage_log._QUEUE, "put_nowait", side_effect=queue.Full):
            usage_log.log_event(account_id=ACCOUNT, event_type="matrix_search", query_text="topic")
        self.assertEqual(usage_log.dropped_count(), before + 1)

    def test_batches_and_database_failures_do_not_reach_the_caller(self):
        seen = []

        def capture(rows):
            seen.append(list(rows))

        usage_log._write_batch = capture
        for index in range(3):
            usage_log.log_event(account_id=ACCOUNT, event_type="matrix_chat_turn", query_text=f"question {index}")
        usage_log.flush(2)
        self.assertEqual(sum(len(batch) for batch in seen), 3)

        def fail(rows):
            raise RuntimeError("database unavailable")

        usage_log._write_batch = fail
        usage_log.log_event(account_id=ACCOUNT, event_type="matrix_search", query_text="still safe")
        usage_log.flush(2)


if __name__ == "__main__":
    unittest.main()
