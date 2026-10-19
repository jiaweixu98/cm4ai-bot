"""Exercise the real route functions without loading a second encoder/index."""
import ast
import asyncio
from pathlib import Path
import time
import unittest

from fastapi import FastAPI, Request


class SessionConcurrencyTests(unittest.IsolatedAsyncioTestCase):
    async def test_slow_history_database_does_not_block_other_requests(self):
        source = ast.parse(Path(__file__).with_name('main.py').read_text())
        route = next(n for n in source.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                     and n.name == 'list_sessions')
        app = FastAPI()

        def slow_database(*_):
            time.sleep(0.2)
            return []

        namespace = {'app':app, 'Request':Request, 'list_chat_sessions':slow_database,
                     '_require_matrix_session_identity':lambda _: {'orcid':'test-owner'}}
        exec(compile(ast.Module(body=[route], type_ignores=[]), '<session route>', 'exec'), namespace)

        @app.get('/ping')
        async def ping():
            return {'ok':True}

        async def request(path):
            messages=[]
            async def receive():
                return {'type':'http.request', 'body':b'', 'more_body':False}
            async def send(message):
                messages.append(message)
            await app({'type':'http', 'asgi':{'version':'3.0'}, 'method':'GET',
                       'path':path, 'query_string':b'', 'headers':[], 'scheme':'http',
                       'server':('local',80), 'client':('127.0.0.1',1), 'root_path':''}, receive, send)
            return messages

        start = time.monotonic()
        async def quick_request():
            await asyncio.sleep(0.02)
            response = await request('/ping')
            return response, time.monotonic() - start

        history, (response, seconds) = await asyncio.gather(request('/api/chat-sessions'), quick_request())
        self.assertEqual(history[0]['status'], 200)
        self.assertEqual(history[1]['body'], b'{"sessions":[]}')
        self.assertEqual(response[0]['status'], 200)
        self.assertLess(seconds, 0.15, 'A slow history query blocked the API event loop')
