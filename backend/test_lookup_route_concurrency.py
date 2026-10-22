"""Exercise actual ASGI routes without starting model/index resources."""
import ast
import asyncio
import json
from pathlib import Path
import threading
import unittest

from fastapi import FastAPI, HTTPException

SOURCE = ast.parse(Path(__file__).with_name('main.py').read_text())

class LookupRouteConcurrencyTests(unittest.IsolatedAsyncioTestCase):
    async def request(self, app, path, query=b''):
        messages = []
        async def receive():
            return {'type': 'http.request', 'body': b'', 'more_body': False}
        async def send(message):
            messages.append(message)
        await app({'type': 'http', 'asgi': {'version': '3.0'}, 'method': 'GET',
                   'path': path, 'query_string': query, 'headers': [], 'scheme': 'http',
                   'server': ('local', 80), 'client': ('127.0.0.1', 1), 'root_path': ''}, receive, send)
        return messages[0]['status'], json.loads(b''.join(m.get('body', b'') for m in messages[1:]))

    def app_for(self, name, helper_name, helper):
        app = FastAPI()
        route = next(n for n in SOURCE.body if getattr(n, 'name', '') == name)
        ns = {'app': app, 'HTTPException': HTTPException, helper_name: helper}
        exec(compile(ast.Module(body=[route], type_ignores=[]), '<actual lookup route>', 'exec'), ns)
        return app

    async def test_slow_lookup_does_not_block_unrelated_request(self):
        for route, helper, path, result, expected in [
            ('find_people', '_find_people_by_name', '/api/people', [{'author_id': '42'}], {'people': [{'author_id': '42'}]}),
            ('author_details', '_get_author_details', '/api/author/42/details', {'name': 'Test', 'papers': [{'Title': 'Kept'}], 'topics': ['Kept']}, {'name': 'Test', 'papers': [{'Title': 'Kept'}], 'topics': ['Kept']}),
            ('graph_path', '_get_shortest_path', '/api/graph-path/42/43', [{'id': '42'}, {'id': '43'}], {'path': [{'id': '42'}, {'id': '43'}], 'hops': 1}),
        ]:
            with self.subTest(route=route):
                started, release, finished = threading.Event(), threading.Event(), threading.Event()
                def slow_helper(*args):
                    started.set()
                    release.wait(2)  # safety bound also makes the old async route fail deterministically
                    finished.set()
                    return result
                app = self.app_for(route, helper, slow_helper)
                @app.get('/ping')
                async def ping():
                    return {'lookup_still_running': not finished.is_set()}
                task = asyncio.create_task(self.request(app, path))
                try:
                    self.assertTrue(await asyncio.to_thread(started.wait, 3))
                    status, payload = await self.request(app, '/ping')
                    self.assertEqual(status, 200)
                    self.assertTrue(payload['lookup_still_running'], 'Lookup blocked the API event loop')
                finally:
                    release.set()
                    response = await task
                self.assertEqual(response, (200, expected))

    async def test_unknown_author_stays_404(self):
        app = self.app_for('author_details', '_get_author_details', lambda _: {'name': 'Unknown'})
        self.assertEqual(await self.request(app, '/api/author/999/details'), (404, {'detail': 'Author not found'}))

    async def test_disconnected_path_stays_empty(self):
        app = self.app_for('graph_path', '_get_shortest_path', lambda *_: [])
        self.assertEqual(await self.request(app, '/api/graph-path/42/43'), (200, {'path': [], 'hops': -1}))

if __name__ == '__main__':
    unittest.main()
