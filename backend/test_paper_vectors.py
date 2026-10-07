import json
import os
import sys
import tempfile
import time
import types
import unittest
from unittest.mock import patch

import numpy  # noqa: F401  (imported before sys.modules is patched)
import paper_vectors


class PaperVectorSourceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.data = self.directory.name
        self.stub = types.ModuleType('data_loader')
        self.stub.LOCAL_DATA_DIR = self.data
        self.modules = patch.dict(sys.modules, {'data_loader': self.stub})
        self.modules.start()
        self.env = patch.dict(os.environ, {'PAPER_VECTOR_DIR': ''})
        self.env.start()
        self.pv = paper_vectors
        self.state = {name: getattr(paper_vectors, name) for name in ('_index', '_missing', '_retry_at', '_source_logged')}

    def tearDown(self):
        for name, value in self.state.items():
            setattr(self.pv, name, value)
        self.env.stop(); self.modules.stop(); self.directory.cleanup()

    def declare(self, files=True):
        vectors = os.path.join(self.data, 'vectors')
        os.makedirs(vectors)
        if files:
            for name in ('paper_embeddings.npy', 'paper_embedding_ids.json'):
                open(os.path.join(vectors, name), 'w').close()
        with open(os.path.join(self.data, 'snapshot_manifest.json'), 'w') as handle:
            json.dump({'paper_vector_source': 'vectors'}, handle)
        return os.path.realpath(vectors)

    def test_without_a_declared_source_vectors_are_unavailable(self):
        self.pv._source_logged = False
        with self.assertLogs('paper_vectors', 'WARNING') as logs:
            self.assertIsNone(self.pv._vector_dir())
            self.assertIsNone(self.pv._vector_dir())
        self.assertEqual(len(logs.records), 1)  # logged once
        self.assertNotIn('20260927', ' '.join(r.getMessage() for r in logs.records))

    def test_older_release_folders_are_never_used(self):
        old = os.path.join(self.data, 'tkg-20260927')
        os.makedirs(old)
        for name in ('paper_embeddings.npy', 'paper_embedding_ids.json'):
            open(os.path.join(old, name), 'w').close()
        self.assertIsNone(self.pv._vector_dir())

    def test_manifest_declared_source_and_explicit_override(self):
        vectors = self.declare()
        self.assertEqual(self.pv._vector_dir(), vectors)

    def test_memory_pressure_is_retried_but_a_missing_source_is_not(self):
        self.pv._index, self.pv._missing, self.pv._retry_at = None, False, 0.0
        self.declare()
        sentinel = object()
        with patch.object(self.pv, '_memory_ok', return_value=False):
            self.assertIsNone(self.pv.load_paper_index())
        self.assertFalse(self.pv._missing)
        self.assertGreater(self.pv._retry_at, time.monotonic())
        with patch.object(self.pv, '_memory_ok', return_value=True), \
                patch.object(self.pv.PaperVectors, 'open', return_value=sentinel):
            self.assertIsNone(self.pv.load_paper_index())  # still inside the retry window
            self.pv._retry_at = 0.0
            self.assertIs(self.pv.load_paper_index(), sentinel)

    def test_missing_source_is_permanent(self):
        self.pv._index, self.pv._missing, self.pv._retry_at = None, False, 0.0
        self.assertIsNone(self.pv.load_paper_index())
        self.assertTrue(self.pv._missing)


if __name__ == '__main__':
    unittest.main()
