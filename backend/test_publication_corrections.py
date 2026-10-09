import json
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from paper_library import PaperLibrary, PublicationDecisionsUnavailable


class CorrectionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name)
        db = sqlite3.connect(self.path / 'papers.sqlite')
        db.executescript('''CREATE TABLE papers(work_id TEXT PRIMARY KEY,title TEXT,abstract TEXT,year INTEGER,venue TEXT,doi TEXT,doi_key TEXT,pmid TEXT,cited_by INTEGER,primary_topic TEXT,primary_field TEXT,title_key TEXT);
            CREATE TABLE paper_authors(work_id TEXT,author_id INTEGER,position INTEGER,affiliation_status TEXT);
            CREATE TABLE publication_groups(author_id INTEGER,group_id TEXT,institution TEXT,ror_id TEXT,status TEXT);
            CREATE TABLE publication_affiliations(author_id INTEGER,work_id TEXT,group_id TEXT);
            INSERT INTO papers VALUES('W1001','Clinical phenotyping with health records','clinical phenotyping electronic records',2025,NULL,NULL,NULL,NULL,0,NULL,NULL,'clinical phenotyping with health records'),
                ('W1002','Portable clinical phenotyping systems','clinical phenotyping electronic records',2024,NULL,NULL,NULL,NULL,0,NULL,NULL,'portable clinical phenotyping systems'),
                ('W1003','Ocean ecosystem biology','coral ocean ecosystems',2020,NULL,NULL,NULL,NULL,0,NULL,NULL,'ocean ecosystem biology');
            INSERT INTO paper_authors VALUES('W1001',1,1,'confirmed'),('W1001',2,2,'unrecognised'),('W1002',3,1,'unknown'),('W1003',1,1,'confirmed'),('W1003',4,2,'unknown');
            INSERT INTO publication_groups VALUES(2,'institution','Clinical Institute','https://ror.org/012345678','unrecognised');
            INSERT INTO publication_affiliations VALUES(2,'W1001','institution');
            CREATE VIRTUAL TABLE papers_fts USING fts5(title,abstract,content='papers',content_rowid='rowid');
            INSERT INTO papers_fts(papers_fts) VALUES('rebuild');''')
        db.commit(); db.close()
        (self.path / 'snapshot_manifest.json').write_text(json.dumps({'snapshot_version':'test-v1'}))
        self.state = self.path / 'decisions.sqlite'
        with sqlite3.connect(self.state) as state:
            state.executescript('''CREATE TABLE excluded_links(snapshot_version TEXT,author_id INTEGER,work_id TEXT);
                CREATE TABLE decisions(author_id INTEGER);
                CREATE TABLE decision_events(revision INTEGER PRIMARY KEY);''')
        self.env = patch.dict(os.environ, {'PROFILE_DECISIONS_DB':str(self.state)})
        self.env.start()
        self.library = PaperLibrary.open(str(self.path))

    def tearDown(self):
        self.env.stop(); self.directory.cleanup()

    def test_decisions_created_after_startup_are_live_and_author_specific(self):
        # A snapshot withholds evidence until the shared store is readable.
        self.state.unlink()
        with patch.dict(os.environ, {'PROFILE_DECISIONS_DB':''}), patch('paper_library.decisions_path',return_value=str(self.state)):
            with self.assertRaises(PublicationDecisionsUnavailable): self.library.owns(2,'W1001')
            with sqlite3.connect(self.state) as db:
                db.executescript('''CREATE TABLE excluded_links(snapshot_version TEXT,author_id INTEGER,work_id TEXT);
                    CREATE TABLE decisions(author_id INTEGER); CREATE TABLE decision_events(revision INTEGER PRIMARY KEY);
                    INSERT INTO excluded_links VALUES('test-v1',2,'W1001');''')
            self.library._local.decisions_retry_at=0
            self.assertFalse(self.library.owns(2,'W1001'))
        self.assertFalse(self.library.owns(2,'W1001'))
        self.assertTrue(self.library.owns(1,'W1001'))
        self.assertEqual(self.library.visible_work_ids(2),set())
        db=sqlite3.connect(self.state); db.execute('DELETE FROM excluded_links'); db.commit(); db.close()
        self.assertTrue(self.library.owns(2,'W1001'))


    def test_decision_store_is_attached_read_only(self):
        self.assertTrue(self.library.owns(2,'W1001'))
        with self.assertRaises(sqlite3.OperationalError):
            self.library._db().execute("INSERT INTO corrections.excluded_links VALUES('test-v1',2,'W1001')")

    def test_replacing_an_attached_store_withholds_evidence(self):
        self.assertTrue(self.library.owns(2,'W1001'))
        old=self.state.with_suffix('.old')
        self.state.rename(old)
        with sqlite3.connect(old) as source,sqlite3.connect(self.state) as replacement:
            source.backup(replacement)
        with self.assertRaises(PublicationDecisionsUnavailable): self.library.owns(2,'W1001')

    def test_configured_store_with_another_schema_withholds_links_and_recovers(self):
        self.state.unlink()
        db=sqlite3.connect(self.state); db.execute('CREATE TABLE something_else(x)'); db.commit(); db.close()
        for lookup in (lambda:self.library.owns(2,'W1001'),lambda:self.library.shared_years(1),lambda:self.library.visible_work_ids(2)):
            with self.assertRaises(PublicationDecisionsUnavailable): lookup()
        with sqlite3.connect(self.state) as db:
            db.executescript('''CREATE TABLE excluded_links(snapshot_version TEXT,author_id INTEGER,work_id TEXT);
                CREATE TABLE decisions(author_id INTEGER); CREATE TABLE decision_events(revision INTEGER PRIMARY KEY);
                INSERT INTO excluded_links VALUES('test-v1',2,'W1001');''')
        self.library._local.decisions_retry_at=0
        self.assertFalse(self.library.owns(2,'W1001'))

    def test_missing_configured_store_withholds_links_during_retry(self):
        self.state.unlink()
        for _ in range(2):
            with self.assertRaises(PublicationDecisionsUnavailable): self.library.owns(2,'W1001')
        self.assertFalse(self.state.exists(),'Read-only access must not create an empty correction store')

    def test_incomplete_correction_schema_is_not_attached(self):
        with sqlite3.connect(self.state) as db: db.execute('DROP TABLE decision_events')
        with self.assertRaises(PublicationDecisionsUnavailable): self.library.owns(2,'W1001')

    def test_default_snapshot_store_with_another_schema_withholds_evidence(self):
        self.state.unlink()
        db=sqlite3.connect(self.state); db.execute('CREATE TABLE something_else(x)'); db.commit(); db.close()
        with patch.dict(os.environ, {'PROFILE_DECISIONS_DB':''}), patch('paper_library.decisions_path',return_value=str(self.state)):
            for lookup in (lambda:self.library.owns(2,'W1001'),lambda:self.library.shared_years(1),lambda:self.library.visible_work_ids(2)):
                with self.assertRaises(PublicationDecisionsUnavailable): lookup()


    def test_vectors_and_paths_refresh_and_undo_without_touching_source(self):
        import faiss
        import networkx as nx
        import numpy as np
        import publication_corrections as corrections
        matrix=np.asarray([[1,0],[0,1]],dtype=np.float32)
        corrections._matrix=matrix
        corrections._row_of={'W1001':0,'W1003':1}
        original=np.asarray([[0.8,0.6],[1,0]],dtype=np.float32)
        index=faiss.IndexFlatL2(2); index.add(original)
        db=sqlite3.connect(self.state)
        db.executescript('''INSERT INTO excluded_links VALUES('test-v1',1,'W1001');
            INSERT INTO decision_events VALUES(1); INSERT INTO decisions VALUES(1);''')
        db.commit(); db.close()
        graph=nx.Graph([('1','2'),('1','4')])
        corrections.sync_index(['1','2'],index,self.library,str(self.path))
        np.testing.assert_allclose(index.reconstruct(0),[0,1])
        np.testing.assert_allclose(index.reconstruct(1),original[1])
        corrections.sync_graph(graph,self.library)
        self.assertFalse(graph.has_edge('1','2')); self.assertTrue(graph.has_edge('1','4'))
        db=sqlite3.connect(self.state); db.executescript('DELETE FROM excluded_links; INSERT INTO decision_events VALUES(2);'); db.commit(); db.close()
        corrections.sync_index(['1','2'],index,self.library,str(self.path))
        np.testing.assert_allclose(index.reconstruct(0),original[0])
        corrections.sync_graph(graph,self.library)
        self.assertTrue(graph.has_edge('1','2'))
        np.testing.assert_allclose(original[0],[0.8,0.6])


if __name__ == '__main__': unittest.main()
