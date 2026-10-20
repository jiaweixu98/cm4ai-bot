"""Run against an explicitly configured, disposable PostgreSQL database only."""
import concurrent.futures
import os
import unittest
import uuid
from urllib.parse import urlsplit
import session_store as store

@unittest.skipUnless(os.getenv('TEST_DATABASE_URL'), 'Requires an isolated TEST_DATABASE_URL')
class RevisionIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        target = os.environ['TEST_DATABASE_URL']
        if not urlsplit(target).path.rsplit('/',1)[-1].startswith('tkg_concurrency_fix_'):
            raise RuntimeError('Refusing to use a non-test database')
        cls.previous = os.environ.get('DATABASE_URL')
        os.environ['DATABASE_URL'] = target
        store._SESSION_STORE_READY = False
        store.ensure_session_store()

    @classmethod
    def tearDownClass(cls):
        if cls.previous is None: os.environ.pop('DATABASE_URL',None)
        else: os.environ['DATABASE_URL']=cls.previous
        store._SESSION_STORE_READY = False

    def setUp(self):
        self.owner = 'google:test-'+str(uuid.uuid4())
        self.messages = [{'role':'user','content':'Original question'}]
        self.session = store.create_chat_session(self.owner,'Test','42','Author',self.messages,{})

    def tearDown(self):
        with store.get_db_conn() as conn:
            conn.execute('DELETE FROM matrix_chat_sessions WHERE owner_orcid=%s',(self.owner,));conn.commit()

    def save(self, text, revision, state=None):
        return store.save_chat_session(self.session['id'],self.owner,'Test','42','Author',
            [{'role':'user','content':text}],state or {},expected_revision=revision)

    def test_equal_length_stale_write_and_old_client_rejected(self):
        self.assertEqual(self.session['revision'],1)
        first=self.save('Device A',1)
        self.assertEqual(first['revision'],2)
        for rev in [1,None]:
            with self.assertRaises(store.SessionConflictError): self.save('Device B',rev)
        self.assertEqual(store.get_chat_session(self.session['id'],self.owner)['messages'][0]['content'],'Device A')

    def test_state_edits_and_idempotent_retry(self):
        saved=self.save('Original question',1,{'context':'new'})
        retry=self.save('Original question',1,{'context':'new'})
        self.assertEqual(saved['revision'],retry['revision'])
        self.assertEqual(saved['last_message_at'],self.session['last_message_at'])
        with self.assertRaises(store.SessionConflictError):self.save('Original question',1,{'context':'stale'})
        self.assertEqual(self.save('Original question',2,{'context':'edited'})['revision'],3)

    def test_concurrent_divergent_writers_have_one_winner(self):
        def write(i):
            try:self.save('Question '+str(i),1);return True
            except store.SessionConflictError:return False
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
            self.assertEqual(sum(pool.map(write,range(16))),1)

    def test_other_owner_and_deleted_chat_cannot_be_changed(self):
        sid=self.session['id']
        self.assertIsNone(store.get_chat_session(sid,'other-owner'))
        self.assertFalse(store.hide_chat_session(sid,'other-owner'))
        with self.assertRaises(PermissionError):
            store.save_chat_session(sid,'other-owner','Test','42','Author',self.messages,{},1)
        self.assertTrue(store.hide_chat_session(sid,self.owner))
        with self.assertRaises(PermissionError):self.save('Late write',1)
        with store.get_db_conn() as conn:
            self.assertEqual(conn.execute('SELECT messages FROM matrix_chat_sessions WHERE id=%s',(sid,)).fetchone()['messages'],self.messages)

    def test_fifty_independent_owners(self):
        def probe(i):
            owner=self.owner+':'+str(i)
            session=store.create_chat_session(owner,'Test','42','Author',self.messages,{})
            try:
                self.assertIsNone(store.get_chat_session(session['id'],self.owner))
                self.assertEqual(store.list_chat_sessions(owner)[0]['revision'],1)
                store.save_chat_session(session['id'],owner,'Test','42','Author',self.messages,{'i':i},1)
                self.assertEqual(store.get_chat_session(session['id'],owner)['state'],{'i':i})
            finally:
                with store.get_db_conn() as conn:
                    conn.execute('DELETE FROM matrix_chat_sessions WHERE id=%s',(session['id'],));conn.commit()
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as pool:list(pool.map(probe,range(50)))
