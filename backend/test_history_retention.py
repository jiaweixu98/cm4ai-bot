"""Integration checks against the isolated development database, never production."""
import os
import unittest
import uuid
from urllib.parse import urlparse
import session_store as store

@unittest.skipUnless(os.getenv('TEST_DATABASE_URL'), 'Set TEST_DATABASE_URL for isolated DB checks')
class HistoryRetentionTests(unittest.TestCase):
    def test_removal_retains_data_and_blocks_late_writes(self):
        url=os.environ['TEST_DATABASE_URL']
        target=urlparse(url)
        self.assertIn(target.hostname, ['127.0.0.1','localhost'])
        self.assertEqual(target.port,55432)
        previous=os.environ.get('DATABASE_URL')
        os.environ['DATABASE_URL']=url
        owner='test-history:'+str(uuid.uuid4())
        session=None
        try:
            messages=[{'role':'user','content':'Retention test question'}]
            session=store.create_chat_session(owner,'Test owner','unlinked','',messages,{'researchInterests':['Topic']})
            sid=session['id']
            self.assertFalse(store.hide_chat_session(sid,'another-owner'))
            self.assertIsNotNone(store.get_chat_session(sid,owner))
            self.assertTrue(store.hide_chat_session(sid,owner))
            self.assertTrue(store.hide_chat_session(sid,owner))
            self.assertIsNone(store.get_chat_session(sid,owner))
            self.assertEqual(store.list_chat_sessions(owner),[])
            with self.assertRaises(PermissionError):
                store.save_chat_session(sid,owner,'Test owner','unlinked','',messages+[{'role':'assistant','content':'Late reply'}],{})
            with store.get_db_conn() as conn:
                row=conn.execute('SELECT messages,state,hidden_at FROM matrix_chat_sessions WHERE id=%s',(sid,)).fetchone()
            self.assertEqual(row['messages'],messages)
            self.assertEqual(row['state'],{'researchInterests':['Topic']})
            self.assertIsNotNone(row['hidden_at'])
        finally:
            if session:
                with store.get_db_conn() as conn:
                    conn.execute('DELETE FROM matrix_chat_sessions WHERE id=%s AND owner_orcid=%s',(session['id'],owner));conn.commit()
            if previous is None: os.environ.pop('DATABASE_URL',None)
            else: os.environ['DATABASE_URL']=previous
