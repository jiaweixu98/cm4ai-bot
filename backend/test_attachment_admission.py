import ast
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
import unittest
import zipfile

from fastapi import FastAPI, HTTPException


class AttachmentAdmissionTests(unittest.IsolatedAsyncioTestCase):
    async def test_pdf_uses_shared_admission_and_docx_needs_no_model_slot(self):
        calls=[]
        @asynccontextmanager
        async def slot(name):
            calls.append(name)
            yield
        def pdf(*_):
            self.assertIn('attachment-extraction',calls)
            return 'PDF proposal text'
        tree=ast.parse(Path(__file__).with_name('main.py').read_text())
        route=next(n for n in tree.body if isinstance(n,ast.AsyncFunctionDef) and n.name=='attachment_text')
        namespace={'app':FastAPI(),'AttachmentTextRequest':object,'Path':Path,'asyncio':asyncio,
          'HTTPException':HTTPException,'zipfile':zipfile,'llm_request_slot':slot,
          'attachments':SimpleNamespace(decode=lambda _:b'document',pdf_text=pdf,docx_text=lambda _:'Word proposal text'),
          '_get_openai_client':lambda:None,'MODEL_NAME_CHAT':'test-model','_CHAT_FALLBACK_MODEL':'test-model'}
        exec(compile(ast.Module(body=[route],type_ignores=[]),'<attachment route>','exec'),namespace)
        handler=namespace['attachment_text']
        self.assertEqual(await handler(SimpleNamespace(filename='proposal.pdf',data_base64='fixture')),{'text':'PDF proposal text'})
        calls.clear()
        self.assertEqual(await handler(SimpleNamespace(filename='proposal.docx',data_base64='fixture')),{'text':'Word proposal text'})
        self.assertEqual(calls,[])

    async def test_busy_pdf_fails_before_calling_the_provider(self):
        @asynccontextmanager
        async def busy(_):
            raise HTTPException(429,'Busy')
            yield
        tree=ast.parse(Path(__file__).with_name('main.py').read_text())
        route=next(n for n in tree.body if isinstance(n,ast.AsyncFunctionDef) and n.name=='attachment_text')
        namespace={'app':FastAPI(),'AttachmentTextRequest':object,'Path':Path,'asyncio':asyncio,
          'HTTPException':HTTPException,'zipfile':zipfile,'llm_request_slot':busy,
          'attachments':SimpleNamespace(decode=lambda _:b'document')}
        exec(compile(ast.Module(body=[route],type_ignores=[]),'<attachment route>','exec'),namespace)
        with self.assertRaises(HTTPException) as failure:
            await namespace['attachment_text'](SimpleNamespace(filename='proposal.pdf',data_base64='fixture'))
        self.assertEqual(failure.exception.status_code,429)
