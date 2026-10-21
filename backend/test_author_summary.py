"""Startup summary retains full-details identity/affiliation without paper reads."""
import ast, unittest
from pathlib import Path
from unittest.mock import Mock
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tkg_publications

source=ast.parse(Path(__file__).with_name('main.py').read_text())
names={'_get_author_summary','get_author','_get_user_name','_get_author_details','_build_catalog_author_details','AuthorResponse'}
class SummaryTests(unittest.TestCase):
 def check(self, features, catalog=None, library=True, exists=True, alias=False):
  nodes={'42':{'title':'Fallback','features':features}} if exists else {}
  lib=Mock() if library else None
  if lib:
   lib.canonical_author.return_value='42';lib.affiliations.return_value=[]
   lib.publication_page.return_value={'papers':[]};lib.topics.return_value=[]
  ns=dict(app=FastAPI(),HTTPException=HTTPException,BaseModel=BaseModel,
          load_paper_library=lambda:lib,load_author_nodes=lambda:nodes,
          _get_catalog_author=lambda aid:catalog,_is_linked_author_id=lambda aid:True)
  exec(compile(ast.Module(body=[n for n in source.body if getattr(n,'name','') in names],type_ignores=[]),'summary','exec'),ns)
  old=ns['_get_author_details']('42')
  if lib:lib.reset_mock();lib.canonical_author.return_value='42'
  result=ns['get_author']('17' if alias else '42')
  self.assertEqual(result.name,ns['_get_user_name']('42'))
  self.assertEqual(result.affiliation,old['affiliation'])
  self.assertEqual(result.id,'42')
  if lib:lib.publication_page.assert_not_called();lib.topics.assert_not_called()
 def test_node(self):self.check({'FullName':'Test','affiliations':[]})
 def test_alias(self):self.check({'FullName':'Test'},alias=True)
 def test_catalog(self):self.check({},dict(fullName='Catalog',primaryAffiliation='Source',source_affiliations=[]),exists=False)
 def test_no_library(self):self.check({'FullName':'Test','Affiliation':'Legacy'},library=False)
 def test_catalog_no_library(self):self.check({},dict(fullName='Catalog',primaryAffiliation='Source'),library=False,exists=False)
 def test_empty_node(self):self.check({})
 def test_route_is_threaded(self):
  route=next(n for n in source.body if getattr(n,'name','')=='get_author')
  self.assertIsInstance(route,ast.FunctionDef)
if __name__=='__main__':unittest.main()
