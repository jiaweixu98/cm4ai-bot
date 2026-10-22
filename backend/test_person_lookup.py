import ast
from pathlib import Path
import re
import unicodedata
import unittest

# Exercise the actual lookup functions without starting model/index resources.
source=ast.parse(Path(__file__).with_name('main.py').read_text())
functions=[n for n in source.body if isinstance(n,ast.FunctionDef) and n.name in
           {'_normalize_person_name','_author_name_records','_find_people_by_name'}]

class PersonLookupTests(unittest.TestCase):
    def setUp(self):
        self.context={'re':re,'unicodedata':unicodedata,'_author_name_index':None,
          'load_author_nodes':lambda:{'42':{'features':{'FullName':'Moni C. Muñoz Torres',
          'name_aliases':['Mónica Muñoz-Torres','Moni C. Muñoz Torres'], 'Bridge2AISeedAuthor':True}},
          '43':{'features':{'FullName':'Monica Muñoz Torres'}}},
          '_get_author_summary':lambda aid:{'affiliation':'Example University'}}
        exec(compile(ast.Module(body=functions,type_ignores=[]),'<lookup>','exec'),self.context)
    def test_aliases_keep_display_name_and_one_result_per_identity(self):
        for query in ['Monica Munoz Torres','Mónica Muñoz-Torres','Moni Munoz Torres']:
            rows=self.context['_find_people_by_name'](query)
            self.assertEqual(rows[0]['author_id'],'42')
            self.assertEqual(rows[0]['name'],'Moni C. Muñoz Torres')
            self.assertEqual(sum(r['author_id']=='42' for r in rows),1)
    def test_ambiguous_names_do_not_merge_distinct_records(self):
        self.assertEqual(len(self.context['_find_people_by_name']('Munoz Torres')),2)
