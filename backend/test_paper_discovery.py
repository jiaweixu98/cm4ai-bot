import json
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from paper_library import PaperLibrary
from paper_discovery import audience, explore
from geography import Geography


class DiscoveryTests(unittest.TestCase):
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
        self.env = patch.dict(os.environ, {'PROFILE_DECISIONS_DB':str(self.state)})
        self.env.start()
        self.library = PaperLibrary.open(str(self.path))

    def tearDown(self):
        self.env.stop(); self.directory.cleanup()

    def test_audience_excludes_source_authors_and_has_real_support(self):
        result = audience(self.library,None,'W1001')
        self.assertEqual([r['author_id'] for r in result['people']],['3'])
        self.assertEqual(result['people'][0]['matched_papers'][0]['work_id'],'W1002')
        self.assertEqual(result,audience(self.library,None,'W1001'))
        self.assertEqual(audience(self.library,None)['status'],'needs_paper')

    def test_explore_dates_apply_to_joint_papers_and_geography_has_a_source(self):
        self.assertEqual(explore(self.library,1,from_year=2021)['people'][0]['author_id'],'2')
        self.assertEqual(explore(self.library,1,to_year=2021)['people'][0]['author_id'],'4')
        self.assertEqual(explore(self.library,1,specialty='clinical phenotyping')['people'][0]['author_id'],'2')
        places=Geography({'012345678':{'country_code':'US','country_name':'United States','subdivision_code':'CA',
            'subdivision_name':'California','city':'San Diego','continent_code':'NA','continent_name':'North America'}},live=None,version='fixture')
        for typed in ('US','U.S.','USA','united states','America','California','US-CA','san diego','North America','San Diego, CA'):
            result=explore(self.library,1,geography=typed,locations=places)
            self.assertEqual([p['author_id'] for p in result['people']],['2'],typed)
        result=explore(self.library,1,geography='US',locations=places)
        self.assertEqual(result['people'][0]['location_evidence'][0]['source'],'https://ror.org/012345678')
        self.assertEqual(result['geography_basis'],'institution_on_shared_paper')
        self.assertEqual(result['geography_locations_version'],'fixture')
        # Person 4 has no ROR on the shared paper, so the answer says it is incomplete.
        self.assertFalse(result['geography_complete']); self.assertEqual(result['geography_unresolved_people'],1)
        self.assertEqual(result['geography_checked_people'],1)
        for typed in ('UK','Germany','Texas','Europe','CA'):  # CA alone is Canada
            self.assertEqual(explore(self.library,1,geography=typed,locations=places)['people'],[],typed)

    def test_institution_filter_ignores_punctuation_and_accepts_initials(self):
        for wanted in ('clinical-institute','CLINICAL  INSTITUTE.'):
            self.assertEqual([p['author_id'] for p in explore(self.library,1,institution=wanted,from_year=2021)['people']],['2'],wanted)
        from paper_discovery import _institution_matches as same
        self.assertTrue(same('UCSF','University of California, San Francisco') and same('MIT','Massachusetts Institute of Technology'))
        self.assertFalse(same('ucsf','Stanford University') or same('UCSD','University of California, San Francisco'))
        self.assertEqual(explore(self.library,1,institution='Other Place',from_year=2021)['people'],[])

    def test_unknown_ror_uses_bounded_live_lookup_and_caches_failures(self):
        calls=[]
        def live(key):
            calls.append(key)
            raise OSError('offline')
        places=Geography({},live=live)
        for _ in range(2): explore(self.library,1,geography='US',locations=places)
        self.assertEqual(calls,['012345678'])  # the failure is cached
        result=explore(self.library,1,geography='US',locations=places)
        self.assertEqual(result['people'],[]); self.assertFalse(result['geography_complete'])

    def test_pasted_title_of_a_catalog_paper_excludes_its_authors(self):
        pasted=audience(self.library,None,'','Clinical phenotyping with health records','')
        self.assertEqual(pasted['excluded_source_authors'],[1,2])
        self.assertEqual([p['author_id'] for p in pasted['people']],['3'])
        self.assertEqual(pasted['source_paper']['work_id'],'W1001')
        # A pasted title with a new abstract still excludes the catalog paper's authors.
        again=audience(self.library,None,'','Clinical phenotyping with health records','clinical phenotyping')
        self.assertNotIn('1',[p['author_id'] for p in again['people']])
        # A new paper that is not in the catalog excludes no one.
        new=audience(self.library,None,'','A brand new study of clinical phenotyping systems','clinical phenotyping')
        self.assertEqual(new['excluded_source_authors'],[])
        with self.assertRaises(ValueError): explore(self.library,1,from_year=2025,to_year=2024)

    def test_prose_only_model_answer_still_publishes_grounded_explore_cards(self):
        from research_tools import ResearchTools
        from research_agent import ResearchAnswer,AnswerBlock,AnswerPart,render_answer
        details=lambda aid:{'name':f'Person {aid}','affiliation':'Institute','papers':[], 'is_bridge2ai_member':True}
        tools=ResearchTools(lambda *_:[],details,lambda *_:[],library=self.library)
        tools.explore_coauthors('1','','','',2021,None)
        answer=ResearchAnswer(blocks=[AnswerBlock(kind='general',person_ids=[],parts=[AnswerPart(text='Here are the shared-paper matches.',evidence_ids=[])])],
            shortlist=[],shortlist_kind='researchers',shortlist_title='',result_update='keep',task_goal='',task_requirements=[],suggested_followups=[])
        rendered=render_answer(answer,tools)
        self.assertEqual([p['author_id'] for p in rendered['shortlist']],['2'])
        self.assertEqual(rendered['shortlist'][0]['papers'][0]['title'],'Clinical phenotyping with health records')
        self.assertTrue(rendered['shortlist'][0]['is_bridge2ai_member'])
        self.assertEqual(rendered['result_update'],'replace')


if __name__ == '__main__': unittest.main()
