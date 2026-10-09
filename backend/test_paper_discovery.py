import json
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from paper_library import PaperLibrary
from paper_discovery import audience, explore as source_explore, explore_people

def current_affiliations(aid):
    return [{"institution":"Clinical Institute","ror_id":"https://ror.org/012345678","source":"orcid","trusted":True,"is_current":True}] if str(aid)=="2" else []

def explore(*args, **kwargs):
    return source_explore(*args, affiliation_lookup=current_affiliations, **kwargs)

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
            UPDATE papers SET primary_topic='clinical phenotyping' WHERE work_id IN ('W1001','W1002');
            UPDATE papers SET primary_topic='ocean' WHERE work_id='W1003';
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
                CREATE TABLE decisions(author_id INTEGER); CREATE TABLE decision_events(revision INTEGER PRIMARY KEY);''')
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

    def test_audience_accepts_description_without_inventing_title(self):
        result = audience(self.library,None,abstract='clinical phenotyping electronic health records')
        self.assertEqual(result['status'],'ok')
        self.assertTrue(result['people'])
        self.assertEqual(result['source_paper']['title'],'')

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
        self.assertEqual(result['geography_basis'],'current_trusted_affiliations')
        self.assertEqual(result['geography_locations_version'],'fixture')
        # Person 4 has no resolved current trusted institution.
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

    def test_historical_and_imported_affiliations_cannot_establish_geography(self):
        places=Geography({'012345678':{'country_code':'US','country_name':'United States'}},live=None)
        for aff in ({'institution':'Clinical Institute','ror_id':'https://ror.org/012345678','trusted':True,'is_current':False,'end_year':2024},
                    {'institution':'Clinical Institute','ror_id':'https://ror.org/012345678','trusted':False,'is_current':True}):
            result=source_explore(self.library,1,geography='US',locations=places,affiliation_lookup=lambda aid:[aff])
            self.assertEqual(result['people'],[])

    def test_specialty_keeps_specific_terms_and_names_a_nearby_topic(self):
        with sqlite3.connect(self.path / 'papers.sqlite') as db:
            db.execute("INSERT INTO papers VALUES('W1005','Clinical glioblastoma radiation','clinical radiation resistance',2022,NULL,NULL,NULL,NULL,0,'glioblastoma',NULL,'clinical glioblastoma radiation')")
            db.execute("INSERT INTO paper_authors VALUES('W1005',1,1,'confirmed'),('W1005',5,2,'unknown')")
            rowid = db.execute("SELECT rowid FROM papers WHERE work_id='W1005'").fetchone()[0]
            db.execute("INSERT INTO papers_fts(rowid,title,abstract) VALUES(?,?,?)", (rowid, 'Clinical glioblastoma radiation', 'clinical radiation resistance'))
        result = explore(self.library, 1, specialty='clinical phenotyping')
        self.assertNotIn('5', [person['author_id'] for person in result['people']])
        matched = next(person for person in result['people'] if person['author_id'] == '2')
        self.assertTrue(all('phenotyp' in (paper['title'] + ' ' + (paper.get('abstract') or '')).casefold() for paper in matched['matched_papers']))
        empty = explore(self.library, 1, specialty='quantum cryptocurrency')
        self.assertEqual(empty['people'], [])
        self.assertIn('clinical phenotyping', empty['nearby_topics'])

    def test_topics_or_combined_categories_and_complete_pagination(self):
        first=explore_people(self.library,'1',topics=['clinical phenotyping','ocean'],coauthors_only=True,limit=1)
        second=explore_people(self.library,'1',topics=['clinical phenotyping','ocean'],coauthors_only=True,offset=1,limit=1)
        self.assertEqual(first['total'],2);self.assertTrue(first['has_more'])
        self.assertEqual([p['author_id'] for p in first['people']+second['people']],['2','4'])
        self.assertEqual(explore_people(self.library,'1',topics=['ocean'],from_year=2025,to_year=2025,coauthors_only=True)['people'],[])
        exact=explore_people(self.library,'1',topics=['clinical phenotyping'],from_year=2025,to_year=2025,coauthors_only=True)
        self.assertEqual([p['author_id'] for p in exact['people']],['2'])

    def test_selected_context_is_live_and_automatic_papers_do_not_migrate(self):
        from context_selection import resolve_references, revalidate_session
        ref={'author_id':'2','work_id':'W1001'}
        self.assertEqual(resolve_references(self.library,[ref])[0]['work_id'],'W1001')
        recent={'aid':'2','state':{'contextChoices':{'paperScope':'profile','paperTitles':['Clinical phenotyping with health records']}}}
        self.assertEqual(revalidate_session(recent,self.library)['state']['contextChoices']['selectedPapers'],[])
        explicit={'focal_author_id':'2','state':{'contextChoices':{'paperScope':'chosen','paperTitles':['Clinical phenotyping with health records']}}}
        self.assertEqual(revalidate_session(explicit,self.library)['state']['contextChoices']['selectedPapers'][0]['work_id'],'W1001')
        with sqlite3.connect(self.state) as db: db.execute("INSERT INTO excluded_links VALUES('test-v1',2,'W1001')")
        self.assertEqual(resolve_references(self.library,[ref]),[])
        self.assertEqual(resolve_references(self.library,[],['Clinical phenotyping with health records'],'2'),[])

    def test_read_context_has_topics_but_only_explicit_profile_papers(self):
        from research_tools import ResearchTools
        tools=ResearchTools(lambda *_:[],lambda aid:{'name':f'Person {aid}',
            'topics':['clinical phenotyping'],'papers':[{'Title':'Injected browser title'}]},lambda *_:[],library=self.library)
        initial=tools.read_context('1',['2'],[])
        self.assertEqual(initial['profile_context']['topics'],['clinical phenotyping'])
        self.assertEqual(initial['profile_context']['papers'],[])
        self.assertEqual(initial['selected_people'][0]['papers'],[])
        self.assertEqual(initial['selected_publications'],[])
        tools.selected_publication_refs=[{'author_id':'1','work_id':'W1001'}]
        chosen=tools.read_context('1',[],[])
        self.assertEqual([p['work_id'] for p in chosen['profile_context']['papers']],['W1001'])
        with sqlite3.connect(self.state) as db: db.execute("INSERT INTO excluded_links VALUES('test-v1',1,'W1001')")
        self.assertEqual(tools.read_context('1',[],[])['selected_publications'],[])

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

    def _tools(self):
        from research_tools import ResearchTools
        return ResearchTools(lambda *_: [], lambda aid: {
            'name': f'Person {aid}', 'affiliation': 'Current Institution', 'papers': []
        }, lambda *_: [], library=self.library)

    def _answer(self, text, kind='general', person_ids=None, shortlist=None):
        from research_agent import ResearchAnswer, AnswerBlock, AnswerPart
        return ResearchAnswer(blocks=[AnswerBlock(kind=kind, person_ids=person_ids or [],
            parts=[AnswerPart(text=text, evidence_ids=[])])], shortlist=shortlist or [],
            shortlist_kind='researchers', shortlist_title='', result_update='keep',
            task_goal='', task_requirements=[], suggested_followups=[])

    def test_explore_requested_count_preserves_deterministic_order(self):
        tools = self._tools()
        full = tools.explore_coauthors('1', '', '', '', None, None)
        self.assertEqual(len(full['people']), 2)
        limited = tools.explore_coauthors('1', '', '', '', None, None, limit=1)
        self.assertEqual([p['author_id'] for p in limited['people']],
                         [full['people'][0]['author_id']])
        self.assertEqual(len(tools.result_people), 1)

    def test_promote_abstention_does_not_publish_nearest_papers(self):
        from research_agent import render_answer
        tools = self._tools()
        with patch('paper_vectors.load_paper_index', return_value=None):
            tools.find_paper_audience('W1001', '', '')
        self.assertTrue(tools.result_people)
        answer = self._answer('No directly relevant published work was found; I am not recommending anyone.')
        rendered = render_answer(answer, tools)
        self.assertNotIn('shortlist', rendered)
        self.assertEqual(rendered['result_update'], 'replace')

    def test_promote_abstention_after_explore_still_withholds_cards(self):
        from research_agent import render_answer
        tools = self._tools()
        tools.explore_coauthors('1', '', '', '', 2021, None)
        with patch('paper_vectors.load_paper_index', return_value=None):
            tools.find_paper_audience('W1001', '', '')
        rendered = render_answer(self._answer('No audience members have directly relevant work.'), tools)
        self.assertNotIn('shortlist', rendered)

    def test_positive_promote_keeps_cited_publication_and_own_author_exclusion(self):
        from research_agent import render_answer, ShortlistEntry
        tools = self._tools()
        with patch('paper_vectors.load_paper_index', return_value=None):
            tools.find_paper_audience('W1001', '', '')
        person = tools.result_people[0]
        entry = ShortlistEntry(author_id=person['author_id'], why='Their clinical phenotyping study addresses this topic.',
                               evidence_ids=[person['papers'][0]['evidence_id']])
        rendered = render_answer(self._answer('Potential audience based on published work.', shortlist=[entry]), tools)
        self.assertEqual([p['author_id'] for p in rendered['shortlist']], ['3'])
        self.assertEqual(rendered['shortlist'][0]['papers'][0]['title'], 'Portable clinical phenotyping systems')

    def test_empty_explore_is_an_answer_even_when_model_mislabels_summary(self):
        from research_agent import render_answer
        tools = self._tools()
        tools.read_person_evidence('1', '')
        tools.explore_coauthors('1', '', '', '', 2035, 2040)
        answer = self._answer('No shared publications were returned for this date range.', 'research', ['1'])
        rendered = render_answer(answer, tools)
        self.assertEqual(rendered['reply'], 'No recorded coauthors have papers on that topic.')
        self.assertNotIn('shortlist', rendered)
        self.assertEqual(rendered['citations'], [])
        self.assertEqual(rendered['result_update'], 'replace')

    def test_new_cards_avoid_duplicate_person_prose_but_followups_keep_it(self):
        from research_agent import render_answer, ShortlistEntry, AnswerBlock, AnswerPart
        tools = self._tools()
        with patch('paper_vectors.load_paper_index', return_value=None):
            tools.find_paper_audience('W1001', '', '')
        person = tools.result_people[0]
        evidence = person['papers'][0]['evidence_id']
        entry = ShortlistEntry(author_id='3', why='Their portable clinical phenotyping study addresses this topic.',
                               evidence_ids=[evidence])
        answer = self._answer('Potential audience based on published work.', shortlist=[entry])
        detail = 'Person 3 published work on portable clinical phenotyping.'
        answer.blocks.append(AnswerBlock(kind='research', person_ids=['3'],
                             parts=[AnswerPart(text=detail, evidence_ids=[evidence])]))
        rendered = render_answer(answer, tools)
        self.assertNotIn(detail, rendered['reply'])
        self.assertEqual(rendered['shortlist'][0]['why'], entry.why)
        # A question about an earlier person has no new result set: retain its answer.
        tools.result_people = None
        followup = render_answer(answer, tools)
        self.assertIn(detail, followup['reply'])
        self.assertNotIn('shortlist', followup)

    def test_local_paper_info_uses_printed_institutions_and_never_needs_network(self):
        tools = self._tools()
        with patch.object(tools, '_openalex_raw', side_effect=AssertionError('Local paper made a network call')):
            result = tools.get_paper_info('W1001')
            by_name = {author['name']: author for author in result['authors']}
            self.assertEqual(by_name['Person 1']['institutions'], [])
            self.assertEqual(by_name['Person 2']['institutions'], ['Clinical Institute'])
            self.assertTrue(all(person['affiliation'] == 'Current Institution' for person in result['catalog_people']))
            abstract = tools.read_abstracts([result['evidence_id']])['abstracts'][result['evidence_id']]
            self.assertEqual(abstract['abstract'], 'clinical phenotyping electronic records')
            exact = tools.get_paper_info('Clinical phenotyping with health records')
            self.assertEqual(exact['title'], result['title'])

    def test_local_title_lookup_does_not_attach_ambiguous_or_short_titles(self):
        self.assertIsNone(self.library.by_title('Ocean ecosystem biology'))
        with sqlite3.connect(self.path / 'papers.sqlite') as db:
            db.execute("INSERT INTO papers SELECT 'W1004',title,abstract,year,venue,doi,doi_key,pmid,cited_by,primary_topic,primary_field,title_key FROM papers WHERE work_id='W1001'")
        self.assertIsNone(self.library.by_title('Clinical phenotyping with health records'))

    def test_context_reads_displayed_paper_instead_of_newest_papers(self):
        with sqlite3.connect(self.path / 'papers.sqlite') as db:
            for index in range(3):
                work = f'W20{index}'
                db.execute("INSERT INTO papers(work_id,title,abstract,year) VALUES(?,?,?,2026)",
                           (work, f'New unrelated study {index}', 'unrelated methods'))
                db.execute("INSERT INTO paper_authors VALUES(?,2,1,'unknown')", (work,))
        tools = self._tools()
        result = tools.read_context('', [], ['2'], [{'author_id': '2', 'papers': [{
            'title': 'Clinical phenotyping with health records', 'year': '2025',
            'abstract': 'An invented browser abstract',
        }]}])
        papers = result['displayed_people_in_order'][0]['papers']
        self.assertEqual([paper['title'] for paper in papers], ['Clinical phenotyping with health records'])
        abstract = tools.read_abstracts([papers[0]['evidence_id']])['abstracts'][papers[0]['evidence_id']]
        self.assertEqual(abstract['abstract'], 'clinical phenotyping electronic records')
        self.assertIsNone(tools.result_people)  # Reading prior cards must not replace them.

    def test_context_does_not_trust_supplied_paper_authorship(self):
        result = self._tools().read_context('', [], ['3'], [{'author_id': '3', 'papers': [{
            'work_id': 'W1001', 'title': 'Clinical phenotyping with health records',
        }]}])
        self.assertEqual(result['displayed_people_in_order'][0]['papers'], [])

    def test_context_withholds_displayed_paper_after_live_correction(self):
        with sqlite3.connect(self.state) as db:
            db.execute("INSERT INTO excluded_links VALUES('test-v1',2,'W1001')")
        result = self._tools().read_context('', [], ['2'], [{'author_id': '2', 'papers': [{
            'work_id': 'W1001', 'title': 'Clinical phenotyping with health records',
        }]}])
        self.assertEqual(result['displayed_people_in_order'][0]['papers'], [])


if __name__ == '__main__': unittest.main()
