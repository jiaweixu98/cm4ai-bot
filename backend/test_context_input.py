import json
import unittest
from types import SimpleNamespace
from research_agent import _build_input

class ContextInputTests(unittest.TestCase):
    def request(self, papers, scope='selected'):
        return SimpleNamespace(conversation_history=[], user_input='Any selected publications?',
            current_query='', working_context=SimpleNamespace(goal='', requirements=[]),
            attached_context=[], aid='unlinked', context_person_ids=[], search_results=[],
            _publication_scope=scope, _publication_filters=[], _resolved_publications=papers)

    def test_empty_selection_is_explicit_even_in_selected_mode(self):
        body=json.loads(_build_input(self.request([]))[-1]['content'])
        self.assertEqual(body['publication_scope'], 'selected')
        self.assertEqual(body['selected_publication_count'], 0)
        self.assertEqual(body['selected_publications'], [])
        self.assertFalse(body['user_publications_on_record'])

    def test_count_uses_resolved_papers(self):
        paper={'author_id':'123','work_id':'W1','title':'Verified paper'}
        body=json.loads(_build_input(self.request([paper]))[-1]['content'])
        self.assertEqual(body['selected_publication_count'], 1)
        self.assertEqual(body['selected_publications'], [paper])

    def test_eligible_mode_is_not_an_explicit_selection(self):
        body=json.loads(_build_input(self.request([], 'eligible'))[-1]['content'])
        self.assertEqual(body['selected_publication_count'], 0)
        self.assertEqual(body['publication_scope'], 'eligible')

class PersonalContextTests(ContextInputTests):
    def test_self_reported_context_is_separate_from_verified_profile(self):
        from context_models import ContextChoices
        req=self.request([])
        req.context_choices=ContextChoices(research_interests='Biomedical standards',known_people=[{'name':'QA person','relationship':'met','note':'Workshop'}])
        body=json.loads(_build_input(req)[-1]['content'])
        self.assertEqual(body['self_reported_research_interests'],'Biomedical standards')
        self.assertEqual(body['self_reported_connections'][0]['relationship'],'met')
        self.assertFalse(body['has_profile_context'])
        self.assertFalse(body['has_selected_people'])
        self.assertFalse(body['user_publications_on_record'])

    def test_current_empty_context_is_explicit(self):
        body=json.loads(_build_input(self.request([]))[-1]['content'])
        self.assertEqual(body['self_reported_research_interests'],'')
        self.assertEqual(body['self_reported_connections'],[])

    def test_context_limits_and_relationship_validation(self):
        from context_models import ContextChoices
        from pydantic import ValidationError
        for data in [{'research_interests':'x'*2001},{'known_people':[{'name':'QA','relationship':'verified'}]},{'known_people':[{'name':'QA','relationship':'met'}]*21}]:
            with self.assertRaises(ValidationError): ContextChoices(**data)

if __name__ == '__main__':
    unittest.main()
