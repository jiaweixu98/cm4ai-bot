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

if __name__ == '__main__':
    unittest.main()
