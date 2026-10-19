import unittest
from research_agent import AnswerBlock, AnswerPart, _part_problem
from research_tools import ResearchTools


class NamedPublicationTests(unittest.TestCase):
    def setUp(self):
        self.tools=ResearchTools(lambda *_:[],lambda *_:{},lambda *_:[])
        self.tools.evidence={'a':{'title':'Federated learning for clinical phenotype prediction'},
                             'b':{'title':'Fairness evaluation across distinct clinical research sites'}}

    def test_existing_but_wrong_citation_cannot_support_a_named_paper(self):
        part=AnswerPart(text='Recent work includes “Federated learning for clinical phenotype prediction”.',evidence_ids=['b'])
        block=AnswerBlock(kind='general',person_ids=[],parts=[part])
        self.assertEqual(_part_problem(block,part,self.tools),'named publication is not cited')

    def test_matching_reference_and_ordinary_quotes_remain_valid(self):
        for text in ['Read “Federated learning for clinical phenotype prediction”.',
                     'Consider “fairness across sites” as a research direction.']:
            part=AnswerPart(text=text,evidence_ids=['a'])
            block=AnswerBlock(kind='general',person_ids=[],parts=[part])
            self.assertIsNone(_part_problem(block,part,self.tools))
