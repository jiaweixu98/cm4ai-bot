import asyncio
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import AsyncMock, patch
from research_agent import (_fallback_answer, _repair_style, ResearchAnswer,
                            render_answer, stream_research_turn, ShortlistEntry)
from research_tools import ResearchTools


def tools():
    return ResearchTools(lambda *_: [], lambda *_: {}, lambda *_: [])


def request():
    return SimpleNamespace(aid='unlinked', context_choices=None, context_person_ids=[],
        search_results=[], conversation_history=[], user_input='Continue comparing the candidates',
        current_query='Single-cell mentors', attached_context=[],
        working_context=SimpleNamespace(goal='Learn single-cell genomics', requirements=['Python', 'beginner']))


class RecoveryTests(TestCase):
    def test_failed_search_preserves_previous_cards_and_task(self):
        service = tools()
        service.calls.append('search_people')
        req = request()
        payload = render_answer(_fallback_answer('Please retry.', req.working_context), service)
        self.assertEqual(payload['result_update'], 'keep')
        self.assertEqual(payload['working_context'], {'goal': req.working_context.goal, 'requirements': ['Python', 'beginner']})
        self.assertNotIn('shortlist', payload)
        self.assertEqual(payload['suggested_followups'], [])
        self.assertTrue(payload['recovered'])

    def test_recovery_marker_is_not_model_writable(self):
        self.assertNotIn('_is_recovery', ResearchAnswer.model_json_schema()['properties'])
        self.assertFalse(ResearchAnswer.model_validate(_fallback_answer('Retry').model_dump())._is_recovery)


class AgentRuntimeTests(IsolatedAsyncioTestCase):
    def setUp(self):
        client = patch('research_agent.AsyncOpenAI', return_value=AsyncMock())
        client.start()
        self.addCleanup(client.stop)

    async def test_scientific_limitations_do_not_trigger_rewriting(self):
        answer = _fallback_answer('This is a plausible niche hypothesis, not a measured gap. Availability is unverified.')
        answer.suggested_followups = ['What evidence could verify this hypothesis?']
        client = SimpleNamespace(responses=SimpleNamespace(create=AsyncMock()))
        before = answer.model_dump()
        await _repair_style(client, 'test-model', answer)
        client.responses.create.assert_not_called()
        self.assertEqual(answer.model_dump(), before)

    async def test_pronoun_repair_instructs_preserving_uncertainty(self):
        answer = _fallback_answer('She may be relevant, but her availability is unknown.')
        client = SimpleNamespace(responses=SimpleNamespace(create=AsyncMock(return_value=SimpleNamespace(
            output_text='{"texts":["They may be relevant, but their availability is unknown."]}'))))
        await _repair_style(client, 'test-model', answer)
        self.assertIn('unknown', answer.blocks[0].parts[0].text)
        instructions = client.responses.create.call_args.kwargs['instructions']
        self.assertIn('Preserve every factual claim', instructions)
        self.assertIn('Do not add claims', instructions)

    async def test_sdk_limit_handlers_preserve_context(self):
        for kind in ('max_turns', 'invalid_final_output'):
            with self.subTest(kind=kind):
                def run(*args, **kwargs):
                    answer = ResearchAnswer.model_validate(kwargs['error_handlers'][kind](None).final_output.model_dump())
                    async def events():
                        if False: yield None
                    return SimpleNamespace(final_output=answer, is_complete=True, stream_events=events)
                with patch('research_agent.Runner.run_streamed', side_effect=run):
                    events = [event async for event in stream_research_turn(request(), tools(), 'test-model')]
                payload = events[-1][1]
                self.assertTrue(payload['recovered'])
                self.assertEqual(payload['working_context']['requirements'], ['Python', 'beginner'])
                self.assertEqual(payload['result_update'], 'keep')

    async def test_cancelling_turn_cancels_sdk_run(self):
        started = asyncio.Event()
        cancelled = []
        async def events():
            started.set()
            await asyncio.Event().wait()
            if False: yield None
        result = SimpleNamespace(is_complete=False, stream_events=events, cancel=lambda: cancelled.append(True))
        async def consume():
            return [event async for event in stream_research_turn(request(), tools(), 'test-model')]
        with patch('research_agent.Runner.run_streamed', return_value=result):
            task = asyncio.create_task(consume())
            await asyncio.wait_for(started.wait(), 5)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError): await task
        self.assertEqual(cancelled, [True])

    async def test_optional_repair_timeout_keeps_answer(self):
        answer = _fallback_answer('A usable answer.', request().working_context)
        answer._is_recovery = False
        async def events():
            if False: yield None
        async def slow_repair(*args):
            await asyncio.Event().wait()
        result = SimpleNamespace(final_output=answer, is_complete=True, stream_events=events)
        real_timeout = asyncio.timeout
        with patch('research_agent.Runner.run_streamed', return_value=result), \
             patch('research_agent._repair_reasons', side_effect=slow_repair), \
             patch('research_agent.asyncio.timeout', side_effect=lambda seconds: real_timeout(0.01 if seconds == 10 else seconds)):
            events = [event async for event in stream_research_turn(request(), tools(), 'test-model')]
        self.assertEqual(events[-1][1]['reply'], 'A usable answer.')
        self.assertEqual(events[-1][1]['working_context']['goal'], request().working_context.goal)


class MultipleSearchTests(TestCase):
    def service(self):
        def details(aid):
            return {'name': 'Researcher ' + aid, 'papers': [
                {'title': 'Single-cell publication by researcher ' + aid, 'year': '2025'}]}
        def search(query, scope, excluded):
            ids = ['1', '2'] if query == 'first' else ['1', '3']
            return [{'author_id': aid, 'score': 1, 'is_bridge2ai_member': True}
                    for aid in ids if aid not in excluded]
        return ResearchTools(lambda *_: [], details, search)

    def test_two_queries_preserve_cited_candidate_from_first(self):
        service = self.service()
        first = service.search_people('first', [], 'all', [])
        second = service.search_people('second', [], 'all', [])
        answer = _fallback_answer('Two researchers.')
        answer._is_recovery = False
        for person in first['people']:
            answer.shortlist.append(ShortlistEntry(author_id=person['author_id'], why='Relevant single-cell publication.',
                evidence_ids=[person['papers'][0]['evidence_id']]))
        payload = render_answer(answer, service)
        self.assertEqual({p['author_id'] for p in payload['shortlist']}, {'1', '2'})
        self.assertEqual([p['author_id'] for p in second['people']], ['1', '3'])
        self.assertEqual([p['author_id'] for p in service.result_people], ['1', '3', '2'])

    def test_hard_scope_and_exclusions_do_not_resurrect_candidates(self):
        for scope, excluded in [('bridge2ai', []), ('all', ['2'])]:
            service = self.service()
            service.search_people('first', [], 'all', [])
            service.search_people('second', [], scope, excluded)
            self.assertNotIn('2', [p['author_id'] for p in service.result_people])
