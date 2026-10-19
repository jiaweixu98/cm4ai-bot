"""Compare explanation guidance with identical, eligibility-checked paper evidence.

Uses the existing development credentials and a read-only catalog connection;
does not import main.py or load another model/index into RAM.
"""
import ast
import json
import os
from pathlib import Path
import sqlite3
import sys
import time
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
from openai import OpenAI
from tkg_publications import resolve_snapshot

load_dotenv(ROOT / '.env')
OUTPUT = Path(os.environ.get('REVIEW_OUTPUT', '/tmp/conference-review-20261016'))
tree = ast.parse((ROOT / 'research_agent.py').read_text())
assignment = next(n for n in tree.body if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == 'INSTRUCTIONS' for t in n.targets))
updated = assignment.value.func.value.value.split('# Output\n', 1)[1].split('- shortlist_kind:', 1)[0]
guidance = {'baseline':(OUTPUT / 'baseline-why-instructions.txt').read_text(), 'proposed':updated}
data = Path(resolve_snapshot(os.environ.get('LOCAL_DATA_DIR', str(ROOT.parent / 'data'))))
catalog = sqlite3.connect(f'file:{data / "papers.sqlite"}?mode=ro', uri=True)
catalog.row_factory = sqlite3.Row
client = OpenAI(timeout=60, max_retries=1)
model = os.environ.get('MATRIX_AGENT_MODEL') or os.environ.get('MATRIX_CHAT_MODEL') or 'gpt-6-luna'
report = {'model':model, 'same_people_and_evidence':True, 'complete':False, 'cases':[]}
if (OUTPUT / 'fixed-evidence-why.json').exists():
    report = json.loads((OUTPUT / 'fixed-evidence-why.json').read_text())
schema = {'type':'object', 'additionalProperties':False, 'required':['results'],
          'properties':{'results':{'type':'array','items':{'type':'object','additionalProperties':False,
            'required':['author_id','why','work_ids'], 'properties':{'author_id':{'type':'string'},
            'why':{'type':'string'}, 'work_ids':{'type':'array','items':{'type':'string'}}}}}}}
for case in json.loads((OUTPUT / 'natural.json').read_text())['cases']:
    if any(row['label'] == case['label'] for row in report['cases']):
        continue
    candidates = []
    for person in case['result'].get('shortlist', [])[:2]:
        refs = [{'author_id':person['author_id'], 'work_id':p['work_id']} for p in person['papers'] if p.get('work_id')]
        request = Request('http://127.0.0.1:8100/api/context-publications/resolve',
                          data=json.dumps({'references':refs}).encode(), headers={'Content-Type':'application/json'})
        with urlopen(request, timeout=10) as response:
            eligible = json.load(response)['selected_publications']
        papers = []
        for paper in eligible:
            row = catalog.execute('SELECT work_id,title,abstract,year FROM papers WHERE work_id=?', (paper['work_id'],)).fetchone()
            if row:
                papers.append(dict(row))
        if papers:
            candidates.append({'author_id':person['author_id'], 'name':person['name'], 'papers':papers})
    if not candidates:
        continue
    row = {'label':case['label'], 'request':case['prompt'], 'candidates':candidates, 'versions':{}}
    for label, instructions in guidance.items():
        started = time.monotonic()
        response = client.chat.completions.create(model=model,
            messages=[{'role':'system','content':instructions + '\nExplain only the supplied people in the supplied order. Return JSON with key results, a list of objects with author_id, why and work_ids. Every work_id must belong to that person. Do not search or add people.'},
                      {'role':'user','content':json.dumps({'request':case['prompt'], 'people':candidates})}],
            response_format={'type':'json_schema','json_schema':{'name':'fixed_evidence_comparison','strict':True,'schema':schema}},
            reasoning_effort='low', max_completion_tokens=3000)
        payload = json.loads(response.choices[0].message.content)
        assert [p['author_id'] for p in payload['results']] == [p['author_id'] for p in candidates]
        for result, person in zip(payload['results'], candidates):
            assert result['work_ids'] and set(result['work_ids']) <= {p['work_id'] for p in person['papers']}
        row['versions'][label] = {'seconds':round(time.monotonic()-started,1), 'results':payload['results']}
        print(case['label'], label, row['versions'][label]['seconds'], flush=True)
    report['cases'].append(row)
    (OUTPUT / 'fixed-evidence-why.json').write_text(json.dumps(report, indent=2))
report['complete'] = True
(OUTPUT / 'fixed-evidence-why.json').write_text(json.dumps(report, indent=2))
final = OUTPUT / 'polished.json'
if final.exists():
    checked = []
    for case in json.loads(final.read_text())['cases']:
        for person in case['result'].get('shortlist', []):
            refs = [{'author_id':person['author_id'], 'work_id':p['work_id']} for p in person['papers']]
            request = Request('http://127.0.0.1:8100/api/context-publications/resolve',
                              data=json.dumps({'references':refs}).encode(), headers={'Content-Type':'application/json'})
            with urlopen(request, timeout=10) as response:
                eligible = json.load(response)['selected_publications']
            assert {p['work_id'] for p in refs} == {p['work_id'] for p in eligible}, 'Displayed paper is not owned/eligible'
            papers = [dict(catalog.execute('SELECT work_id,title,abstract,year FROM papers WHERE work_id=?',
                                          (p['work_id'],)).fetchone()) for p in eligible]
            checked.append({'workflow':case['label'], 'author_id':person['author_id'],
                            'name':person['name'], 'why':person['why'], 'papers':papers})
    (OUTPUT / 'final-catalog-evidence.json').write_text(json.dumps({'ownership_and_eligibility_checked':True,
        'semantic_review_is_separate':True, 'people':checked}, indent=2))
    print('Final catalog ownership/eligibility checked:', len(checked), 'cards', flush=True)
