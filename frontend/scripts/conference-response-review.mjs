// Sequential local-model review. Results contain complete replies and owned evidence,
// not OAuth acceptance or an assertion of scientific correctness.
import assert from 'node:assert/strict';
import {mkdir, readFile, writeFile} from 'node:fs/promises';
import {contextStarters} from '../app/lib/contextStarters.mjs';

const phase = process.argv[2] || 'natural';
assert(['baseline', 'natural', 'polished'].includes(phase));
const base = process.env.REVIEW_BACKEND || 'http://127.0.0.1:8100';
assert.equal(new URL(base).hostname, '127.0.0.1');
assert.equal(new URL(base).port, '8100');
const directory = process.env.REVIEW_OUTPUT || '/tmp/conference-review-20261016';
await mkdir(directory, {recursive:true});
const scientific = [
  "I'm a wet-lab biologist moving into AI-ready data standards. Who could I learn from?",
  "I'm planning a study to deploy a sepsis early-warning model across hospitals. Who could help with the clinical, machine learning, and implementation work?",
  'I work on federated learning with EHR data. What promising research directions could I explore?',
  'Who are my top 10 recorded coauthors?',
  "I'm planning a study combining clinical phenotyping and responsible AI. Which Bridge2AI researchers could I collaborate with?",
  "I'm proposing a paper on pathway enrichment and gene-expression networks for interpreting disease-associated molecular phenotypes. Who would be a relevant research audience? Please suggest up to four researchers.",
];
const starters = contextStarters({linked:true, profileName:'Jake Y. Chen'});
let manifest = starters.map((s, i) => ({label:s.label, aid:i === 3 ? '5058469124' : 'unlinked',
  intent:s.intent, prompt:phase === 'baseline' ? (i === 3 ? s.prompt : contextStarters()[i].prompt) : scientific[i]}));
if (phase === 'baseline') {
  const captured = await readFile(directory+'/baseline-prompts.json','utf8').then(JSON.parse).catch(()=>null);
  if (captured) manifest = captured;
  else await writeFile(directory+'/baseline-prompts.json', JSON.stringify(manifest,null,2));
}
const report = {phase, base, started_at:new Date().toISOString(), complete:false, cases:[]};
const previous = await readFile(directory+'/'+phase+'.json','utf8').then(JSON.parse).catch(()=>null);
if (previous?.cases) report.cases = previous.cases.filter(c=>c.status === 200 && c.result?.reply);
for (const entry of manifest) {
  if (report.cases.some(c=>c.label === entry.label)) continue;
  const start = performance.now();
  try {
    const response = await fetch(base+'/api/chat', {method:'POST',
      headers:{'Content-Type':'application/json'}, signal:AbortSignal.timeout(195000),
      body:JSON.stringify({aid:entry.aid, user_input:entry.prompt, intent:entry.intent})});
    const result = await response.json();
    report.cases.push({...entry, status:response.status, seconds:Math.round((performance.now()-start)/100)/10, result});
    console.log(JSON.stringify({phase,label:entry.label,status:response.status,seconds:report.cases.at(-1).seconds,
      people:result.shortlist?.length || 0,tools:result.tool_calls || [],reply:result.reply?.slice(0,200)}));
  } catch(error) {
    report.cases.push({...entry,status:0,error:error.name,seconds:Math.round((performance.now()-start)/100)/10});
    console.log(JSON.stringify({phase,label:entry.label,error:error.name}));
  }
  await writeFile(directory+'/'+phase+'.json',JSON.stringify(report,null,2));
}
report.complete = report.cases.length === 6 && report.cases.every(c=>c.status === 200 && c.result?.reply);
await writeFile(directory+'/'+phase+'.json',JSON.stringify(report,null,2));
if (!report.complete) process.exitCode=1;
