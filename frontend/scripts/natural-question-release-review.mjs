// Sequential real-model acceptance for identity, follow-ups and explicit paper scope.
// Use only the isolated development backend; this does not exercise provider login.
import assert from 'node:assert/strict';
import {mkdir,writeFile} from 'node:fs/promises';

const base=process.env.NATURAL_REVIEW_BACKEND || 'http://127.0.0.1:8100';
assert.equal(new URL(base).hostname,'127.0.0.1');assert.equal(new URL(base).port,'8100');
const directory=process.env.NATURAL_REVIEW_OUTPUT || '/tmp/natural-question-release-review';
await mkdir(directory,{recursive:true});
const report={started_at:new Date().toISOString(),source:'Sequential local model/catalog; no OAuth or production mutation',cases:[],complete:false};
async function send(label,body){
  const started=performance.now();
  const response=await fetch(base+'/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body),signal:AbortSignal.timeout(195000)});
  const result=await response.json();
  report.cases.push({label,prompt:body.user_input,status:response.status,seconds:Math.round((performance.now()-started)/100)/10,result});
  await writeFile(directory+'/context-review.json',JSON.stringify(report,null,2));
  assert.equal(response.status,200,label);assert(result.reply&&!result.recovered,label);
  console.log(label,JSON.stringify({tools:result.tool_calls,people:result.shortlist?.length||0,seconds:report.cases.at(-1).seconds}));
  return result;
}
const source='5106027423';
const selectedPrompt='Show up to 3 recorded coauthors of the researcher selected for this chat, with their shared publications.';
const selected=await send('Guest selected researcher has a distinct identity',{
  aid:'unlinked',intent:'collaborator',context_person_ids:[source],user_input:selectedPrompt});
assert(selected.tool_calls.includes('explore_coauthors'));
assert.equal(selected.explore_filters.author_id,source);assert(selected.shortlist?.length&&selected.shortlist.length<=3);
const readerOwnership=/\byour (?:coauthors?\b|coauthorship\b|papers\b|publications\b|research record\b)/i;
const prose=result=>[result.reply,...(result.shortlist||[]).map(person=>person.why),...(result.suggested_followups||[])].join('\n');
assert(!readerOwnership.test(prose(selected)),'Selected-person publications must not become the guest reader\'s own');
const follow=await send('Follow-up retains people and the selected-person identity',{
  aid:'unlinked',intent:'collaborator',context_person_ids:[source],
  conversation_history:[{role:'user',content:selectedPrompt},{role:'assistant',content:selected.reply}],
  search_results:selected.shortlist,working_context:selected.working_context,explore_filters:selected.explore_filters,
  user_input:'Which shared paper should I read to understand the first person\'s work?'});
assert.equal(follow.result_update,'keep');
assert(follow.citations?.some(citation=>selected.shortlist[0].papers.some(paper=>paper.work_id===citation.work_id||paper.title===citation.title)));
assert(!readerOwnership.test(prose(follow)),'Follow-up must keep the selected researcher separate from the guest');
const viewer='5058469124';
const empty=await send('Explicitly empty own-publication scope stays empty',{
  aid:viewer,intent:'collaborator',user_input:'Who are my top 10 recorded coauthors? Which publications do we share?',
  context_choices:{publication_scope:'selected',selected_publications:[]}});
assert(!empty.shortlist?.length,'An empty publication selection must not broaden into all papers');
const page=await (await fetch(base+'/api/author/'+viewer+'/publications?limit=1')).json();assert(page.papers.length);
const target=page.papers[0];
const promoted=await send('Audience uses the explicitly selected paper',{
  aid:viewer,intent:'collaborator',user_input:"Who would be a relevant research audience for the first paper I've selected? Please suggest up to four researchers.",
  context_choices:{publication_scope:'selected',selected_publications:[{author_id:viewer,work_id:target.work_id}]}});
assert(promoted.tool_calls.includes('find_paper_audience'));assert(promoted.shortlist?.length&&promoted.shortlist.length<=4);
assert(promoted.shortlist.every(person=>person.author_id!==viewer));
const attached=await send('Audience uses the attached proposed paper',{
  aid:'unlinked',intent:'collaborator',user_input:'Who would be a relevant research audience for the attached proposed paper? Please suggest up to four researchers.',
  attached_context:['Proposed paper: pathway and gene-set enrichment with gene-expression networks for interpreting disease-associated molecular phenotypes.']});
assert(attached.tool_calls.includes('find_paper_audience'));assert(attached.shortlist?.length&&attached.shortlist.length<=4);
report.complete=true;report.completed_at=new Date().toISOString();
await writeFile(directory+'/context-review.json',JSON.stringify(report,null,2));
console.log('PASS guest identity, preserved follow-up cards, explicit empty scope, selected-paper and attached-proposal audiences');
