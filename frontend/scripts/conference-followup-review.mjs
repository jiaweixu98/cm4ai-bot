// Focused live acceptance after the citation-title guard. Local backend only.
import assert from 'node:assert/strict';
import {readFile,writeFile,copyFile} from 'node:fs/promises';
const directory=process.env.REVIEW_OUTPUT || '/tmp/conference-review-20261016';
const base='http://127.0.0.1:8100';
const report=await readFile(directory+'/followups-and-attachments.json','utf8').then(JSON.parse).catch(()=>({source:'Live local catalog and model; no OAuth acceptance',cases:[]}));
async function send(label,body) {
 const previous=report.cases.find(c=>c.label===label);
 if(previous) return previous;
 const start=performance.now();
 const response=await fetch(base+'/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body),signal:AbortSignal.timeout(195000)});
 const result=await response.json();
 assert.equal(response.status,200,label);assert(result.reply && !result.recovered,label);
 const row={label,prompt:body.user_input,seconds:Math.round((performance.now()-start)/100)/10,result};
 report.cases.push(row);await writeFile(directory+'/followups-and-attachments.json',JSON.stringify(report,null,2));
 console.log(label,JSON.stringify({tools:result.tool_calls,people:result.shortlist?.length || 0,seconds:row.seconds}));
 return row;
}
const selected=JSON.parse(await readFile(directory+'/polished.json','utf8'));
const niche=selected.cases.find(c=>c.label==='Find a niche');
const reviewed=await send('Niche after citation-title validation',{aid:'unlinked',user_input:niche.prompt,intent:'collaborator'});
assert(reviewed.result.tool_calls.includes('analyze_niche'));
if(!selected.niche_citation_guard_reviewed) await copyFile(directory+'/polished.json',directory+'/polished-before-citation-fix.json');
selected.cases[selected.cases.indexOf(niche)]={...niche,...reviewed,label:'Find a niche'};
selected.niche_citation_guard_reviewed=true;
await writeFile(directory+'/polished.json',JSON.stringify(selected,null,2));
const explore=selected.cases.find(c=>c.label==='Explore');
const follow=await send('Explain first coauthor without replacing cards',{
 aid:'5058469124',user_input:'Which publication do I share with the first person in this list?',
 conversation_history:[{role:'user',content:explore.prompt},{role:'assistant',content:explore.result.reply}],
 search_results:explore.result.shortlist,working_context:explore.result.working_context,
 explore_filters:explore.result.explore_filters,intent:'collaborator'});
assert.equal(follow.result.result_update,'keep');
assert(follow.result.citations.some(c=>explore.result.shortlist[0].papers.some(p=>p.title===c.title)));
const papers=await (await fetch(base+'/api/author/5058469124/publications?limit=1')).json();
assert(papers.papers.length);
const target=papers.papers[0];
const promoted=await send('Promote explicit selected publication',{
 aid:'5058469124',intent:'collaborator',user_input:"Who would be a relevant research audience for the first paper I've selected? Please suggest up to four researchers.",
 context_choices:{publication_scope:'selected',selected_publications:[{author_id:'5058469124',work_id:target.work_id}]}});
assert(promoted.result.tool_calls.includes('find_paper_audience'));
assert(promoted.result.shortlist?.length && promoted.result.shortlist.length<=4);
assert(promoted.result.shortlist.every(p=>p.author_id!=='5058469124'));
const attached=await send('Explicit audience for attached proposal',{
 aid:'unlinked',intent:'collaborator',user_input:'Who would be a relevant research audience for the attached proposed paper? Please suggest up to four researchers.',
 attached_context:['Proposed paper: pathway and gene-set enrichment with gene-expression networks for interpreting disease-associated molecular phenotypes.']});
assert(attached.result.tool_calls.includes('find_paper_audience'));
assert(attached.result.shortlist?.length && attached.result.shortlist.length<=4);
report.complete=true;
await writeFile(directory+'/followups-and-attachments.json',JSON.stringify(report,null,2));
