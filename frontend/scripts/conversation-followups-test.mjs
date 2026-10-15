import assert from 'node:assert/strict';
import {conversationFollowUps} from '../app/lib/conversationFollowUps.mjs';
import {contextStarters} from '../app/lib/contextStarters.mjs';
for(const options of [{},{linked:true,profileName:'Jiawei Xu'},{choices:{personalContext:{interests:'Biomedical data standards',contacts:[]}}}]) {
 const starters=contextStarters(options);
 assert.equal(starters.length,6);
 for(const starter of starters) {
  const messages=[{role:'user',content:starter.prompt},{role:'assistant',content:'What would you like to learn?'}];
  assert.deepEqual(conversationFollowUps({messages,starters}),[],'No fallback to opening cards');
  assert.deepEqual(conversationFollowUps({messages,starters,suggestions:[starter.prompt]}),[],'No repeated starter');
  assert.deepEqual(conversationFollowUps({messages,starters,suggestions:['  What would you like to learn?  ']}),[]);
 }
}
const messages=[{role:'user',content:'Compare the two researchers.'},{role:'assistant',content:'Here is their evidence-based comparison.'}];
const suggestions=['COMPARE THE TWO RESEARCHERS!','Which paper supports the first recommendation?','Which paper supports the first recommendation?',{},'x'.repeat(121),'   '];
assert.deepEqual(conversationFollowUps({messages,suggestions}),['Which paper supports the first recommendation?']);
assert.deepEqual(conversationFollowUps({messages,suggestions,busy:true}),[]);
assert.deepEqual(conversationFollowUps({messages:[...messages,{role:'user',content:'New topic'}],suggestions}),[]);
assert.deepEqual(conversationFollowUps({messages,suggestions:[]}),[]);
const longest=contextStarters({people:Array.from({length:8},(_,i)=>({authorId:i,name:'R'.repeat(160)})),personIds:Array.from({length:8},(_,i)=>i),linked:true,profileName:'X'.repeat(160),choices:{publicationScope:'selected',selectedPapers:Array.from({length:8},()=>({title:'P'.repeat(400)})),personalContext:{interests:'I'.repeat(2000),contacts:Array.from({length:20},()=>({name:'N'.repeat(160),included:true}))}}});
assert.ok(longest.every(s=>s.prompt.length<=2000));
console.log('PASS six workflows across empty/linked/personalized accounts; no restart loops, duplicates, stale/busy suggestions or oversized drafts');
