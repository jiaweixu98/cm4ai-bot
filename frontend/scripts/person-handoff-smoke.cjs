// Exercise the actual startup effect without a browser or live model calls.
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const {createRequire} = require('node:module');
const root=require('node:path').resolve(__dirname, '..');
const req=createRequire(root+'/package.json');
const parser=req('next/dist/compiled/babel/parser');
const source=fs.readFileSync(process.argv[2] || root+'/app/page.js','utf8');
const ast=parser.parse(source,{sourceType:'module',plugins:['jsx']});
const home=ast.program.body.find(n=>n.type==='ExportDefaultDeclaration').declaration;
const effect=home.body.body.find(n=>n.type==='ExpressionStatement' && n.expression.callee?.name==='useEffect' && n.expression.arguments[1]?.elements?.some(e=>e.name==='applySessionSnapshot')).expression.arguments[0];
const effectSource=source.slice(effect.start,effect.end);
const flush=()=>new Promise(r=>setImmediate(r));
async function scenario({replay=false,token=false,historyFail=false,personFail=false,handoff=true,fresh=false}={}) {
 const state={}; const queued=[]; let restores=0;
 const search=handoff?'?embedded=1&handoff=person&context_person_id=5089373684&fresh=1':fresh?'?intent=mentor&fresh=1':'?intent=collaborator';
 const window={location:{search,pathname:'/',hash:''},
   sessionStorage:{getItem:()=>token?'test-token':'',setItem(){}},
   history:{replaceState(a,b,url){window.location.search=new URL(url,'http://test.local').search}},clearTimeout(){}};
 const setters=Object.fromEntries([...effectSource.matchAll(/\b(set[A-Z]\w+)\(/g)].map(m=>[m[1],v=>state[m[1]]=v]));
 const sandbox={...setters,window,URLSearchParams,Number,String,Array,Promise,Date,
   console:{error(){}},SESSION_TOKEN_STORAGE_KEY:'test',UNLINKED_AID:'unlinked',entryParamsRef:{current:null},
   sessionBootstrappedRef:{current:false},sessionSaveTimerRef:{current:null},
   parseAidParam:v=>/^\d+$/.test(v||'')?v:'unlinked',parsePersonaIntent:v=>v==='mentor'?'mentor':'collaborator',inBridgeIframe:()=>true,
   fetchAuthor:()=>new Promise((resolve,reject)=>queued.push(()=>personFail?reject(new Error('unavailable')):resolve({name:'Trey Ideker',affiliation:'UC San Diego',papers:[]}))),
   resetWorkflowState(){state.setContextPersonIds=[];state.setGraphHandoffPersonId=null;state.setProfileNotice='';state.setMessages=[]},
   applySessionSnapshot(session){restores++;state.setGraphContextPeople=[];state.setContextPersonIds=[];state.setGraphHandoffPersonId=null;state.setMessages=['Old chat']},
   listChatSessions:async()=>{if(historyFail)throw new Error('History unavailable');return {sessions:[{id:'old',intent:'collaborator',last_message_at:'2026-01-01'}]}},
   getChatSession:async()=>({session:{id:'old'}}),
 };
 vm.createContext(sandbox);
 const initialize=vm.runInContext(`(${effectSource})`,sandbox);
 const cleanup=initialize();
 if(replay){cleanup();initialize()}
 queued.forEach(resolve=>resolve()); await flush();await flush();
 if(personFail){assert.match(state.setProfileNotice,/could not be loaded/);assert.equal(state.setContextPersonIds.length,0)}
 else if(handoff){
   assert.equal(state.setGraphContextPeople?.[0]?.name,'Trey Ideker','handoff retains the named person');
   assert.deepEqual(Array.from(state.setContextPersonIds),[5089373684]);
   assert.equal(state.setGraphHandoffPersonId,5089373684);
   assert.equal(state.setInputValue,'');
   assert.equal(restores,0,'graph handoff starts a fresh conversation');
 } else if(fresh){assert.equal(restores,0,'fresh graph entry starts a blank conversation')}
 else {assert.equal(restores,1,'ordinary direct entry still restores history')}
 assert.equal(state.setUiReady,true);
 assert(!window.location.search.includes('context_person_id'));
}
(async()=>{
 for(const test of [{replay:true},{token:true,replay:true},{token:true,historyFail:true},{personFail:true},{token:true,handoff:false},{token:true,handoff:false,fresh:true}])await scenario(test);
 console.log('PASS: repeated initialization, authenticated handoff, graph-fresh entry, history failure, missing person, and ordinary history restoration.');
})().catch(e=>{console.error(e.message);process.exitCode=1});
