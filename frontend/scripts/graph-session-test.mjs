import assert from 'node:assert/strict';
import fs from 'node:fs';
import {bootstrapGraphSession} from '../app/lib/graphSession.mjs';
function fixture(responses){const calls=[];return {calls,fetchImpl:async(path,options)=>{calls.push(path);assert.equal(options.cache,'no-store');assert.equal(options.credentials,'same-origin');options.signal?.throwIfAborted();const [status,body]=responses[path]||[500,{}];return {status,ok:status>=200&&status<300,json:async()=>body};}};}
const auth='/api/matrix/session-auth',account='/api/account',mapping='/api/node-mapping';
let f=fixture({[auth]:[401,{}]});assert.deepEqual(await bootstrapGraphSession(f),{token:'',name:'',authorId:null});assert.deepEqual(f.calls,[auth]);
for(const id of ['5101834496',-12,null]){
 f=fixture({[auth]:[200,{token:'fresh-token'}],[account]:[200,{account:{identities:[{provider:'orcid',name:'Test researcher'}]}}],[mapping]:[200,{authorId:id}]});
 assert.deepEqual(await bootstrapGraphSession(f),{token:'fresh-token',name:'Test researcher',authorId:id===null?null:String(id)});
}
f=fixture({[auth]:[200,{token:'google-token'}],[account]:[200,{account:{identities:[{provider:'google',name:'Google researcher'}]}}]});
assert.deepEqual(await bootstrapGraphSession(f),{token:'google-token',name:'Google researcher',authorId:null});assert(!f.calls.includes(mapping));
for(const path of [auth,account,mapping]){
 f=fixture({[auth]:[200,{token:'fresh'}],[account]:[200,{account:{identities:[{provider:'orcid',name:'Test'}]}}],[mapping]:[200,{authorId:1}],[path]:[503,{}]});
 await assert.rejects(()=>bootstrapGraphSession(f));
}
await assert.rejects(()=>bootstrapGraphSession({fetchImpl:async()=>{throw new TypeError('network');}}));
const controller=new AbortController();controller.abort();f=fixture({[auth]:[200,{token:'unused'}]});await assert.rejects(()=>bootstrapGraphSession({...f,signal:controller.signal}),{name:'AbortError'});
f=fixture({[auth]:[200,{}]});await assert.rejects(()=>bootstrapGraphSession(f));
const source=fs.readFileSync(new URL('../app/page.js',import.meta.url),'utf8');
const standalone=source.slice(source.indexOf('if (!isEmbedded &&'),source.indexOf('      setIntent(nextIntent);',source.indexOf('if (!isEmbedded &&')));
assert(standalone.indexOf('removeItem')<standalone.indexOf('await bootstrapGraphSession'));
assert(standalone.includes('aidParam = parseAidParam(identity.authorId)'));
assert(standalone.includes('resolvedToken = identity.token'));
assert(standalone.includes('seekerNameParam = identity.name'));
console.log('PASS current-cookie standalone bootstrap: ORCID, Google-only, catalog-only, unlinked, guest, malformed/session/upstream failures and abort; stale storage cleared and embedded handoff preserved');
assert(source.includes('/^\\/matrix(?:\\/|$)/.test(window.location.pathname)'), 'root-hosted previews keep their existing entry contract');
