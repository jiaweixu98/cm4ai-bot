import assert from 'node:assert/strict';
import {graphAccountProxy} from '../app/lib/graphAccountProxy.mjs';
const calls=[];
const fetchImpl=async (url,options)=>{calls.push({url:String(url),...options});return Response.json({ok:true});};
const request=(origin,extras={})=>new Request('http://127.0.0.1:3100/api/saved-people',{
  method:'PUT',headers:{Host:'127.0.0.1:3100',...(origin ? {Origin:origin} : {}),Cookie:'bridge_session=test-fixture',...extras},body:'{"authorId":-42}'});
for (const origin of [null,'https://unrelated.example','null','http://127.0.0.1:3100.evil.example']) {
  assert.equal((await graphAccountProxy(request(origin),'/api/saved-people',{fetchImpl})).status,403);
}
assert.equal(calls.length,0,'Cross-origin mutations must never reach Graph');
assert.equal((await graphAccountProxy(request('https://unrelated.example',{'x-forwarded-host':'unrelated.example'}),
  '/api/saved-people',{fetchImpl})).status,403,'Forwarded headers cannot authorize a cross-site write');
assert.equal((await graphAccountProxy(request('http://127.0.0.1:3100'),'/api/saved-people',{fetchImpl,graphUrl:'http://127.0.0.1:4173'})).status,200);
assert.equal(calls[0].headers.get('origin'),'http://127.0.0.1:4173');
assert.equal(calls[0].headers.get('cookie'),'bridge_session=test-fixture');
assert.equal(calls[0].body,'{"authorId":-42}');
const production=new Request('http://internal:3000/api/saved-people',{method:'PUT',headers:{Host:'research.example',Origin:'https://research.example'},body:'{}'});
await graphAccountProxy(production,'/api/saved-people',{fetchImpl,graphUrl:'http://127.0.0.1:5173'});
assert.equal(calls[1].headers.get('origin'),'https://research.example','TLS proxy keeps the validated public origin');
const failing=async()=>{throw new Error('Unavailable');};
assert.equal((await graphAccountProxy(request('http://127.0.0.1:3100'),'/api/saved-people',{fetchImpl:failing})).status,502);
console.log('PASS Graph account proxy: local save origin, cookies, negative IDs, TLS public origin, cross-site rejection and outage');
