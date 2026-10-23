import assert from 'node:assert/strict';
import {readAttachedFiles} from '../app/lib/attachedPapers.js';
let finish;
const controller=new AbortController();
const pending=readAttachedFiles([{name:'draft.txt',size:20,text:()=>new Promise(r=>{finish=r})}],0,controller.signal);
controller.abort();finish('synthetic test content');
await assert.rejects(pending,{name:'AbortError'});
await assert.rejects(()=>readAttachedFiles([{name:'unused.txt',size:1}],0,controller.signal));
console.log('PASS actual text reader discards late completion and never begins pre-cancelled work');
