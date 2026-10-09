// Real per-conversation choices in local Postgres; the chat reply is stubbed.
import assert from 'node:assert/strict';
import {randomUUID} from 'node:crypto';
import {createRequire} from 'node:module';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
const root=path.resolve('../bridge2aikg'),require=createRequire(path.join(root,'package.json')),{chromium}=require('playwright');
const {createServer}=await import(pathToFileURL(path.join(root,'node_modules/vite/dist/node/index.js')));
process.env.DATABASE_URL ||= 'postgresql://bridge2ai_dev@127.0.0.1:55432/postgres';
const dbUrl=new URL(process.env.DATABASE_URL);assert.equal(dbUrl.port,'55432');assert.ok(['127.0.0.1','localhost'].includes(dbUrl.hostname));
process.chdir(root);const server=await createServer({root,server:{middlewareMode:true,hmr:false},appType:'custom',logLevel:'error'});
const subject=`showcase-context-${randomUUID()}`,account=randomUUID(),jar=new Map(),cookies={get:k=>jar.get(k),set:(k,v)=>jar.set(k,v),delete:k=>jar.delete(k)};
let auth,sql,browser;
try {
 auth=await server.ssrLoadModule('/src/lib/server/authSession.ts');sql=(await server.ssrLoadModule('/src/lib/db.ts')).getDb();await auth.getAccountSummary(undefined);
 await sql`INSERT INTO bridge_accounts(id) VALUES(${account})`;
 await sql`INSERT INTO bridge_account_identities(account_id,provider,subject,display_name) VALUES(${account},'google',${subject},'Context showcase')`;
 await auth.createSession(cookies,new URL('http://127.0.0.1:4173/'),{provider:'google',subject,name:'Context showcase',accountId:account});
 const response=await fetch('http://127.0.0.1:4173/api/matrix/session-auth',{headers:{Cookie:`bridge_session=${jar.get('bridge_session')}`}});assert.equal(response.status,200);const token=(await response.json()).token;
 browser=await chromium.launch({headless:true,args:['--disable-dev-shm-usage']});
 const context=await browser.newContext({viewport:{width:1440,height:900}});await context.addInitScript(t=>sessionStorage.setItem('matrix_user_token',t),token);
 await context.route('**/api/chat-lite',route=>route.fulfill({json:{action:'agent',reply:'Context history acceptance response.',result_update:'keep',working_context:{goal:'',requirements:[]}}}));
 const page=await context.newPage(),errors=[];page.on('pageerror',e=>errors.push(e.message));
 await page.goto('http://127.0.0.1:3100/?aid=5058469124&fresh=1',{waitUntil:'networkidle'});
 const rail=page.getByRole('complementary',{name:'Chat history'});
 assert.equal(await page.getByRole('complementary',{name:'Conversation context'}).count(),0);
 assert.equal(await rail.getByText('Context',{exact:true}).count(),0);
 const open=async()=>{await page.getByRole('button',{name:'Manage context'}).click();const panel=page.getByRole('complementary',{name:'Conversation context'});await panel.locator('.you-paper-choice').first().waitFor();return panel;};
 let panel=await open();await panel.getByText('No publications selected.',{exact:true}).waitFor();
 await panel.getByText('Topics',{exact:true}).waitFor();
 await panel.getByRole('button',{name:'Next',exact:true}).click();await panel.getByRole('button',{name:'Previous',exact:true}).click();
 await panel.getByLabel('Search publications').fill('PAGER');await panel.locator('.you-paper-choice').first().waitFor();
 const titleA=await panel.locator('.you-paper-choice span').first().innerText();await panel.locator('.you-paper-choice input').first().check();await panel.getByText('1/8 selected',{exact:true}).waitFor();
 await panel.getByRole('button',{name:'Close context',exact:true}).click();
 async function send(text) {
   await page.getByLabel('Describe the research help you need').fill(text);
   const saved=page.waitForResponse(r=>r.url().includes('/api/chat-sessions/')&&r.request().method()==='PUT'&&r.status()===200);
   await page.getByRole('button',{name:'Send message',exact:true}).click();await saved;
 }
 await send('Showcase conversation A');
 await rail.getByRole('button',{name:'New chat',exact:true}).click();panel=await open();await panel.getByText('No publications selected.',{exact:true}).waitFor();
 const titleB=await panel.locator('.you-paper-choice span').first().innerText();assert.notEqual(titleA,titleB);
 await panel.locator('.you-paper-choice input').first().check();await panel.getByText('1/8 selected',{exact:true}).waitFor();
 await panel.getByRole('button',{name:'Close context',exact:true}).click();await send('Showcase conversation B');
 const records=await sql`SELECT id,state FROM matrix_chat_sessions WHERE owner_orcid=${'account:'+account} OR owner_orcid=${'google:'+subject}`;
 assert.equal(records.length,2);const choices=records.map(r=>r.state.contextChoices.selectedPapers);assert.ok(choices.every(c=>c.length===1));assert.notEqual(choices[0][0].work_id,choices[1][0].work_id);
 await rail.getByRole('button',{name:/Showcase conversation A/}).click();panel=await open();await panel.getByText(/PAGER/).first().waitFor();assert.equal(await panel.getByText('1/8 selected',{exact:true}).count(),1);
 await page.screenshot({path:'/tmp/showcase-context-desktop.png'});await page.setViewportSize({width:390,height:844});assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));await page.screenshot({path:'/tmp/showcase-context-mobile.png'});
 await page.keyboard.press('Escape');assert.equal(await page.getByRole('complementary',{name:'Conversation context'}).count(),0);
 await page.reload({waitUntil:'networkidle'});await page.getByRole('button',{name:'Manage context'}).click();await page.getByRole('complementary',{name:'Conversation context'}).getByText('1/8 selected',{exact:true}).waitFor();
 assert.deepEqual(errors,[]);console.log('PASS: right context starts closed, topics with no default papers, live searchable/paginated picker, two distinct durable selections, restored conversation, mobile fit and Escape. Chat reply stubbed; catalog, selections, auth and history are live.');
} finally {
 await browser?.close();if(auth) await auth.revokeSession(cookies);if(sql){await sql`DELETE FROM matrix_chat_sessions WHERE owner_orcid=${'account:'+account} OR owner_orcid=${'google:'+subject}`;await sql`DELETE FROM bridge_accounts WHERE id=${account}`;await sql.end({timeout:5});}await server.close();
}
