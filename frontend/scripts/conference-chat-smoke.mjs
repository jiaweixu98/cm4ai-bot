// Real local auth, session APIs and Postgres. Only the conversational reply is mocked.
import assert from 'node:assert/strict';
import {randomUUID} from 'node:crypto';
import {createRequire} from 'node:module';
import {pathToFileURL,fileURLToPath} from 'node:url';
const graphRoot=process.env.GRAPH_REPO || fileURLToPath(new URL('../../../bridge2aikg/',import.meta.url));
const require=createRequire(graphRoot+'/package.json');
const {chromium}=require('playwright');
const {createServer}=await import(pathToFileURL(graphRoot+'/node_modules/vite/dist/node/index.js'));
process.env.DATABASE_URL ||= 'postgresql://bridge2ai_dev@127.0.0.1:55432/postgres';
assert.equal(new URL(process.env.DATABASE_URL).port,'55432');
assert.equal(new URL(process.env.DATABASE_URL).hostname,'127.0.0.1');
process.chdir(graphRoot);
const server=await createServer({root:graphRoot,server:{middlewareMode:true,hmr:false},appType:'custom',logLevel:'error'});
const account=randomUUID(),subject='conference-chat-'+randomUUID(),owner='google:'+subject;
const values=new Map(),cookies={get:k=>values.get(k),set:(k,v)=>values.set(k,v),delete:k=>values.delete(k)};
let auth,sql,browser,page;
const errors=[],requests=[];
const base='http://127.0.0.1:3100';
try {
  auth=await server.ssrLoadModule('/src/lib/server/authSession.ts');
  sql=(await server.ssrLoadModule('/src/lib/db.ts')).getDb();
  await auth.createSession(cookies,new URL('http://127.0.0.1:4173'),{provider:'google',subject,name:'Chat browser test'});
  await auth.revokeSession(cookies);
  await sql`INSERT INTO bridge_accounts(id) VALUES(${account})`;
  await sql`INSERT INTO bridge_account_identities(account_id,provider,subject,display_name) VALUES(${account},'google',${subject},'Chat browser test')`;
  await auth.createSession(cookies,new URL('http://127.0.0.1:4173'),{provider:'google',subject,name:'Chat browser test',accountId:account});
  const tokenResponse=await fetch('http://127.0.0.1:4173/api/matrix/session-auth',{headers:{Cookie:`bridge_session=${values.get('bridge_session')}`}});
  assert.equal(tokenResponse.status,200);const {token}=await tokenResponse.json();
  browser=await chromium.launch({headless:true,args:['--disable-dev-shm-usage']});
  const context=await browser.newContext({viewport:{width:1440,height:1000}});
  await context.addCookies([{name:'bridge_session',value:values.get('bridge_session'),url:base}]);
  await context.addInitScript(value=>sessionStorage.setItem('matrix_user_token',value),token);
  await context.route('**/api/chat-lite',async route=>{
    requests.push(route.request().postDataJSON());
    await route.fulfill({json:{action:'agent',reply:'Conference test response.',result_update:'keep',working_context:{goal:'Find biomedical collaborators',requirements:[]}}});
  });
  page=await context.newPage();page.setDefaultTimeout(30000);page.on('pageerror',e=>errors.push(e.message));
  await page.goto(base+'/?aid=5106027423&embedded=1&fresh=1',{waitUntil:'networkidle'});
  await page.getByRole('button',{name:'Manage context',exact:true}).click();
  const panel=page.getByRole('complementary',{name:'Conversation context'});
  await panel.locator('.you-paper-choice').first().waitFor();
  const total=(await (await context.request.get(base+'/api/author/5106027423/publications?limit=20')).json()).total;
  assert(total>20);
  const firstTitle=await panel.locator('.you-paper-choice').first().innerText();
  await panel.locator('.you-paper-choice input').first().check();
  const nextResponse=page.waitForResponse(r=>r.url().includes('/publications?')&&new URL(r.url()).searchParams.get('offset')==='20');
  await panel.getByRole('button',{name:'Next',exact:true}).click();assert.equal((await nextResponse).status(),200);
  await panel.getByText(`21–${Math.min(40,total)} of ${total}`,{exact:true}).waitFor();
  assert.notEqual(await panel.locator('.you-paper-choice').first().innerText(),firstTitle);
  await panel.getByText('1/8 selected',{exact:true}).waitFor();
  await panel.getByRole('button',{name:'Previous',exact:true}).click();
  await panel.getByText(`1–20 of ${total}`,{exact:true}).waitFor();
  assert(await panel.locator('.you-paper-choice input').first().isChecked());
  let failPage=true;
  await page.route('**/api/author/5106027423/publications?*',async route=>{
    if(failPage && new URL(route.request().url()).searchParams.get('offset')==='20') {failPage=false;await route.fulfill({status:503,json:{detail:'Temporary test outage'}});}
    else await route.continue();
  });
  await panel.getByRole('button',{name:'Next',exact:true}).click();
  await panel.getByRole('alert').filter({hasText:'Publications could not be loaded'}).waitFor();
  assert.equal(await panel.locator('.you-paper-choice').count(),0,'An outage cannot show stale papers');
  await panel.getByRole('button',{name:'Retry',exact:true}).click();
  await panel.getByText(`21–${Math.min(40,total)} of ${total}`,{exact:true}).waitFor();
  await page.unroute('**/api/author/5106027423/publications?*');
  await panel.getByLabel('Search publications',{exact:true}).fill('NoSuchPublicationForConferenceSmoke');
  await panel.getByText('0–0 of 0',{exact:true}).waitFor();
  assert(await panel.getByRole('button',{name:'Next',exact:true}).isDisabled());
  console.log('Real publication pagination, cross-page selection, empty results and outage retry passed');
  const topic='Biomedical Text Mining and Ontologies';
  const graphBase='http://127.0.0.1:4173';
  assert.equal((await context.request.put(graphBase+'/api/research-interests',{headers:{Origin:graphBase},data:{topics:[topic]}})).status(),200);
  assert.equal((await context.request.put(graphBase+'/api/saved-people',{headers:{Origin:graphBase},data:{authorId:5106027423,name:'Moni C. Muñoz Torres',savedAt:new Date().toISOString(),source:'graph',intent:'collaborator'}})).status(),200);
  await page.goto(base+'/?fresh=1',{waitUntil:'networkidle'});
  await page.getByRole('button',{name:'Add people to this chat',exact:true}).click();
  const picker=page.getByRole('dialog',{name:'Add people to this chat'});
  await picker.locator('.people-picker-option').filter({hasText:'Moni C. Muñoz Torres'}).getByRole('button',{name:'Add',exact:true}).click();
  await page.keyboard.press('Escape');
  await page.getByRole('button',{name:'Manage context',exact:true}).click();
  await panel.getByRole('button',{name:'Moni C. Muñoz Torres',exact:true}).waitFor();
  assert.equal(await panel.getByLabel('Publication affiliation filter',{exact:true}).count(),0);
  assert.equal(await panel.getByLabel('Search publications',{exact:true}).count(),0);
  assert.equal(await panel.locator('textarea').count(),0);
  await panel.getByRole('button',{name:'Close context',exact:true}).click();

  await page.locator('.chat-input').fill('Conference retained conversation');
  await page.getByRole('button',{name:'Send message',exact:true}).click();
  await page.getByText('Conference test response.',{exact:true}).waitFor();
  assert.equal(requests.at(-1).context_choices.research_interests,topic);
  assert.deepEqual(requests.at(-1).context_person_ids,['5106027423']);
  assert.deepEqual(requests.at(-1).context_choices.publication_filters,[]);
  let rows=[];
  for(let i=0;i<80;i++) {rows=await sql`SELECT id,messages,state FROM matrix_chat_sessions WHERE owner_orcid=${owner}`;if(rows[0]?.messages.length===2)break;await new Promise(r=>setTimeout(r,100));}
  assert.equal(rows.length,1);assert.equal(rows[0].messages.length,2);assert.deepEqual(rows[0].state.contextChoices.personalContext.topics,[topic]);
  const sid=rows[0].id;
  const available=(await (await context.request.get(graphBase+'/api/research-topics')).json()).topics;
  const other=available.find(t=>t!==topic);assert(other);
  assert.equal((await context.request.put(graphBase+'/api/research-interests',{headers:{Origin:graphBase},data:{topics:[other]}})).status(),200);
  await page.reload({waitUntil:'networkidle'});
  await page.getByRole('button',{name:'Manage context',exact:true}).click();
  await panel.getByText(topic,{exact:true}).waitFor();
  assert.equal(await panel.getByText(other,{exact:true}).count(),0,'Restored conversation keeps its own topic snapshot');
  assert.equal(await panel.getByLabel('Publication affiliation filter',{exact:true}).count(),0);
  await panel.getByRole('button',{name:'Close context',exact:true}).click();
  console.log('Whole-person saved context and topic snapshot restoration passed');

  const rail=page.getByRole('complementary',{name:'Chat history'});
  await rail.getByText('Conference retained conversation',{exact:true}).waitFor();
  const removed=page.waitForResponse(r=>r.request().method()==='DELETE'&&r.url().endsWith(sid));
  await rail.getByRole('button',{name:'Remove chat Conference retained conversation',exact:true}).click();
  await page.getByRole('alertdialog').getByRole('button',{name:'Remove',exact:true}).click();
  assert.equal((await removed).status(),200);
  await rail.getByText('Conference retained conversation',{exact:true}).waitFor({state:'hidden'});
  const [retained]=await sql`SELECT messages,state,hidden_at FROM matrix_chat_sessions WHERE id=${sid}`;
  assert.equal(retained.messages.length,2);assert.deepEqual(retained.state.contextChoices.personalContext.topics,[topic]);assert(retained.hidden_at);
  assert.equal((await fetch(base+'/api/chat-sessions/'+sid,{headers:{'x-matrix-user-token':token}})).status,404);
  const late=await fetch(base+'/api/chat-sessions/'+sid,{method:'PUT',headers:{'Content-Type':'application/json','x-matrix-user-token':token},body:JSON.stringify({aid:'unlinked',messages:[...retained.messages,{role:'user',content:'Late browser write'}],state:{}})});
  assert.equal(late.status,403);
  await page.reload({waitUntil:'networkidle'});
  assert.equal(await rail.getByText('Conference retained conversation',{exact:true}).count(),0);
  await page.setViewportSize({width:390,height:844});
  assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1));
  assert.deepEqual(errors,[]);
  console.log('PASS: real Next/backend pagination; selected papers retained across pages; no-publication interests sent and stored; history removal in UI and API; messages retained for analysis; reload and late writes cannot restore hidden chats; mobile fit.');
} catch(e) {
  if(page) {console.log('Page diagnostic:',(await page.locator('body').innerText()).slice(0,1800));await page.screenshot({path:'/tmp/conference-chat-failure.png'}).catch(()=>{});}
  throw e;
} finally {
  await browser?.close();if(auth)await auth.revokeSession(cookies);
  if(sql) {await sql`DELETE FROM matrix_chat_sessions WHERE owner_orcid=${owner}`;await sql`DELETE FROM bridge_usage_events WHERE account_id=${account}`;await sql`DELETE FROM bridge_accounts WHERE id=${account}`;await sql.end({timeout:5});}
  await server.close();
}
