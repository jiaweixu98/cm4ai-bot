// Real Graph-issued sessions, Next proxies, MATRIX APIs and isolated Postgres.
// Only the model response is mocked. Never point this test at production.
import assert from 'node:assert/strict';
import {randomUUID} from 'node:crypto';
import {createRequire} from 'node:module';
import {fileURLToPath, pathToFileURL} from 'node:url';
import path from 'node:path';

const graphRoot = process.env.GRAPH_REPO || path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../../bridge2aikg');
const require = createRequire(path.join(graphRoot, 'package.json'));
const {chromium} = require('playwright');
const {createServer} = await import(pathToFileURL(path.join(graphRoot, 'node_modules/vite/dist/node/index.js')));
const database = process.env.DATABASE_URL || 'postgresql://bridge2ai_dev@127.0.0.1:55432/postgres';
const databaseUrl = new URL(database);
assert.ok(['127.0.0.1', 'localhost'].includes(databaseUrl.hostname) && databaseUrl.port === '55432', 'Use isolated development Postgres');
process.env.DATABASE_URL = database;
const base = 'http://127.0.0.1:3100';
const graph = 'http://127.0.0.1:4173';
// SvelteKit resolves its plugin peers from the project's working directory.
process.chdir(graphRoot);
const server = await createServer({root: graphRoot, server: {middlewareMode: true, hmr: false}, appType: 'custom', logLevel: 'error'});
const account = randomUUID();
const subject = `durable-history-${randomUUID()}`;
const owner = `google:${subject}`;
const jars = [];
let auth, sql, browser, graphCookie;

async function credential() {
  const jar = new Map();
  const cookies = {get: key => jar.get(key), set: (key, value) => jar.set(key, value), delete: key => jar.delete(key)};
  jars.push(cookies);
  await auth.createSession(cookies, new URL(graph), {provider: 'google', subject, name: 'History test', accountId: account});
  graphCookie = jar.get('bridge_session');
  const response = await fetch(`${graph}/api/matrix/session-auth`, {headers: {Cookie: `bridge_session=${jar.get('bridge_session')}`}});
  assert.equal(response.status, 200);
  return (await response.json()).token;
}

const reply = 'Local history acceptance response.';
const citation = {title: 'History acceptance publication', url: 'https://doi.org/10.1000/history-test', evidence_id: 'history-test-paper'};
async function prepare(context, token) {
  context.setDefaultTimeout(30000);
  await context.addInitScript(value => sessionStorage.setItem('matrix_user_token', value), token);
  await context.route('http://127.0.0.1:8100/**', route => route.abort('connectionrefused'));
  await context.route('**/api/chat-lite', route => route.fulfill({json: {
    action: 'agent', reply, citations: [citation], result_update: 'keep',
    working_context: {goal: 'Study clinical data quality', requirements: ['Published methods']},
  }}));
}

async function send(page, question) {
  const previousReplies = await page.getByText(reply, {exact: true}).count();
  await page.locator('.chat-input').fill(question);
  await page.getByRole('button', {name: 'Send message', exact: true}).click();
  await page.getByText(reply, {exact: true}).nth(previousReplies).waitFor();
  await page.getByRole('button', {name: 'Send message', exact: true}).waitFor();
}

try {
  auth = await server.ssrLoadModule('/src/lib/server/authSession.ts');
  sql = (await server.ssrLoadModule('/src/lib/db.ts')).getDb();
  await auth.getAccountSummary(undefined);
  await sql`INSERT INTO bridge_accounts(id) VALUES(${account})`;
  await sql`INSERT INTO bridge_account_identities(account_id, provider, subject, display_name) VALUES(${account}, 'google', ${subject}, 'History test')`;
  const token = await credential();
  // Prove the running API reads this isolated database before any API writes.
  const probe = randomUUID();
  await sql`INSERT INTO matrix_chat_sessions(id, owner_orcid, focal_author_id) VALUES(${probe}, ${owner}, 'unlinked')`;
  try {
    const check = await fetch(`${base}/api/chat-sessions/${probe}`, {headers: {'x-matrix-user-token': token}, signal: AbortSignal.timeout(10000)});
    assert.equal(check.status, 200, 'The frontend must proxy to the isolated development session store');
    assert.equal((await check.json()).session.id, probe);
  } finally {
    await sql`DELETE FROM matrix_chat_sessions WHERE id=${probe}`;
  }
  browser = await chromium.launch({headless: true, args: ['--disable-dev-shm-usage']});
  const context = await browser.newContext({viewport: {width: 1440, height: 900}});
  await prepare(context, token);
  const page = await context.newPage();
  const errors = [], directBackend = [];
  page.on('pageerror', error => errors.push(error.message));
  page.on('request', request => { if (new URL(request.url()).port === '8100') directBackend.push(new URL(request.url()).pathname); });
  await page.goto(`${base}/?fresh=1&seeker_name=History+test`, {waitUntil: 'networkidle'});
  const rail = page.getByRole('complementary', {name: 'Chat history'});
  await rail.getByText('Context', {exact: true}).click();
  await rail.getByText('History test', {exact: true}).waitFor();
  assert.equal(await rail.getByText('Same institution', {exact: true}).count(), 0);
  const toggle = rail.getByRole('button', {name: 'Close sidebar', exact: true});
  const bounds = await toggle.boundingBox();
  assert.ok(bounds.width >= 44 && bounds.height >= 44, 'Collapse has a usable target');
  assert.equal(await toggle.locator('svg').getAttribute('width'), '22');

  const question = 'Keep this clinical data quality conversation';
  await send(page, question);
  await page.getByRole('button', {name: 'New chat', exact: true}).click();
  await page.getByText('What should we work on?', {exact: true}).waitFor();
  await rail.getByText(question, {exact: true}).click();
  await page.getByText(reply, {exact: true}).waitFor();
  await page.getByRole('link', {name: citation.title, exact: true}).waitFor();
  await rail.getByText('Study clinical data quality', {exact: true}).waitFor();
  let rows = await sql`SELECT id, messages, state FROM matrix_chat_sessions WHERE owner_orcid=${owner} ORDER BY created_at`;
  assert.equal(rows.length, 1, 'A first message creates one durable chat');
  assert.equal(rows[0].messages.length, 2);
  const firstSessionId = rows[0].id;
  assert.deepEqual(rows[0].messages[1].citations, [citation], 'Saving retains citation metadata');
  assert.equal(rows[0].state.workingContext.goal, 'Study clinical data quality');

  // Delayed saves must finish before New chat clears the current conversation.
  await page.route('**/api/chat-sessions/*', async route => {
    if (route.request().method() === 'PUT') await new Promise(resolve => setTimeout(resolve, 350));
    await route.continue();
  });
  await send(page, 'Continue with published methods');
  await page.getByRole('button', {name: 'New chat', exact: true}).click();
  await page.getByText('What should we work on?', {exact: true}).waitFor();
  rows = await sql`SELECT messages FROM matrix_chat_sessions WHERE id=${rows[0].id}`;
  assert.equal(rows[0].messages.length, 4, 'Switching chats keeps the last question and reply');
  const stale = await fetch(`${base}/api/chat-sessions/${firstSessionId}`, {
    method: 'PUT', headers: {'Content-Type': 'application/json', 'x-matrix-user-token': token},
    body: JSON.stringify({aid: 'unlinked', messages: [{role: 'user', content: question}], state: {intent: 'collaborator'}}),
  });
  assert.equal(stale.status, 200);
  assert.equal((await stale.json()).session.messages.length, 4, 'A stale request cannot truncate the saved conversation');
  await page.unroute('**/api/chat-sessions/*');

  // The same signed-in account sees history in a separate browser storage context.
  const other = await browser.newContext({viewport: {width: 390, height: 844}});
  await prepare(other, await credential());
  const otherPage = await other.newPage();
  otherPage.on('pageerror', error => errors.push(error.message));
  await otherPage.goto(`${base}/?fresh=1&seeker_name=History+test`, {waitUntil: 'networkidle'});
  if (!await otherPage.getByRole('complementary', {name: 'Chat history'}).count()) await otherPage.getByRole('button', {name: 'Open sidebar', exact: true}).click();
  await otherPage.getByText(question, {exact: true}).click();
  await otherPage.getByText('Continue with published methods', {exact: true}).waitFor();
  await otherPage.getByRole('button', {name: 'Open sidebar', exact: true}).click();
  await otherPage.getByText('Context', {exact: true}).click();
  await otherPage.getByText('Study clinical data quality', {exact: true}).waitFor();
  assert.ok(await otherPage.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1));
  if (process.env.MATRIX_SHOTS) await otherPage.screenshot({path: `${process.env.MATRIX_SHOTS}/matrix-durable-history-mobile.png`});
  await other.close();

  // A temporary failure retains the conversation and Retry saves both turns.
  let unavailable = true;
  await page.route('**/api/chat-sessions', async route => {
    if (unavailable && route.request().method() === 'POST') return route.fulfill({status: 503, json: {detail: 'Local test outage'}});
    return route.continue();
  });
  await send(page, 'Recover this chat after a storage outage');
  await page.locator('.session-status-error').waitFor();
  unavailable = false;
  await page.getByRole('button', {name: 'Try again', exact: true}).click();
  await rail.getByText('Recover this chat after a storage outage', {exact: true}).waitFor();
  await page.waitForFunction(() => !document.querySelector('.session-status-error'));
  rows = await sql`SELECT messages FROM matrix_chat_sessions WHERE owner_orcid=${owner} ORDER BY created_at DESC LIMIT 1`;
  assert.equal(rows[0].messages.length, 2, 'Retry saves the whole unsaved conversation');
  await page.unroute('**/api/chat-sessions');

  // Reopen immediately after closing a tab, without waiting for a debounce.
  await send(page, 'Save the final turn before closing');
  await page.close();
  const reopened = await context.newPage();
  await reopened.goto(`${base}/?fresh=1&seeker_name=History+test`, {waitUntil: 'networkidle'});
  await reopened.getByRole('complementary', {name: 'Chat history'}).getByText('Recover this chat after a storage outage', {exact: true}).click();
  await reopened.getByText('Save the final turn before closing', {exact: true}).waitFor();
  assert.equal(await reopened.getByText(reply, {exact: true}).count(), 2);
  assert.equal(await reopened.locator('.session-status-error').count(), 0);
  if (process.env.MATRIX_SHOTS) {
    await reopened.waitForTimeout(350);
    await reopened.screenshot({path: `${process.env.MATRIX_SHOTS}/matrix-durable-history-desktop.png`});
  }

  const guest = await context.newPage();
  await guest.goto(`${base}/?embedded=1&fresh=1&auth_handoff=1`, {waitUntil: 'networkidle'});
  await guest.getByText('Sign in on the graph to keep chats.', {exact: true}).waitFor();
  assert.equal(await guest.locator('.session-chip').count(), 0, 'A guest Graph handoff must not reuse the previous account history');
  assert.equal(await guest.evaluate(() => sessionStorage.getItem('matrix_user_token')), null);
  await guest.close();

  // Exercise the actual Graph modal X button, with slow writes and an outage.
  const embeddedContext = await browser.newContext({viewport: {width: 1440, height: 1000}});
  await embeddedContext.addCookies([{name: 'bridge_session', value: graphCookie, url: `${graph}/`}]);
  await prepare(embeddedContext, token);
  let storageDown = false;
  await embeddedContext.route('**/api/chat-sessions/*', async route => {
    if (route.request().method() === 'PUT') {
      if (storageDown) return route.fulfill({status: 503, json: {detail: 'Local test outage'}});
      await new Promise(resolve => setTimeout(resolve, 300));
    }
    return route.continue();
  });
  const graphPage = await embeddedContext.newPage();
  graphPage.on('pageerror', error => errors.push(error.message));
  graphPage.on('dialog', async dialog => { errors.push('Unexpected close confirmation'); await dialog.accept(); });
  await graphPage.goto(graph, {waitUntil: 'domcontentloaded'});
  const launch = graphPage.getByRole('button', {name: 'Ask MATRIX', exact: true});
  await launch.waitFor({timeout: 60000});
  await launch.click();
  assert.equal(new URL(await graphPage.locator('.matrix-modal-iframe').getAttribute('src')).origin, base, 'Graph must use the local MATRIX frontend');
  const iframe = graphPage.frameLocator('.matrix-modal-iframe');
  await iframe.getByText('What should we work on?', {exact: true}).waitFor({timeout: 30000});
  await send(iframe, 'Keep the chat when the Graph dialog closes');
  await graphPage.locator('button.matrix-modal-close').click();
  await graphPage.locator('.matrix-modal-iframe').waitFor({state: 'detached'});
  await launch.click();
  await iframe.getByRole('complementary', {name: 'Chat history'}).getByText('Keep the chat when the Graph dialog closes', {exact: true}).click();
  await iframe.getByText(reply, {exact: true}).waitFor();
  await iframe.getByRole('link', {name: citation.title, exact: true}).waitFor();
  const matrixFrame = graphPage.frames().find(frame => frame.url().startsWith(base));
  assert.equal(new URL(matrixFrame.url()).searchParams.has('auth_handoff'), false);
  await matrixFrame.goto(matrixFrame.url(), {waitUntil: 'networkidle'});
  await iframe.getByText(reply, {exact: true}).waitFor();
  storageDown = true;
  await send(iframe, 'Keep an unsaved turn open during an outage');
  await graphPage.locator('button.matrix-modal-close').click();
  await iframe.locator('.session-status-error').waitFor();
  await graphPage.waitForFunction(() => !document.querySelector('.matrix-modal-close')?.disabled);
  assert.equal(await graphPage.locator('.matrix-modal-iframe').count(), 1, 'Failed save leaves the Graph dialog open');
  storageDown = false;
  await iframe.getByRole('button', {name: 'Try again', exact: true}).click();
  await iframe.locator('.session-status-error').waitFor({state: 'hidden'});
  await graphPage.locator('button.matrix-modal-close').click();
  await graphPage.locator('.matrix-modal-iframe').waitFor({state: 'detached'});
  rows = await sql`SELECT messages FROM matrix_chat_sessions WHERE owner_orcid=${owner} AND title='Keep the chat when the Graph dialog closes'`;
  assert.equal(rows[0].messages.length, 4, 'The Graph close acknowledgement follows the final committed save');
  await embeddedContext.close();
  assert.deepEqual(directBackend, [], 'Browser uses the frontend proxy for history');
  assert.deepEqual(errors, []);
  console.log('PASS: actual signed-in persistence, two browser stores, citation/context restoration, stale-write protection, delayed saves, outage retry, immediate close/reopen, Graph close acknowledgement and usable responsive sidebar; model mocked.');
} finally {
  if (browser) await browser.close();
  if (auth) {
    for (const cookies of jars) await auth.revokeSession(cookies);
    await auth.flushUsageEvents();
  }
  if (sql) {
    await sql`DELETE FROM matrix_chat_sessions WHERE owner_orcid=${owner}`;
    await sql`DELETE FROM bridge_usage_events WHERE account_id=${account}`;
    await sql`DELETE FROM bridge_accounts WHERE id=${account}`;
    await sql.end({timeout: 5});
  }
  await server.close();
}
