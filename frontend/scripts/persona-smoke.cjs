// Run against local Next with NODE_PATH pointing to an installed Playwright package.
// All application APIs are mocked: no model, database or retrieval requests.
const { chromium } = require('playwright');
const assert = require('node:assert/strict');

(async () => {
  const browser = await chromium.launch({ headless: true, args: ['--disable-dev-shm-usage'] });
  try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
    const errors = [];
    page.on('pageerror', (error) => errors.push(error.message));
    const sessions = {};
    const chats = [];
    const searches = [];
    let failSave = false;
    await page.addInitScript(() => sessionStorage.setItem('matrix_user_token', 'mock-only'));
    await page.route('**/api/**', async (route) => {
      const req = route.request();
      const url = new URL(req.url());
      const data = req.postDataJSON() || {};
      if (failSave && req.method() === 'PUT' && url.pathname.startsWith('/api/chat-sessions')) {
        await route.fulfill({ status: 503, json: { error: 'Mock save failure' } });
        return;
      }
      let payload = {};
      if (url.pathname === '/api/chat-sessions' && req.method() === 'GET') {
        payload = { sessions: Object.values(sessions).filter((s) => s.state.intent === url.searchParams.get('intent')) };
      } else if (url.pathname.startsWith('/api/chat-sessions')) {
        const id = url.pathname.split('/')[3] || `s${Object.keys(sessions).length + 1}`;
        if (req.method() !== 'GET') sessions[id] = { ...data, id, title: `${data.state.intent} chat` };
        payload = { session: sessions[id] };
      } else if (url.pathname === '/api/chat-lite') {
        chats.push(data);
        payload = data.user_input === 'Search that' ? { action: 'confirm' }
          : data.user_input.includes('Compare') ? { action: 'chat', reply: 'The included researcher has a relevant publication.' }
          : { action: 'search', query: 'Spatial integration', justification: 'For your research goal.' };
      } else if (url.pathname === '/api/search-people') {
        searches.push(data);
        payload = { candidates: [{ author_id: '22', name: 'Researcher Example', affiliation: 'Example University', retrieval_score: 0.72,
          search_basis: data.representative_titles?.length ? 'representative_papers' : 'research_need', papers: [{ Title: 'Spatial integration methods', PubYear: 2025 }, { Title: 'Multimodal analysis', PubYear: 2024 }] }] };
      } else if (url.pathname === '/api/why-lines') {
        payload = { context_basis: data.seeker_papers?.length ? 'need_profile' : 'need', results: [{ author_id: '22', source: 'llm', explanation: 'They have published methods you can learn for spatial integration in your stated goal.', evidence_paper_index: 0 }] };
      }
      await route.fulfill({ json: payload });
    });
    await page.goto('http://127.0.0.1:3100/?intent=mentor&embedded=1&fresh=1', { waitUntil: 'domcontentloaded', timeout: 45000 });
    await page.getByRole('button', { name: 'Mentor', exact: true }).waitFor();
    assert.equal(await page.getByText('Match from your publications', { exact: false }).count(), 0);
    assert.equal(await page.getByText('Include papers from my profile', { exact: false }).count(), 0);
    await page.evaluate(() => window.dispatchEvent(new MessageEvent('message', { origin: 'http://127.0.0.1:4173', data: { type: 'bridge2ai-saved-people', people: [{ authorId: 11, name: 'Saved Researcher', affiliation: 'University' }] } })));
    await page.getByRole('button', { name: 'Include Saved Researcher in this chat', exact: true }).click();
    await page.locator('.chat-input').fill('Compare the included people');
    await page.getByRole('button', { name: 'Send', exact: true }).click();
    await page.getByText('The included researcher has a relevant publication.', { exact: true }).waitFor();
    assert.deepEqual(chats.at(-1).context_person_ids, ['11']);
    await page.getByRole('button', { name: 'Exclude Saved Researcher in this chat', exact: true }).click();
    await page.getByRole('button', { name: 'Include Saved Researcher in this chat', exact: true }).waitFor();
    require('fs').writeFileSync('/tmp/matrix-draft.txt', 'Toward cross-platform EHR phenotyping\n\nDraft abstract body for one-time search context.\n');
    await page.locator('.chat-input-container input[type="file"]').setInputFiles('/tmp/matrix-draft.txt');
    await page.locator('.attach-chip').waitFor();
    await page.locator('.chat-input').fill('Find mentors');
    await page.getByRole('button', { name: 'Send', exact: true }).click();
    await page.getByRole('button', { name: 'Search that', exact: true }).click();
    await page.getByText('Why this person', { exact: true }).waitFor();
    assert.equal(await page.getByText('Publication context', { exact: true }).count(), 0);
    assert.deepEqual(searches.at(-1).representative_titles, ['Toward cross-platform EHR phenotyping']);
    assert.equal(await page.locator('.attach-chip').count(), 0);
    await page.screenshot({ path: '/tmp/matrix-persona-desktop.png' });
    for (const width of [1440, 1024, 768, 390]) {
      await page.setViewportSize({ width, height: 900 });
      assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1), `overflow at ${width}`);
      const box = await page.locator('.chat-input').boundingBox();
      assert(box && box.y >= 0 && box.y + box.height <= 900, `composer clipped at ${width}`);
    }
    await page.screenshot({ path: '/tmp/matrix-persona-mobile.png' });
    failSave = true;
    await page.getByRole('button', { name: 'Collaborator', exact: true }).click();
    await page.getByText('Could not save this conversation.', { exact: true }).waitFor();
    assert(new URL(page.url()).searchParams.get('intent') === 'mentor');
    assert.equal(await page.locator('.collab-card').count(), 1);
    failSave = false;
    await page.getByRole('button', { name: 'Collaborator', exact: true }).click();
    await page.waitForURL('**/*intent=collaborator*');
    await page.getByText('Find the right collaborator for this work', { exact: true }).waitFor();
    assert.equal(await page.locator('.collab-card').count(), 0);
    assert.equal(await page.locator('.message-user').count(), 0);
    assert.equal(await page.locator('.chat-input').inputValue(), '');
    await page.getByRole('button', { name: 'Mentor', exact: true }).click();
    await page.waitForURL('**/*intent=mentor*');
    await page.getByText('Find the right professor or mentor', { exact: true }).waitFor();
    assert.equal(await page.locator('.collab-card').count(), 0);
    await page.locator('.session-sidebar').getByRole('option').filter({ hasText: 'mentor chat' }).first().click();
    await page.getByText('Why this person', { exact: true }).waitFor();
    assert.deepEqual(errors, []);
    console.log('PASS: one-time draft attach, include/exclude, search payload, save-failure retention, mode isolation both directions, sidebar mentor restoration, four responsive widths, no page errors.');
  } finally { await browser.close(); }
})().catch((error) => { console.error(error); process.exitCode = 1; });
