// Live local catalog/model acceptance; URL identities exercise the handoff, not OAuth.
const assert = require('node:assert/strict');
const {chromium} = require('playwright');
const fs = require('node:fs');
const base = (process.env.CHAT_SMOKE_BASE || 'http://127.0.0.1:3100/').replace(/\/?$/, '/');
const evidencePath = process.env.CHAT_SMOKE_EVIDENCE || '/tmp/personalized-showcase-evidence.json';

(async () => {
  const browser = await chromium.launch({headless: true, args: ['--disable-dev-shm-usage']});
  const evidence = [], errors = [];
  try {
    const page = await browser.newPage({viewport: {width: 1440, height: 1000}});
    page.on('pageerror', error => errors.push(error.message));
    async function open(params) {
      await page.goto(`${base}?fresh=1&${params}`, {waitUntil: 'domcontentloaded'});
      await page.locator('.showcase-examples summary').click();
    }
    async function starter(label) {
      await page.locator('.showcase-examples .starter-card').filter({has: page.getByText(label, {exact: true})}).click();
      return page.getByLabel('Describe the research help you need').inputValue();
    }
    async function send(question, label) {
      const pending = page.waitForResponse(r => r.url().endsWith('/api/chat-lite') && r.request().method() === 'POST', {timeout: 180000});
      await page.getByLabel('Describe the research help you need').fill(question);
      await page.getByRole('button', {name: 'Send message', exact: true}).click();
      const response = await pending;
      await response.finished();
      const raw = await response.text();
      const result = raw.includes('event: result')
        ? JSON.parse(raw.match(/event: result\r?\ndata: (.+)/)?.[1] || 'null') : JSON.parse(raw);
      assert.equal(response.status(), 200, label);
      assert.ok(result?.reply, `${label}: a published answer is required`);
      await page.getByRole('button', {name: 'Send message', exact: true}).waitFor({timeout: 180000});
      evidence.push({case: label, tools: result.tool_calls, filters: result.explore_filters, citations: result.citations, reply: result.reply,
        people: (result.shortlist || []).map(p => ({id: p.author_id, name: p.name, why: p.why, titles: p.papers.map(w => w.title)}))});
      fs.writeFileSync(evidencePath, JSON.stringify(evidence, null, 2));
      console.log(`PASS ${label}: ${(result.shortlist || []).length} cards`);
      return result;
    }

    await open('aid=unlinked');
    const proposal = await starter('Promote');
    assert.match(proposal, /proposing a paper/);
    const audience = await send(proposal, 'Exact Promote starter');
    assert.ok(audience.tool_calls.includes('find_paper_audience'));
    assert.ok(audience.shortlist.length >= 3 && audience.shortlist.length <= 4);
    assert.ok(new Set(audience.shortlist.map(p => p.papers[0].title.toLowerCase())).size >= 3);
    assert.ok(audience.shortlist.every(p => p.why && !/see supporting publication/i.test(p.why)));
    assert.ok(!/rather than confirmed|not confirmed/i.test(audience.reply));
    await page.locator('.collab-card').first().waitFor();
    assert.equal(await page.getByRole('button', {name: /^Save /}).count(), audience.shortlist.length);
    assert.equal(await page.getByRole('button', {name: /^View on graph/}).count(), audience.shortlist.length);
    const more = page.locator('.more-evidence').first();
    if (await more.count()) {await more.locator('summary').click(); assert.ok(await more.locator('.paper-title').count());}
    await page.screenshot({path: '/tmp/personalized-promote.png'});
    const followup = await send('Explain the specific supporting publication shown in the first card, using its abstract.', 'Supporting-paper follow-up');
    assert.equal(followup.result_update, 'keep');
    assert.ok(followup.citations.some(c => c.title === audience.shortlist[0].papers[0].title));

    // Profile-name failure must not turn a linked identity into the guest starter.
    // This one identity-display fixture leaves catalog retrieval and the model live.
    await page.route('**/api/author/5058469124', route => route.fulfill({status: 503, json: {error: 'Profile fixture unavailable'}}));
    await open('aid=5058469124&handoff=person&context_person_id=5010789029');
    await page.getByRole('button', {name: /^Explore Show my top 10/}).waitFor();
    assert.match(await starter('Explore'), /^Show my top 10 recorded coauthors, with/);
    await page.unroute('**/api/author/5058469124');
    await page.getByRole('button', {name: 'Remove Gary D. Bader from this chat', exact: true}).waitFor();
    const explored = await send(await starter('Explore'), 'Viewer Explore with another selected graph person');
    assert.equal(explored.explore_filters.author_id, '5058469124');
    assert.ok(explored.shortlist.length > 0 && explored.shortlist.length <= 10);
    assert.ok(!/which researcher/i.test(explored.reply));
    assert.ok(!/in the order returned|catalog.s ranking/i.test(explored.reply));
    assert.ok(explored.suggested_followups.some(s => /year/i.test(s)));
    assert.ok(explored.suggested_followups.some(s => /institution/i.test(s)));
    const dated = await send('Only shared papers from 2020 through 2026.', 'Explore date follow-up');
    assert.equal(dated.explore_filters.author_id, '5058469124');
    assert.equal(dated.explore_filters.from_year, 2020);
    assert.equal(dated.explore_filters.to_year, 2026);
    assert.ok((dated.shortlist || []).every(p => p.papers.every(w => Number(w.year) >= 2020 && Number(w.year) <= 2026)));
    const institutional = await send('Show up to four coauthors currently at University of Alabama at Birmingham.', 'Explore combined institution follow-up');
    assert.equal(institutional.explore_filters.author_id, '5058469124');
    assert.equal(institutional.explore_filters.from_year, 2020);
    assert.equal(institutional.explore_filters.to_year, 2026);
    assert.match(institutional.explore_filters.institution, /Alabama|UAB/i);
    const api = await page.request.get(`${base}api/author/5058469124/explore-coauthors?from_year=2020&to_year=2026&institution=University%20of%20Alabama%20at%20Birmingham`);
    assert.equal(api.status(), 200);
    const matchingIds = new Set((await api.json()).people.map(p => p.author_id));
    assert.ok((institutional.shortlist || []).every(p => matchingIds.has(p.author_id)));
    await page.screenshot({path: '/tmp/personalized-explore.png'});

    await open('aid=5010789029&handoff=person&context_person_id=5058469124');
    await page.getByRole('button', {name: /Explore Show my top 10 recorded coauthors for Gary/}).waitFor();
    const second = await send(await starter('Explore'), 'Second viewer identity');
    assert.equal(second.explore_filters.author_id, '5010789029');

    await open('aid=unlinked');
    assert.match(await starter('Explore'), /choose a researcher/);
    const guest = await send(await starter('Explore'), 'Unlinked researcher selection');
    assert.ok(!guest.shortlist?.length);

    await open('aid=unlinked');
    // Text attachments are read in the browser; PDF/Word use the extraction API.
    await page.locator('input[type=file]').setInputFiles({name: 'proposed-paper.txt', mimeType: 'text/plain',
      buffer: Buffer.from('Interpretable pathway networks for disease-associated molecular phenotypes\n\nA proposed approach combines pathway and gene-set enrichment with gene-expression networks to interpret disease-associated molecular phenotypes.')});
    await page.locator('.attach-chip').waitFor();
    const attached = await send('Find up to four audience members for the attached proposed paper, with related publications.', 'Attached promotion target');
    assert.ok(attached.tool_calls.includes('find_paper_audience') && attached.shortlist?.length);
    await page.locator('.attach-chip').waitFor({state: 'detached'});
    const unrelated = await send('Find an audience for a proposed paper about translating Sumerian cuneiform tax records and reconstructing Bronze Age irrigation in Mesopotamia. Only include directly relevant published work.', 'Unrelated topic abstention');
    assert.ok(!unrelated.shortlist?.length);
    assert.equal(await page.locator('.collab-card').count(), 0);
    await page.setViewportSize({width: 390, height: 844});
    assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1));
    assert.deepEqual(errors, []);
    fs.writeFileSync(evidencePath, JSON.stringify(evidence, null, 2));
    console.log('PASS: live examples, distinct papers, identity separation, filter retention, attachments, abstention and mobile fit. URL identities do not validate OAuth.');
  } finally {await browser.close();}
})().catch(error => {console.error(error); process.exitCode = 1;});
