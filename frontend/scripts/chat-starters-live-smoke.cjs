// All six exact starters through the running composer/catalog/model. No auth fixtures.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {chromium} = require('playwright');
const base = (process.env.CHAT_SMOKE_BASE || 'http://127.0.0.1:3100/').replace(/\/?$/, '/');
const evidencePath = process.env.CHAT_SMOKE_EVIDENCE || '/tmp/chat-starters-live-evidence.json';

(async () => {
  const config = await import('data:text/javascript;base64,' + fs.readFileSync(path.join(__dirname, '../app/lib/personaConfig.js')).toString('base64'));
  const starters = [...config.GENERAL_STARTERS, ...config.showcaseStarters('Jake Y. Chen', true)];
  const expectedTools = {'Find a mentor': 'search_people', 'Build a team': 'assemble_team',
    'Find a niche': 'analyze_niche', Explore: 'explore_coauthors', Collaborate: 'search_people', Promote: 'find_paper_audience'};
  const browser = await chromium.launch({headless: true, args: ['--disable-dev-shm-usage']});
  const evidence = {base, started_at: new Date().toISOString(), source: 'Live catalog and model; URL linked identity for Explore, no OAuth validation', cases: [], page_errors: []};
  try {
    const page = await browser.newPage({viewport: {width: 1440, height: 1000}});
    page.on('pageerror', e => evidence.page_errors.push(e.message));
    for (const starter of starters) {
      const aid = starter.label === 'Explore' ? '5058469124' : 'unlinked';
      await page.goto(`${base}?fresh=1&aid=${aid}`, {waitUntil: 'domcontentloaded'});
      await page.getByLabel('Describe the research help you need').waitFor();
      if (config.SHOWCASE_STARTERS.some(s => s.label === starter.label)) await page.locator('.showcase-examples summary').click();
      const card = page.locator('.starter-card').filter({has: page.getByText(starter.label, {exact: true})});
      if (starter.label === 'Explore') await page.getByRole('button', {name: /Explore Show my top 10 recorded coauthors for Jake/}).waitFor();
      await card.click();
      const prompt = await page.getByLabel('Describe the research help you need').inputValue();
      assert.equal(prompt, starter.prompt);
      const responsePromise = page.waitForResponse(r => r.url().endsWith('/api/chat-lite') && r.request().method() === 'POST', {timeout: 180000});
      await page.getByRole('button', {name: 'Send message', exact: true}).click();
      const response = await responsePromise;
      await response.finished();
      const raw = await response.text();
      const result = raw.includes('event: result') ? JSON.parse(raw.match(/event: result\r?\ndata: (.+)/)?.[1] || 'null') : JSON.parse(raw);
      assert.equal(response.status(), 200);
      assert.ok(result?.reply, starter.label);
      evidence.cases.push({label: starter.label, prompt, ...result});
      fs.writeFileSync(evidencePath, JSON.stringify(evidence, null, 2));
      assert.ok(result.tool_calls?.includes(expectedTools[starter.label]), `${starter.label}: task-specific routing`);
      if (starter.label !== 'Find a niche') assert.ok(result.shortlist?.length, `${starter.label}: grounded recommendations`);
      if (starter.label === 'Find a mentor') assert.equal(result.shortlist_kind, 'mentors');
      if (starter.label === 'Build a team') assert.ok(result.shortlist.every(p => p.role), 'Team members have explicit roles');
      if (starter.label === 'Explore') {
        assert.equal(result.explore_filters.author_id, aid);
        assert.ok(result.shortlist.length <= 10);
        assert.ok(!/in the order returned|catalog.s ranking/i.test(result.reply));
      }
      if (starter.label === 'Collaborate') assert.ok(result.shortlist.every(p => p.is_bridge2ai_member), 'Collaborate retains Bridge2AI scope');
      if (starter.label === 'Promote') {
        assert.ok(result.shortlist.length >= 3 && result.shortlist.length <= 4);
        assert.ok(new Set(result.shortlist.map(p => p.papers[0].title.toLowerCase())).size >= 3);
      }
      for (const person of result.shortlist || []) {
        assert.ok(person.name && person.why && person.papers?.length);
        assert.ok(person.papers.every(p => p.title && p.evidence_id));
        assert.ok(!/see supporting publication/i.test(person.why));
      }
      await page.getByRole('button', {name: 'Send message', exact: true}).waitFor();
      if (result.shortlist?.length) {
        await page.waitForFunction(n => document.querySelectorAll('.collab-card').length === n, result.shortlist.length);
        await page.getByRole('button', {name: `Save ${result.shortlist[0].name}`, exact: true}).click();
        await page.getByRole('button', {name: `Remove from saved people ${result.shortlist[0].name}`, exact: true}).waitFor();
        await page.getByRole('button', {name: /^View on graph/}).first().click();
        await page.getByText('Open this from the graph to view the profile.', {exact: true}).waitFor();
      }
      console.log(`PASS ${starter.label}: ${(result.shortlist || []).length} cards; ${result.tool_calls.join(', ')}`);
    }
    assert.deepEqual(evidence.page_errors, []);
    evidence.completed_at = new Date().toISOString();
    evidence.complete = true;
    console.log('PASS all six exact starters: routing, grounded cards, scope, local save interaction and profile action. No OAuth or persistent-save claim.');
  } finally {fs.writeFileSync(evidencePath, JSON.stringify(evidence, null, 2)); await browser.close();}
})().catch(error => {console.error(error); process.exitCode = 1;});
