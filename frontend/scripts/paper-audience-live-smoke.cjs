// Live local chat acceptance. Uses the configured model, not a mocked response.
const assert = require('node:assert/strict');
const {chromium} = require('playwright');
(async()=>{
  const browser=await chromium.launch({headless:true,args:['--disable-dev-shm-usage']});
  try {
    const page=await browser.newPage({viewport:{width:1440,height:1000}}),errors=[];
    page.on('pageerror',error=>errors.push(error.message));
    await page.goto('http://127.0.0.1:3100/?aid=unlinked',{waitUntil:'domcontentloaded'});
    const input=page.getByLabel('Describe the research help you need');
    async function send(question) {
      const responsePromise=page.waitForResponse(response=>response.url().endsWith('/api/chat-lite')
        && response.request().method()==='POST',{timeout:180000});
      await input.fill(question);
      await page.getByLabel('Send message',{exact:true}).click();
      const response=await responsePromise;
      await response.finished();
      const raw=await response.text();
      const result=raw.includes('event: result')
        ? JSON.parse(raw.match(/event: result\r?\ndata: (.+)/)?.[1]||'null') : JSON.parse(raw);
      assert.equal(response.status(),200);
      assert.ok(result?.reply,'A live turn must finish with a published answer');
      await page.getByLabel('Send message',{exact:true}).waitFor({timeout:180000});
      return result;
    }
    const audience=await send('I want to promote paper W2111573647. Find up to four potential audience members based on related published work, with supporting titles.');
    assert.ok(audience.shortlist?.length>0&&audience.shortlist.length<=4);
    await page.locator('.collab-card').first().waitFor();
    const text=await page.locator('body').innerText();
    assert.ok(/PAGER|gene.signature|gene.set|pathway/i.test(text),'Scientific paper evidence must be displayed');
    assert.ok(!/Considered \d+ matching researchers/.test(text),'No redundant candidate-pool metric');
    const followup=await send('What did the supporting publication shown in the first card actually do? Explain that specific study’s methods briefly, using its abstract.');
    assert.equal(followup.result_update,'keep','A paper follow-up must retain the displayed cards');
    assert.ok(followup.citations?.some(citation=>citation.title===audience.shortlist[0].papers[0].title),
      'The explanation must cite the publication shown in the first card');
    assert.equal(await page.locator('.collab-card').count(),audience.shortlist.length);
    const unrelated=await send('Find an audience in the local researcher catalog for a paper about translating Sumerian cuneiform tax records and reconstructing Bronze Age irrigation management in Mesopotamia. Only recommend people with directly relevant publications; if none exist, give no people.');
    assert.equal(unrelated.shortlist?.length||0,0,'No forced audience from unrelated nearest papers');
    assert.equal(await page.locator('.collab-card').count(),0,'An empty search clears earlier cards');
    const empty=await send('Explore Jake Y. Chen’s coauthors on shared publications dated 2035 through 2040. Only list people if those records exist.');
    assert.equal(empty.shortlist?.length||0,0);
    assert.match(empty.reply,/no|not|none/i);
    assert.deepEqual(errors,[]);
    console.log('PASS: live Promote shows supporting publications, a follow-up keeps cards and cites the displayed paper, unrelated papers give no cards, empty Explore answers normally, and no page errors.');
  } finally {await browser.close();}
})().catch(error=>{console.error(error.message);process.exitCode=1});
