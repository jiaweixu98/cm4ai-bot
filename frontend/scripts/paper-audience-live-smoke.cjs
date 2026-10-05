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
    await input.fill('I want to promote paper W2111573647. Find a potential audience based on related published work, with supporting titles.');
    await page.getByLabel('Send message',{exact:true}).click();
    await page.getByText('Aik Choon Tan',{exact:true}).first().waitFor({timeout:90000});
    const text=await page.locator('body').innerText();
    assert.ok(/PAGER|gene.signature|gene.set|pathway/i.test(text),'Scientific paper evidence must be displayed');
    assert.deepEqual(errors,[]);
    console.log('PASS: live Promote through the actual composer displays an audience and supporting publication evidence; no page errors');
  } finally {await browser.close();}
})().catch(error=>{console.error(error.message);process.exitCode=1});
