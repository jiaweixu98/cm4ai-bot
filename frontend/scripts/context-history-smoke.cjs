// Run against local Next with NODE_PATH pointing to an installed Playwright package.
// All application APIs are mocked. No model, database, or retrieval calls are made.
const { chromium } = require("playwright");
const assert = require("node:assert/strict");

const BASE = process.env.MATRIX_BASE || "http://127.0.0.1:3100";
const SHOTS = process.env.MATRIX_SHOTS || "";
const shot = (page, name) => (SHOTS ? page.screenshot({ path: `${SHOTS}/matrix-${name}.png`, fullPage: false }) : Promise.resolve());

const now = Date.now();
const summaries = [
  {
    id: "mentor-old",
    title: "Older mentor chat",
    intent: "mentor",
    last_message_at: new Date(now - 60_000).toISOString(),
  },
  {
    id: "team-old",
    title: "Earlier team chat",
    intent: "collaborator",
    last_message_at: new Date(now - 120_000).toISOString(),
  },
];

function fullSession(id) {
  const summary = summaries.find((item) => item.id === id);
  return {
    ...summary,
    messages: [{ role: "user", content: "Existing question", at: now - 60_000 }],
    state: {
      intent: summary.intent,
      searchIntent: summary.intent,
      phase: "idle",
      contextChoices: { samePlace: false, recentYears: 0, paperScope: "profile", paperTitles: [] },
    },
  };
}

(async () => {
  const browser = await chromium.launch({ headless: true, args: ["--disable-dev-shm-usage"] });
  try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 900 }, colorScheme: "light" });
    const capture = { listUrls: [], chats: [], puts: [] };
    const errors = [];
    page.on("pageerror", (error) => errors.push(error.message));
    await page.addInitScript(() => sessionStorage.setItem("matrix_user_token", "mock-only"));
    await page.route("**/api/**", async (route) => {
      const request = route.request();
      const url = new URL(request.url());
      const data = request.postDataJSON() || {};
      let payload = {};

      if (url.pathname === "/api/chat-sessions" && request.method() === "GET") {
        capture.listUrls.push(url.toString());
        payload = { sessions: summaries };
      } else if (url.pathname.startsWith("/api/chat-sessions/") && request.method() === "GET") {
        payload = { session: fullSession(url.pathname.split("/").at(-1)) };
      } else if (url.pathname.startsWith("/api/chat-sessions/") && request.method() === "PUT") {
        capture.puts.push(data);
        const id = url.pathname.split("/").at(-1);
        payload = { session: { ...fullSession(id), ...data, id, title: data.messages?.[0]?.content || "Chat", last_message_at: new Date().toISOString() } };
      } else if (url.pathname === "/api/chat-sessions" && request.method() === "POST") {
        payload = { session: { ...data, id: "new-chat", title: "Draft chat", last_message_at: new Date().toISOString(), intent: data.state?.intent } };
      } else if (url.pathname === "/api/author/42") {
        payload = { author_id: "42", name: "Avery Researcher", affiliation: "Example University", papers: [] };
      } else if (url.pathname === "/api/author/42/details") {
        payload = {
          author_id: "42",
          name: "Avery Researcher",
          affiliation: "Example University",
          topics: ["Clinical informatics", "Data standards"],
          mesh: ["Electronic Health Records"],
          papers: [
            { Title: "Clinical data standards", PubYear: 2025 },
            { Title: "Portable phenotyping", PubYear: 2024 },
            { Title: "Health data quality", PubYear: 2023 },
          ],
        };
      } else if (url.pathname === "/api/author/42/publications") {
        payload={author_id:'43',total:3,papers:[{work_id:'W1',title:'Clinical data standards',year:2025},{work_id:'W2',title:'Portable phenotyping',year:2024},{work_id:'W3',title:'Health data quality',year:2023}]};
      } else if(url.pathname === '/api/context-publications/resolve') {
        payload={selected_publications:(data.references || []).map(p=>({...p,title:p.work_id==='W1'?'Clinical data standards':p.work_id==='W2'?'Portable phenotyping':'Health data quality'}))};
      } else if (url.pathname === "/api/chat-lite") {
        capture.chats.push(data);
        payload = { action: "chat", reply: "I will use that context for this conversation." };
      }
      await route.fulfill({ json: payload });
    });

    await page.goto(`${BASE}/?aid=42&intent=mentor&embedded=1&fresh=1`, { waitUntil: "networkidle", timeout: 45_000 });
    await page.getByText("What should we work on?", { exact: true }).waitFor();
    assert.equal(capture.listUrls.length, 1);
    const historyUrl = new URL(capture.listUrls[0]);
    assert.equal(historyUrl.searchParams.has("aid"), false);
    assert.equal(historyUrl.searchParams.has("intent"), false);
    assert.equal(await page.getByText("Older mentor chat", { exact: true }).count(), 1);
    assert.equal(await page.getByText("Earlier team chat", { exact: true }).count(), 1);

    await page.getByText("Older mentor chat", { exact: true }).click();
    await page.getByText("Existing question", { exact: true }).waitFor();
    await page.waitForTimeout(1_000);
    assert.equal(capture.puts.length, 0, "opening a chat must not save it");

    const rail = page.getByRole("complementary", { name: "Chat history" });
    assert.equal(await rail.getByText('Context',{exact:true}).count(),0);
    assert.equal(await page.getByRole('complementary',{name:'Conversation context'}).count(),0);
    await page.getByRole('button',{name:'Manage context'}).click();
    const panel=page.getByRole('complementary',{name:'Conversation context'});
    await panel.getByRole('button',{name:'Avery Researcher',exact:true}).waitFor();
    await panel.getByText('Example University',{exact:true}).waitFor();
    await panel.getByText('Clinical informatics · Data standards',{exact:true}).waitFor();
    await panel.getByText('No publications selected.',{exact:true}).waitFor();
    await panel.locator('.you-paper-choice').first().waitFor();
    assert.equal(await panel.locator('.you-paper-choice').count(),3);
    await panel.locator('.you-paper-choice input').first().check();
    await panel.getByText('1/8 selected',{exact:true}).waitFor();
    await shot(page,'desktop-context');
    await panel.getByRole('button',{name:'Close context'}).click();

    await page.locator(".chat-input").fill("Use my profile and focus settings");
    await page.getByRole("button", { name: "Send message", exact: true }).click();
    await page.getByText("I will use that context for this conversation.", { exact: true }).waitFor();
    const sent = capture.chats.at(-1).context_choices;
    assert.equal(sent.same_place, false);
    assert.equal(sent.recent_years, 0);
    assert.equal(sent.paper_scope, "chosen");
    assert.deepEqual(sent.paper_titles, []);
    assert.deepEqual(sent.selected_publications, [{author_id:'43',work_id:'W1'}]);
    await page.waitForTimeout(1_000);
    assert(capture.puts.length > 0, "a changed conversation should save");

    await page.setViewportSize({width:390,height:844});
    await page.waitForTimeout(250);
    await page.keyboard.press('Escape');
    await page.getByRole('button',{name:'Manage context'}).click();
    await panel.getByText('1/8 selected',{exact:true}).waitFor();
    assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1));
    await shot(page,'mobile-context');
    await page.keyboard.press('Escape');
    assert.equal(await panel.count(),0,'Escape closes the context drawer');
    await page.getByRole('button',{name:'Open sidebar'}).click();
    await rail.getByText('Earlier team chat',{exact:true}).waitFor();
    await page.keyboard.press('Escape');
    assert.equal(await rail.count(),0,'Escape closes the mobile history rail');

    assert.deepEqual(errors, []);
    console.log("PASS: owner-wide history, unchanged-open guard, topics-only migration, explicit paper request, changed-chat save, right context and mobile history.");
  } finally {
    await browser.close();
  }
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
