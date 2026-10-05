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
    await rail.getByText("Context", { exact: true }).click();
    assert.equal(await page.getByRole("button", { name: /^Context/ }).count(), 0, "top-bar Context button is gone");
    assert.equal(await page.getByText("Coauthors", { exact: true }).count(), 0, "coauthor list is gone");
    await rail.getByText("Avery Researcher", { exact: true }).waitFor();
    await rail.getByText("Example University", { exact: true }).waitFor();
    await rail.getByText("Sent with questions: 3 recent papers \u00b7 2 topics", { exact: true }).waitFor();
    await rail.getByText("Today", { exact: true }).waitFor();
    await shot(page, "desktop-rail");
    assert.equal(await rail.getByRole("button", { name: "Recent", exact: true }).count(), 1);
    await rail.getByRole("button", { name: "Pick", exact: true }).click();
    assert.equal(await rail.locator(".you-papers").getByRole("checkbox").count(), 3, "Pick shows selectable papers");
    assert.equal(await rail.getByText("Same institution", { exact: true }).count(), 0);
    assert.equal(await rail.getByRole("button", { name: "5 years", exact: true }).count(), 0);
    await shot(page, "desktop-edit-open");

    await page.locator(".chat-input").fill("Use my profile and focus settings");
    await page.getByRole("button", { name: "Send message", exact: true }).click();
    await page.getByText("I will use that context for this conversation.", { exact: true }).waitFor();
    const sent = capture.chats.at(-1).context_choices;
    assert.equal(sent.same_place, false);
    assert.equal(sent.recent_years, 0);
    assert.equal(sent.paper_scope, "chosen");
    assert.equal(sent.paper_titles.length, 3);
    await page.waitForTimeout(1_000);
    assert(capture.puts.length > 0, "a changed conversation should save");

    await page.setViewportSize({ width: 390, height: 844 });
    await page.waitForTimeout(300);
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1));
    await rail.getByRole("button", { name: "Close sidebar", exact: true }).click();
    await shot(page, "mobile-chat");
    await page.getByRole("button", { name: "Open sidebar", exact: true }).click();
    await rail.getByText("Context", { exact: true }).click();
    await rail.getByText("Avery Researcher", { exact: true }).waitFor();
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1));
    await shot(page, "mobile-rail-open");
    await page.keyboard.press("Escape");
    await page.waitForTimeout(200);
    assert.equal(await rail.count(), 0, "Escape closes the mobile rail");

    assert.deepEqual(errors, []);
    console.log("PASS: owner-wide history, unchanged-open guard, You card controls, request payload, changed-chat save, and responsive rail.");
  } finally {
    await browser.close();
  }
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
