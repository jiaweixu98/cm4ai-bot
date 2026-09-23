const { chromium } = require("playwright");
const assert = require("node:assert/strict");

async function mockApi(page, capture) {
  await page.addInitScript(() => sessionStorage.setItem("matrix_user_token", "mock-only"));
  await page.route("**/api/**", async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    const data = request.postDataJSON() || {};
    let payload = {};

    if (url.pathname === "/api/chat-sessions" && request.method() === "GET") {
      payload = { sessions: [] };
    } else if (url.pathname.startsWith("/api/chat-sessions")) {
      payload = {
        session: {
          ...data,
          id: "unified-session",
          title: "Research guidance",
          last_message_at: new Date().toISOString(),
          intent: data.state?.intent || "collaborator",
        },
      };
    } else if (url.pathname === "/api/chat-lite") {
      capture.chats.push(data);
      if (data.user_input === "Search that") {
        payload = { action: "confirm" };
      } else {
        const mentor = data.intent === "mentor";
        payload = {
          action: "search",
          query: mentor ? "federated clinical analysis" : "clinical model validation",
          justification: "I focused the search on the research need you described.",
          research_plan: {
            schema_version: "research-fit-v1",
            question: mentor ? "Learn federated analysis across clinical sites" : "Find a collaborator for clinical model validation",
            intent: mentor ? "mentor" : "collaborator",
            fields: {
              topic: mentor ? ["federated analysis"] : ["clinical model"],
              method: [],
              population: [],
              setting: mentor ? ["clinical sites"] : [],
              evidence_stage: mentor ? [] : ["validation"],
              needed_capability: mentor ? ["learn federated analysis"] : ["model validation"],
              constraints: [],
            },
            affiliation_filters: [],
            scope: mentor ? "bridge2ai" : "all",
            context: { team_author_ids: [], selected_work_ids: [], exclude_recorded_direct_coauthors: !mentor },
            retrieval_query: mentor ? "federated clinical analysis" : "clinical model validation",
            status: "ready",
            clarification_question: null,
            clarification_options: [],
          },
        };
      }
    } else if (url.pathname === "/api/search-people") {
      capture.searches.push(data);
      const mentor = Boolean(data.bridge2ai_only);
      payload = {
        candidates: [{
          author_id: mentor ? "22" : "33",
          name: mentor ? "Priya Raman" : "Mateo Alvarez",
          affiliation: mentor ? "Example Medical Center" : "Regional Health Network",
          is_bridge2ai_member: mentor,
          papers: [{
            Title: mentor ? "Federated analysis across clinical sites" : "Prospective validation across health systems",
            PubYear: 2025,
          }],
        }],
      };
    } else if (url.pathname === "/api/why-lines") {
      payload = {
        context_basis: data.intent === "mentor" || !data.team_member_ids?.length ? "need" : "need_team",
        team_count: data.team_member_ids?.length || 0,
        results: data.candidates.map((candidate) => ({
          author_id: candidate.author_id,
          source: "llm",
          explanation: data.intent === "mentor"
            ? "This publication addresses the learning goal through federated clinical analysis."
            : "This publication adds multi-site validation evidence relevant to the stated team need.",
          evidence_paper_index: 0,
        })),
      };
    }
    await route.fulfill({ json: payload });
  });
}

(async () => {
  const browser = await chromium.launch({ headless: true, args: ["--disable-dev-shm-usage"] });
  try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 900 }, colorScheme: "light" });
    const capture = { chats: [], searches: [] };
    const errors = [];
    page.on("pageerror", (error) => errors.push(error.message));
    await mockApi(page, capture);
    await page.route("http://127.0.0.1:3100/test-shell", (route) => route.fulfill({
      contentType: "text/html",
      body: `<!doctype html><html><body style="margin:0">
        <script>
          window.addEventListener("message", function (event) {
            if (event.data && event.data.type === "bridge2ai-request-saved") {
              event.source.postMessage({ type: "bridge2ai-saved-people", people: [{ authorId: 11, name: "Saved Researcher", affiliation: "University" }] }, event.origin);
            }
          });
        </script>
        <iframe title="MATRIX" src="http://127.0.0.1:3100/?embedded=1&fresh=1" style="border:0;width:100vw;height:100vh;display:block"></iframe>
      </body></html>`,
    }));
    await page.goto("http://127.0.0.1:3100/test-shell", { waitUntil: "networkidle", timeout: 45000 });
    const app = page.frameLocator('iframe[title="MATRIX"]');

    await app.getByRole("heading", { name: "Who do you need for your research?" }).waitFor();
    assert.equal(await app.locator(".session-sidebar").count(), 0);
    assert.equal(await app.getByRole("button", { name: "Learn from a researcher" }).count(), 1);
    assert.equal(await app.getByRole("button", { name: "Strengthen a research team" }).count(), 1);
    assert.equal(await app.getByRole("button", { name: "Mentor", exact: true, pressed: false }).count(), 1);
    assert.equal(await app.getByRole("button", { name: "History", exact: true }).count(), 1);
    await app.locator(".saved-strip-label").waitFor();
    await app.getByRole("button", { name: "Include Saved Researcher in team context", exact: true }).click();
    await page.screenshot({ path: "/tmp/matrix-unified-empty-light.png", fullPage: false });

    require("fs").writeFileSync(
      "/tmp/matrix-one-time-context.txt",
      "Cross-site EHR phenotyping draft\n\nUse this only as a short, one-time research context."
    );
    await app.locator('.chat-input-container input[type="file"]').setInputFiles("/tmp/matrix-one-time-context.txt");
    await app.getByText("Optional one-time context.", { exact: false }).waitFor();
    await app.locator(".attach-chip").waitFor();
    await app.getByRole("button", { name: "Learn from a researcher" }).click();
    await app.getByText("Topic:", { exact: false }).waitFor();
    await app.getByRole("button", { name: "Search that", exact: true }).click();
    await app.getByRole("heading", { name: /Potential mentors/ }).waitFor();
    assert.equal(capture.chats[0].intent, "mentor");
    assert.match(capture.chats[0].attached_context[0], /Cross-site EHR phenotyping draft/);
    assert.equal(capture.searches[0].bridge2ai_only, true);
    assert.equal(capture.searches[0].research_plan.schema_version, "research-fit-v1");
    assert.deepEqual(capture.searches[0].research_plan.fields.topic, ["federated analysis"]);
    assert.equal(capture.searches[0].outside_network, false);
    assert.equal("representative_titles" in capture.searches[0], false);
    assert.equal(await app.locator(".attach-chip").count(), 0);

    await page.screenshot({ path: "/tmp/matrix-unified-light.png", fullPage: false });

    await app.getByRole("button", { name: "Team", exact: true }).click();
    await app.locator(".chat-input").fill("Find a collaborator for clinical validation");
    await app.getByRole("button", { name: "Send", exact: true }).click();
    await app.getByRole("button", { name: "Search that", exact: true }).click();
    await app.getByRole("heading", { name: /Potential collaborators/ }).waitFor();
    assert.equal(capture.chats.at(-2).intent, "collaborator");
    assert.equal(capture.chats.at(-1).intent, "collaborator");
    assert.equal(capture.searches.at(-1).outside_network, true);
    assert.deepEqual(capture.searches.at(-1).team_member_ids, ["11"]);
    assert.equal(capture.searches.at(-1).research_plan.fields.evidence_stage[0], "validation");

    for (const width of [1440, 1024, 768, 390]) {
      await page.setViewportSize({ width, height: 900 });
      assert(await app.locator("html").evaluate((root) => root.scrollWidth <= root.clientWidth + 1), `horizontal overflow at ${width}`);
      const composer = await app.locator(".chat-input-wrapper").boundingBox();
      assert(composer && composer.y >= 0 && composer.y + composer.height <= 900, `composer clipped at ${width}`);
      if (width === 390) await page.screenshot({ path: "/tmp/matrix-unified-mobile.png", fullPage: false });
    }

    await page.emulateMedia({ colorScheme: "dark", reducedMotion: "reduce" });
    await page.setViewportSize({ width: 1440, height: 900 });
    await page.screenshot({ path: "/tmp/matrix-unified-dark.png", fullPage: false });
    assert.deepEqual(errors, []);
    console.log("PASS: unified mentor/team chat, focus switching, evidence results, light/dark rendering, reduced motion, and responsive widths.");
  } finally {
    await browser.close();
  }
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
