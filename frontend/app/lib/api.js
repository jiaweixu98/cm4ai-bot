// Same-origin by default so the browser talks to Next on :3100, which proxies
// chat/rerank to the already-loaded SPECTER. Set NEXT_PUBLIC_API_URL only when
// you really want a cross-origin backend.
const rawApiBase = process.env.NEXT_PUBLIC_API_URL;
const API_BASE = (rawApiBase !== undefined ? rawApiBase : "").replace(/\/$/, "");

function withMatrixAuth(headers = {}, authToken) {
  if (!authToken) return headers;
  return { ...headers, "x-matrix-user-token": authToken };
}

export async function fetchAuthor(aid) {
  const res = await fetch(`${API_BASE}/api/author/${aid}`);
  if (!res.ok) throw new Error("Author not found");
  return res.json();
}

export async function generateQuery({ aid, userInput, currentQuery, pastQueries, priorInputs }) {
  const res = await fetch(`${API_BASE}/api/generate-query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      aid,
      user_input: userInput,
      current_query: currentQuery || null,
      past_queries: pastQueries || [],
      prior_inputs: priorInputs || [],
    }),
  });
  if (!res.ok) throw new Error("Query generation failed");
  return res.json();
}

export async function checkConfirmation({ userText, currentQuery, priorInputs }) {
  const res = await fetch(`${API_BASE}/api/check-confirmation`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      user_text: userText,
      current_query: currentQuery,
      prior_inputs: priorInputs || [],
    }),
  });
  if (!res.ok) throw new Error("Confirmation check failed");
  return res.json();
}

export async function searchCandidates({
  aid,
  query,
  topK = 8,
  bridge2aiOnly = false,
  outsideNetwork = false,
  teamMemberIds = [],
  researchPlan = null,
  signal,
}) {
  // Same-origin Next route talks to the already-loaded MATRIX backend.
  const res = await fetch("/api/search-people", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    signal,
    body: JSON.stringify({
      aid: aid || "unlinked",
      query,
      top_k: topK,
      bridge2ai_only: Boolean(bridge2aiOnly),
      outside_network: Boolean(outsideNetwork),
      team_member_ids: Array.isArray(teamMemberIds) ? teamMemberIds.map(String) : [],
      research_plan: researchPlan && typeof researchPlan === "object" ? researchPlan : null,
    }),
  });
  if (!res.ok) throw new Error("Search failed");
  return res.json();
}

export async function explainCandidates({
  aid,
  teamMemberIds,
  teamPeople,
  seekerPapers,
  query,
  intent,
  candidates,
  signal,
}) {
  const res = await fetch("/api/why-lines", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    signal: AbortSignal.any([...(signal ? [signal] : []), AbortSignal.timeout(22000)]),
    body: JSON.stringify({
      aid,
      team_member_ids: teamMemberIds,
      team_people: teamPeople || [],
      seeker_papers: seekerPapers || [],
      query,
      intent,
      candidates,
    }),
  });
  if (!res.ok) throw new Error("Match notes failed");
  return res.json();
}

export function rerankCandidates({ aid, query, candidates }, onBatch, onComplete, onError, signal) {
  const url = `${API_BASE}/api/rerank`;
  fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    signal,
    body: JSON.stringify({ aid, query, candidates }),
  }).then((res) => {
    if (!res.ok || !res.body) {
      if (onError) onError({ error: `Rerank failed (${res.status})` });
      return;
    }
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    function processLines(lines, eventState) {
      for (const line of lines) {
        if (line.startsWith("event:")) {
          eventState.eventType = line.slice(6).trim();
        } else if (line.startsWith("data:")) {
          eventState.dataStr = line.slice(5).trim();
        } else if (line.trim() === "" && eventState.dataStr) {
          try {
            const data = JSON.parse(eventState.dataStr);
            if (eventState.eventType === "batch" && onBatch) onBatch(data);
            else if (eventState.eventType === "complete" && onComplete) onComplete(data);
            else if (eventState.eventType === "error" && onError) onError(data);
          } catch (e) {
            // skip malformed
          }
          eventState.eventType = "";
          eventState.dataStr = "";
        }
      }
    }

    const eventState = { eventType: "", dataStr: "" };

    function read() {
      reader.read().then(({ done, value }) => {
        if (done) {
          // Flush remaining buffer when stream ends
          if (buffer.trim()) {
            const lines = buffer.split("\n");
            lines.push("");  // add empty line to trigger event dispatch
            processLines(lines, eventState);
          }
          return;
        }
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";
        processLines(lines, eventState);
        read();
      });
    }
    read();
  }).catch((err) => {
    if (err?.name === "AbortError") return;
    if (onError) onError({ error: err.message });
  });
}

export async function chatMessage({
  attachedContext = [],
  contextPersonIds = [],
  aid,
  userInput,
  conversationHistory,
  currentQuery,
  pastQueries,
  priorInputs,
  searchResults,
  searchPhase,
  intent,
  pendingResearchPlan = null,
  signal,
}) {
  const res = await fetch("/api/chat-lite", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    signal,
    body: JSON.stringify({
      aid: aid || "unlinked",
      user_input: userInput,
      context_person_ids: contextPersonIds.map(String),
      attached_context: Array.isArray(attachedContext)
        ? attachedContext.map((text) => String(text || "").trim().slice(0, 1200)).filter(Boolean).slice(0, 5)
        : [],
      conversation_history: conversationHistory || [],
      current_query: currentQuery || null,
      past_queries: pastQueries || [],
      prior_inputs: priorInputs || [],
      search_results: searchResults || [],
      search_phase: searchPhase || null,
      intent: intent || null,
      pending_research_plan: pendingResearchPlan && typeof pendingResearchPlan === "object"
        ? pendingResearchPlan
        : null,
    }),
  });
  const payload = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error(payload.error || "Chat is unavailable. Try again.");
  }
  return payload;
}

export async function listChatSessions({ aid, intent, authToken }) {
  const params = new URLSearchParams({ aid: String(aid || "unlinked") });
  if (intent) params.set("intent", String(intent));
  const res = await fetch(`${API_BASE}/api/chat-sessions?${params.toString()}`, {
    headers: withMatrixAuth({}, authToken),
  });
  if (!res.ok) throw new Error("Session list request failed");
  return res.json();
}

export async function getChatSession({ sessionId, authToken }) {
  const res = await fetch(`${API_BASE}/api/chat-sessions/${encodeURIComponent(sessionId)}`, {
    headers: withMatrixAuth({}, authToken),
  });
  if (!res.ok) throw new Error("Session fetch failed");
  return res.json();
}

export async function createChatSession({ aid, focalAuthorName, messages, state, authToken }) {
  const res = await fetch(`${API_BASE}/api/chat-sessions`, {
    method: "POST",
    headers: withMatrixAuth({ "Content-Type": "application/json" }, authToken),
    body: JSON.stringify({
      aid,
      focal_author_name: focalAuthorName || "",
      messages: messages || [],
      state: state || {},
    }),
  });
  if (!res.ok) throw new Error("Session creation failed");
  return res.json();
}

export async function saveChatSession({ sessionId, aid, focalAuthorName, messages, state, authToken }) {
  const res = await fetch(`${API_BASE}/api/chat-sessions/${encodeURIComponent(sessionId)}`, {
    method: "PUT",
    signal: AbortSignal.timeout(8000),
    headers: withMatrixAuth({ "Content-Type": "application/json" }, authToken),
    body: JSON.stringify({
      aid,
      focal_author_name: focalAuthorName || "",
      messages: messages || [],
      state: state || {},
    }),
  });
  if (!res.ok) throw new Error("Session save failed");
  return res.json();
}

export async function fetchGraphPath(aid, collaboratorId) {
  const res = await fetch(`${API_BASE}/api/graph-path/${aid}/${collaboratorId}`);
  if (!res.ok) throw new Error("Graph path fetch failed");
  return res.json();
}

export async function submitErrorReport({
  project = "cm4ai-bot",
  page,
  reportFolder = "matrix_error",
  feedback,
  context = {},
  currentUrl,
  userAgent,
}) {
  const payload = {
    project,
    page,
    report_folder: reportFolder,
    feedback,
    context,
    current_url: currentUrl || null,
    user_agent: userAgent || null,
  };

  // Primary: backend API (same base as chat/search/rerank)
  const primaryRes = await fetch(`${API_BASE}/api/report-error`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (primaryRes.ok) return primaryRes.json();

  // Fallback: Next.js local API route in the frontend app
  const fallbackRes = await fetch(`/api/report-error`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (fallbackRes.ok) return fallbackRes.json();

  throw new Error(
    `Error report submission failed (backend ${primaryRes.status}, fallback ${fallbackRes.status})`
  );
}
