// Prefer explicit API URL when provided.
// Otherwise: same-origin in production (for nginx /api reverse proxy),
// and localhost backend in local development.
const rawApiBase = process.env.NEXT_PUBLIC_API_URL;
const API_BASE =
  rawApiBase !== undefined
    ? rawApiBase.replace(/\/$/, "")
    : process.env.NODE_ENV === "production"
      ? ""
      : "http://localhost:8000";

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

export async function searchCandidates({ aid, query, topK = 100 }) {
  const res = await fetch(`${API_BASE}/api/search`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ aid, query, top_k: topK }),
  });
  if (!res.ok) throw new Error("Search failed");
  return res.json();
}

export function rerankCandidates({ aid, query, candidates }, onBatch, onComplete, onError) {
  const url = `${API_BASE}/api/rerank`;
  fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ aid, query, candidates }),
  }).then((res) => {
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
    if (onError) onError({ error: err.message });
  });
}

export async function chatMessage({ aid, userInput, conversationHistory, currentQuery, pastQueries, priorInputs, searchResults, searchPhase }) {
  const res = await fetch(`${API_BASE}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      aid,
      user_input: userInput,
      conversation_history: conversationHistory || [],
      current_query: currentQuery || null,
      past_queries: pastQueries || [],
      prior_inputs: priorInputs || [],
      search_results: searchResults || [],
      search_phase: searchPhase || null,
    }),
  });
  if (!res.ok) throw new Error("Chat request failed");
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
