// In production, Vercel rewrites proxy /api/* to the backend, so we use "" (relative path).
// In local dev, we call the backend directly at localhost:8000.
const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

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
