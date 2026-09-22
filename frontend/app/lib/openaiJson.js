const DEFAULT_MODEL = "gpt-4.1-mini";

export function paperTitles(papers) {
  return (Array.isArray(papers) ? papers : [])
    .map((paper) => {
      if (!paper) return "";
      if (typeof paper === "string") return paper.trim();
      return String(paper.Title || paper.title || "").trim();
    })
    .filter(Boolean)
    .slice(0, 3);
}

function parseJsonContent(text) {
  const jsonText = String(text || "").replace(/^```json\s*|\s*```$/g, "").trim();
  return JSON.parse(jsonText);
}

export async function completeJson({ system, user, maxTokens = 700, signal }) {
  const key = String(process.env.OPENAI_API_KEY || "").trim();
  if (!key) {
    throw new Error("LLM is not configured");
  }
  const model = String(process.env.MATRIX_WHY_MODEL || DEFAULT_MODEL).trim() || DEFAULT_MODEL;
  const messages = [
    { role: "system", content: system },
    { role: "user", content: user },
  ];
  const attempts = [
    { model, max_tokens: maxTokens, response_format: { type: "json_object" }, messages },
    { model, max_completion_tokens: maxTokens, response_format: { type: "json_object" }, messages },
    { model, max_tokens: maxTokens, messages },
  ];
  let lastError = "LLM request failed";
  for (const body of attempts) {
    const response = await fetch("https://api.openai.com/v1/chat/completions", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${key}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify(body),
      signal,
      cache: "no-store",
    });
    if (response.status === 400) {
      lastError = "LLM request failed (400)";
      continue;
    }
    if (!response.ok) {
      throw new Error(`LLM request failed (${response.status})`);
    }
    const payload = await response.json();
    return parseJsonContent(payload?.choices?.[0]?.message?.content);
  }
  throw new Error(lastError);
}
