export const UNLINKED_AID = "unlinked";

export function parsePersonaIntent(value) {
  return value === "mentor" ? "mentor" : "collaborator";
}

export function parseAidParam(value) {
  const aid = String(value || "").trim();
  if (!aid || aid.toLowerCase() === UNLINKED_AID) return UNLINKED_AID;
  return aid;
}

export const PERSONA_COPY = {
  mentor: {
    title: "Find a professor or mentor",
    placeholder: "Ask a question",
    resultsTitle: "People",
    emptyHeadline: (name) =>
      name ? `Find the right professor or mentor for ${name}` : "Find the right professor or mentor",
    emptyBody: "Ask what you want to learn. MATRIX matches that topic to Bridge2AI researchers and their papers.",
    exampleLabel: "Example questions",
    promptsIdle: [
      "Mentor for privacy-preserving federated analysis",
      "Mentor in responsible AI and clinical data governance",
      "Mentor for multimodal clinical AI and imaging",
    ],
    promptMeta: [
      { icon: "packaging", label: "Federated analysis", hint: "Ask for a mentor on privacy-preserving federated analysis" },
      { icon: "governance", label: "Responsible AI", hint: "Ask for a mentor in responsible AI and clinical data governance" },
      { icon: "grants", label: "Clinical AI", hint: "Ask for a mentor on multimodal clinical AI and imaging" },
    ],
    promptsConfirm: ["Search that", "Make it more specific"],
    promptsResults: [
      "Which titles are about governance?",
      "Who is at a different institution?",
      "Summarize the strongest publication match",
    ],
  },
  collaborator: {
    title: "Find collaborators",
    placeholder: "Ask a question",
    resultsTitle: "People",
    emptyHeadline: (name) =>
      name ? `Find the right collaborator for ${name}` : "Find the right collaborator for this work",
    emptyBody: "Ask for a method, population, or skill your team needs. MATRIX finds researchers whose papers cover that gap.",
    exampleLabel: "Example questions",
    promptsIdle: [
      "Single-cell and spatial omics integration",
      "Prospective clinical validation across health systems",
      "Clinical NLP and EHR phenotyping",
    ],
    promptMeta: [
      { icon: "multimodal", label: "Spatial omics", hint: "Ask for help with single-cell and spatial omics integration" },
      { icon: "clinical", label: "Clinical validation", hint: "Ask for prospective clinical validation across health systems" },
      { icon: "ehr", label: "Clinical NLP", hint: "Ask for clinical NLP and EHR phenotyping" },
    ],
    promptsConfirm: ["Search that", "Make it more specific"],
    promptsResults: [
      "Which titles support the top person?",
      "Who looks most complementary?",
      "Compare the first two using their papers",
    ],
  },
};

export function paperTitle(paper) {
  if (typeof paper === 'string') return paper;
  return paper?.Title || paper?.title || "Untitled";
}

export function normalizePaperTitles(value, limit = 10) {
  const rows = Array.isArray(value) ? value : String(value || "").split("\n");
  const seen = new Set();
  const titles = [];
  for (const row of rows) {
    const title = typeof row === "string"
      ? row.trim()
      : String(row?.title || row?.Title || "").trim();
    const key = title.toLowerCase();
    if (!title || seen.has(key)) continue;
    seen.add(key);
    titles.push(title.slice(0, 500));
    if (titles.length >= limit) break;
  }
  return titles;
}

export function paperMeta(paper) {
  const venue = paper?.Venue || paper?.journal || paper?.venue || "";
  const year = paper?.PubYear || paper?.year || "";
  return [venue, year].filter(Boolean).join(", ");
}
