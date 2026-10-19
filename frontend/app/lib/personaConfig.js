export const UNLINKED_AID = "unlinked";

export function parsePersonaIntent(value) {
  return value === "mentor" ? "mentor" : "collaborator";
}

export function inferPersonaIntent(value, fallback = "collaborator") {
  const text = String(value || "").toLowerCase();
  const mentorSignals = [
    /\bmentor(?:ship)?\b/,
    /\b(advisor|adviser|supervisor|professor)\b/,
    /\blearn (?:from|with|about)\b/,
    /\bhelp me learn\b/,
    /\b(guidance|coaching|training)\b/,
    /\bwho (?:can|should) i (?:learn from|ask)\b/,
  ];
  const collaboratorSignals = [
    /\b(collaborator|collaboration|co-?investigator|partner)\b/,
    /\b(build|form|strengthen|complete) (?:a |my |our )?team\b/,
    /\b(team member|team[- ]building|research team)\b/,
    /\b(expert|expertise|specialist) (?:in|for)\b/,
    /\b(missing skill|capability|complementary)\b/,
    /\bwho (?:can|could) (?:join|complement|add to)\b/,
  ];
  const mentorScore = mentorSignals.reduce((score, pattern) => score + Number(pattern.test(text)), 0);
  const collaboratorScore = collaboratorSignals.reduce((score, pattern) => score + Number(pattern.test(text)), 0);
  if (mentorScore > collaboratorScore) return "mentor";
  if (collaboratorScore > mentorScore) return "collaborator";
  return fallback === "mentor" ? "mentor" : "collaborator";
}

export const GENERAL_STARTERS = [
  {
    intent: "mentor",
    icon: "mentor",
    label: "Find a mentor",
    prompt: "I'm a wet-lab biologist moving into AI-ready data standards. Who could I learn from?",
  },
  {
    intent: "collaborator",
    icon: "team",
    label: "Build a team",
    prompt: "I'm planning a study to deploy a sepsis early-warning model across hospitals. Who could help with the clinical, machine learning, and implementation work?",
  },
  {
    intent: "collaborator",
    icon: "sparkle",
    label: "Find a niche",
    prompt: "I work on federated learning with EHR data. What promising research directions could I explore?",
  },
];

export const SHOWCASE_STARTERS = [
  { intent: "collaborator", icon: "sparkle", label: "Explore", prompt: "Who are my top 10 recorded coauthors? Which publications do we share?" },
  { intent: "collaborator", icon: "team", label: "Collaborate", prompt: "I'm planning a study combining clinical phenotyping and responsible AI. Which Bridge2AI researchers could I collaborate with?" },
  { intent: "collaborator", icon: "sparkle", label: "Promote", prompt: "I'm proposing a paper on pathway enrichment and gene-expression networks for interpreting disease-associated molecular phenotypes. Who would be a relevant research audience? Please suggest up to four researchers." },
];

export function showcaseStarters(profileName, linked = Boolean(profileName)) {
  return SHOWCASE_STARTERS.map(starter => {
    if (starter.label !== 'Explore') return starter;
    if (!linked) return {...starter, prompt: "Help me choose a researcher and explore their recorded coauthors, with shared publications. I’d like to filter by publication date or current affiliation."};
    if (!profileName) return starter;
    return {...starter, prompt: 'Who are my top 10 recorded coauthors? Which publications do we share?'};
  });
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
      "I want to learn privacy-preserving federated analysis. Who could I learn from?",
      "I'm interested in responsible AI and clinical data governance. Who could mentor me?",
      "I want to learn how to combine clinical records and imaging with AI. Who should I look into?",
    ],
    promptMeta: [
      { icon: "packaging", label: "Federated analysis", hint: "Ask for a mentor on privacy-preserving federated analysis" },
      { icon: "governance", label: "Responsible AI", hint: "Ask for a mentor in responsible AI and clinical data governance" },
      { icon: "grants", label: "Clinical AI", hint: "Ask for a mentor on multimodal clinical AI and imaging" },
    ],
    promptsConfirm: ["Search that", "Make it more specific"],
    promptsResults: [
      "Which of these papers should I read to learn about governance?",
      "Who is at a different institution?",
      "Which paper should I start with for my learning goal?",
    ],
  },
  collaborator: {
    title: "Find collaborators",
    placeholder: "Ask a question",
    resultsTitle: "People",
    emptyHeadline: (name) =>
      name ? `Find the right collaborator for ${name}` : "Find the right collaborator for this work",
    emptyBody: "Ask for a method, population, or skill you need. MATRIX finds researchers whose papers cover that need.",
    exampleLabel: "Example questions",
    promptsIdle: [
      "Who could help me integrate single-cell and spatial omics data?",
      "I'm planning prospective clinical validation across health systems. Who could I collaborate with?",
      "I'm working on clinical NLP and EHR phenotyping. Who could help?",
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
