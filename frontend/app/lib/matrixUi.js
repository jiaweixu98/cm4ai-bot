export const RESULT_LIMIT = 8;
export const MAX_COMPARE = 3;
export const CHAT_WINDOW = 80;

const GOVERNANCE_RE = /govern|elsi|ethic|privacy|irb|hipaa|consent|steward/i;

export function initials(name) {
  const parts = String(name || "")
    .trim()
    .split(/\s+/)
    .filter(Boolean);
  if (parts.length === 0) return "M";
  if (parts.length === 1) return parts[0].slice(0, 2).toUpperCase();
  return `${parts[0][0]}${parts[parts.length - 1][0]}`.toUpperCase();
}

export function affiliationKey(value) {
  return String(value || "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, " ")
    .trim()
    .split(/\s+/)
    .slice(0, 4)
    .join(" ");
}

export function paperYear(paper) {
  const raw = paper?.PubYear || paper?.year || paper?.Year || "";
  const year = Number(String(raw).match(/\d{4}/)?.[0]);
  return Number.isFinite(year) ? year : 0;
}

export function newestPaperYear(papers) {
  return (Array.isArray(papers) ? papers : []).reduce((max, paper) => Math.max(max, paperYear(paper)), 0);
}

export function matchStrength({ order, retrievalScore, rank, total }) {
  const count = Math.max(Number(total) || 1, 1);
  if (Number(order) > 0) {
    const step = 68 / Math.max(count - 1, 1);
    return Math.round(Math.min(96, Math.max(28, 96 - (Number(order) - 1) * step)));
  }
  const score = Number(retrievalScore);
  if (score > 0 && score <= 1) return Math.round(Math.min(96, Math.max(28, score * 100)));
  if (score > 1 && score <= 10) return Math.round(Math.min(96, Math.max(28, score * 10)));
  const position = Number(rank) || 0;
  return Math.round(Math.min(90, Math.max(32, 88 - position * (56 / Math.max(count - 1, 1)))));
}

export function matchLabel(strength) {
  if (strength >= 78) return "Strong fit";
  if (strength >= 52) return "Good fit";
  return "Possible fit";
}

export function reasonTags({ candidate, ranked, intent, focalAffiliation }) {
  const tags = [];
  const affiliation = ranked?.institution || ranked?.affiliation || candidate.affiliation;
  const hops = ranked?.hops ?? candidate.hops;
  const papers = ranked?.papers || candidate.papers || [];
  const why = String(ranked?.justification || "");
  const titles = papers.map((paper) => `${paper?.Title || paper?.title || ""}`).join(" ");

  if (intent === "collaborator" || (Number(hops) || 0) > 1) {
    tags.push({ id: "outside", label: "Outside network" });
  }
  if (intent === "mentor" || candidate.is_bridge2ai_member) {
    tags.push({ id: "bridge", label: "Bridge2AI" });
  }
  const focalKey = affiliationKey(focalAffiliation);
  const otherKey = affiliationKey(affiliation);
  if (focalKey && otherKey && focalKey !== otherKey) {
    tags.push({ id: "diff-inst", label: "Different institution" });
  }
  if (GOVERNANCE_RE.test(`${why} ${titles}`)) {
    tags.push({ id: "gov", label: "Governance titles" });
  } else if (papers.length > 0) {
    tags.push({ id: "titles", label: "Title overlap" });
  }
  return tags.slice(0, 3);
}

export function hasGovernanceTitles(candidate, ranked) {
  const papers = ranked?.papers || candidate.papers || [];
  const why = String(ranked?.justification || "");
  const titles = papers.map((paper) => `${paper?.Title || paper?.title || ""}`).join(" ");
  return GOVERNANCE_RE.test(`${why} ${titles}`);
}

export function isDifferentInstitution(candidate, ranked, focalAffiliation) {
  const affiliation = ranked?.institution || ranked?.affiliation || candidate.affiliation;
  const focalKey = affiliationKey(focalAffiliation);
  const otherKey = affiliationKey(affiliation);
  return Boolean(focalKey && otherKey && focalKey !== otherKey);
}

export function relativeTime(value) {
  if (!value) return "";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "";
  const delta = Date.now() - date.getTime();
  const minutes = Math.round(delta / 60000);
  if (minutes < 1) return "Just now";
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.round(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.round(hours / 24);
  if (days < 7) return `${days}d ago`;
  return date.toLocaleDateString();
}

export function normalizeMessage(message, index) {
  return {
    id: message?.id || `msg-${index}-${message?.role || "assistant"}`,
    role: message?.role === "user" ? "user" : "assistant",
    content: String(message?.content || ""),
    at: message?.at || null,
    stopped: Boolean(message?.stopped),
    ...(message?.hasResults ? { hasResults: true } : {}),
    ...(Array.isArray(message?.citations) && message.citations.length > 0
      ? { citations: message.citations.slice(0, 40) }
      : {}),
  };
}

export function stageForPhase(phase) {
  if (phase === "generating") return { step: 1, label: "Reading your request…" };
  if (phase === "searching") return { step: 2, label: "Searching…" };
  if (phase === "explaining") return { step: 3, label: "Writing notes…" };
  return { step: 0, label: "" };
}

export function followUpPrompts(candidates, rerankedMap, copy) {
  if (!candidates.length) return copy.promptsIdle;
  const top = candidates[0];
  const rankedTop = rerankedMap[top?.author_id] || {};
  const topName = rankedTop.name || top?.name || "the top person";
  return [...new Set([`Why is ${topName} a strong option for this need?`, ...(copy.promptsResults || [])])].slice(0, 3);
}
