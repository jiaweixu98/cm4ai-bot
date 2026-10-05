import { useEffect, useState } from "react";
import { fetchAuthorDetails } from "../lib/api";

const PAPER_PREVIEW = 3;
const SCOPE_OPTIONS = [
  { id: "profile", label: "Recent" },
  { id: "papers", label: "All" },
  { id: "chosen", label: "Pick" },
];
const YEAR_OPTIONS = [
  { id: 0, label: "Any time" },
  { id: 5, label: "5 years" },
  { id: 10, label: "10 years" },
];

function shownAffiliation(value) {
  const text = String(value || "").trim();
  if (!text || ["unknown", "affiliation unavailable"].includes(text.toLowerCase())) return "";
  return text;
}

function uniqueCount(value) {
  return new Set((Array.isArray(value) ? value : []).map((item) => String(item || "").trim().toLowerCase()).filter(Boolean)).size;
}

function paperEntries(papers, limit = 24) {
  const rows = [];
  const seen = new Set();
  for (const paper of Array.isArray(papers) ? papers : []) {
    const title = String(paper?.Title || paper?.title || "").trim();
    const key = title.toLowerCase();
    if (!title || title === "Untitled" || seen.has(key)) continue;
    seen.add(key);
    const year = String(paper?.PubYear || paper?.year || "").match(/\d{4}/)?.[0] || "";
    rows.push({ title, year });
    if (rows.length >= limit) break;
  }
  return rows;
}

function recentPapers(papers) {
  const floor = new Date().getFullYear() - 5;
  const recent = papers.filter((paper) => Number(paper.year) >= floor);
  return recent.length ? recent : papers;
}

function sentSummary(details, papers, scope, selectedTitles) {
  const parts = [];
  const topics = uniqueCount(details?.topics);
  const paperCount = scope === "chosen" ? selectedTitles.length : scope === "papers" ? papers.length : recentPapers(papers).slice(0, 8).length;
  if (paperCount) parts.push(`${paperCount} ${scope === "profile" ? "recent " : scope === "chosen" ? "picked " : ""}${paperCount === 1 ? "paper" : "papers"}`);
  if (topics) parts.push(`${topics} ${topics === 1 ? "topic" : "topics"}`);
  return parts.length ? `Sent with questions: ${parts.join(" · ")}` : "";
}

function PaperChoices({ papers, scope, selectedTitles, onToggle }) {
  const [shown, setShown] = useState(PAPER_PREVIEW);
  const pool = scope === "profile" ? recentPapers(papers).slice(0, 8) : scope === "papers" ? papers.slice(0, 8) : papers;
  const visible = pool.slice(0, shown);
  if (!pool.length) return <p className="you-note">No papers are linked.</p>;
  return (
    <div className="you-paper-block">
      <ul className="you-papers">
        {visible.map((paper) => {
          const picked = selectedTitles.some((title) => title.toLowerCase() === paper.title.toLowerCase());
          return (
            <li key={`${paper.title}-${paper.year}`}>
              {scope === "chosen" ? (
                <label className="you-paper-choice">
                  <input type="checkbox" checked={picked} onChange={() => onToggle(paper.title)} />
                  <span>{paper.title}{paper.year ? <small>{paper.year}</small> : null}</span>
                </label>
              ) : <span className="you-paper-line">{paper.title}{paper.year ? <small>{paper.year}</small> : null}</span>}
            </li>
          );
        })}
      </ul>
      {pool.length > visible.length && <button type="button" className="you-more" onClick={() => setShown((count) => count + 6)}>Show more</button>}
    </div>
  );
}

export default function YouCard({ signedIn, linked, aid, authorInfo, seekerName, choices, onChoices, onOpenProfile }) {
  // The record carries the aid it was loaded for, so a previous person never renders.
  const [record, setRecord] = useState(null);
  const [failedAid, setFailedAid] = useState(null);

  useEffect(() => {
    if (!linked || !aid) return undefined;
    const controller = new AbortController();
    let current = true;
    fetchAuthorDetails(aid, controller.signal)
      .then((details) => {
        if (current) setRecord({ aid, details });
      })
      .catch((error) => {
        if (current && error?.name !== "AbortError") setFailedAid(aid);
      });
    return () => {
      current = false;
      controller.abort();
    };
  }, [aid, linked]);

  if (!linked) {
    return signedIn ? null : <p className="you-hint">Sign in to personalize.</p>;
  }

  const fresh = record && String(record.aid) === String(aid) ? record : null;
  const failed = failedAid !== null && String(failedAid) === String(aid) && !fresh;
  const details = fresh?.details || null;
  const papers = paperEntries(details?.papers);
  const authorMatches = authorInfo && (authorInfo.author_id === undefined || String(authorInfo.author_id) === String(aid));
  const info = authorMatches ? authorInfo : null;
  const name = (details?.name && details.name !== "Unknown" ? details.name : info?.name) || seekerName || "";
  const affiliation = shownAffiliation(details?.affiliation) || shownAffiliation(info?.affiliation);
  const summary = fresh ? sentSummary(details, papers, choices.paperScope, choices.paperTitles) : "";

  function setScope(paperScope) {
    if (paperScope === "chosen" && choices.paperTitles.length === 0) {
      onChoices({ ...choices, paperScope, paperTitles: recentPapers(papers).slice(0, 3).map((paper) => paper.title) });
      return;
    }
    onChoices({ ...choices, paperScope });
  }

  function togglePaper(title) {
    const key = title.toLowerCase();
    const has = choices.paperTitles.some((item) => item.toLowerCase() === key);
    const paperTitles = has ? choices.paperTitles.filter((item) => item.toLowerCase() !== key) : [...choices.paperTitles, title].slice(0, 8);
    onChoices({ ...choices, paperScope: "chosen", paperTitles });
  }

  return (
    <section className="you-card" aria-label="You">
      {name && <button type="button" className="you-name" onClick={() => onOpenProfile(aid)}>{name}</button>}
      {affiliation && <p className="you-affiliation">{affiliation}</p>}
      {failed ? <p className="you-note">Profile could not be loaded.</p> : !fresh ? (
        <div className="you-loading" aria-label="Loading profile"><span /><span /></div>
      ) : summary && <p className="you-summary">{summary}</p>}
      <details className="you-edit">
        <summary>Edit</summary>
        <div className="you-edit-body">
          {fresh && (
            <>
              <div className="you-seg" role="group" aria-label="Papers sent with your question">
                {SCOPE_OPTIONS.map((option) => <button key={option.id} type="button" className={choices.paperScope === option.id ? "is-active" : ""} aria-pressed={choices.paperScope === option.id} onClick={() => setScope(option.id)}>{option.label}</button>)}
              </div>
              <PaperChoices key={choices.paperScope} papers={papers} scope={choices.paperScope} selectedTitles={choices.paperTitles} onToggle={togglePaper} />
            </>
          )}
          <label className={`you-switch-row ${!affiliation ? "is-disabled" : ""}`}>
            <span>Same institution</span>
            <input type="checkbox" checked={choices.samePlace} disabled={!affiliation} onChange={(event) => onChoices({ ...choices, samePlace: event.target.checked })} />
            <span className="you-switch" aria-hidden="true" />
          </label>
          <div className="you-seg" role="group" aria-label="Recommendation time range">
            {YEAR_OPTIONS.map((option) => <button key={option.id} type="button" className={choices.recentYears === option.id ? "is-active" : ""} aria-pressed={choices.recentYears === option.id} onClick={() => onChoices({ ...choices, recentYears: option.id })}>{option.label}</button>)}
          </div>
        </div>
      </details>
    </section>
  );
}
