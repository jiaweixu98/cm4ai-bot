import { useEffect, useState } from "react";
import { fetchAuthorDetails } from "../lib/api";

const PAPER_PREVIEW = 3;
const SCOPE_OPTIONS = [
  { id: "profile", label: "Recent" },
  { id: "papers", label: "Papers only" },
  { id: "chosen", label: "Pick" },
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
  const topics = scope === "papers" ? 0 : Math.min(uniqueCount(details?.topics), 6);
  const paperCount = Math.min(8, scope === "chosen" ? selectedTitles.length : scope === "papers" ? papers.length : recentPapers(papers).length);
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

export default function YouCard({ signedIn, linked, aid, authorInfo, seekerName, choices, onChoices, onOpenProfile, people = [], workingContext = {}, attachments = [] }) {
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

  const fresh = record && String(record.aid) === String(aid) ? record : null;
  const failed = failedAid !== null && String(failedAid) === String(aid) && !fresh;
  const details = fresh?.details || null;
  const papers = paperEntries(details?.papers);
  const authorMatches = authorInfo && (authorInfo.author_id === undefined || String(authorInfo.author_id) === String(aid));
  const info = authorMatches ? authorInfo : null;
  const name = (linked && ((details?.name && details.name !== "Unknown" ? details.name : info?.name))) || seekerName || "";
  const affiliation = linked ? shownAffiliation(details?.affiliation) || shownAffiliation(info?.affiliation) : "";
  const summary = fresh ? sentSummary(details, papers, choices.paperScope, choices.paperTitles) : "";
  const topics = [...new Set((Array.isArray(details?.topics) ? details.topics : []).map(String).filter(Boolean))].slice(0, 6);
  const terms = [...new Set((Array.isArray(details?.mesh) ? details.mesh : []).map(String).filter(Boolean))].slice(0, 8);
  const requirements = Array.isArray(workingContext.requirements) ? workingContext.requirements.filter(Boolean) : [];
  const overview = [linked ? "Your profile" : "", people.length ? `${people.length} ${people.length === 1 ? "person" : "people"}` : "", attachments.length ? `${attachments.length} ${attachments.length === 1 ? "file" : "files"}` : ""].filter(Boolean).join(" · ");

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
    <details className="you-card context-disclosure" aria-label="Chat context">
      <summary className="context-summary">
        <span>Context</span>
        {overview && <small>{overview}</small>}
      </summary>
      <div className="you-edit-body">
        {name && (
          <div>
            <h2 className="context-section-label">You</h2>
            {linked ? <button type="button" className="you-name" onClick={() => onOpenProfile(aid)}>{name}</button> : <p className="you-name">{name}</p>}
            {affiliation && <p className="you-affiliation">{affiliation}</p>}
          </div>
        )}
        {linked && (failed ? <p className="you-note">Profile could not be loaded.</p> : !fresh ? (
          <div className="you-loading" aria-label="Loading profile"><span /><span /></div>
        ) : summary && <p className="you-summary">{summary}</p>)}
          {fresh && (
            <>
              <h2 className="context-section-label">Papers</h2>
              <div className="you-seg" role="group" aria-label="Papers sent with your question">
                {SCOPE_OPTIONS.map((option) => <button key={option.id} type="button" className={choices.paperScope === option.id ? "is-active" : ""} aria-pressed={choices.paperScope === option.id} onClick={() => setScope(option.id)}>{option.label}</button>)}
              </div>
              <PaperChoices key={choices.paperScope} papers={papers} scope={choices.paperScope} selectedTitles={choices.paperTitles} onToggle={togglePaper} />
              {choices.paperScope !== "papers" && topics.length > 0 && <div><h2 className="context-section-label">Topics</h2><p className="you-note">{topics.join(" · ")}</p></div>}
              {choices.paperScope !== "papers" && terms.length > 0 && <div><h2 className="context-section-label">Research terms</h2><p className="you-note">{terms.join(" · ")}</p></div>}
            </>
          )}
        {people.length > 0 && <div><h2 className="context-section-label">People in this chat</h2><ul className="context-people">{people.map((person) => <li key={person.authorId}><button type="button" className="you-name" onClick={() => onOpenProfile(person.authorId)}>{person.name}</button>{shownAffiliation(person.affiliation) && <p className="you-affiliation">{shownAffiliation(person.affiliation)}</p>}</li>)}</ul></div>}
        {(workingContext.goal || requirements.length > 0) && <div><h2 className="context-section-label">Current focus</h2>{workingContext.goal && <p className="you-note">{workingContext.goal}</p>}{requirements.length > 0 && <ul className="context-requirements">{requirements.map((item, index) => <li key={index}>{item}</li>)}</ul>}</div>}
        {attachments.length > 0 && <div><h2 className="context-section-label">For the next question</h2><ul className="context-requirements">{attachments.map((file) => <li key={file.id}>{file.title || file.filename}</li>)}</ul></div>}
        {!linked && people.length === 0 && !workingContext.goal && requirements.length === 0 && attachments.length === 0 && <p className="you-note">Add people or a document in the message box, or describe your research in chat.</p>}
        {!signedIn && !linked && <p className="you-note">Sign in on the graph to use your profile.</p>}
      </div>
    </details>
  );
}
