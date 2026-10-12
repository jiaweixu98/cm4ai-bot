import { useEffect, useRef, useState } from "react";
import { fetchAuthorDetails } from "../lib/api";
import { matrixApiPath } from "../lib/apiPath.mjs";

const affiliation = value => ["unknown", "affiliation unavailable"].includes(String(value || "").toLowerCase()) ? "" : value;
const key = paper => `${paper.author_id}:${paper.work_id}`;

export default function YouCard({ linked, aid, authorInfo, seekerName, choices, onChoices, onOpenProfile, people = [], workingContext = {}, attachments = [], onClose }) {
  const [details, setDetails] = useState(null);
  const [livePeople, setLivePeople] = useState([]);
  const [author, setAuthor] = useState(linked ? String(aid) : String(people[0]?.authorId || ""));
  const [query, setQuery] = useState("");
  const [offset, setOffset] = useState(0);
  const [page, setPage] = useState(null);
  const [error, setError] = useState("");
  const panel = useRef(null);
  const selected = choices.selectedPapers || [];
  const publicationFilters = choices.publicationFilters || [];
  const affiliationFilter = publicationFilters.find(f => String(f.author_id) === author)?.affiliation || 'all';
  const selectedKeys = new Set(selected.map(key));
  const pickerAuthor = String(page?.author_id || author);
  const authors = [...(linked ? [{ authorId: String(aid), name: details?.name || authorInfo?.name || seekerName || "Your profile" }] : []), ...people]
    .filter((p, i, all) => all.findIndex(other => String(other.authorId) === String(p.authorId)) === i);
  const authorKeys = authors.map(p => String(p.authorId)).join(',');
  useEffect(() => {
    if (!authors.some(p => String(p.authorId) === author)) {setAuthor(String(authors[0]?.authorId || ''));setOffset(0);}
  }, [authorKeys, author]);

  useEffect(() => {
    const previous = document.activeElement;
    panel.current?.querySelector("button")?.focus();
    const close = event => {
      if (event.key === "Escape") onClose();
      if (event.key === "Tab" && window.innerWidth <= 960) {
        const controls = [...panel.current.querySelectorAll('button:not(:disabled),input,select,[tabindex="0"]')];
        const first = controls[0], last = controls[controls.length - 1];
        if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last?.focus(); }
        if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first?.focus(); }
      }
    };
    document.addEventListener("keydown", close);
    return () => { document.removeEventListener("keydown", close); previous?.focus?.(); };
  }, []);

  useEffect(() => {
    if (!linked) return;
    const controller = new AbortController();
    fetchAuthorDetails(aid, controller.signal).then(setDetails).catch(() => setError("Profile could not be loaded."));
    return () => controller.abort();
  }, [aid, linked]);

  const peopleKey = JSON.stringify(people.map(p => String(p.authorId)));
  useEffect(() => {
    const controller = new AbortController();
    setLivePeople([]);
    Promise.all(people.map(async person => {
      try { const live = await fetchAuthorDetails(person.authorId, controller.signal);
        return {...person, name: live.name, affiliation: live.affiliation || ''};
      } catch { return {...person, affiliation: ''}; }
    })).then(rows => {if(!controller.signal.aborted) setLivePeople(rows);});
    return () => controller.abort();
  }, [peopleKey]);

  useEffect(() => {
    if (!author) { setPage(null); return; }
    const controller = new AbortController();
    setPage(null); setError("");
    const timer = setTimeout(async () => {
      try {
        const params = new URLSearchParams({ q: query, offset: String(offset), limit: "20", affiliation: affiliationFilter });
        const response = await fetch(matrixApiPath(`/api/author/${encodeURIComponent(author)}/publications?${params}`), { signal: controller.signal, cache: "no-store" });
        if (!response.ok) throw new Error("Publications could not be loaded.");
        setPage(await response.json());
      } catch (e) { if (e.name !== "AbortError") setError(e.message); }
    }, 180);
    return () => { clearTimeout(timer); controller.abort(); };
  }, [author, query, offset, affiliationFilter]);

  const toggle = paper => {
    const ref = { author_id: pickerAuthor, work_id: paper.work_id, title: paper.title, year: paper.year };
    const next = selectedKeys.has(key(ref)) ? selected.filter(p => key(p) !== key(ref)) : [...selected, ref].slice(0, 8);
    onChoices({ ...choices, version: 3, paperScope: "chosen", publicationScope: next.length ? 'selected' : 'eligible', paperTitles: [], selectedPapers: next });
  };
  return <>
    <button className="context-scrim" type="button" aria-label="Close context panel" onClick={onClose} />
    <aside ref={panel} className="conversation-context" aria-label="Conversation context">
      <div className="context-panel-head"><h2>Conversation context</h2><button type="button" className="ghost-btn" onClick={onClose} aria-label="Close context">×</button></div>
      <div className="you-edit-body">
        {(details?.name || authorInfo?.name || seekerName) && <section><h3 className="context-section-label">Profile</h3>
          <button className="you-name" type="button" onClick={() => onOpenProfile(aid)}>{details?.name || authorInfo?.name || seekerName}</button>
          {affiliation(details?.affiliation) && <p className="you-affiliation">{details.affiliation}</p>}
        </section>}
        {!!details?.topics?.length && <section><h3 className="context-section-label">Topics</h3><p className="you-note">{details.topics.join(" · ")}</p></section>}
        {!!livePeople.length && <section><h3 className="context-section-label">Selected people</h3><ul className="context-people">{livePeople.map(p => <li key={p.authorId}><button className="you-name" onClick={() => onOpenProfile(p.authorId)}>{p.name}</button>{affiliation(p.affiliation) && <p className="you-affiliation">{p.affiliation}</p>}</li>)}</ul></section>}
        <section><h3 className="context-section-label">Selected publications</h3>
          {selected.length ? <ul className="you-papers">{selected.map(p => <li className="you-paper-block" key={key(p)}><span className="you-paper-line">{p.title}{p.year && <small>{p.year}</small>}</span><button type="button" className="you-more" aria-label={`Remove ${p.title}`} onClick={() => {const next=selected.filter(other => key(other) !== key(p));onChoices({ ...choices, selectedPapers:next,publicationScope:next.length?'selected':'eligible' });}}>Remove</button></li>)}</ul> : <p className="you-note">No publications selected.</p>}
          {choices.publicationScope === 'selected' && <button className="you-more" onClick={() => onChoices({...choices,publicationScope:'eligible',selectedPapers:[]})}>Use all eligible publications</button>}
        </section>
        {!!authors.length && <section><h3 className="context-section-label">Choose publications</h3>
          <label className="context-picker-label">Researcher<select aria-label="Publication researcher" value={author} onChange={e => { setAuthor(e.target.value); setOffset(0); }}>{authors.map(p => <option key={p.authorId} value={String(p.authorId)}>{p.name}</option>)}</select></label>
          <label className="context-picker-label">Affiliation<select aria-label="Publication affiliation filter" value={affiliationFilter} onChange={e => {setOffset(0);onChoices({...choices,publicationFilters:[...publicationFilters.filter(f => String(f.author_id)!==author),{author_id:author,affiliation:e.target.value}]});}}><option value="all">All publications</option>{(page?.filter_options || []).map(f => <option key={f.group_id} value={f.group_id}>{f.institution}</option>)}</select></label>
          <label className="context-picker-label">Search publications<input type="search" value={query} onChange={e => { setQuery(e.target.value); setOffset(0); }} /></label>
          <ul className="you-papers">{page?.papers?.map(p => <li key={p.work_id}><label className="you-paper-choice"><input type="checkbox" checked={selectedKeys.has(`${pickerAuthor}:${p.work_id}`)} disabled={selected.length >= 8 && !selectedKeys.has(`${pickerAuthor}:${p.work_id}`)} onChange={() => toggle(p)} /><span>{p.title}{p.year && <small>{p.year}</small>}</span></label></li>)}</ul>
          {page && <div className="context-pagination"><button className="you-more" disabled={!offset} onClick={() => setOffset(Math.max(0, offset - 20))}>Previous</button><span>{selected.length}/8 selected</span><button className="you-more" disabled={offset + 20 >= page.total} onClick={() => setOffset(offset + 20)}>Next</button></div>}
        </section>}
        {(workingContext.goal || workingContext.requirements?.length > 0) && <section><h3 className="context-section-label">Current focus</h3><p className="you-note">{workingContext.goal}</p><ul className="context-requirements">{workingContext.requirements?.map((r, i) => <li key={i}>{r}</li>)}</ul></section>}
        {!!attachments.length && <section><h3 className="context-section-label">Pending attachments</h3><ul className="context-requirements">{attachments.map(p => <li key={p.id}>{p.title || p.filename}</li>)}</ul></section>}
        {error && <p role="alert" className="you-note">{error}</p>}
      </div>
    </aside>
  </>;
}
