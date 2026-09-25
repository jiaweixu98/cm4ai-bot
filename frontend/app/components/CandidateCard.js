import { memo } from 'react';
import { paperMeta, paperTitle } from '../lib/personaConfig';
import { fallbackNote } from '../lib/matchNotes.mjs';
import { IconBookmark, IconExternal } from './icons';

function Paper({ paper }) {
  const raw = paper?.url || paper?.URL || (paper?.doi ? `https://doi.org/${paper.doi}` : '');
  const url = /^https?:\/\//i.test(raw) ? raw : '';
  return <div className="collab-paper">
    {url ? <a className="paper-title" href={url} target="_blank" rel="noreferrer">{paperTitle(paper)} <IconExternal /></a> : <span className="paper-title">{paperTitle(paper)}</span>}
    <span className="paper-meta">{paperMeta(paper)}</span>
  </div>;
}

function connectionText({ hops, via = [] }) {
  if (hops === 1) return 'Your coauthor';
  const names = via.filter(Boolean);
  if (hops === 2 && names.length) return `Shared coauthor: ${names[0]}`;
  return names.length ? `Coauthor path via ${names.join(' → ')}` : `${hops} coauthor steps from you`;
}

function CandidateCard({ candidate, ranked, index, saved, intent, currentQuery, teamNames, onOpenProfile, onSave, onUnsave, pending }) {
  const papers = (candidate.papers || []).slice(0, 3);
  const note = pending
    ? null
    : ranked?.explanation ? ranked : ranked?.justification
    ? { explanation: ranked.justification, evidence_paper_index: 0, source: 'llm' } : fallbackNote(candidate, { intent, query: currentQuery, teamNames });
  const lead = note && Number.isInteger(note.evidence_paper_index) ? note.evidence_paper_index : 0;
  const remaining = papers.filter((_, i) => i !== lead);
  return <article className="collab-card" aria-label={candidate.name}>
    <div className="collab-card-header">
      <div className="collab-card-rank" title="Search result position">{index + 1}</div>
      <div className="collab-card-info">
        <h3 className="collab-card-name">
          <button type="button" className="collab-card-name-link" title="Open graph profile"
            onClick={() => onOpenProfile(candidate.author_id)}>{candidate.name}</button>
        </h3>
        {candidate.affiliation && <div className="collab-card-affiliation">{candidate.affiliation}</div>}
        {candidate.role && <div className="collab-card-focus">{candidate.role}</div>}
        {(candidate.latest_year || candidate.connection) && <div className="collab-card-facts">
          {candidate.latest_year && <span>Latest paper {candidate.latest_year}</span>}
          {candidate.connection && <span>{connectionText(candidate.connection)}</span>}
        </div>}
        {candidate.is_bridge2ai_member && <span className="collab-card-role">Bridge2AI</span>}
      </div>
      <button type="button" className={`person-save ${saved ? 'is-saved' : ''}`} aria-pressed={saved}
        aria-label={`${saved ? 'Remove from saved people' : 'Save'} ${candidate.name}`}
        title={saved ? 'Remove from saved people' : 'Save this person'}
        onClick={() => (saved ? onUnsave(candidate.author_id) : onSave(candidate, ranked))}>
        <IconBookmark filled={saved} />
        <span>{saved ? 'Saved' : 'Save'}</span>
      </button>
    </div>
    <div className="collab-card-body">
      {pending || !note ? (
        <div className="collab-match-reason">
          <span className="evidence-label">Why this person</span>
          <div className="why-skeleton" role="status" aria-label="Loading explanation" />
        </div>
      ) : (
        <div className="collab-match-reason"><span className="evidence-label">Why this person</span><p>{note.explanation}</p></div>
      )}
      {papers[lead] && <div className="lead-evidence"><span className="evidence-label">Supporting publication</span><Paper paper={papers[lead]} /></div>}
      {remaining.length > 0 && <details className="more-evidence"><summary>More publications <span>{remaining.length}</span></summary>{remaining.map((paper, i) => <Paper key={i} paper={paper} />)}</details>}
    </div>
    <div className="collab-actions"><button type="button" className="card-action primary" onClick={() => onOpenProfile(candidate.author_id)}>View on graph <IconExternal /></button></div>
  </article>;
}
export default memo(CandidateCard);
