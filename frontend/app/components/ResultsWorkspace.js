import CandidateCard from "./CandidateCard";
import { stageForPhase } from "../lib/matrixUi";

export default function ResultsWorkspace({
  copy,
  candidates,
  orderedCandidates,
  rerankedMap,
  phase,
  currentQuery,
  savedIds,
  teamBuilding,
  teamNames = [],
  onOpenProfile,
  onSave,
  onUnsave,
  onRetryNotes,
  rerankError,
  selectionDisabled,
}) {
  const stage = stageForPhase(phase);
  const context = Object.values(rerankedMap).find((note) => note.source === 'llm');
  const contextText = context?.context_basis === 'need_profile_team'
    ? `Using one-time attached context and ${context.team_count} team members.`
    : context?.context_basis === 'need_profile' ? 'Using one-time attached context.'
    : context?.context_basis === 'need_team' ? `Using ${context.team_count} team ${context.team_count === 1 ? 'member' : 'members'}.`
    : '';
  const resultHeading = teamBuilding ? "Potential collaborators" : "Potential mentors";

  return (
    <section className="results-pane inline-results-pane" aria-live="polite">
      {candidates.length === 0 && (phase === "searching" || phase === "explaining") ? (
        <div className="results-header">
          {currentQuery && <div className="query-text">{currentQuery}</div>}
          <div className="stage-progress">
            <div className="progress-text">{stage.label || "Searching…"}</div>
            <div className="progress-bar-container">
              <div className="progress-bar-fill is-indeterminate" />
            </div>
          </div>
          <div className="skeleton-stack" aria-hidden="true">
            {[0, 1, 2].map((key) => (
              <div key={key} className="skeleton-card" />
            ))}
          </div>
        </div>
      ) : candidates.length > 0 ? (
        <>
          <div className="results-header">
            <h2>{resultHeading} <span className="count-badge">{candidates.length}</span></h2>
            {currentQuery && <div className="query-text">For {currentQuery}</div>}
            {contextText && <p className="results-context">{contextText}</p>}
            {phase === "explaining" && (
              <div className="stage-progress">
                <div className="progress-text">{stage.label}</div>
                <div className="progress-bar-container">
                  <div className="progress-bar-fill is-indeterminate" />
                </div>
              </div>
            )}
            {rerankError && phase !== "explaining" && (
              <div className="stage-progress">
                <div className="progress-text">Notes failed to load.</div>
                <button type="button" className="ghost-btn" onClick={onRetryNotes} disabled={selectionDisabled}>
                  Retry notes
                </button>
              </div>
            )}
          </div>
          <div className="results-list">
            {orderedCandidates.map((candidate, index) => (
              <CandidateCard
                key={candidate.author_id}
                candidate={candidate}
                ranked={rerankedMap[candidate.author_id]}
                pending={phase === "explaining" && !rerankedMap[candidate.author_id]}
                index={index}
                copy={copy}
                intent={teamBuilding ? "collaborator" : "mentor"}
                currentQuery={currentQuery}
                teamNames={teamBuilding ? teamNames : []}
                saved={savedIds.has(Number(candidate.author_id))}
                onOpenProfile={onOpenProfile}
                onSave={onSave}
                onUnsave={onUnsave}
              />
            ))}
          </div>
        </>
      ) : null}
    </section>
  );
}
