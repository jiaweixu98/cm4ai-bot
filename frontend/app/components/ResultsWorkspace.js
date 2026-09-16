import CandidateCard from "./CandidateCard";
import { IconClose } from "./icons";
import { initials, stageForPhase } from "../lib/matrixUi";

export default function ResultsWorkspace({
  copy,
  candidates,
  orderedCandidates,
  rerankedMap,
  phase,
  currentQuery,
  savedPeople,
  savedIds,
  teamBuilding,
  teamMemberIds,
  profileNotice,
  onDismissNotice,
  onOpenProfile,
  onSave,
  onUnsave,
  onToggleTeam,
  onRetryNotes,
  rerankError,
  selectionDisabled,
}) {
  const stage = stageForPhase(phase);
  const context = Object.values(rerankedMap).find((note) => note.source === 'llm');
  const contextText = candidates[0]?.search_basis === 'representative_papers'
    ? 'Using papers from your profile.'
    : context?.context_basis === 'need_profile_team'
    ? `Using your papers and ${context.team_count} team members.`
    : context?.context_basis === 'need_profile' ? 'Using papers from your profile.'
    : context?.context_basis === 'need_team' ? `Using ${context.team_count} team ${context.team_count === 1 ? 'member' : 'members'}.`
    : '';
  const teamNames = savedPeople
    .filter((person) => teamMemberIds.has(Number(person.authorId)))
    .map((person) => person.name);

  return (
    <div className="panel panel-right results-pane">
      {savedPeople.length > 0 && (
        <div className="saved-strip">
          <div className="saved-strip-label">
            Saved people <span className="count-badge">{savedPeople.length}</span>
          </div>
          <div className="saved-strip-list">
            {savedPeople.map((person) => (
              <span
                key={person.authorId}
                className={`saved-chip-wrap ${teamMemberIds.has(Number(person.authorId)) ? "is-team-member" : ""}`}
              >
                <button type="button" className="saved-chip" onClick={() => onOpenProfile(person.authorId)}>
                  <span className="saved-avatar">{initials(person.name)}</span>
                  {person.name}
                </button>
                  <button
                    type="button"
                    className="saved-team-toggle"
                    aria-pressed={teamMemberIds.has(Number(person.authorId))}
                    onClick={() => onToggleTeam(person.authorId)}
                    disabled={selectionDisabled}
                    aria-label={`${teamMemberIds.has(Number(person.authorId)) ? 'Exclude' : 'Include'} ${person.name} ${teamBuilding ? 'in team context' : 'in this chat'}`}
                  >
                    {teamMemberIds.has(Number(person.authorId)) ? 'Included ✓' : 'Include +'}
                  </button>
                  <button
                    type="button"
                    className="saved-remove"
                    onClick={() => onUnsave(person.authorId)}
                    aria-label={`Remove ${person.name} from saved people`}
                    title="Remove from saved people"
                  >
                    ×
                  </button>
              </span>
            ))}
          </div>
          {teamMemberIds.size > 0 && (
            <div className="team-context-note" role="status">
              {teamMemberIds.size} included
            </div>
          )}
        </div>
      )}

      {profileNotice && (
        <div className="toast" role="status">
          <span>{profileNotice}</span>
          <button type="button" className="icon-btn" onClick={onDismissNotice} aria-label="Dismiss">
            <IconClose />
          </button>
        </div>
      )}

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
            <h2>{copy.resultsTitle} <span className="count-badge">{candidates.length}</span></h2>
            {currentQuery && <div className="query-text">{currentQuery}</div>}
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
    </div>
  );
}
