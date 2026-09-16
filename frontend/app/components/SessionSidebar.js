import { relativeTime } from "../lib/matrixUi";
import { IconPlus } from "./icons";

export default function SessionSidebar({
  sessions,
  currentSessionId,
  sessionStatus,
  matrixUserToken,
  isLoading,
  onNewSession,
  onSelectSession,
}) {
  const intentLabel = (session) => (
    session?.intent === "mentor" ? "Mentor" : session?.intent === "collaborator" ? "Collaborator" : ""
  );
  return (
    <aside className="session-sidebar" aria-label="Previous chats">
      <div className="session-sidebar-head">
        <div className="session-sidebar-title">Previous chats</div>
        <button
          className="session-new-btn"
          onClick={onNewSession}
          type="button"
          disabled={sessionStatus.loading || isLoading}
        >
          <IconPlus />
          New
        </button>
      </div>
      <div className="session-sidebar-body" role="listbox" aria-label="Saved sessions">
        {sessionStatus.error && <div className="session-status session-status-error">{sessionStatus.error}</div>}
        {sessionStatus.loading ? (
          <div className="session-status">Loading…</div>
        ) : matrixUserToken ? (
          sessions.length > 0 ? (
            <div className="session-list">
              {sessions.map((session) => (
                <button
                  key={session.id}
                  className={`session-chip ${session.id === currentSessionId ? "active" : ""}`}
                  onClick={() => onSelectSession(session.id)}
                  type="button"
                  role="option"
                  aria-selected={session.id === currentSessionId}
                >
                  <span className="session-chip-title">{session.title || "Untitled session"}</span>
                  <span className="session-chip-meta">
                    {intentLabel(session) && (
                      <span className={`session-intent session-intent-${session.intent}`}>{intentLabel(session)}</span>
                    )}
                    <span>{relativeTime(session.last_message_at)}</span>
                  </span>
                </button>
              ))}
            </div>
          ) : (
            <div className="session-empty">No previous chats yet.</div>
          )
        ) : (
          <div className="session-empty">Sign in on the graph to keep chats.</div>
        )}
      </div>
    </aside>
  );
}
