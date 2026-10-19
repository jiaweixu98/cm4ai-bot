import { IconPlus, IconSidebar } from "./icons";
import { relativeTime } from "../lib/matrixUi";

function groupSessions(sessions) {
  const groups = [
    { id: "today", label: "Today", items: [] },
    { id: "week", label: "Previous 7 days", items: [] },
    { id: "older", label: "Older", items: [] },
  ];
  const start = new Date();
  start.setHours(0, 0, 0, 0);
  const week = new Date(start);
  week.setDate(week.getDate() - 7);
  for (const session of sessions) {
    const date = new Date(session.last_message_at);
    const index = Number.isNaN(date.getTime()) || date < week ? 2 : date >= start ? 0 : 1;
    groups[index].items.push(session);
  }
  return groups.filter((group) => group.items.length > 0);
}

export default function HistorySidebar({
  sessions,
  currentSessionId,
  sessionStatus,
  signedIn,
  isLoading,
  expanded,
  overlay,
  onToggle,
  onNewSession,
  onSelectSession,
  onRemoveSession,
  onRetry,
  you,
}) {
  const groups = groupSessions(sessions);
  const className = ["history-rail", expanded ? "is-open" : "is-collapsed", overlay ? "is-overlay" : ""]
    .filter(Boolean)
    .join(" ");

  return (
    <aside className={className} aria-label="Chat history">
      <div className="history-rail-head">
        <button
          className="icon-btn sidebar-toggle"
          type="button"
          onClick={onToggle}
          aria-expanded={expanded}
          aria-label={expanded ? "Close sidebar" : "Open sidebar"}
        >
          <IconSidebar />
        </button>
        {expanded && (
          <div className="history-brand" aria-label="Ask TalentKG chats">
            <div className="product-mark" aria-hidden="true">KG</div>
            <span className="product-name">Ask TalentKG</span>
          </div>
        )}
      </div>
      <button
        className="history-new"
        type="button"
        onClick={onNewSession}
        disabled={sessionStatus.loading || isLoading}
        aria-label="New chat"
        title="New chat"
      >
        <IconPlus />
        {expanded && <span>New chat</span>}
      </button>
      {expanded && (
        <div className="history-sessions">
          {sessionStatus.error && (
            <div className="session-status session-status-error">
              {sessionStatus.error}
              <button type="button" className="you-more" disabled={sessionStatus.saving || sessionStatus.loading} onClick={onRetry}>Try again</button>
            </div>
          )}
          {!signedIn ? (
            <div className="session-empty">Sign in on the graph to keep chats.</div>
          ) : sessions.length === 0 ? (
            <div className="session-empty">No previous chats yet.</div>
          ) : (
            groups.map((group) => (
              <section key={group.id} className="history-group" aria-label={group.label}>
                <h2 className="history-group-label">{group.label}</h2>
                {group.items.map((session) => (
                  <div className="history-session-row" key={session.id}>
                  <button
                    className={`session-chip ${session.id === currentSessionId ? "active" : ""}`}
                    onClick={() => onSelectSession(session.id)}
                    type="button"
                  >
                    <span className="session-chip-title">{session.title || "Untitled chat"}</span>
                    <span className="session-chip-time">{relativeTime(session.last_message_at)}</span>
                  </button>
                  <button className="session-remove" type="button" aria-label={`Remove chat ${session.title || "Untitled chat"}`} title="Remove from history" disabled={sessionStatus.loading || sessionStatus.saving} onClick={() => onRemoveSession(session.id)}>×</button>
                  </div>
                ))}
              </section>
            ))
          )}
        </div>
      )}
    </aside>
  );
}
