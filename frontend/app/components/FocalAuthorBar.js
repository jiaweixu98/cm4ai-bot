import { useEffect, useRef } from "react";
import { initials } from "../lib/matrixUi";
import { IconClose, IconHistory, IconPlus } from "./icons";

export default function FocalAuthorBar({
  copy,
  intent,
  authorInfo,
  seekerName,
  profileContextEnabled,
  inIframe,
  sessionStatus,
  matrixUserToken,
  historyOpen,
  historyMenu,
  isLoading,
  onIntentChange,
  onOpenFocal,
  onNewSession,
  onToggleHistory,
}) {
  const name =
    (intent === "mentor" ? seekerName : "") ||
    authorInfo?.name ||
    seekerName ||
    "";
  const affiliation = intent === "mentor" ? "" : authorInfo?.affiliation || "";
  const saving = sessionStatus.saving;
  const statusLabel = sessionStatus.error
    ? "Save error"
    : !matrixUserToken
      ? "Not signed in"
      : saving
        ? "Saving…"
        : "Saved";
  const statusTone = sessionStatus.error ? "error" : !matrixUserToken ? "muted" : saving ? "busy" : "ok";
  const historyRef = useRef(null);

  useEffect(() => {
    if (!historyOpen) return undefined;
    const onDown = (event) => {
      if (!historyRef.current?.contains(event.target)) onToggleHistory();
    };
    document.addEventListener("mousedown", onDown);
    return () => document.removeEventListener("mousedown", onDown);
  }, [historyOpen, onToggleHistory]);

  return (
    <header className="focal-bar">
      <div className="focal-identity">
        <div className="focal-avatar" aria-hidden="true">
          {initials(name || "MATRIX")}
        </div>
        <div className="focal-copy">
          <div className="focal-kicker">{copy.title}</div>
          {name && <div className="focal-name">{name}</div>}
          {affiliation && <div className="focal-affiliation">{affiliation}</div>}
        </div>
        {profileContextEnabled && (
          <button type="button" className="ghost-btn" onClick={onOpenFocal}>
            Open in graph
          </button>
        )}
      </div>

      <div className="intent-switch" role="group" aria-label="Start a new conversation in another mode">
        <button
          type="button"
          aria-pressed={intent === "collaborator"}
          disabled={isLoading || sessionStatus.loading}
          title="Start a new collaborator conversation"
          className={intent === "collaborator" ? "active" : ""}
          onClick={() => onIntentChange("collaborator")}
        >
          Collaborator
        </button>
        <button
          type="button"
          aria-pressed={intent === "mentor"}
          disabled={isLoading || sessionStatus.loading}
          title="Start a new mentor conversation"
          className={intent === "mentor" ? "active" : ""}
          onClick={() => onIntentChange("mentor")}
        >
          Mentor
        </button>
      </div>

      <div className="focal-actions">
        {statusTone !== "ok" && (
          <span className={`status-pill status-${statusTone}`} title={sessionStatus.error || undefined}>
            <span className="status-dot" />
            {statusLabel}
          </span>
        )}
        <button className="ghost-btn focal-new" type="button" onClick={onNewSession} disabled={sessionStatus.loading || isLoading}>
          <IconPlus />
          New
        </button>
        <div className="history-wrap" ref={historyRef}>
          <button
            className="ghost-btn"
            type="button"
            onClick={onToggleHistory}
            aria-expanded={historyOpen}
            aria-haspopup="menu"
          >
            <IconHistory />
            Previous
          </button>
          {historyOpen && historyMenu}
        </div>
        {inIframe && historyOpen && (
          <button type="button" className="icon-btn history-dismiss" onClick={onToggleHistory} aria-label="Close history">
            <IconClose />
          </button>
        )}
      </div>
    </header>
  );
}
