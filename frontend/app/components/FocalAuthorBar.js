import { IconBack, IconSidebar } from "./icons";

export default function FocalAuthorBar({
  authorInfo,
  seekerName,
  profileContextEnabled,
  sessionStatus,
  matrixUserToken,
  showSidebarToggle,
  onOpenFocal,
  onToggleSidebar,
  onReturnToGraph,
}) {
  const name = authorInfo?.name || seekerName || "";
  const saving = sessionStatus.saving;
  const statusLabel = sessionStatus.error
    ? "Chat not saved"
    : !matrixUserToken
      ? "Sign in on the graph to save chats"
      : saving
        ? "Saving"
        : "";

  return (
    <header className="focal-bar">
      {showSidebarToggle && (
        <button className="icon-btn" type="button" onClick={onToggleSidebar} aria-label="Open sidebar">
          <IconSidebar />
        </button>
      )}

      {name && (
        profileContextEnabled ? (
          <button type="button" className="context-button" onClick={onOpenFocal}>
            <span>Research context</span>
            <strong>{name}</strong>
          </button>
        ) : (
          <div className="context-label">
            <span>Working with</span>
            <strong>{name}</strong>
          </div>
        )
      )}

      <div className="focal-actions">
        {statusLabel && (
          <span className={`save-status ${sessionStatus.error ? "is-error" : ""}`} title={sessionStatus.error || undefined}>
            {statusLabel}
          </span>
        )}
        {onReturnToGraph && (
          <button className="ghost-btn focal-return" type="button" onClick={onReturnToGraph}>
            <IconBack />
            Back to graph
          </button>
        )}
      </div>
    </header>
  );
}
