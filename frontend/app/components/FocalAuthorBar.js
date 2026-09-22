import { useEffect, useRef } from "react";
import { IconBack, IconHistory, IconPlus } from "./icons";

export default function FocalAuthorBar({
  intent,
  authorInfo,
  seekerName,
  profileContextEnabled,
  sessionStatus,
  matrixUserToken,
  historyOpen,
  historyMenu,
  isLoading,
  onOpenFocal,
  onNewSession,
  onToggleHistory,
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
      <div className="product-identity">
        <div className="product-mark" aria-hidden="true">M</div>
        <div className="focal-copy">
          <div className="product-name">MATRIX</div>
          <div className="product-description">Research guide from publication evidence</div>
        </div>
      </div>

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
        <button className="ghost-btn focal-new" type="button" onClick={onNewSession} disabled={sessionStatus.loading || isLoading}>
          <IconPlus />
          New chat
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
            History
          </button>
          {historyOpen && historyMenu}
        </div>
      </div>
    </header>
  );
}
