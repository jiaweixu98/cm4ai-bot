export default function ConfirmPopover({ open, title, body, confirmLabel, onConfirm, onCancel }) {
  if (!open) return null;
  return (
    <div className="confirm-popover" role="alertdialog" aria-modal="true" aria-labelledby="confirm-title">
      <div className="confirm-popover-card">
        <h3 id="confirm-title">{title}</h3>
        <p>{body}</p>
        <div className="confirm-popover-actions">
          <button type="button" className="ghost-btn" onClick={onCancel}>
            Keep going
          </button>
          <button type="button" className="ghost-btn danger" onClick={onConfirm}>
            {confirmLabel}
          </button>
        </div>
      </div>
    </div>
  );
}
