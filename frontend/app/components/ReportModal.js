export default function ReportModal({
  open,
  pageContext,
  feedback,
  status,
  submitting,
  onFeedbackChange,
  onClose,
  onSubmit,
}) {
  if (!open) return null;
  return (
    <div className="report-modal-backdrop" onClick={onClose} role="presentation">
      <div className="report-modal-card" onClick={(event) => event.stopPropagation()} role="dialog" aria-labelledby="report-title">
        <h3 id="report-title">Report a problem</h3>
        <p className="report-modal-desc">
          {pageContext === "author-info"
            ? "Describe the issue you found in this conversation."
            : "Describe the issue you found in a recommendation."}
        </p>
        <textarea
          className="report-textarea"
          rows={5}
          placeholder="What looked wrong?"
          value={feedback}
          onChange={(event) => onFeedbackChange(event.target.value)}
        />
        {status && <div className="report-status">{status}</div>}
        <div className="report-modal-actions">
          <button type="button" onClick={onClose} disabled={submitting}>
            Cancel
          </button>
          <button type="button" onClick={onSubmit} disabled={submitting || !feedback.trim()}>
            {submitting ? "Submitting..." : "Submit"}
          </button>
        </div>
      </div>
    </div>
  );
}
