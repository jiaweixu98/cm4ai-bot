import { useState } from "react";
import { FormattedText } from "../lib/formatMessage";
import { relativeTime } from "../lib/matrixUi";
import { IconCopy, IconFlag, IconRetry } from "./icons";

export default function MessageBubble({
  message,
  citations,
  onCopy,
  onRetry,
  onReport,
}) {
  const full = String(message.content || "");
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(full);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1200);
    } catch {
      onCopy?.(full);
    }
  };

  return (
    <div className={`message message-${message.role} ${message.stopped ? "is-stopped" : ""}`}>
      <div className={`message-avatar message-avatar-${message.role}`} aria-hidden="true">
        {message.role === "assistant" ? "M" : "You"}
      </div>
      <div className="message-stack">
        <div className={`message-bubble message-bubble-${message.role}`}>
          <FormattedText content={full} citations={citations} />
        </div>
        <div className="message-meta">
          {message.at && <time dateTime={new Date(message.at).toISOString()}>{relativeTime(message.at)}</time>}
          <div className="message-actions">
            <button type="button" className="msg-action" onClick={handleCopy} aria-label="Copy message">
              <IconCopy />
              {copied ? "Copied" : "Copy"}
            </button>
            {message.role === "user" && (
              <button type="button" className="msg-action" onClick={() => onRetry(message.content)} aria-label="Retry">
                <IconRetry />
                Retry
              </button>
            )}
            {message.role === "assistant" && (
              <button type="button" className="msg-action" onClick={() => onReport(message.content)} aria-label="Report this reply">
                <IconFlag />
                Report
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
