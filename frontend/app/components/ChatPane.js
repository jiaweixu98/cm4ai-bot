import { useEffect, useRef, useState } from "react";
import { CHAT_WINDOW } from "../lib/matrixUi";
import { StarterIcon, IconSend, IconStop } from "./icons";
import MessageBubble from "./MessageBubble";

export default function ChatPane({
  copy,
  authorName,
  messages,
  phase,
  isLoading,
  canStop,
  inputValue,
  setInputValue,
  suggestedPrompts,
  promptMeta,
  queuedFollowUp,
  onSend,
  onStop,
  onRetry,
  onReport,
  citations,
  attachedPapers = [],
  onAttachFiles,
  onRemoveAttachment,
}) {
  const chatEndRef = useRef(null);
  const inputRef = useRef(null);
  const fileInputRef = useRef(null);
  const [dragDepth, setDragDepth] = useState(0);
  const scrollerRef = useRef(null);
  const pinnedRef = useRef(true);
  const [showEarlier, setShowEarlier] = useState(false);
  const empty = messages.length === 0 && !isLoading;
  const queueable = phase === "searching" || phase === "explaining";
  const visibleMessages = showEarlier || messages.length <= CHAT_WINDOW ? messages : messages.slice(-CHAT_WINDOW);
  const phaseLabel =
    phase === "generating"
      ? "Reading your request"
      : phase === "searching"
        ? "Searching"
        : phase === "explaining"
          ? "Writing notes"
          : "";

  useEffect(() => {
    if (!pinnedRef.current) return;
    chatEndRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, phase, visibleMessages.length]);

  const onScroll = () => {
    const node = scrollerRef.current;
    if (!node) return;
    pinnedRef.current = node.scrollHeight - node.scrollTop - node.clientHeight < 80;
  };

  const handleDrop = (event) => {
    event.preventDefault();
    setDragDepth(0);
    if (event.dataTransfer?.files?.length) onAttachFiles?.(event.dataTransfer.files);
  };

  const handleKeyDown = (event) => {
    if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
      event.preventDefault();
      onSend();
      return;
    }
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      onSend();
    }
  };

  return (
    <div className="panel panel-left chat-pane">
      <div className="sr-only" aria-live="polite">
        {phaseLabel}
      </div>
      <div
        className="chat-messages"
        ref={scrollerRef}
        onScroll={onScroll}
        role="log"
        aria-live="polite"
        aria-relevant="additions"
      >
        {messages.length > CHAT_WINDOW && !showEarlier && (
          <button type="button" className="ghost-btn show-earlier" onClick={() => setShowEarlier(true)}>
            Show earlier messages
          </button>
        )}
        {empty && (
          <div className="chat-empty-state">
            <h2>{copy.emptyHeadline(authorName)}</h2>
            {copy.emptyBody && <p className="empty-body">{copy.emptyBody}</p>}
            <p className="example-label">{copy.exampleLabel}</p>
            <div className="starter-grid">
              {copy.promptsIdle.map((prompt, index) => (
                <button
                  key={prompt}
                  type="button"
                  className="starter-card"
                  title={prompt}
                  onClick={() => onSend(prompt)}
                >
                  <span className="starter-icon">
                    <StarterIcon name={promptMeta?.[index]?.icon} />
                  </span>
                  <span className="starter-copy">
                    <span className="starter-label">{promptMeta?.[index]?.label || prompt}</span>
                    <span className="starter-prompt">{promptMeta?.[index]?.hint || prompt}</span>
                  </span>
                </button>
              ))}
            </div>
          </div>
        )}
        {visibleMessages.map((msg) => (
          <MessageBubble
            key={msg.id}
            message={msg}
            citations={citations}
            onRetry={onRetry}
            onReport={onReport}
          />
        ))}
        {isLoading && (
          <div className="message message-assistant">
            <div className="message-avatar message-avatar-assistant">M</div>
            <div className="message-bubble message-bubble-assistant">
              <div className="typing-row">
                <div className="typing-indicator" aria-hidden="true">
                  <div className="typing-dot" />
                  <div className="typing-dot" />
                  <div className="typing-dot" />
                </div>
                <span className="typing-label">{phaseLabel || "Working…"}</span>
              </div>
            </div>
          </div>
        )}
        <div ref={chatEndRef} />
      </div>

      <div className="chat-input-container">
        {queuedFollowUp && <div className="queue-chip">Queued: {queuedFollowUp}</div>}
        {suggestedPrompts.length > 0 && !empty && (
          <div className="prompt-carousel" aria-label="Suggested prompts">
            {suggestedPrompts.map((prompt) => (
              <button key={prompt} type="button" className="prompt-chip" disabled={phase === "generating"} onClick={() => onSend(prompt)}>
                {prompt}
              </button>
            ))}
          </div>
        )}
        {attachedPapers.length > 0 && (
          <div className="attach-chips" aria-label="Attached files">
            {attachedPapers.map((paper) => (
              <span key={paper.id} className="attach-chip" title={paper.filename || paper.title}>
                <span className="attach-chip-title">{paper.title}</span>
                <button
                  type="button"
                  className="attach-chip-remove"
                  onClick={() => onRemoveAttachment?.(paper.id)}
                  aria-label={`Remove ${paper.title}`}
                >
                  ×
                </button>
              </span>
            ))}
          </div>
        )}
        <div
          className={`chat-input-wrapper ${dragDepth > 0 ? "is-dragging" : ""}`}
          onDragEnter={(event) => {
            event.preventDefault();
            if (event.dataTransfer?.types?.includes("Files")) setDragDepth((depth) => depth + 1);
          }}
          onDragOver={(event) => event.preventDefault()}
          onDragLeave={() => setDragDepth((depth) => Math.max(0, depth - 1))}
          onDrop={handleDrop}
        >
          {dragDepth > 0 && <div className="drop-overlay">Drop files</div>}
          <textarea
            ref={inputRef}
            className="chat-input"
            placeholder={queueable ? "Queue a follow-up" : copy.placeholder}
            value={inputValue}
            onChange={(event) => setInputValue(event.target.value)}
            onKeyDown={handleKeyDown}
            disabled={phase === "generating"}
            rows={2}
            maxLength={2000}
          />
          {inputValue.length > 200 && (
            <div className="chat-input-meta">
              <span className="char-hint">{inputValue.length}/2000</span>
            </div>
          )}
          <div className="chat-action-group">
            <input
              ref={fileInputRef}
              type="file"
              className="sr-only"
              multiple
              accept=".txt,.md,.markdown,.tex"
              aria-label="Attach files"
              onChange={(event) => {
                if (event.target.files?.length) onAttachFiles?.(event.target.files);
                event.target.value = "";
              }}
            />
            <button
              className="chat-attach-btn"
              onClick={() => fileInputRef.current?.click()}
              type="button"
              aria-label="Attach files"
              title="Attach .txt, .md, or .tex"
            >
              <span aria-hidden="true">📎</span>
              <span className="chat-stop-label">Attach</span>
            </button>
            <button className="chat-stop-btn" onClick={onStop} disabled={!canStop} type="button" aria-label="Stop current run">
              <IconStop />
              <span className="chat-stop-label">Stop</span>
            </button>
            <button
              className="chat-send-btn"
              onClick={() => onSend()}
              disabled={phase === "generating" || !inputValue.trim()}
              aria-label={queueable ? "Queue follow-up" : "Send"}
              type="button"
            >
              <IconSend />
              <span className="chat-send-label">{queueable ? "Queue" : "Send"}</span>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
