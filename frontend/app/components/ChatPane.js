import { useEffect, useRef, useState } from "react";
import { CHAT_WINDOW } from "../lib/matrixUi";
import { StarterIcon, IconClose, IconPaperclip, IconSend, IconStop, IconTeam } from "./icons";
import MessageBubble from "./MessageBubble";

export default function ChatPane({
  starters,
  messages,
  phase,
  isLoading,
  canStop,
  inputValue,
  setInputValue,
  suggestedPrompts,
  draftPrompts = [],
  searchPlan = null,
  onConfirmSearchPlan,
  onClearSearchContext,
  queuedFollowUp,
  onSend,
  onStop,
  onRetry,
  onReport,
  citations,
  attachedPapers = [],
  onAttachFiles,
  onRemoveAttachment,
  savedPeople = [],
  contextPeople = [],
  contextPersonIds = [],
  onToggleContextPerson,
  resultsWorkspace = null,
  notice = "",
  onDismissNotice,
}) {
  const chatEndRef = useRef(null);
  const inputRef = useRef(null);
  const fileInputRef = useRef(null);
  const scrollerRef = useRef(null);
  const pinnedRef = useRef(true);
  const [showEarlier, setShowEarlier] = useState(false);
  const [peoplePickerOpen, setPeoplePickerOpen] = useState(false);
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

  const contextPersonSet = new Set(contextPersonIds.map(Number));
  const selectedPeople = contextPeople.filter((person) => contextPersonSet.has(Number(person.authorId)));
  const toggleContextFromPicker = (authorId) => {
    onToggleContextPerson?.(authorId);
    setPeoplePickerOpen(false);
  };
  const contextPicker = (savedPeople.length > 0 || selectedPeople.length > 0) ? (
    <div className="context-picker-wrap">
      {selectedPeople.length > 0 && (
        <div className="selected-context-block" aria-label="People added to this chat">
          <span className="selected-context-label">In this chat</span>
          <div className="selected-context-chips">
          {selectedPeople.map((person) => (
            <button
              key={person.authorId}
              type="button"
              className="selected-context-chip"
              onClick={() => onToggleContextPerson?.(person.authorId)}
              aria-label={`Remove ${person.name} from this chat`}
            >
              <span className={person.source === "graph" ? "context-source-dot" : ""} aria-hidden="true" />
              {person.name} <IconClose />
            </button>
          ))}
          </div>
        </div>
      )}
      {peoplePickerOpen && (
        <div className="people-picker" role="dialog" aria-label="Add saved people to this chat">
          <div className="people-picker-heading">Add saved people</div>
          <div className="people-picker-list">
            {savedPeople.map((person) => {
              const included = contextPersonSet.has(Number(person.authorId));
              return (
                <button
                  key={person.authorId}
                  type="button"
                  className={`people-picker-option ${included ? "is-included" : ""}`}
                  aria-pressed={included}
                  onClick={() => toggleContextFromPicker(person.authorId)}
                >
                  <span>{person.name}</span>
                  <span>{included ? "Added" : "Add"}</span>
                </button>
              );
            })}
          </div>
        </div>
      )}
    </div>
  ) : null;

  const composer = (landing = false) => (
    <div className={`chat-input-wrapper ${landing ? "landing-composer" : ""}`}>
      <textarea
        ref={inputRef}
        className="chat-input"
        aria-label="Describe the research help you need"
        placeholder={queueable ? "Queue a follow-up" : landing ? "Work on anything" : "Message MATRIX"}
        value={inputValue}
        onChange={(event) => setInputValue(event.target.value)}
        onKeyDown={handleKeyDown}
        disabled={phase === "generating"}
        rows={landing ? 3 : 2}
        maxLength={2000}
      />
      {inputValue.length > 200 && (
        <div className="chat-input-meta">
          <span className="char-hint">{inputValue.length}/2000</span>
        </div>
      )}
      <div className="chat-action-group">
        {savedPeople.length > 0 && (
          <button
            className={`chat-context-btn ${selectedPeople.length > 0 ? "has-people" : ""}`}
            type="button"
            aria-expanded={peoplePickerOpen}
            aria-label="Add saved people to this chat"
            onClick={() => setPeoplePickerOpen((open) => !open)}
          >
            <IconTeam />
            <span>{selectedPeople.length ? `${selectedPeople.length} people` : "Add people"}</span>
          </button>
        )}
        <input
          ref={fileInputRef}
          type="file"
          className="sr-only"
          multiple
          accept=".txt,.md,.markdown,.tex"
          aria-label="Attach research context files"
          onChange={(event) => {
            if (event.target.files?.length) onAttachFiles?.(event.target.files);
            event.target.value = "";
          }}
        />
        <button
          className="chat-plus-btn chat-attach-btn"
          onClick={() => fileInputRef.current?.click()}
          type="button"
          aria-label="Attach research context files"
          title="Attach .txt, .md, or .tex research context"
        >
          <IconPaperclip />
        </button>
        {canStop && (
          <button className="chat-stop-btn" onClick={onStop} type="button" aria-label="Stop current request">
            <IconStop />
          </button>
        )}
        <button
          className="chat-send-btn"
          onClick={() => onSend()}
          disabled={phase === "generating" || !inputValue.trim()}
          aria-label={queueable ? "Queue follow-up" : "Send message"}
          type="button"
        >
          <IconSend />
        </button>
      </div>
    </div>
  );

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
        <div className="chat-stream">
        {messages.length > CHAT_WINDOW && !showEarlier && (
          <button type="button" className="ghost-btn show-earlier" onClick={() => setShowEarlier(true)}>
            Show earlier messages
          </button>
        )}
        {empty && (
          <div className="chat-empty-state">
            <h1>What should we work on?</h1>
            {contextPicker}
            {composer(true)}
            {draftPrompts.length > 0 && (
              <div className="handoff-prompt-row" aria-label="Questions about the selected researcher">
                <span className="handoff-prompt-label">Try a focused question</span>
                <div className="handoff-prompt-chips">
                  {draftPrompts.map((prompt) => (
                    <button
                      key={prompt}
                      type="button"
                      className="prompt-chip handoff-prompt-chip"
                      onClick={() => setInputValue(prompt)}
                    >
                      {prompt}
                    </button>
                  ))}
                </div>
              </div>
            )}
            {attachedPapers.length > 0 && (
              <div className="attach-chips" aria-label="Attached research context">
                {attachedPapers.map((paper) => (
                  <span key={paper.id} className="attach-chip" title={paper.filename || paper.title}>
                    <span className="attach-chip-title">{paper.title}</span>
                    <button type="button" className="attach-chip-remove" onClick={() => onRemoveAttachment?.(paper.id)} aria-label={`Remove ${paper.title}`}>
                      <IconClose />
                    </button>
                  </span>
                ))}
              </div>
            )}
            <p className="starter-heading">See what MATRIX can do</p>
            <div className="starter-grid">
              {starters.map((starter) => (
                <button
                  key={`${starter.intent}-${starter.label}`}
                  type="button"
                  className="starter-card"
                  onClick={() => onSend(starter.prompt, starter.intent)}
                >
                  <span className="starter-icon">
                    <StarterIcon name={starter.icon} />
                  </span>
                  <span className="starter-copy">
                    <span className="starter-label">{starter.label}</span>
                    <span className="starter-prompt">{starter.prompt}</span>
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
        {resultsWorkspace}
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
      </div>

      {!empty && <div className="chat-input-container">
        {!searchPlan && suggestedPrompts.length > 0 && !empty && (
          <div className="prompt-rail">
            <div className="prompt-carousel" aria-label="Suggested prompts">
              {suggestedPrompts.map((prompt) => (
                <button key={prompt} type="button" className="prompt-chip" disabled={phase === "generating"} onClick={() => onSend(prompt)}>
                  {prompt}
                </button>
              ))}
            </div>
          </div>
        )}
        <div className="chat-composer-content">
          {notice && (
            <div className="chat-notice" role="status">
              <span>{notice}</span>
              <button type="button" onClick={onDismissNotice} aria-label="Dismiss notice"><IconClose /></button>
            </div>
          )}
          {queuedFollowUp && <div className="queue-chip">Queued: {queuedFollowUp}</div>}
          {searchPlan && (
            <section className="search-plan" aria-label="Proposed search plan">
              <div className="search-plan-copy">
                <span className="search-plan-label">Search plan</span>
                <strong>{searchPlan.query}</strong>
                <span>
                  {searchPlan.intent === "mentor" ? "Mentor search" : "Team search"}
                  {searchPlan.contextCount > 0 ? ` · ${searchPlan.contextCount} ${searchPlan.contextCount === 1 ? "person" : "people"} in context` : ""}
                </span>
              </div>
              <div className="search-plan-actions">
                <button type="button" className="search-plan-edit" onClick={() => {
                  setInputValue(searchPlan.query);
                  inputRef.current?.focus();
                }}>
                  Edit
                </button>
                {searchPlan.contextCount > 0 && (
                  <button type="button" className="search-plan-edit" onClick={onClearSearchContext}>
                    Clear people
                  </button>
                )}
                <button type="button" className="search-plan-confirm" onClick={onConfirmSearchPlan}>
                  Search
                </button>
              </div>
            </section>
          )}
          {contextPicker}
          {attachedPapers.length > 0 && (
            <div className="attach-chips" aria-label="Attached research context">
              {attachedPapers.map((paper) => (
                <span key={paper.id} className="attach-chip" title={paper.filename || paper.title}>
                  <span className="attach-chip-title">{paper.title}</span>
                  <button type="button" className="attach-chip-remove" onClick={() => onRemoveAttachment?.(paper.id)} aria-label={`Remove ${paper.title}`}>
                    <IconClose />
                  </button>
                </span>
              ))}
            </div>
          )}
          {composer()}
        </div>
      </div>}
    </div>
  );
}
