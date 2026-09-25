import { useEffect, useRef, useState } from "react";
import { CHAT_WINDOW } from "../lib/matrixUi";
import { searchPeopleByName } from "../lib/api";
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
  intent = "collaborator",
  onAddContextPerson,
  onSavePerson,
  onPrepareContextAction,
  onUseStarter,
  resultsWorkspace = null,
  notice = "",
  onDismissNotice,
}) {
  const chatEndRef = useRef(null);
  const inputRef = useRef(null);
  const fileInputRef = useRef(null);
  const scrollerRef = useRef(null);
  const peoplePickerRef = useRef(null);
  const peoplePickerButtonRef = useRef(null);
  const pinnedRef = useRef(true);
  const [showEarlier, setShowEarlier] = useState(false);
  const [peoplePickerOpen, setPeoplePickerOpen] = useState(false);
  const [personSearch, setPersonSearch] = useState("");
  const [personMatches, setPersonMatches] = useState([]);
  const [personSearchState, setPersonSearchState] = useState("idle");
  const [peoplePickerStyle, setPeoplePickerStyle] = useState({});
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
  useEffect(() => {
    const query = personSearch.trim();
    if (!peoplePickerOpen || query.length < 2) {
      setPersonMatches([]);
      setPersonSearchState("idle");
      return undefined;
    }
    const controller = new AbortController();
    const timer = window.setTimeout(async () => {
      setPersonSearchState("loading");
      try {
        const matches = await searchPeopleByName(query, controller.signal);
        if (!controller.signal.aborted) {
          setPersonMatches(matches);
          setPersonSearchState("ready");
        }
      } catch {
        if (!controller.signal.aborted) setPersonSearchState("error");
      }
    }, 180);
    return () => {
      window.clearTimeout(timer);
      controller.abort();
    };
  }, [peoplePickerOpen, personSearch]);

  const closePeoplePicker = () => {
    setPeoplePickerOpen(false);
    setPersonSearch("");
    setPersonMatches([]);
    setPersonSearchState("idle");
  };

  useEffect(() => {
    if (!peoplePickerOpen) return undefined;

    const positionPicker = () => {
      const trigger = peoplePickerButtonRef.current;
      if (!trigger) return;
      const rect = trigger.getBoundingClientRect();
      const inset = 12;
      const width = Math.min(320, Math.max(220, window.innerWidth - inset * 2));
      const left = Math.min(Math.max(inset, rect.left), Math.max(inset, window.innerWidth - width - inset));
      const spaceAbove = rect.top - inset;
      const spaceBelow = window.innerHeight - rect.bottom - inset;
      const openAbove = spaceAbove > spaceBelow;
      const availableHeight = Math.max(96, Math.floor(openAbove ? spaceAbove : spaceBelow));

      setPeoplePickerStyle(openAbove
        ? { bottom: `${Math.max(inset, window.innerHeight - rect.top + 7)}px`, left: `${left}px`, width: `${width}px`, maxHeight: `${availableHeight}px` }
        : { top: `${Math.max(inset, rect.bottom + 7)}px`, left: `${left}px`, width: `${width}px`, maxHeight: `${availableHeight}px` }
      );
    };
    const closeOnPointerDown = (event) => {
      if (peoplePickerRef.current?.contains(event.target) || peoplePickerButtonRef.current?.contains(event.target)) return;
      closePeoplePicker();
    };
    const closeOnEscape = (event) => {
      if (event.key === "Escape") closePeoplePicker();
    };

    positionPicker();
    window.addEventListener("resize", positionPicker);
    document.addEventListener("pointerdown", closeOnPointerDown);
    document.addEventListener("keydown", closeOnEscape);
    return () => {
      window.removeEventListener("resize", positionPicker);
      document.removeEventListener("pointerdown", closeOnPointerDown);
      document.removeEventListener("keydown", closeOnEscape);
    };
  }, [peoplePickerOpen]);

  const personId = (person) => Number(person?.authorId ?? person?.author_id);
  const selectPickerPerson = (person) => {
    const authorId = personId(person);
    if (!Number.isInteger(authorId)) return;
    const included = contextPersonSet.has(authorId);
    if (included) {
      onToggleContextPerson?.(authorId);
    } else {
      onAddContextPerson?.({ ...person, authorId });
    }
    closePeoplePicker();
  };
  const savePickerPerson = (person) => {
    const authorId = personId(person);
    if (!Number.isInteger(authorId)) return;
    onSavePerson?.({ ...person, author_id: String(authorId) });
  };
  const prepareContextAction = (action) => {
    onPrepareContextAction?.(action);
    requestAnimationFrame(() => inputRef.current?.focus());
  };
  const contextPicker = (
    <div className="context-picker-wrap">
      {selectedPeople.length > 0 && (
        <div className="selected-context-block" aria-label="People added to this chat">
          <span className="selected-context-label">{intent === "collaborator" ? "People for this research" : "In this chat"}</span>
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
      {intent === "collaborator" && selectedPeople.length > 0 && (
        <div className="context-next-step" aria-label="Choose what to do with selected people">
          <span>Next step</span>
          <button type="button" onClick={() => prepareContextAction("assess")}>Assess fit</button>
          <button type="button" onClick={() => prepareContextAction("find")}>Find someone to add</button>
        </div>
      )}
      {peoplePickerOpen && (
        <div ref={peoplePickerRef} className="people-picker" style={peoplePickerStyle} role="dialog" aria-label="Add people to this chat">
          <div className="people-picker-heading">Add people</div>
          {savedPeople.length > 0 && (
            <>
              <div className="people-picker-section-label">Saved people</div>
              <div className="people-picker-list people-picker-saved-list">
                {savedPeople.slice(0, 5).map((person) => {
                  const authorId = personId(person);
                  const included = contextPersonSet.has(authorId);
                  return (
                    <div key={authorId} className="people-picker-option">
                      <span className="people-picker-person">
                        <strong>{person.name}</strong>
                        {person.affiliation && <small>{person.affiliation}</small>}
                      </span>
                      <button
                        type="button"
                        className="people-picker-action"
                        onClick={() => selectPickerPerson(person)}
                        aria-pressed={included}
                      >
                        {included ? "Added" : "Add"}
                      </button>
                    </div>
                  );
                })}
              </div>
              {savedPeople.length > 5 && (
                <p className="people-picker-more">{savedPeople.length - 5} more saved people. Search by name to find one.</p>
              )}
            </>
          )}
          <label className="people-picker-search-label" htmlFor="people-picker-search">
            {savedPeople.length > 0 ? "Find someone else" : "Search by name"}
          </label>
          <input
            id="people-picker-search"
            className="people-picker-search"
            type="search"
            value={personSearch}
            onChange={(event) => setPersonSearch(event.target.value)}
            placeholder="Search by name"
            autoComplete="off"
            autoFocus
          />
          {personSearch.trim().length < 2 ? (
            <p className="people-picker-status">Start typing a name to search the research catalog.</p>
          ) : personSearchState === "loading" ? (
            <p className="people-picker-status">Searching people...</p>
          ) : personSearchState === "error" ? (
            <p className="people-picker-status">Search is unavailable. Try again.</p>
          ) : personMatches.length === 0 ? (
            <p className="people-picker-status">No matching people found.</p>
          ) : (
            <div className="people-picker-list">
              {personMatches.map((person) => {
                const authorId = personId(person);
                const included = contextPersonSet.has(authorId);
                const saved = savedPeople.some((savedPerson) => Number(savedPerson.authorId) === authorId);
                return (
                  <div key={authorId} className="people-picker-option">
                    <span className="people-picker-person">
                      <strong>{person.name}</strong>
                      {person.affiliation && <small>{person.affiliation}</small>}
                    </span>
                    <span className="people-picker-actions">
                      <button
                        type="button"
                        className="people-picker-action"
                        onClick={() => selectPickerPerson(person)}
                        aria-pressed={included}
                      >
                        {included ? "Added" : "Add"}
                      </button>
                      {!saved && (
                        <button
                          type="button"
                          className="people-picker-save"
                          onClick={() => savePickerPerson(person)}
                        >
                          Save
                        </button>
                      )}
                    </span>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      )}
    </div>
  );

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
        <button
          ref={peoplePickerButtonRef}
          className={`chat-context-btn ${selectedPeople.length > 0 ? "has-people" : ""}`}
          type="button"
          aria-expanded={peoplePickerOpen}
          aria-label="Add people to this chat"
          onClick={() => peoplePickerOpen ? closePeoplePicker() : setPeoplePickerOpen(true)}
        >
          <IconTeam />
          <span>{selectedPeople.length ? `${selectedPeople.length} people` : "Add people"}</span>
        </button>
        <input
          ref={fileInputRef}
          type="file"
          className="sr-only"
          multiple
          accept=".pdf,.docx,.txt,.md,.markdown,.tex"
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
          title="Attach a paper, draft, CV or abstract (PDF, Word, .txt, .md, .tex)"
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
            <p className="starter-heading">Try an example</p>
            <div className="starter-grid">
              {starters.map((starter) => (
                <button
                  key={`${starter.intent}-${starter.label}`}
                  type="button"
                  className="starter-card"
                  onClick={() => {
                    if (onUseStarter) {
                      onUseStarter(starter);
                      return;
                    }
                    setInputValue(starter.prompt);
                    requestAnimationFrame(() => inputRef.current?.focus());
                  }}
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
                  {searchPlan.intent === "mentor" ? "Mentor search" : "Collaborator search"}
                  {searchPlan.contextCount > 0 ? ` · ${searchPlan.contextCount} ${searchPlan.contextCount === 1 ? "person" : "people"} selected` : ""}
                </span>
              </div>
              <div className="search-plan-actions">
                <button type="button" className="search-plan-edit" onClick={() => {
                  setInputValue(searchPlan.question || searchPlan.query);
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
