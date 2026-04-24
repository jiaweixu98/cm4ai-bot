"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import {
  fetchAuthor,
  chatMessage,
  createChatSession,
  getChatSession,
  listChatSessions,
  searchCandidates,
  rerankCandidates,
  saveChatSession,
  submitErrorReport,
} from "./lib/api";

// ─── State machine phases ───
const PHASE = {
  IDLE: "idle",                 // waiting for first user input
  GENERATING: "generating",     // LLM drafting a query
  AWAITING_CONFIRM: "confirm",  // showing proposed query, waiting for user
  SEARCHING: "searching",       // FAISS retrieval running
  RERANKING: "reranking",       // LLM reranking in progress
  DONE: "done",                 // results ready, can start new query
};

const SESSION_TOKEN_STORAGE_KEY = "matrix_user_token";

export default function Home() {
  // ─── URL params ───
  const [aid, setAid] = useState("6052561");
  const [authorInfo, setAuthorInfo] = useState(null);
  const [matrixUserToken, setMatrixUserToken] = useState("");

  // ─── Chat state ───
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState("");
  const [phase, setPhase] = useState(PHASE.IDLE);
  const [currentSessionId, setCurrentSessionId] = useState(null);
  const [sessions, setSessions] = useState([]);
  const [sessionStatus, setSessionStatus] = useState({ loading: true, saving: false, error: "" });

  // ─── Search state ───
  const [currentQuery, setCurrentQuery] = useState("");
  const [pastQueries, setPastQueries] = useState([]);
  const [priorInputs, setPriorInputs] = useState([]);
  const [candidates, setCandidates] = useState([]);
  const [rerankedMap, setRerankedMap] = useState({});
  const [rerankProgress, setRerankProgress] = useState({ done: 0, total: 0 });
  const [expandedCards, setExpandedCards] = useState({});
  const [reportModalOpen, setReportModalOpen] = useState(false);
  const [reportFeedback, setReportFeedback] = useState("");
  const [reportPageContext, setReportPageContext] = useState("author-info");
  const [reportSubmitting, setReportSubmitting] = useState(false);
  const [reportStatus, setReportStatus] = useState("");

  const chatEndRef = useRef(null);
  const inputRef = useRef(null);
  const sessionHydratingRef = useRef(false);
  const sessionBootstrappedRef = useRef(false);
  const sessionSaveTimerRef = useRef(null);
  const currentSessionIdRef = useRef(null);
  const chatAbortRef = useRef(null);
  const searchAbortRef = useRef(null);
  const rerankAbortRef = useRef(null);

  useEffect(() => {
    currentSessionIdRef.current = currentSessionId;
  }, [currentSessionId]);

  const buildWelcomeMessage = useCallback((data) => {
    const greeting = data?.greeting;
    if (greeting) {
      return greeting;
    }
    return "Welcome! I am a scientific teaming assistant. How can I help you find collaborators?";
  }, []);

  const resetWorkflowState = useCallback((greeting) => {
    setMessages([{ role: "assistant", content: greeting }]);
    setPhase(PHASE.IDLE);
    setCurrentQuery("");
    setPastQueries([]);
    setPriorInputs([]);
    setCandidates([]);
    setRerankedMap({});
    setRerankProgress({ done: 0, total: 0 });
    setExpandedCards({});
  }, []);

  const upsertSessionSummary = useCallback((session) => {
    if (!session) return;
    setSessions((prev) => {
      const next = [session, ...prev.filter((item) => item.id !== session.id)];
      next.sort((a, b) => new Date(b.last_message_at).getTime() - new Date(a.last_message_at).getTime());
      return next;
    });
  }, []);

  const buildSessionStateSnapshot = useCallback(() => ({
    phase,
    currentQuery,
    pastQueries,
    priorInputs,
    candidates,
    rerankedMap,
    rerankProgress,
    expandedCards,
  }), [phase, currentQuery, pastQueries, priorInputs, candidates, rerankedMap, rerankProgress, expandedCards]);

  const applySessionSnapshot = useCallback((session, fallbackGreeting) => {
    const snapshot = session?.state || {};
    const restoredMessages = Array.isArray(session?.messages) && session.messages.length
      ? session.messages
      : [{ role: "assistant", content: fallbackGreeting }];

    sessionHydratingRef.current = true;
    currentSessionIdRef.current = session?.id || null;
    setCurrentSessionId(session?.id || null);
    setMessages(restoredMessages);
    setPhase(Object.values(PHASE).includes(snapshot.phase) ? snapshot.phase : PHASE.IDLE);
    setCurrentQuery(typeof snapshot.currentQuery === "string" ? snapshot.currentQuery : "");
    setPastQueries(Array.isArray(snapshot.pastQueries) ? snapshot.pastQueries : []);
    setPriorInputs(Array.isArray(snapshot.priorInputs) ? snapshot.priorInputs : []);
    setCandidates(Array.isArray(snapshot.candidates) ? snapshot.candidates : []);
    setRerankedMap(snapshot.rerankedMap && typeof snapshot.rerankedMap === "object" ? snapshot.rerankedMap : {});
    setRerankProgress(
      snapshot.rerankProgress && typeof snapshot.rerankProgress === "object"
        ? {
            done: Number(snapshot.rerankProgress.done) || 0,
            total: Number(snapshot.rerankProgress.total) || 0,
          }
        : { done: 0, total: 0 }
    );
    setExpandedCards(snapshot.expandedCards && typeof snapshot.expandedCards === "object" ? snapshot.expandedCards : {});
    setTimeout(() => {
      sessionHydratingRef.current = false;
    }, 0);
  }, []);

  // ─── Init: parse URL and load author ───
  useEffect(() => {
    let cancelled = false;

    const initialize = async () => {
      const params = new URLSearchParams(window.location.search);
      const aidParam = params.get("aid") || "6052561";
      const tokenFromUrl = params.get("mx_user_token") || "";
      const storedToken =
        typeof window !== "undefined" ? window.sessionStorage.getItem(SESSION_TOKEN_STORAGE_KEY) || "" : "";
      const resolvedToken = tokenFromUrl || storedToken;

      if (tokenFromUrl && typeof window !== "undefined") {
        window.sessionStorage.setItem(SESSION_TOKEN_STORAGE_KEY, tokenFromUrl);
        params.delete("mx_user_token");
        const nextSearch = params.toString();
        const nextUrl = `${window.location.pathname}${nextSearch ? `?${nextSearch}` : ""}${window.location.hash}`;
        window.history.replaceState({}, "", nextUrl);
      }

      setAid(aidParam);
      setMatrixUserToken(resolvedToken);

      let fetchedAuthor = null;
      try {
        fetchedAuthor = await fetchAuthor(aidParam);
        if (cancelled) return;
        setAuthorInfo(fetchedAuthor);
      } catch {
        if (cancelled) return;
        setAuthorInfo(null);
      }

      const fallbackGreeting = buildWelcomeMessage(fetchedAuthor);

      if (!resolvedToken) {
        if (cancelled) return;
        resetWorkflowState(fallbackGreeting);
        setSessionStatus({ loading: false, saving: false, error: "" });
        sessionBootstrappedRef.current = true;
        return;
      }

      try {
        const listPayload = await listChatSessions({ aid: aidParam, authToken: resolvedToken });
        if (cancelled) return;
        const sessionItems = Array.isArray(listPayload.sessions) ? listPayload.sessions : [];
        setSessions(sessionItems);
        if (sessionItems.length > 0) {
          const latestPayload = await getChatSession({
            sessionId: sessionItems[0].id,
            authToken: resolvedToken,
          });
          if (cancelled) return;
          applySessionSnapshot(latestPayload.session, fallbackGreeting);
        } else {
          resetWorkflowState(fallbackGreeting);
        }
        setSessionStatus({ loading: false, saving: false, error: "" });
      } catch (error) {
        if (cancelled) return;
        console.error("Failed to restore chat sessions:", error);
        resetWorkflowState(fallbackGreeting);
        setSessionStatus({ loading: false, saving: false, error: "Session history is temporarily unavailable." });
      } finally {
        sessionBootstrappedRef.current = true;
      }
    };

    initialize();

    return () => {
      cancelled = true;
      if (sessionSaveTimerRef.current) {
        window.clearTimeout(sessionSaveTimerRef.current);
      }
    };
  }, [applySessionSnapshot, buildWelcomeMessage, resetWorkflowState]);

  // ─── Auto-scroll & auto-focus ───
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, phase]);

  useEffect(() => {
    if (phase !== PHASE.GENERATING && phase !== PHASE.SEARCHING) {
      inputRef.current?.focus();
    }
  }, [phase, messages]);

  // ─── Add message helper ───
  const addMessage = useCallback((role, content) => {
    setMessages((prev) => [...prev, { role, content }]);
  }, []);

  const ensureCurrentSession = useCallback(async () => {
    if (!matrixUserToken) {
      return null;
    }
    if (currentSessionIdRef.current) {
      return currentSessionIdRef.current;
    }

    const sessionPayload = await createChatSession({
      aid,
      focalAuthorName: authorInfo?.name || "",
      messages: Array.isArray(messages) ? messages : [],
      state: buildSessionStateSnapshot(),
      authToken: matrixUserToken,
    });
    const created = sessionPayload.session;
    currentSessionIdRef.current = created.id;
    setCurrentSessionId(created.id);
    upsertSessionSummary(created);
    return created.id;
  }, [aid, authorInfo?.name, buildSessionStateSnapshot, matrixUserToken, messages, upsertSessionSummary]);

  const handleStop = useCallback(() => {
    chatAbortRef.current?.abort();
    searchAbortRef.current?.abort();
    rerankAbortRef.current?.abort();
    chatAbortRef.current = null;
    searchAbortRef.current = null;
    rerankAbortRef.current = null;

    setPhase(candidates.length > 0 ? PHASE.DONE : PHASE.IDLE);
    addMessage("assistant", "Stopped the current run.");
  }, [addMessage, candidates.length]);

  const handleSelectSession = useCallback(async (sessionId) => {
    if (!matrixUserToken || !sessionId) return;
    if (sessionId === currentSessionIdRef.current) return;
    if (phase === PHASE.GENERATING || phase === PHASE.SEARCHING || phase === PHASE.RERANKING) {
      const confirmed = window.confirm("Stop the current run and switch sessions?");
      if (!confirmed) return;
      handleStop();
    }
    try {
      setSessionStatus((prev) => ({ ...prev, loading: true, error: "" }));
      const payload = await getChatSession({ sessionId, authToken: matrixUserToken });
      applySessionSnapshot(payload.session, buildWelcomeMessage(authorInfo));
      setSessionStatus((prev) => ({ ...prev, loading: false }));
    } catch (error) {
      console.error("Failed to load chat session:", error);
      setSessionStatus((prev) => ({
        ...prev,
        loading: false,
        error: "Failed to load that saved session.",
      }));
    }
  }, [applySessionSnapshot, authorInfo, buildWelcomeMessage, handleStop, matrixUserToken, phase]);

  const handleNewSession = useCallback(() => {
    if (phase === PHASE.GENERATING || phase === PHASE.SEARCHING || phase === PHASE.RERANKING) {
      const confirmed = window.confirm("Stop the current run and start a new session?");
      if (!confirmed) return;
      handleStop();
    }
    currentSessionIdRef.current = null;
    setCurrentSessionId(null);
    resetWorkflowState(buildWelcomeMessage(authorInfo));
  }, [authorInfo, buildWelcomeMessage, handleStop, phase, resetWorkflowState]);

  useEffect(() => {
    if (!sessionBootstrappedRef.current || sessionHydratingRef.current) {
      return;
    }
    if (!matrixUserToken || !currentSessionId) {
      return;
    }

    if (sessionSaveTimerRef.current) {
      window.clearTimeout(sessionSaveTimerRef.current);
    }

    sessionSaveTimerRef.current = window.setTimeout(async () => {
      try {
        setSessionStatus((prev) => ({ ...prev, saving: true, error: "" }));
        const payload = await saveChatSession({
          sessionId: currentSessionId,
          aid,
          focalAuthorName: authorInfo?.name || "",
          messages,
          state: buildSessionStateSnapshot(),
          authToken: matrixUserToken,
        });
        upsertSessionSummary(payload.session);
        setSessionStatus((prev) => ({ ...prev, saving: false, error: "" }));
      } catch (error) {
        console.error("Failed to save chat session:", error);
        setSessionStatus((prev) => ({
          ...prev,
          saving: false,
          error: "Failed to save the current session.",
        }));
      }
    }, 800);

    return () => {
      if (sessionSaveTimerRef.current) {
        window.clearTimeout(sessionSaveTimerRef.current);
      }
    };
  }, [
    aid,
    authorInfo?.name,
    buildSessionStateSnapshot,
    currentSessionId,
    matrixUserToken,
    messages,
    upsertSessionSummary,
  ]);

  // ─── Send handler ───
  const handleSend = async () => {
    const text = inputValue.trim();
    if (!text) return;
    try {
      await ensureCurrentSession();
    } catch (sessionError) {
      console.error("Failed to create chat session:", sessionError);
      setSessionStatus((prev) => ({
        ...prev,
        error: "Session persistence is unavailable right now. You can still keep chatting in this tab.",
      }));
    }
    setInputValue("");
    addMessage("user", text);
    setPriorInputs((prev) => [...prev.slice(-199), text]);

    // Route ALL user input through the unified /api/chat endpoint
    const previousPhase = phase;  // capture before we change it
    setPhase(PHASE.GENERATING);
    const chatController = new AbortController();
    chatAbortRef.current = chatController;

    // Build search results summary for LLM context
    const resultsForLLM = candidates
      .map((c) => {
        const r = rerankedMap[c.author_id];
        if (!r) return null;
        return {
          name: r.name || c.name,
          affiliation: r.institution || r.affiliation || c.affiliation,
          score: r.score,
          justification: r.justification || "",
          hops: r.hops,
          mutual_coauthors: r.mutual_coauthors || [],
        };
      })
      .filter(Boolean)
      .sort((a, b) => (b.score || 0) - (a.score || 0))
      .filter((r) => r.score >= 6.5);  // only include relevant results

    try {
      const result = await chatMessage({
        aid,
        userInput: text,
        conversationHistory: [...messages, { role: "user", content: text }],
        currentQuery: currentQuery || null,
        pastQueries,
        priorInputs: [...priorInputs, text],
        searchResults: resultsForLLM,
        searchPhase: previousPhase,
        signal: chatController.signal,
      });
      chatAbortRef.current = null;

      if (result.action === "confirm" && currentQuery) {
        // User confirmed the pending query → run search
        addMessage("assistant", "Query confirmed. Searching for collaborators…");
        setPhase(PHASE.SEARCHING);
        await runSearch();
      } else if (result.action === "search") {
        // LLM generated a new/refined query
        const query = result.query;
        const justification = result.justification || "";
        setCurrentQuery(query);
        setPastQueries((prev) => [...prev, query]);

        // Clear previous results if starting fresh
        if (previousPhase === PHASE.DONE || previousPhase === PHASE.IDLE) {
          setCandidates([]);
          setRerankedMap({});
          setRerankProgress({ done: 0, total: 0 });
          setExpandedCards({});
        }

        let msg = `🔍 **${query}**`;
        if (justification) msg += `\n\n${justification}`;
        addMessage("assistant", msg);
        setPhase(PHASE.AWAITING_CONFIRM);
      } else {
        // Chat response — show reply and stay in previous phase
        const reply = result.reply || "I'm here to help! Feel free to describe your research interests.";
        addMessage("assistant", reply);
        setPhase(previousPhase === PHASE.GENERATING ? PHASE.IDLE : previousPhase);
      }
    } catch (e) {
      chatAbortRef.current = null;
      if (e?.name === "AbortError") {
        setPhase(previousPhase === PHASE.GENERATING ? PHASE.IDLE : previousPhase);
        return;
      }
      addMessage("assistant", `⚠️ Error: ${e.message}`);
      setPhase(previousPhase === PHASE.GENERATING ? PHASE.IDLE : previousPhase);
    }
  };

  // ─── Search ───
  const runSearch = async () => {
    try {
      const searchController = new AbortController();
      searchAbortRef.current = searchController;
      const { candidates: cands } = await searchCandidates({
        aid,
        query: currentQuery,
        signal: searchController.signal,
      });
      searchAbortRef.current = null;
      setCandidates(cands);
      addMessage("assistant", `✅ Found ${cands.length} potential collaborators. Now analyzing for best matches…`);
      // Auto-expand top 3
      const expanded = {};
      cands.slice(0, 3).forEach((c) => { expanded[c.author_id] = true; });
      setExpandedCards(expanded);

      // Start reranking
      setPhase(PHASE.RERANKING);
      runRerank(cands);
    } catch (e) {
      searchAbortRef.current = null;
      if (e?.name === "AbortError") {
        setPhase(candidates.length > 0 ? PHASE.DONE : PHASE.IDLE);
        return;
      }
      addMessage("assistant", `⚠️ Search error: ${e.message}`);
      setPhase(PHASE.IDLE);
    }
  };

  // ─── Rerank (SSE) ───
  const runRerank = (cands) => {
    const rerankController = new AbortController();
    rerankAbortRef.current = rerankController;
    const candidatePayload = cands.map((c) => ({
      author_id: c.author_id,
      retrieval_score: c.retrieval_score,
    }));

    rerankCandidates(
      { aid, query: currentQuery, candidates: candidatePayload },
      // onBatch
      (data) => {
        const batchResults = data.results || [];
        setRerankedMap((prev) => {
          const next = { ...prev };
          for (const r of batchResults) {
            next[r.author_id] = r;
          }
          return next;
        });
        setRerankProgress(data.progress || { done: 0, total: 0 });
      },
      // onComplete
      (data) => {
        const finalResults = data.results || [];
        setRerankedMap((prev) => {
          const next = { ...prev };
          for (const r of finalResults) {
            next[r.author_id] = r;
          }
          return next;
        });
        setRerankProgress((prev) => ({ ...prev, done: prev.total }));
        addMessage(
          "assistant",
          "🏁 Analysis complete! Review the ranked collaborators on the right. Would you like to run another query?"
        );
        rerankAbortRef.current = null;
        // Add a slight delay to allow the React batch update to finish
        // before transitioning the phase which changes the UI layout.
        setTimeout(() => setPhase(PHASE.DONE), 100);
      },
      // onError
      (data) => {
        rerankAbortRef.current = null;
        console.error("Rerank error:", data.error);
      },
      rerankController.signal
    );
  };

  // ─── Key handler ───
  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  // ─── Render helpers ───
  const isLoading = phase === PHASE.GENERATING || phase === PHASE.SEARCHING || phase === PHASE.RERANKING;
  const canStop = phase === PHASE.GENERATING || phase === PHASE.SEARCHING || phase === PHASE.RERANKING;

  const formatMessage = (content) => {
    // Bold: **text**
    return content.split(/(\*\*[^*]+\*\*)/g).map((part, i) => {
      if (part.startsWith("**") && part.endsWith("**")) {
        return <strong key={i}>{part.slice(2, -2)}</strong>;
      }
      return part;
    });
  };

  // ─── Build ordered candidate list for results panel ───
  const orderedCandidates = (() => {
    if (!candidates.length) return [];

    // Split into reranked and pending
    const rerankedList = [];
    const pendingList = [];

    candidates.forEach(c => {
      const r = rerankedMap[c.author_id];
      if (r) {
        // Only include if score >= 65, like the streamlit logic: 
        // if score >= 65: filtered_ordered_ids.append(rid)
        if (r.score >= 6.5) {
          rerankedList.push(c);
        }
      } else {
        pendingList.push(c);
      }
    });

    // Sort reranked by LLM score desc
    rerankedList.sort((a, b) => rerankedMap[b.author_id].score - rerankedMap[a.author_id].score);

    // Sort pending by base retrieval score desc (they are already sorted but just to be safe)
    pendingList.sort((a, b) => b.retrieval_score - a.retrieval_score);

    return [...rerankedList, ...pendingList];
  })();

  const toggleCard = (id) => {
    setExpandedCards((prev) => ({ ...prev, [id]: !prev[id] }));
  };

  const getScoreClass = (score) => {
    if (score >= 8.0) return "score-high";
    if (score >= 6.5) return "score-medium";
    return "score-low";
  };

  const openReportModal = (page) => {
    setReportPageContext(page);
    setReportFeedback("");
    setReportStatus("");
    setReportModalOpen(true);
  };

  const closeReportModal = () => {
    if (reportSubmitting) return;
    setReportModalOpen(false);
  };

  const handleSubmitReport = async () => {
    const feedbackText = reportFeedback.trim();
    if (!feedbackText || reportSubmitting) return;

    setReportSubmitting(true);
    setReportStatus("");
    try {
      await submitErrorReport({
        project: "cm4ai-bot",
        page: reportPageContext,
        reportFolder: "matrix_error",
        feedback: feedbackText,
        currentUrl: window.location.href,
        userAgent: navigator.userAgent,
        context: {
          aid,
          phase,
          current_query: currentQuery || null,
          candidate_count: candidates.length,
          top_candidate_id: orderedCandidates[0]?.author_id || null,
        },
      });
      setReportStatus("Thanks! Your feedback has been recorded.");
      setReportFeedback("");
      setTimeout(() => {
        setReportModalOpen(false);
        setReportStatus("");
      }, 900);
    } catch (e) {
      setReportStatus(`Submission failed: ${e.message}`);
    } finally {
      setReportSubmitting(false);
    }
  };

  return (
    <div className="app-container">
      <aside className="session-sidebar">
        <div className="session-sidebar-head">
          <div className="session-sidebar-title">Saved sessions</div>
          <div className="session-sidebar-subtitle">
            {matrixUserToken
              ? "Stored per signed-in Bridge user for this author."
              : "Open MATRIX from Bridge to enable saved sessions."}
          </div>
          <button
            className="session-new-btn"
            onClick={handleNewSession}
            type="button"
            disabled={sessionStatus.loading || isLoading}
          >
            New session
          </button>
        </div>
        <div className="session-sidebar-body">
          {sessionStatus.error && <div className="session-status session-status-error">{sessionStatus.error}</div>}
          {sessionStatus.loading ? (
            <div className="session-status">Loading saved sessions…</div>
          ) : matrixUserToken ? (
            sessions.length > 0 ? (
              <div className="session-list">
                {sessions.map((session) => (
                  <button
                    key={session.id}
                    className={`session-chip ${session.id === currentSessionId ? "active" : ""}`}
                    onClick={() => handleSelectSession(session.id)}
                    type="button"
                  >
                    <span className="session-chip-title">{session.title || "Untitled session"}</span>
                    <span className="session-chip-meta">
                      {new Date(session.last_message_at).toLocaleDateString()}
                    </span>
                  </button>
                ))}
              </div>
            ) : (
              <div className="session-empty">No saved sessions for this author yet.</div>
            )
          ) : (
            <div className="session-empty">Signed session history is available when MATRIX is opened from the Bridge app.</div>
          )}
          {sessionStatus.saving && currentSessionId && <div className="session-status">Saving session…</div>}
        </div>
      </aside>

      {/* ─── Left: Chat ─── */}
      <div className="panel panel-left">
        <div className="app-title-bar">
          <div className="app-title">🔬 CM4AI Teaming Assistant</div>
          <div className="app-tagline">Tell me your research needs, I'll find the right collaborators for you.</div>
        </div>
        <div className="panel-header">
          <h1>
            {authorInfo
              ? `Hi ${authorInfo.name}!`
              : "Welcome"}
          </h1>
          <button
            className="report-error-btn"
            onClick={() => openReportModal("author-info")}
            type="button"
          >
            Report Error
          </button>
        </div>

        <div className="chat-messages">
          {messages.map((msg, i) => (
            <div key={i} className={`message message-${msg.role}`}>
              <div className={`message-avatar message-avatar-${msg.role}`}>
                {msg.role === "assistant" ? "🤖" : "👤"}
              </div>
              <div className={`message-bubble message-bubble-${msg.role}`}>
                {formatMessage(msg.content)}
              </div>
            </div>
          ))}
          {isLoading && (
            <div className="message message-assistant">
              <div className="message-avatar message-avatar-assistant">🤖</div>
              <div className="message-bubble message-bubble-assistant">
                <div className="typing-indicator">
                  <div className="typing-dot" />
                  <div className="typing-dot" />
                  <div className="typing-dot" />
                </div>
              </div>
            </div>
          )}
          <div ref={chatEndRef} />
        </div>

        <div className="chat-input-container">
          <div className="chat-input-wrapper">
            <textarea
              ref={inputRef}
              className="chat-input"
              placeholder="Describe your research interests and collaboration needs…"
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyDown={handleKeyDown}
              disabled={isLoading}
              rows={1}
              autoFocus
            />
            <div className="chat-action-group">
              <button
                className="chat-stop-btn"
                onClick={handleStop}
                disabled={!canStop}
                type="button"
                aria-label="Stop current run"
              >
                <span className="chat-stop-icon">■</span>
                <span className="chat-stop-label">Stop</span>
              </button>
              <button
                className="chat-send-btn"
                onClick={handleSend}
                disabled={isLoading || !inputValue.trim()}
                aria-label="Send"
                type="button"
              >
                <span className="chat-send-icon">↑</span>
                <span className="chat-send-label">Send</span>
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* ─── Right: Results ─── */}
      <div className="panel panel-right">
        {candidates.length > 0 ? (
          <>
            <div className="results-header">
              <h2>Top Recommended Collaborators</h2>
              <div className="query-text">Based on: {currentQuery}</div>
              <button
                className="report-error-btn"
                onClick={() => openReportModal("teaming-recommendation")}
                type="button"
              >
                Report Error
              </button>
              {phase === PHASE.RERANKING && rerankProgress.total > 0 && (
                <>
                  <div className="progress-bar-container">
                    <div
                      className="progress-bar-fill"
                      style={{
                        width: `${Math.round(
                          (rerankProgress.done / rerankProgress.total) * 100
                        )}%`,
                      }}
                    />
                  </div>
                  <div className="progress-text">
                    Analyzing… {rerankProgress.done}/{rerankProgress.total} batches
                  </div>
                </>
              )}
            </div>
            <div className="results-list">
              {orderedCandidates.map((candidate, i) => {
                const r = rerankedMap[candidate.author_id];
                const isExpanded = expandedCards[candidate.author_id];
                const name = r?.name || candidate.name;
                const affiliation = r?.institution || r?.affiliation || candidate.affiliation;
                const score = r?.score;
                const scoreLabel = score != null ? score.toFixed(1) : null;

                return (
                  <div key={candidate.author_id} className="collab-card">
                    <div
                      className="collab-card-header"
                      onClick={() => toggleCard(candidate.author_id)}
                    >
                      <div className="collab-card-rank">{i + 1}</div>
                      <div className="collab-card-info">
                        <div className="collab-card-name">{name}</div>
                        <div className="collab-card-affiliation">{affiliation}</div>
                      </div>
                      {scoreLabel != null ? (
                        <div className={`collab-card-score ${getScoreClass(score)}`}>
                          {scoreLabel}
                        </div>
                      ) : (
                        <div className="collab-card-score score-pending">
                          <span className="loading-spinner" />
                        </div>
                      )}
                      <div className={`collab-card-chevron ${isExpanded ? "expanded" : ""}`}>
                        ▼
                      </div>
                    </div>
                    {isExpanded && r && (
                      <div className="collab-card-body">
                        {r.justification && (
                          <div className="collab-match-reason">
                            {formatMessage(r.justification)}
                          </div>
                        )}
                        {r.mutual_coauthors?.length > 0 && (
                          <div className="collab-section">
                            <div className="collab-detail-label">Mutual Co-Authors</div>
                            <div className="collab-tags">
                              {r.mutual_coauthors.map((name, ci) => (
                                <span key={ci} className="collab-tag">{name}</span>
                              ))}
                            </div>
                          </div>
                        )}
                        {r.papers?.length > 0 && (
                          <div className="collab-section">
                            <div className="collab-detail-label">Selected Publications</div>
                            {r.papers.slice(0, 3).map((p, pi) => (
                              <div key={pi} className="collab-paper">
                                <span className="paper-title">{p.Title || "Untitled"}</span>
                                <span className="paper-meta">
                                  {p.Venue || ""}{p.PubYear ? `, ${p.PubYear}` : ""}
                                  {p.CitedCount > 0 && <span className="paper-cite">✦ {p.CitedCount}</span>}
                                </span>
                              </div>
                            ))}
                          </div>
                        )}
                      </div>
                    )}
                    {isExpanded && !r && (
                      <div className="collab-card-body">
                        <div className="collab-detail-text" style={{ padding: "12px 0" }}>
                          <span className="loading-spinner" /> Analyzing this candidate…
                        </div>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </>
        ) : (
          <div className="results-empty">
            <div className="results-empty-icon">🔬</div>
            <h2>Results will appear here</h2>
            <button
              className="report-error-btn"
              onClick={() => openReportModal("teaming-recommendation")}
              type="button"
            >
              Report Error
            </button>
            <p>
              Describe your needs on the left. I'll propose a query to confirm, then search and
              rank results.
            </p>
          </div>
        )}
      </div>

      {reportModalOpen && (
        <div className="report-modal-backdrop" onClick={closeReportModal}>
          <div className="report-modal-card" onClick={(e) => e.stopPropagation()}>
            <h3>Report Error</h3>
            <p className="report-modal-desc">
              {reportPageContext === "author-info"
                ? "Describe the issue you found in Author Info."
                : "Describe the issue you found in Teaming Recommendation."}
            </p>
            <textarea
              className="report-textarea"
              rows={5}
              placeholder="Please enter your feedback..."
              value={reportFeedback}
              onChange={(e) => setReportFeedback(e.target.value)}
            />
            {reportStatus && <div className="report-status">{reportStatus}</div>}
            <div className="report-modal-actions">
              <button type="button" onClick={closeReportModal} disabled={reportSubmitting}>
                Cancel
              </button>
              <button
                type="button"
                onClick={handleSubmitReport}
                disabled={reportSubmitting || !reportFeedback.trim()}
              >
                {reportSubmitting ? "Submitting..." : "Submit"}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
