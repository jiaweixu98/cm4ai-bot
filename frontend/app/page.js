"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import {
  fetchAuthor,
  chatMessage,
  searchCandidates,
  rerankCandidates,
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

export default function Home() {
  // ─── URL params ───
  const [aid, setAid] = useState("6052561");
  const [authorInfo, setAuthorInfo] = useState(null);
  const [returnTo, setReturnTo] = useState("/");

  // ─── Chat state ───
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState("");
  const [phase, setPhase] = useState(PHASE.IDLE);

  // ─── Search state ───
  const [currentQuery, setCurrentQuery] = useState("");
  const [pastQueries, setPastQueries] = useState([]);
  const [priorInputs, setPriorInputs] = useState([]);
  const [candidates, setCandidates] = useState([]);
  const [rerankedMap, setRerankedMap] = useState({});
  const [rerankProgress, setRerankProgress] = useState({ done: 0, total: 0 });
  const [expandedCards, setExpandedCards] = useState({});

  const chatEndRef = useRef(null);
  const inputRef = useRef(null);

  // ─── Init: parse URL and load author ───
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const aidParam = params.get("aid") || "6052561";
    const returnToParam = params.get("return_to");
    if (returnToParam) {
      try {
        const parsed = new URL(returnToParam, window.location.origin);
        if (parsed.origin === window.location.origin) {
          setReturnTo(`${parsed.pathname}${parsed.search}${parsed.hash}`);
        }
      } catch {
        // Keep default "/" when return_to is invalid.
      }
    }
    setAid(aidParam);

    fetchAuthor(aidParam)
      .then((data) => {
        setAuthorInfo(data);
        setMessages([
          {
            role: "assistant",
            content: data.greeting,
          },
        ]);
      })
      .catch(() => {
        setMessages([
          {
            role: "assistant",
            content:
              "Welcome! I am a scientific teaming assistant. How can I help you find collaborators?",
          },
        ]);
      });
  }, []);

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

  // ─── Send handler ───
  const handleSend = async () => {
    const text = inputValue.trim();
    if (!text) return;
    setInputValue("");
    addMessage("user", text);
    setPriorInputs((prev) => [...prev.slice(-199), text]);

    // Route ALL user input through the unified /api/chat endpoint
    const previousPhase = phase;  // capture before we change it
    setPhase(PHASE.GENERATING);

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
      });

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
      addMessage("assistant", `⚠️ Error: ${e.message}`);
      setPhase(previousPhase === PHASE.GENERATING ? PHASE.IDLE : previousPhase);
    }
  };

  // ─── Search ───
  const runSearch = async () => {
    try {
      const { candidates: cands } = await searchCandidates({
        aid,
        query: currentQuery,
      });
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
      addMessage("assistant", `⚠️ Search error: ${e.message}`);
      setPhase(PHASE.IDLE);
    }
  };

  // ─── Rerank (SSE) ───
  const runRerank = (cands) => {
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
        // Add a slight delay to allow the React batch update to finish
        // before transitioning the phase which changes the UI layout.
        setTimeout(() => setPhase(PHASE.DONE), 100);
      },
      // onError
      (data) => {
        console.error("Rerank error:", data.error);
      }
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
  const isLoading = phase === PHASE.GENERATING || phase === PHASE.SEARCHING;

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

  return (
    <div className="app-container">
      {/* ─── Left: Chat ─── */}
      <div className="panel panel-left">
        <div className="app-title-bar">
          <div className="app-title">🔬 CM4AI Teaming Assistant</div>
          <div className="app-tagline">Tell me your research needs, I'll find the right collaborators for you.</div>
        </div>
        <div style={{ margin: "10px 0 2px" }}>
          <a
            href={returnTo}
            style={{
              display: "inline-block",
              background: "#1f2937",
              color: "#fff",
              textDecoration: "none",
              padding: "8px 12px",
              borderRadius: "8px",
              fontWeight: 600,
              fontSize: "14px",
            }}
          >
            ← Back to Knowledge Graph
          </a>
        </div>
        <div className="panel-header">
          <h1>
            {authorInfo
              ? `Hi ${authorInfo.name}!`
              : "Welcome"}
          </h1>
          <div className="subtitle">
            <a href={returnTo}>
              🌎 Open Knowledge Graph
            </a>
          </div>
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
            <button
              className="chat-send-btn"
              onClick={handleSend}
              disabled={isLoading || !inputValue.trim()}
              aria-label="Send"
            >
              ↑
            </button>
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
            <p>
              Describe your needs on the left. I'll propose a query to confirm, then search and
              rank results.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
