"use client";

import { useState, useEffect, useRef, useCallback, useMemo } from "react";
import {
  fetchAuthor,
  chatMessage,
  createChatSession,
  getChatSession,
  listChatSessions,
  searchCandidates,
  explainCandidates,
  saveChatSession,
  submitErrorReport,
} from "./lib/api";
import { BRIDGE_MSG, inBridgeIframe, isBridgeOrigin, postToBridge } from "./lib/bridgeHost";
import {
  PERSONA_COPY,
  UNLINKED_AID,
  parseAidParam,
  parsePersonaIntent,
} from "./lib/personaConfig";
import {
  RESULT_LIMIT,
  followUpPrompts,
  normalizeMessage,
  relativeTime,
} from "./lib/matrixUi";
import { MAX_ATTACHED_FILES, readAttachedFiles } from "./lib/attachedPapers";
import FocalAuthorBar from "./components/FocalAuthorBar";
import ChatPane from "./components/ChatPane";
import ResultsWorkspace from "./components/ResultsWorkspace";
import SessionSidebar from "./components/SessionSidebar";
import ConfirmPopover from "./components/ConfirmPopover";
import ReportModal from "./components/ReportModal";

const PHASE = {
  IDLE: "idle",
  GENERATING: "generating",
  AWAITING_CONFIRM: "confirm",
  SEARCHING: "searching",
  RERANKING: "explaining",
  DONE: "done",
};

const SESSION_TOKEN_STORAGE_KEY = "matrix_user_token";

export default function Home() {
  const [aid, setAid] = useState(UNLINKED_AID);
  const [intent, setIntent] = useState("collaborator");
  const [switchingPersona, setSwitchingPersona] = useState(false);
  const switchingPersonaRef = useRef(false);
  const [searchIntent, setSearchIntent] = useState("collaborator");
  const [embedded, setEmbedded] = useState(false);
  const [uiReady, setUiReady] = useState(false);
  const [authorInfo, setAuthorInfo] = useState(null);
  const [seekerName, setSeekerName] = useState("");
  const [matrixUserToken, setMatrixUserToken] = useState("");
  const [savedPeople, setSavedPeople] = useState([]);
  const [savedPeopleReady, setSavedPeopleReady] = useState(false);
  const [teamMemberIds, setTeamMemberIds] = useState([]);
  const [mentorContextIds, setMentorContextIds] = useState([]);
  const [attachedPapers, setAttachedPapers] = useState([]);
  const [profileNotice, setProfileNotice] = useState("");
  const [historyOpen, setHistoryOpen] = useState(false);
  const [queuedFollowUp, setQueuedFollowUp] = useState("");
  const [pendingConfirm, setPendingConfirm] = useState(null);

  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState("");
  const [phase, setPhase] = useState(PHASE.IDLE);
  const [currentSessionId, setCurrentSessionId] = useState(null);
  const [sessions, setSessions] = useState([]);
  const [sessionStatus, setSessionStatus] = useState({ loading: true, saving: false, error: "" });

  const [currentQuery, setCurrentQuery] = useState("");
  const [pastQueries, setPastQueries] = useState([]);
  const [priorInputs, setPriorInputs] = useState([]);
  const [candidates, setCandidates] = useState([]);
  const [rerankedMap, setRerankedMap] = useState({});
  const [rerankProgress, setRerankProgress] = useState({ done: 0, total: 0 });
  const [rerankError, setRerankError] = useState(false);
  const [expandedCards, setExpandedCards] = useState({});
  const [reportModalOpen, setReportModalOpen] = useState(false);
  const [reportFeedback, setReportFeedback] = useState("");
  const [reportPageContext, setReportPageContext] = useState("author-info");
  const [reportSubmitting, setReportSubmitting] = useState(false);
  const [reportStatus, setReportStatus] = useState("");

  const sessionHydratingRef = useRef(false);
  const sessionBootstrappedRef = useRef(false);
  const sessionSaveTimerRef = useRef(null);
  const currentSessionIdRef = useRef(null);
  const chatAbortRef = useRef(null);
  const searchAbortRef = useRef(null);
  const rerankAbortRef = useRef(null);
  const handleSendRef = useRef(null);

  const copy = PERSONA_COPY[intent] || PERSONA_COPY.collaborator;
  const resultsCopy = PERSONA_COPY[searchIntent] || copy;
  const linked = aid !== UNLINKED_AID;
  const inIframe = embedded;
  const focalName = authorInfo?.name || seekerName;

  useEffect(() => {
    currentSessionIdRef.current = currentSessionId;
  }, [currentSessionId]);

  const resetWorkflowState = useCallback(() => {
    setMentorContextIds([]);
    setTeamMemberIds([]);
    setAttachedPapers([]);
    setMessages([]);
    setPhase(PHASE.IDLE);
    setCurrentQuery("");
    setPastQueries([]);
    setPriorInputs([]);
    setCandidates([]);
    setRerankedMap({});
    setRerankProgress({ done: 0, total: 0 });
    setExpandedCards({});
    setProfileNotice("");
    setQueuedFollowUp("");
  }, []);

  const upsertSessionSummary = useCallback((session) => {
    if (!session) return;
    setSessions((prev) => {
      const tagged = session.intent === "mentor" || session.intent === "collaborator"
        ? session
        : { ...session, intent };
      const next = [tagged, ...prev.filter((item) => item.id !== tagged.id)];
      next.sort((a, b) => new Date(b.last_message_at).getTime() - new Date(a.last_message_at).getTime());
      return next;
    });
  }, [intent]);

  const buildSessionStateSnapshot = useCallback(
    () => ({
      phase,
      currentQuery,
      pastQueries,
      priorInputs,
      candidates,
      rerankedMap,
      rerankProgress,
      expandedCards,
      intent,
      searchIntent,
      teamMemberIds,
      mentorContextIds,
    }),
    [phase, currentQuery, pastQueries, priorInputs, candidates, rerankedMap, rerankProgress, expandedCards, intent, searchIntent, teamMemberIds, mentorContextIds]
  );

  const applySessionSnapshot = useCallback((session) => {
    const snapshot = session?.state || {};
    const restoredMessages = (Array.isArray(session?.messages) ? session.messages : []).map(normalizeMessage);

    sessionHydratingRef.current = true;
    currentSessionIdRef.current = session?.id || null;
    setCurrentSessionId(session?.id || null);
    setMentorContextIds(Array.isArray(snapshot.mentorContextIds) ? snapshot.mentorContextIds.map(Number).filter(Number.isInteger).slice(0, 8) : []);
    setMessages(restoredMessages);
    setPhase(Array.isArray(snapshot.candidates) && snapshot.candidates.length ? PHASE.DONE : PHASE.IDLE);
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
    setExpandedCards(
      snapshot.expandedCards && typeof snapshot.expandedCards === "object" ? snapshot.expandedCards : {}
    );
    if (snapshot.searchIntent === "mentor" || snapshot.searchIntent === "collaborator") {
      setSearchIntent(snapshot.searchIntent);
    }
    setTeamMemberIds(
      Array.isArray(snapshot.teamMemberIds)
        ? snapshot.teamMemberIds.map(Number).filter(Number.isInteger).slice(0, 25)
        : []
    );
    setTimeout(() => {
      sessionHydratingRef.current = false;
    }, 0);
  }, []);

  useEffect(() => {
    let cancelled = false;

    const initialize = async () => {
      const params = new URLSearchParams(window.location.search);
      const nextIntent = parsePersonaIntent(params.get("intent"));
      const freshConversation = params.get('fresh') === '1';
      if (freshConversation) {
        params.delete('fresh');
        window.history.replaceState({}, '', `${window.location.pathname}?${params.toString()}`);
      }
      const aidParam = parseAidParam(params.get("aid"));
      const tokenFromUrl = params.get("mx_user_token") || "";
      const seekerNameParam = String(params.get("seeker_name") || "").trim();
      const storedToken =
        typeof window !== "undefined" ? window.sessionStorage.getItem(SESSION_TOKEN_STORAGE_KEY) || "" : "";
      const resolvedToken = tokenFromUrl || storedToken;
      const isEmbedded = params.get("embedded") === "1" || inBridgeIframe();

      if (tokenFromUrl && typeof window !== "undefined") {
        window.sessionStorage.setItem(SESSION_TOKEN_STORAGE_KEY, tokenFromUrl);
        params.delete("mx_user_token");
        const nextSearch = params.toString();
        const nextUrl = `${window.location.pathname}${nextSearch ? `?${nextSearch}` : ""}${window.location.hash}`;
        window.history.replaceState({}, "", nextUrl);
      }

      setIntent(nextIntent);
      setSearchIntent(nextIntent);
      setAid(aidParam);
      setSeekerName(seekerNameParam);
      setEmbedded(isEmbedded);
      setMatrixUserToken(resolvedToken);

      let fetchedAuthor = null;
      if (aidParam !== UNLINKED_AID) {
        try {
          fetchedAuthor = await fetchAuthor(aidParam);
          if (cancelled) return;
          setAuthorInfo(fetchedAuthor);
        } catch {
          if (cancelled) return;
          setAuthorInfo(null);
        }
      } else {
        setAuthorInfo(null);
      }
      if (cancelled) return;
      setUiReady(true);

      if (!resolvedToken) {
        if (cancelled) return;
        resetWorkflowState();
        setSessionStatus({ loading: false, saving: false, error: "" });
        sessionBootstrappedRef.current = true;
        return;
      }

      try {
        const [mentorPayload, collaboratorPayload] = await Promise.all([
          listChatSessions({ aid: aidParam, intent: "mentor", authToken: resolvedToken }),
          listChatSessions({ aid: aidParam, intent: "collaborator", authToken: resolvedToken }),
        ]);
        if (cancelled) return;
        const tagSessions = (items, fallbackIntent) => (Array.isArray(items) ? items : []).map((item) => (
          item?.intent === "mentor" || item?.intent === "collaborator" ? item : { ...item, intent: fallbackIntent }
        ));
        const mergedSessions = [
          ...tagSessions(mentorPayload.sessions, "mentor"),
          ...tagSessions(collaboratorPayload.sessions, "collaborator"),
        ].sort((a, b) => new Date(b.last_message_at).getTime() - new Date(a.last_message_at).getTime());
        setSessions(mergedSessions);
        let effectiveIntent = nextIntent;
        const requestedSessionId = params.get("session") || "";
        const requestedSession = requestedSessionId
          ? mergedSessions.find((item) => item.id === requestedSessionId)
          : null;
        if (requestedSession && (requestedSession.intent === "mentor" || requestedSession.intent === "collaborator")) {
          effectiveIntent = requestedSession.intent;
        }
        if (requestedSessionId) {
          params.delete("session");
          params.set("intent", effectiveIntent);
          window.history.replaceState({}, "", `${window.location.pathname}?${params.toString()}`);
        }
        setIntent(effectiveIntent);
        setSearchIntent(effectiveIntent);
        const restoreTarget = requestedSession
          || (!freshConversation ? mergedSessions.find((item) => item.intent === effectiveIntent) : null);
        if (restoreTarget) {
          const latestPayload = await getChatSession({
            sessionId: restoreTarget.id,
            authToken: resolvedToken,
          });
          if (cancelled) return;
          applySessionSnapshot(latestPayload.session);
        } else {
          resetWorkflowState();
        }
        setSessionStatus({ loading: false, saving: false, error: "" });
      } catch (error) {
        if (cancelled) return;
        console.error("Failed to restore chat sessions:", error);
        resetWorkflowState();
        setSessionStatus({
          loading: false,
          saving: false,
          error: "Could not load chat history.",
        });
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
  }, [applySessionSnapshot, resetWorkflowState]);

  useEffect(() => {
    if (!inIframe) return undefined;
    postToBridge({ type: BRIDGE_MSG.REQUEST_SAVED });
    const onMessage = (event) => {
      if (!isBridgeOrigin(event.origin)) return;
      const data = event.data;
      if (!data || typeof data !== "object") return;
      if (data.type === BRIDGE_MSG.SAVED_PEOPLE && Array.isArray(data.people)) {
        setSavedPeople(data.people);
        setSavedPeopleReady(true);
      }
      if (data.type === BRIDGE_MSG.OPEN_PERSON_RESULT) {
        if (!data.ok) {
          setProfileNotice("That researcher is not on the loaded map.");
        }
      }
    };
    window.addEventListener("message", onMessage);
    return () => window.removeEventListener("message", onMessage);
  }, [inIframe]);

  useEffect(() => {
    if (!profileNotice) return undefined;
    const timer = window.setTimeout(() => setProfileNotice(""), 4200);
    return () => window.clearTimeout(timer);
  }, [profileNotice]);

  useEffect(() => {
    if (!savedPeopleReady) return;
    const availableIds = new Set(savedPeople.map((person) => Number(person.authorId)));
    setTeamMemberIds((current) => current.filter((authorId) => availableIds.has(Number(authorId))));
    setMentorContextIds((current) => current.filter((authorId) => availableIds.has(Number(authorId))));
  }, [savedPeople, savedPeopleReady]);

  useEffect(() => {
    const role = intent === "mentor" ? "mentor" : "collaborator";
    const name = focalName;
    document.title = name ? `MATRIX · ${role} for ${name}` : `MATRIX · Find a ${role}`;
  }, [intent, focalName]);

  const addMessage = useCallback((role, content, extra = {}) => {
    setMessages((prev) => [
      ...prev,
      normalizeMessage({ role, content, at: Date.now(), ...extra }, prev.length),
    ]);
  }, []);

  const ensureCurrentSession = useCallback(async () => {
    if (!matrixUserToken) return null;
    if (currentSessionIdRef.current) return currentSessionIdRef.current;

    const sessionPayload = await createChatSession({
      aid,
      focalAuthorName: focalName || "",
      messages: Array.isArray(messages) ? messages : [],
      state: buildSessionStateSnapshot(),
      authToken: matrixUserToken,
    });
    const created = sessionPayload.session;
    currentSessionIdRef.current = created.id;
    setCurrentSessionId(created.id);
    upsertSessionSummary(created);
    return created.id;
  }, [aid, focalName, buildSessionStateSnapshot, matrixUserToken, messages, upsertSessionSummary]);

  const persistSessionNow = useCallback(async () => {
    if (!matrixUserToken) return null;
    try {
      const sessionId = await ensureCurrentSession();
      if (!sessionId) return null;
      const payload = await saveChatSession({
        sessionId,
        aid,
        focalAuthorName: focalName || "",
        messages,
        state: buildSessionStateSnapshot(),
        authToken: matrixUserToken,
      });
      upsertSessionSummary(payload.session);
      return sessionId;
    } catch (error) {
      console.error("Failed to persist chat session:", error);
      return currentSessionIdRef.current;
    }
  }, [
    aid,
    focalName,
    buildSessionStateSnapshot,
    ensureCurrentSession,
    matrixUserToken,
    messages,
    upsertSessionSummary,
  ]);

  const handleStop = useCallback(() => {
    chatAbortRef.current?.abort();
    searchAbortRef.current?.abort();
    rerankAbortRef.current?.abort();
    chatAbortRef.current = null;
    searchAbortRef.current = null;
    rerankAbortRef.current = null;
    setPhase(candidates.length > 0 ? PHASE.DONE : PHASE.IDLE);
    addMessage("assistant", "Stopped.", { stopped: true });
  }, [addMessage, candidates.length]);

  const startNewSession = useCallback(() => {
    currentSessionIdRef.current = null;
    setCurrentSessionId(null);
    setHistoryOpen(false);
    resetWorkflowState();
  }, [resetWorkflowState]);

  const handleNewSession = useCallback(() => {
    if (phase === PHASE.GENERATING || phase === PHASE.SEARCHING || phase === PHASE.RERANKING) {
      setPendingConfirm({ type: "new" });
      return;
    }
    startNewSession();
  }, [phase, startNewSession]);

  const navigateToSession = useCallback((targetIntent, sessionId) => {
    const params = new URLSearchParams(window.location.search);
    params.set("intent", targetIntent);
    params.delete("fresh");
    if (sessionId) params.set("session", sessionId);
    else params.delete("session");
    // A fresh document prevents old requests, save callbacks and selected context leaking across modes.
    window.location.assign(`${window.location.pathname}?${params.toString()}`);
  }, []);

  const handleSelectSession = useCallback(
    async (sessionId) => {
      if (!matrixUserToken || !sessionId) return;
      if (sessionId === currentSessionIdRef.current) return;
      const selected = sessions.find((item) => item.id === sessionId);
      const selectedIntent = selected?.intent === "mentor" || selected?.intent === "collaborator" ? selected.intent : null;
      if (selectedIntent && selectedIntent !== intent) {
        if (phase === PHASE.GENERATING || phase === PHASE.SEARCHING || phase === PHASE.RERANKING) {
          setPendingConfirm({ type: "switch", sessionId, sessionIntent: selectedIntent });
          return;
        }
        navigateToSession(selectedIntent, sessionId);
        return;
      }
      if (phase === PHASE.GENERATING || phase === PHASE.SEARCHING || phase === PHASE.RERANKING) {
        setPendingConfirm({ type: "switch", sessionId });
        return;
      }
      try {
        setHistoryOpen(false);
        setSessionStatus((prev) => ({ ...prev, loading: true, error: "" }));
        const payload = await getChatSession({ sessionId, authToken: matrixUserToken });
        applySessionSnapshot(payload.session);
        setSessionStatus((prev) => ({ ...prev, loading: false }));
      } catch (error) {
        console.error("Failed to load chat session:", error);
        setSessionStatus((prev) => ({
          ...prev,
          loading: false,
          error: "Could not load that chat.",
        }));
      }
    },
    [applySessionSnapshot, intent, matrixUserToken, navigateToSession, phase, sessions]
  );

  const confirmPending = useCallback(async () => {
    const pending = pendingConfirm;
    setPendingConfirm(null);
    if (!pending) return;
    handleStop();
    if (pending.type === "new") {
      startNewSession();
      return;
    }
    if (pending.type === "switch" && pending.sessionId) {
      if (pending.sessionIntent && pending.sessionIntent !== intent) {
        navigateToSession(pending.sessionIntent, pending.sessionId);
        return;
      }
      try {
        setSessionStatus((prev) => ({ ...prev, loading: true, error: "" }));
        const payload = await getChatSession({ sessionId: pending.sessionId, authToken: matrixUserToken });
        applySessionSnapshot(payload.session);
        setSessionStatus((prev) => ({ ...prev, loading: false }));
      } catch (error) {
        console.error("Failed to load chat session:", error);
        setSessionStatus((prev) => ({
          ...prev,
          loading: false,
          error: "Could not load that chat.",
        }));
      }
    }
  }, [applySessionSnapshot, handleStop, intent, matrixUserToken, navigateToSession, pendingConfirm, startNewSession]);

  useEffect(() => {
    if (!sessionBootstrappedRef.current || sessionHydratingRef.current) return;
    if (!matrixUserToken || !currentSessionId) return;

    if (sessionSaveTimerRef.current) window.clearTimeout(sessionSaveTimerRef.current);

    sessionSaveTimerRef.current = window.setTimeout(async () => {
      try {
        setSessionStatus((prev) => ({ ...prev, saving: true, error: "" }));
        const payload = await saveChatSession({
          sessionId: currentSessionId,
          aid,
          focalAuthorName: focalName || "",
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
          error: "Could not save this chat.",
        }));
      }
    }, 800);

    return () => {
      if (sessionSaveTimerRef.current) window.clearTimeout(sessionSaveTimerRef.current);
    };
  }, [
    aid,
    focalName,
    buildSessionStateSnapshot,
    currentSessionId,
    matrixUserToken,
    messages,
    upsertSessionSummary,
  ]);

  useEffect(() => {
    const flush = () => {
      if (!matrixUserToken || !currentSessionIdRef.current) return;
      void persistSessionNow();
    };
    window.addEventListener("pagehide", flush);
    return () => window.removeEventListener("pagehide", flush);
  }, [matrixUserToken, persistSessionNow]);

  const shortlistForChat = useCallback(() => {
    return candidates.slice(0, RESULT_LIMIT).map((candidate) => {
      const ranked = rerankedMap[candidate.author_id] || {};
      return {
        author_id: candidate.author_id,
        name: candidate.name,
        affiliation: candidate.affiliation,
        justification: ranked.justification || "",
        hops: ranked.hops,
        mutual_coauthors: ranked.mutual_coauthors || [],
        papers: ranked.papers || candidate.papers || [],
        intent,
      };
    });
  }, [candidates, intent, rerankedMap]);

  const runRerank = useCallback(
    async (cands) => {
      const rerankController = new AbortController();
      rerankAbortRef.current = rerankController;
      setRerankProgress({ done: 0, total: 1 });
      setRerankError(false);
      try {
        const payload = await explainCandidates({
          aid: intent === 'collaborator' && linked ? aid : UNLINKED_AID,
          teamMemberIds: intent === 'collaborator' ? teamMemberIds.map(String) : [],
          teamPeople: intent === 'collaborator'
            ? savedPeople.filter((person) => teamMemberIds.includes(Number(person.authorId))).map((person) => {
                const fromResults = cands.find((candidate) => Number(candidate.author_id) === Number(person.authorId));
                return {
                  author_id: String(person.authorId),
                  name: person.name,
                  papers: fromResults?.papers || person.papers || [],
                };
              })
            : [],
          seekerPapers: attachedPapers.map((paper) => paper.title),
          query: currentQuery,
          intent,
          candidates: cands,
          signal: rerankController.signal,
        });
        if (rerankController.signal.aborted) return;
        const finalResults = Array.isArray(payload.results) ? payload.results : [];
        setRerankedMap((prev) => {
          const next = { ...prev };
          for (const result of finalResults) next[result.author_id] = { ...result, justification: result.explanation, context_basis: payload.context_basis, team_count: payload.team_count };
          return next;
        });
        const expanded = {};
        [...finalResults]
          .sort((left, right) => (left.order || 99) - (right.order || 99))
          .slice(0, 3)
          .forEach((result) => {
            expanded[result.author_id] = true;
          });
        setExpandedCards(expanded);
        setRerankProgress({ done: 1, total: 1 });
        setRerankError(false);
        setPhase(PHASE.DONE);
      } catch (error) {
        if (rerankController.signal.aborted) return;
        if (error?.name !== "AbortError") {
          setRerankError(true);
        }
        setPhase(PHASE.DONE);
      } finally {
        if (rerankAbortRef.current === rerankController) rerankAbortRef.current = null;
      }
    },
    [attachedPapers, currentQuery, intent, aid, linked, teamMemberIds, savedPeople]
  );

  const handleRetryNotes = useCallback(() => {
    if (!candidates.length || phase === PHASE.RERANKING || phase === PHASE.SEARCHING) return;
    setPhase(PHASE.RERANKING);
    runRerank(candidates);
  }, [candidates, phase, runRerank]);

  const runSearch = useCallback(async () => {
    const searchController = new AbortController();
    try {
      searchAbortRef.current = searchController;
      setSearchIntent(intent);
      setRerankError(false);
      const attachedTitles = attachedPapers.map((paper) => paper.title);
      const { candidates: cands } = await searchCandidates({
        representativeTitles: attachedTitles.join("\n"),
        aid: linked ? aid : UNLINKED_AID,
        query: currentQuery,
        topK: RESULT_LIMIT,
        bridge2aiOnly: intent === "mentor",
        outsideNetwork: intent === "collaborator",
        teamMemberIds: intent === "collaborator" ? teamMemberIds : [],
        signal: searchController.signal,
      });
      if (searchController.signal.aborted) return;
      searchAbortRef.current = null;
      const nextCandidates = Array.isArray(cands) ? cands.slice(0, RESULT_LIMIT) : [];
      setCandidates(nextCandidates);
      const expanded = {};
      nextCandidates.slice(0, 3).forEach((candidate) => {
        expanded[candidate.author_id] = true;
      });
      setExpandedCards(expanded);
      if (nextCandidates.length === 0) {
        addMessage("assistant", "No researchers found for that topic.");
        setPhase(PHASE.IDLE);
        return;
      }
      const teamContext =
        intent === "collaborator" && teamMemberIds.length > 0
          ? ` Using ${teamMemberIds.length} team ${teamMemberIds.length === 1 ? "member" : "members"}.`
          : "";
      addMessage(
        "assistant",
        attachedTitles.length > 0
          ? `Researchers for **${currentQuery}**, using your attached ${attachedTitles.length === 1 ? "draft" : "drafts"}.${teamContext}`
          : `Researchers for **${currentQuery}**.${teamContext}`
      );
      setAttachedPapers([]);
      setPhase(PHASE.RERANKING);
      runRerank(nextCandidates);
    } catch (error) {
      if (searchController.signal.aborted) return;
      searchAbortRef.current = null;
      if (error?.name === "AbortError") {
        setPhase(candidates.length > 0 ? PHASE.DONE : PHASE.IDLE);
        return;
      }
      addMessage("assistant", `Search error: ${error.message}`);
      setPhase(PHASE.IDLE);
    }
  }, [
    addMessage,
    aid,
    attachedPapers,
    candidates.length,
    currentQuery,
    intent,
    linked,
    runRerank,
    teamMemberIds,
  ]);

  const handleSend = async (presetText) => {
    if (switchingPersonaRef.current) return;
    const text = String(presetText ?? inputValue).trim();
    if (!text) return;
    if (phase === PHASE.SEARCHING || phase === PHASE.RERANKING) {
      setQueuedFollowUp(text);
      setInputValue("");
      return;
    }
    try {
      await ensureCurrentSession();
    } catch (sessionError) {
      console.error("Failed to create chat session:", sessionError);
      setSessionStatus((prev) => ({
        ...prev,
        error: "Chats are not being saved in this tab.",
      }));
    }
    setInputValue("");
    addMessage("user", text);
    setPriorInputs((prev) => [...prev.slice(-199), text]);

    const previousPhase = phase;
    setPhase(PHASE.GENERATING);
    const chatController = new AbortController();
    chatAbortRef.current = chatController;

    try {
      const result = await chatMessage({
        representativeTitles: attachedPapers.map((paper) => paper.title).join('\n'),
        contextPersonIds: intent === 'mentor' ? mentorContextIds : teamMemberIds,
        aid: linked ? aid : UNLINKED_AID,
        userInput: text,
        conversationHistory: [...messages, { role: "user", content: text }],
        currentQuery: currentQuery || null,
        pastQueries,
        priorInputs: [...priorInputs, text],
        searchResults: shortlistForChat(),
        searchPhase: previousPhase,
        intent,
        signal: chatController.signal,
      });
      if (chatController.signal.aborted) return;
      chatAbortRef.current = null;

      if (result.action === "confirm" && currentQuery) {
        addMessage("assistant", "Searching now.");
        setPhase(PHASE.SEARCHING);
        await runSearch();
      } else if (result.action === "search") {
        const query = result.query;
        const justification = result.justification || "";
        setCurrentQuery(query);
        setPastQueries((prev) => [...prev, query]);
        if (previousPhase === PHASE.DONE || previousPhase === PHASE.IDLE) {
          setCandidates([]);
          setRerankedMap({});
          setRerankProgress({ done: 0, total: 0 });
          setExpandedCards({});
        }
        let msg = `Search **${query}**?`;
        if (justification) msg += ` ${justification.replace(/\s+/g, " ").trim()}`;
        addMessage("assistant", msg);
        setPhase(PHASE.AWAITING_CONFIRM);
      } else {
        const reply = result.reply || "Ask about someone in the list, or describe a new topic.";
        addMessage("assistant", reply);
        setPhase(previousPhase === PHASE.GENERATING ? PHASE.IDLE : previousPhase);
      }
    } catch (error) {
      if (chatController.signal.aborted) return;
      chatAbortRef.current = null;
      if (error?.name === "AbortError") {
        setPhase(previousPhase === PHASE.GENERATING ? PHASE.IDLE : previousPhase);
        return;
      }
      addMessage("assistant", error.message || "Chat is unavailable. Try again.");
      setPhase(previousPhase === PHASE.GENERATING ? PHASE.IDLE : previousPhase);
    }
  };

  handleSendRef.current = handleSend;

  useEffect(() => {
    if (!queuedFollowUp) return;
    if (phase === PHASE.GENERATING || phase === PHASE.SEARCHING || phase === PHASE.RERANKING) return;
    const text = queuedFollowUp;
    setQueuedFollowUp("");
    void handleSendRef.current?.(text);
  }, [phase, queuedFollowUp]);

  const isLoading = switchingPersona || phase === PHASE.GENERATING || phase === PHASE.SEARCHING || phase === PHASE.RERANKING;
  const canStop = isLoading;

  const orderedCandidates = useMemo(() => {
    return candidates;
  }, [candidates]);

  const savedIds = useMemo(
    () => new Set(savedPeople.map((person) => Number(person.authorId))),
    [savedPeople]
  );

  const suggestedPrompts =
    intent === 'mentor' && mentorContextIds.length > 0 && phase !== PHASE.AWAITING_CONFIRM
      ? [
          `How does ${savedPeople.find((person) => Number(person.authorId) === mentorContextIds[0])?.name || 'the included person'} fit my learning goal?`,
          mentorContextIds.length > 1 ? 'Compare the included people using their publications' : 'What should I ask this person about mentoring?',
          'Find more mentors for my learning goal',
        ]
      : phase === PHASE.AWAITING_CONFIRM
      ? copy.promptsConfirm
      : candidates.length > 0
        ? followUpPrompts(orderedCandidates, rerankedMap, copy)
        : copy.promptsIdle;

  const handleAttachFiles = useCallback(async (fileList) => {
    const { attached, skippedType, skippedSize } = await readAttachedFiles(fileList, attachedPapers.length);
    if (attached.length) {
      setAttachedPapers((prev) => [...prev, ...attached].slice(0, MAX_ATTACHED_FILES));
    }
    const skipped = skippedType + skippedSize;
    if (skipped > 0) {
      setProfileNotice(`Skipped ${skipped} file${skipped === 1 ? "" : "s"}: only .txt, .md or .tex under 200 KB.`);
    }
  }, [attachedPapers.length]);

  const removeAttachment = useCallback((id) => {
    setAttachedPapers((prev) => prev.filter((paper) => paper.id !== id));
  }, []);

  const openProfile = async (authorId) => {    setProfileNotice("");
    await persistSessionNow();
    postToBridge({ type: BRIDGE_MSG.OPEN_PERSON, authorId: Number(authorId) });
    if (!inIframe) {
      setProfileNotice("Open this from the graph to view the profile.");
    }
  };

  const savePerson = (candidate, ranked) => {
    const authorId = Number(candidate.author_id);
    const person = {
      authorId,
      name: candidate.name,
      affiliation: candidate.affiliation,
      intent,
      query: currentQuery,
    };
    postToBridge({
      type: BRIDGE_MSG.SAVE_PERSON,
      person,
    });
    setSavedPeople((prev) => [person, ...prev.filter((item) => Number(item.authorId) !== authorId)]);
    setSavedPeopleReady(true);
    setProfileNotice("Saved.");
  };

  const unsavePerson = (authorId) => {
    postToBridge({ type: BRIDGE_MSG.UNSAVE_PERSON, authorId: Number(authorId) });
    setSavedPeople((prev) => prev.filter((person) => Number(person.authorId) !== Number(authorId)));
  };

  const toggleTeamMember = (authorId) => {
    const normalizedId = Number(authorId);
    if (!Number.isInteger(normalizedId)) return;
    setTeamMemberIds((current) =>
      current.includes(normalizedId)
        ? current.filter((id) => id !== normalizedId)
        : [...current, normalizedId].slice(-25)
    );
  };

  const openReportModal = (page, seed = "") => {
    setReportPageContext(page);
    setReportFeedback(seed);
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
          intent,
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
    } catch (error) {
      setReportStatus(`Submission failed: ${error.message}`);
    } finally {
      setReportSubmitting(false);
    }
  };

  const handleIntentChange = async (nextIntent) => {
    if (nextIntent === intent || switchingPersonaRef.current || isLoading || sessionStatus.loading) return;
    switchingPersonaRef.current = true;
    setSwitchingPersona(true);
    window.clearTimeout(sessionSaveTimerRef.current);
    try {
      if (matrixUserToken && messages.length) {
        const sessionId = await ensureCurrentSession();
        await saveChatSession({ sessionId, aid, focalAuthorName: focalName || '', messages,
          state: buildSessionStateSnapshot(), authToken: matrixUserToken });
      }
    } catch {
      switchingPersonaRef.current = false;
      setSwitchingPersona(false);
      setProfileNotice('Could not save this conversation.');
      return;
    }
    const params = new URLSearchParams(window.location.search);
    params.set("intent", nextIntent);
    params.set('fresh', '1');
    params.delete("session");
    // A fresh document prevents old requests, save callbacks and selected context leaking across modes.
    window.location.assign(`${window.location.pathname}?${params.toString()}`);
  };

  useEffect(() => {
    const onKey = (event) => {
      if (event.key !== "Escape") return;
      setHistoryOpen(false);
      setPendingConfirm(null);
      setReportModalOpen((open) => (reportSubmitting ? open : false));
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [reportSubmitting]);

  const historyMenu = (
    <div className="history-menu" role="menu">
      {sessionStatus.error && <div className="session-status session-status-error">{sessionStatus.error}</div>}
      {!matrixUserToken ? (
        <div className="session-empty">Sign in on the graph to keep chats.</div>
      ) : sessions.filter((session) => (session.intent || intent) === intent).length === 0 ? (
        <div className="session-empty">No previous chats yet.</div>
      ) : (
        sessions.filter((session) => (session.intent || intent) === intent).map((session) => (
          <button
            key={session.id}
            className={`session-chip ${session.id === currentSessionId ? "active" : ""}`}
            onClick={() => {
              setHistoryOpen(false);
              handleSelectSession(session.id);
            }}
            type="button"
            role="menuitem"
          >
            <span className="session-chip-title">{session.title || "Untitled session"}</span>
            <span className="session-chip-meta">{relativeTime(session.last_message_at)}</span>
          </button>
        ))
      )}
    </div>
  );

  if (!uiReady) {
    return (
      <div className="app-container is-embedded">
        <header className="focal-bar">
          <div className="focal-copy">
            <div className="focal-kicker">MATRIX</div>
            <div className="focal-name">Loading…</div>
          </div>
        </header>
      </div>
    );
  }

  const shellClass = ["app-container", inIframe ? "is-embedded" : ""].filter(Boolean).join(" ");

  return (
    <div className={shellClass}>
      <FocalAuthorBar
        copy={copy}
        intent={intent}
        authorInfo={authorInfo}
        seekerName={seekerName}
        profileContextEnabled={intent === "collaborator" && linked}
        inIframe={inIframe}
        sessionStatus={sessionStatus}
        matrixUserToken={matrixUserToken}
        historyOpen={historyOpen}
        historyMenu={historyMenu}
        isLoading={isLoading}
        onIntentChange={handleIntentChange}
        onOpenFocal={() => openProfile(aid)}
        onNewSession={handleNewSession}
        onToggleHistory={() => setHistoryOpen((open) => !open)}
      />

      <div className="workspace">
        <SessionSidebar
          sessions={sessions}
          currentSessionId={currentSessionId}
          sessionStatus={sessionStatus}
          matrixUserToken={matrixUserToken}
          isLoading={isLoading}
          onNewSession={handleNewSession}
          onSelectSession={handleSelectSession}
        />
        <ChatPane
          copy={copy}
          authorName={focalName || ""}
          messages={messages}
          phase={phase}
          isLoading={isLoading}
          canStop={canStop}
          inputValue={inputValue}
          setInputValue={setInputValue}
          attachedPapers={attachedPapers}
          onAttachFiles={handleAttachFiles}
          onRemoveAttachment={removeAttachment}
          suggestedPrompts={suggestedPrompts}
          promptMeta={copy.promptMeta}
          queuedFollowUp={queuedFollowUp}
          onSend={handleSend}
          onStop={handleStop}
          onRetry={handleSend}
          onReport={(content) => openReportModal("author-info", content)}
          citations={orderedCandidates.flatMap((candidate) => rerankedMap[candidate.author_id]?.papers || candidate.papers || []).slice(0, 8)}
        />

        <ResultsWorkspace
          copy={resultsCopy}
          candidates={candidates}
          orderedCandidates={orderedCandidates}
          rerankedMap={rerankedMap}
          phase={phase}
          currentQuery={currentQuery}
          savedPeople={savedPeople}
          savedIds={savedIds}
          teamBuilding={intent === "collaborator"}
          teamMemberIds={new Set(intent === 'mentor' ? mentorContextIds : teamMemberIds)}
          selectionDisabled={isLoading}
          profileNotice={profileNotice}
          onDismissNotice={() => setProfileNotice("")}
          onOpenProfile={openProfile}
          onSave={savePerson}
          onUnsave={unsavePerson}
          onRetryNotes={handleRetryNotes}
          rerankError={rerankError}
          onToggleTeam={intent === 'mentor' ? (id) => {
            const value = Number(id);
            if (!mentorContextIds.includes(value) && mentorContextIds.length >= 8) {
              setProfileNotice('Include up to 8 people.');
              return;
            }
            setMentorContextIds((current) => current.includes(value) ? current.filter((item) => item !== value) : [...current, value]);
          } : toggleTeamMember}
        />
      </div>

      <ConfirmPopover
        open={Boolean(pendingConfirm)}
        title={pendingConfirm?.type === "switch" ? "Switch sessions?" : "Start a new session?"}
        body="This stops the current search."
        confirmLabel={pendingConfirm?.type === "switch" ? "Switch" : "Start new"}
        onConfirm={confirmPending}
        onCancel={() => setPendingConfirm(null)}
      />

      <ReportModal
        open={reportModalOpen}
        pageContext={reportPageContext}
        feedback={reportFeedback}
        status={reportStatus}
        submitting={reportSubmitting}
        onFeedbackChange={setReportFeedback}
        onClose={closeReportModal}
        onSubmit={handleSubmitReport}
      />
    </div>
  );
}
