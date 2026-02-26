"""CM4AI Bot — FastAPI backend.

Replicates the full Streamlit chat flow as REST + SSE endpoints.
"""

import os
import torch  # IMPORT FIRST to prevent macOS segfaults with faiss/asyncio
import re
import json
import logging
import concurrent.futures
from collections import deque
from contextlib import asynccontextmanager
from typing import List, Tuple

import numpy as np
import networkx as nx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

load_dotenv()

from data_loader import (
    load_all,
    load_author_nodes,
    load_knowledge_graph_nx,
    load_embeddings_and_index,
    load_specter_model,
    load_publication_counts,
)
from retriever import Retriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Silence noisy httpx logs from openai client
logging.getLogger("httpx").setLevel(logging.WARNING)

# ---------- models ----------
MODEL_NAME_EXPERTISE = "gpt-4.1"
MODEL_NAME_RERANKING = "gpt-4.1-mini"

# Pre-load synchronously BEFORE the event loop starts
# This avoids PyTorch segfaults inside asyncio loops on Apple Silicon macOS
logger.info("Pre-loading resources BEFORE starting the FastAPI event loop...")
load_all()


# ---------- lifespan ----------
@asynccontextmanager
async def lifespan(app: FastAPI):
    import time
    t0 = time.time()
    logger.info("Starting up — server is ready. Warming up encoder…")
    # Warm up encoder
    try:
        t1 = time.time()
        _model_encode("warmup")
        logger.info("Encoder warmup done in %.1fs.", time.time() - t1)
    except Exception as e:
        logger.warning("Encoder warmup failed: %s", e)
    logger.info("✅ Startup complete in %.1fs.", time.time() - t0)
    yield


app = FastAPI(title="CM4AI Bot API", lifespan=lifespan)
# In production, restrict to your Vercel domain; "*" kept for dev convenience
_ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "https://*.vercel.app",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- helpers ----------
def _get_openai_client():
    from openai import OpenAI

    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set")
    return OpenAI(api_key=key)


def _model_encode(query_text: str) -> np.ndarray:
    import torch

    tokenizer, model = load_specter_model()
    if tokenizer is None or model is None:
        raise HTTPException(status_code=500, detail="SPECTER model not loaded")
    inputs = tokenizer(
        [str(query_text)],
        padding=True,
        truncation=True,
        return_tensors="pt",
        return_token_type_ids=False,
        max_length=512,
    )
    with torch.no_grad():
        output = model(**inputs)
        embeddings = output.last_hidden_state[:, 0, :].cpu().detach().numpy().astype(np.float32)
    return embeddings


def normalize_query_text(text: str) -> str:
    try:
        s = str(text or "")
        s = re.sub(r"\[/?QUERY\]", "", s, flags=re.IGNORECASE)
        s = s.replace("**", "")
        s = " ".join(s.split()).strip().lower()
        return s
    except Exception:
        return (text or "").strip().lower()


def _get_user_background(user_id: str) -> str:
    nodes = load_author_nodes()
    user = nodes.get(user_id, {})
    features = user.get("features", {})
    name = features.get("FullName", user.get("title", "Unknown"))
    affiliation = features.get("Affiliation", "Unknown")
    papers = features.get("Top Cited or Most Recent Papers", [])
    bg = f"Name: {name}\nAffiliation: {affiliation}\nTop Cited or Most Recent Papers:\n"
    for p in papers:
        bg += f"- {p.get('Title', 'Untitled')} ({p.get('Venue', '')}, {p.get('PubYear', '')}) - Cited {p.get('CitedCount', 0)} times\n"
    return bg


def _get_user_name(user_id: str) -> str:
    nodes = load_author_nodes()
    user = nodes.get(user_id, {})
    return user.get("features", {}).get("FullName", user.get("title", "Researcher"))


def _get_author_details(author_id: str) -> dict:
    nodes = load_author_nodes()
    info = nodes.get(author_id, {})
    features = info.get("features", {})
    return {
        "name": features.get("FullName", info.get("title", "Unknown")),
        "affiliation": features.get("Affiliation", "Unknown"),
        "papers": features.get("Top Cited or Most Recent Papers", []),
    }


# ---------- graph helpers ----------
def _get_authors_within_n_hops(user_id: str, max_distance: int = 7) -> dict:
    graph = load_knowledge_graph_nx()
    visited = {user_id: 0}
    q = deque([(user_id, 0)])
    while q:
        node, depth = q.popleft()
        if depth >= max_distance:
            continue
        for nb in graph.neighbors(node) if node in graph else []:
            if nb not in visited or depth + 1 < visited[nb]:
                visited[nb] = depth + 1
                q.append((nb, depth + 1))
    visited.pop(user_id, None)
    return visited


def _get_hops(a: str, b: str) -> int:
    try:
        return int(nx.shortest_path_length(load_knowledge_graph_nx(), source=a, target=b))
    except Exception:
        return -1


def _get_mutual_coauthors(a: str, b: str, n: int = 3) -> list[str]:
    graph = load_knowledge_graph_nx()
    nodes = load_author_nodes()
    try:
        a_nb = set(str(x) for x in graph.neighbors(a))
        b_nb = set(str(x) for x in graph.neighbors(b))
        mutual = a_nb & b_nb
        ranked = []
        for mid in mutual:
            feats = nodes.get(mid, {}).get("features", {})
            ranked.append((feats.get("FullName", "Unknown"), feats.get("H-index", 0)))
        ranked.sort(key=lambda x: x[1], reverse=True)
        return [name for name, _ in ranked[:n]]
    except Exception:
        return []


def _get_shortest_path(a: str, b: str) -> list[dict]:
    """Return shortest path with node details for graph visualization."""
    graph = load_knowledge_graph_nx()
    nodes = load_author_nodes()
    try:
        valid = {n for n in graph if n in nodes}
        sub = graph.subgraph(valid)
        path = nx.shortest_path(sub, source=a, target=b)
        result = []
        for nid in path:
            feats = nodes.get(nid, {}).get("features", {})
            result.append({
                "id": nid,
                "name": feats.get("FullName", "Unknown"),
                "is_source": nid == a,
                "is_target": nid == b,
            })
        return result
    except Exception:
        return []


# ---------- retrieval ----------
def _retrieve_candidates(query_text: str, user_id: str, top_k: int = 50) -> list[tuple]:
    author_ids, faiss_index = load_embeddings_and_index()
    if faiss_index is None or not author_ids:
        raise HTTPException(status_code=500, detail="Search index not available")
    retriever = Retriever(author_ids, faiss_index)
    q_emb = _model_encode(query_text)
    results = retriever.search(q_emb, 5000)

    # Deduplicate by base id
    best_by_id: dict[str, float] = {}
    for key, score in results:
        base_id = str(key).split("_")[0]
        if base_id not in best_by_id or score > best_by_id[base_id]:
            best_by_id[base_id] = float(score)

    # Network-aware weighting
    pub_counts = load_publication_counts()
    hop_info = _get_authors_within_n_hops(user_id, max_distance=7)
    exclude = {aid for aid, dist in hop_info.items() if dist < 2}
    exclude.add(user_id)

    weighted = []
    for aid, sim in best_by_id.items():
        if aid in exclude:
            continue
        dist = hop_info.get(aid, 7)
        dist_weight = 1.0 / float(dist ** 2) if dist > 0 else 0.0
        pub_weight = 0.05 if int(pub_counts.get(aid, 0)) == 1 else 1.0
        weighted_score = sim * dist_weight * pub_weight if dist_weight > 0 else 0.0
        weighted.append((aid, weighted_score))

    return sorted(weighted, key=lambda x: x[1], reverse=True)[:top_k]


# ---------- LLM helpers ----------
def _generate_query(
    user_input: str,
    user_background: str,
    past_queries: list[str] | None = None,
    prior_inputs: list[str] | None = None,
) -> tuple[str, str]:
    client = _get_openai_client()
    system_message = (
        "You are a scientific teaming assistant helping a researcher find collaborators. "
        "Generate a SHORT, FOCUSED search query for BERT-based vector retrieval.\n\n"
        "QUERY RULES:\n"
        "- MAXIMUM 3-8 words. Pick ONE specific topic, not a list.\n"
        "- Use ONLY concrete research topic keywords (e.g., 'single cell transcriptomics cancer').\n"
        "- NEVER include meta-words: 'expertise', 'gap', 'exploration', 'recent', 'collaboration', 'research on', 'application of'.\n"
        "- NEVER list multiple topics with commas. Focus on the SINGLE most important topic the user needs.\n"
        "- BAD: 'computational systems biology, network analysis, multi-omics integration, clinical data application'\n"
        "- GOOD: 'clinical translational medicine'\n"
        "- Phrase positively (no negation). BERT cannot understand 'not' or 'without'.\n"
        "- If the user specifies a topic, preserve it with minimal changes.\n\n"
        "LANGUAGE: Always write the JUSTIFICATION in English unless the user is clearly writing in another language.\n\n"
        "STRICT OUTPUT FORMAT:\n"
        "[QUERY]short focused topic keywords in English[/QUERY]\n"
        "[JUSTIFICATION]one concise sentence speaking directly to the user (use 'I' and 'you', e.g. 'I focused on X because it complements your work in Y')[/JUSTIFICATION]"
    )
    past_queries = past_queries or []
    past_block = ""
    if past_queries:
        uniq = []
        seen: set[str] = set()
        for q in reversed(past_queries):
            nq = normalize_query_text(q)
            if nq and nq not in seen:
                uniq.append(q)
                seen.add(nq)
        uniq = list(reversed(uniq[-5:]))
        past_block = "\nPreviously searched queries (do NOT repeat these):\n- " + "\n- ".join(uniq) + "\n\n"

    user_message = (
        f"The RESEARCHER's BACKGROUND:\n{user_background}\n\n"
        + past_block
        + f"The researcher's input:\n{user_input}\n\n"
        "Follow the STRICT OUTPUT FORMAT."
    )
    messages = [{"role": "system", "content": system_message}]
    for txt in (prior_inputs or [])[-20:]:
        if txt:
            messages.append({"role": "user", "content": str(txt)})
    messages.append({"role": "user", "content": user_message})

    response = client.chat.completions.create(model=MODEL_NAME_EXPERTISE, messages=messages)
    full = response.choices[0].message.content or ""

    query = full.strip()
    justification = ""
    q_match = re.search(r"\[QUERY\](.*?)(\[/QUERY\]|$)", full, re.DOTALL | re.IGNORECASE)
    if q_match:
        query = q_match.group(1).strip()
    j_match = re.search(r"\[JUSTIFICATION\](.*?)(\[/JUSTIFICATION\]|$)", full, re.DOTALL | re.IGNORECASE)
    if j_match:
        justification = j_match.group(1).strip()
    else:
        e_match = re.search(r"\[EXPLANATION\](.*?)(\[/EXPLANATION\]|$)", full, re.DOTALL | re.IGNORECASE)
        if e_match:
            justification = e_match.group(1).strip()
    return query, justification


def _is_confirmation(user_text: str, query: str, prior_inputs: list[str] | None = None) -> bool:
    client = _get_openai_client()
    system_message = (
        "You are a strict binary intent classifier for confirmation. "
        "Output exactly one word: YES or NO. "
        "Return YES if the user approves/proceeds/affirms. "
        "Return NO if they reject, ask to refine/change, ask questions, or anything uncertain."
    )
    user_message = f"Current query: {query}. User message: {user_text}\nAnswer with only YES or NO."
    messages = [{"role": "system", "content": system_message}]
    for txt in (prior_inputs or [])[-20:]:
        if txt:
            messages.append({"role": "user", "content": str(txt)})
    messages.append({"role": "user", "content": user_message})
    response = client.chat.completions.create(model=MODEL_NAME_EXPERTISE, messages=messages, max_tokens=2)
    ans = (response.choices[0].message.content or "").strip().upper()
    return ans.startswith("Y")


def _classify_and_respond(
    user_input: str,
    user_background: str,
    conversation_history: list[dict] | None = None,
    current_query: str | None = None,
    past_queries: list[str] | None = None,
    prior_inputs: list[str] | None = None,
    search_results: list[dict] | None = None,
    search_phase: str | None = None,
) -> dict:
    """Classify user intent and respond accordingly.

    Returns one of:
      {"action": "search",  "query": ..., "justification": ...}
      {"action": "confirm"}
      {"action": "chat",    "reply": ...}
    """
    client = _get_openai_client()

    # Build the system prompt with 3-way classification
    pending_block = ""
    if current_query:
        pending_block = (
            f"\n\nIMPORTANT — there is a PENDING search query the user has not yet confirmed:\n"
            f"  \"{current_query}\"\n"
            "If the user approves/confirms/agrees to proceed with this query, output CONFIRM.\n"
            "If the user wants to adjust/refine/change the query, output SEARCH with an improved query.\n"
            "If the user is just chatting or asking unrelated questions, output CHAT — "
            "the pending query will remain for them to confirm later."
        )

    past_queries = past_queries or []
    past_block = ""
    if past_queries:
        uniq = []
        seen: set[str] = set()
        for q in reversed(past_queries):
            nq = normalize_query_text(q)
            if nq and nq not in seen:
                uniq.append(q)
                seen.add(nq)
        uniq = list(reversed(uniq[-5:]))
        past_block = "\nPreviously searched queries (do NOT repeat these):\n- " + "\n- ".join(uniq)

    system_message = (
        "You are a friendly, intelligent scientific teaming assistant. "
        "You help researchers find collaborators and can also chat naturally.\n\n"
        "LANGUAGE RULE: Always respond in English by default. "
        "Only switch to another language if the user's latest message is CLEARLY written in that language.\n\n"
        "Classify the user's INTENT into one of three actions:\n\n"
        "1. SEARCH — user wants to find collaborators or refine a query.\n"
        "   Generate a SHORT query (3-8 words max, ONE focused topic, English only).\n"
        "   NEVER list multiple topics with commas. NEVER use filler words like "
        "'expertise', 'gap', 'exploration', 'collaboration'.\n"
        "   BAD: 'systems biology, network analysis, multi-omics, clinical data'\n"
        "   GOOD: 'clinical translational proteomics'\n"
        "2. CONFIRM — user approves the pending query (only when one exists).\n"
        "3. CHAT — anything else: greetings, thanks, questions, casual talk.\n\n"
        "STRICT OUTPUT FORMAT:\n"
        "[ACTION]SEARCH or CONFIRM or CHAT[/ACTION]\n"
        "If SEARCH: [QUERY]short focused English keywords, 3-8 words max[/QUERY]\n"
        "[JUSTIFICATION]one sentence speaking directly to the user (use 'I' and 'you')[/JUSTIFICATION]\n"
        "If CONFIRM: nothing else needed.\n"
        "If CHAT: [REPLY]natural response in English (or match user's language if not English)[/REPLY]\n"
        + pending_block
        + past_block
    )

    # Build search results context so the LLM can reference them
    if search_results:
        results_block = f"\n\nSEARCH RESULTS (query: \"{current_query or 'N/A'}\""
        if search_phase:
            results_block += f", status: {search_phase}"
        results_block += "):\n"
        for i, r in enumerate(search_results[:20], 1):  # cap at 20 to avoid token overflow
            name = r.get("name", "Unknown")
            affiliation = r.get("affiliation", "")
            score = r.get("score")
            justification = r.get("justification", "")
            hops = r.get("hops")
            mutual = r.get("mutual_coauthors", [])
            results_block += f"  {i}. {name}"
            if affiliation:
                results_block += f" ({affiliation})"
            if score is not None:
                results_block += f" — Score: {score}"
            if justification:
                results_block += f" — {justification}"
            if hops is not None and hops > 0:
                results_block += f" — {hops} hops away"
            if mutual:
                results_block += f" — Mutual co-authors: {', '.join(mutual[:3])}"
            results_block += "\n"
        results_block += (
            "\nWhen the user asks about these results, answer based on this data. "
            "You can reference candidates by rank number or name."
        )
        system_message += results_block
    elif search_phase:
        system_message += f"\n\nSearch status: {search_phase}. No results available yet."

    messages = [{"role": "system", "content": system_message}]

    # Add conversation history for context (last 20 messages)
    for msg in (conversation_history or [])[-20:]:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})

    # Add the current user message with background
    user_message = (
        f"The RESEARCHER's BACKGROUND:\n{user_background}\n\n"
        f"The researcher's latest message:\n{user_input}\n\n"
        "Follow the STRICT OUTPUT FORMAT."
    )
    messages.append({"role": "user", "content": user_message})

    response = client.chat.completions.create(model=MODEL_NAME_EXPERTISE, messages=messages)
    full = response.choices[0].message.content or ""

    # Parse action
    action_match = re.search(r"\[ACTION\](.*?)\[/ACTION\]", full, re.DOTALL | re.IGNORECASE)
    action_raw = (action_match.group(1).strip().upper() if action_match else "").split()[0] if action_match else ""

    # Fallback: detect action from raw output if no [ACTION] tags found
    if not action_raw:
        full_upper = full.strip().upper()
        # Check for bare CONFIRM / [CONFIRM]
        if current_query and re.search(r"\[?CONFIRM\]?", full_upper):
            action_raw = "CONFIRM"
        # Check for [QUERY] tag without [ACTION] wrapper
        elif re.search(r"\[QUERY\]", full, re.IGNORECASE):
            action_raw = "SEARCH"

    if action_raw == "CONFIRM" and current_query:
        return {"action": "confirm"}

    if action_raw == "SEARCH":
        query = full.strip()
        justification = ""
        q_match = re.search(r"\[QUERY\](.*?)(\[/QUERY\]|$)", full, re.DOTALL | re.IGNORECASE)
        if q_match:
            query = q_match.group(1).strip()
        j_match = re.search(r"\[JUSTIFICATION\](.*?)(\[/JUSTIFICATION\]|$)", full, re.DOTALL | re.IGNORECASE)
        if j_match:
            justification = j_match.group(1).strip()
        return {"action": "search", "query": query, "justification": justification}

    # Default to CHAT (also handles explicit CHAT action)
    reply = ""
    r_match = re.search(r"\[REPLY\](.*?)(\[/REPLY\]|$)", full, re.DOTALL | re.IGNORECASE)
    if r_match:
        reply = r_match.group(1).strip()
    else:
        # Fallback: use the whole response if no tags found
        reply = re.sub(r"\[ACTION\].*?\[/ACTION\]", "", full, flags=re.DOTALL | re.IGNORECASE).strip()
        # Strip any raw tags that leaked through
        reply = re.sub(r"\[/?(?:ACTION|QUERY|CONFIRM|REPLY|JUSTIFICATION)\]", "", reply, flags=re.IGNORECASE).strip()
        if not reply:
            reply = "I'm here to help you find collaborators! Feel free to describe your research interests or ask me anything."
    return {"action": "chat", "reply": reply}


def _rerank_batch(
    candidates: list[tuple], query: str, user_background: str, batch_size: int = 5
) -> list[dict]:
    client = _get_openai_client()
    batch = candidates[:batch_size]
    batch_authors = []
    for author_id, _ in batch:
        details = _get_author_details(author_id)
        info = f"Name: {details['name']}\nAffiliation: {details['affiliation']}\nPapers:"
        for p in details["papers"]:
            info += f"\n- {p.get('Title', 'Untitled')} ({p.get('Venue', '')}, {p.get('PubYear', '')}) - Cited {p.get('CitedCount', 0)} times"
        batch_authors.append((author_id, info))

    prompt = f"""
    RESEARCHER's NEEDS (Most important): {query}.
    For each candidate, provide:
    1. A score from 1.0 to 10.0 (use 0.5 increments, e.g. 7.5, 8.0, 6.5).
    2. A brief but specific justification.
    3. INSTITUTION: The candidate's institution in the format "Institution Name, Country" only.

    CANDIDATE 1's name:
    SCORE: [score]
    JUSTIFICATION: [justification]
    INSTITUTION: [Institution Name, Country]
    ...

    The researcher's BACKGROUND:
    {user_background}

    EVALUATE THE FOLLOWING CANDIDATES:
    """
    for idx, (_, info) in enumerate(batch_authors):
        prompt += f"\n\nCANDIDATE {idx + 1}:\n{info}\n"

    response = client.chat.completions.create(
        model=MODEL_NAME_RERANKING,
        messages=[
            {
                "role": "system",
                "content": f"You are an academic collaboration expert. The researcher's needs: {prompt}",
            }
        ],
    )
    full = response.choices[0].message.content or ""
    results: list[dict] = []
    current_score = 0.0
    current_just = ""
    current_inst = ""
    current_idx = -1
    for line in full.split("\n"):
        text = line.strip()
        if text.startswith("CANDIDATE"):
            if current_idx >= 0 and current_idx < len(batch_authors):
                results.append(
                    {
                        "author_id": batch_authors[current_idx][0],
                        "score": current_score,
                        "justification": current_just,
                        "institution": current_inst,
                    }
                )
            current_idx += 1
            current_score, current_just, current_inst = 0.0, "", ""
        elif text.upper().startswith("SCORE:"):
            try:
                current_score = float(text.split(":", 1)[1].strip())
            except Exception:
                current_score = 5.0
        elif text.upper().startswith("JUSTIFICATION:"):
            current_just = text.split(":", 1)[1].strip()
        elif text.upper().startswith("INSTITUTION:"):
            current_inst = text.split(":", 1)[1].strip()
    if current_idx >= 0 and current_idx < len(batch_authors):
        results.append(
            {
                "author_id": batch_authors[current_idx][0],
                "score": current_score,
                "justification": current_just,
                "institution": current_inst,
            }
        )
    return results


# ====================== API ENDPOINTS ======================


class AuthorResponse(BaseModel):
    id: str
    name: str
    affiliation: str
    greeting: str


@app.get("/api/author/{aid}")
async def get_author(aid: str):
    nodes = load_author_nodes()
    if aid not in nodes:
        raise HTTPException(status_code=404, detail="Author not found")
    name = _get_user_name(aid)
    feats = nodes[aid].get("features", {})
    return AuthorResponse(
        id=aid,
        name=name,
        affiliation=feats.get("Affiliation", "Unknown"),
        greeting=f"Hi {name}! What kind of collaborators are you looking for?",
    )


class GenerateQueryRequest(BaseModel):
    aid: str
    user_input: str
    current_query: str | None = None
    past_queries: list[str] = []
    prior_inputs: list[str] = []


class GenerateQueryResponse(BaseModel):
    query: str
    justification: str


@app.post("/api/generate-query")
async def generate_query(req: GenerateQueryRequest):
    user_bg = _get_user_background(req.aid)
    if req.current_query:
        improved = (
            f"Original query: {req.current_query}\n"
            f"User feedback (Most important, follow it as much as possible): {req.user_input}"
        )
        query, justification = _generate_query(improved, user_bg, req.past_queries, req.prior_inputs)
    else:
        query, justification = _generate_query(req.user_input, user_bg, req.past_queries, req.prior_inputs)

    # Deduplicate check
    existing_norm = {normalize_query_text(q) for q in req.past_queries}
    attempts = 0
    while normalize_query_text(query) in existing_norm and attempts < 2:
        reinforce = (
            (f"Original query: {req.current_query}\n" if req.current_query else "")
            + f"User feedback: {req.user_input}\nAvoid repeating any previous queries."
        )
        query, justification = _generate_query(reinforce, user_bg, req.past_queries, req.prior_inputs)
        attempts += 1

    return GenerateQueryResponse(query=query, justification=justification)


class ConfirmRequest(BaseModel):
    user_text: str
    current_query: str
    prior_inputs: list[str] = []


@app.post("/api/check-confirmation")
async def check_confirmation(req: ConfirmRequest):
    confirmed = _is_confirmation(req.user_text, req.current_query, req.prior_inputs)
    return {"confirmed": confirmed}


# ---------- Unified chat endpoint (intent-aware) ----------
class ChatMessageItem(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    aid: str
    user_input: str
    conversation_history: list[ChatMessageItem] = []
    current_query: str | None = None
    past_queries: list[str] = []
    prior_inputs: list[str] = []
    search_results: list[dict] = []
    search_phase: str | None = None


@app.post("/api/chat")
async def chat(req: ChatRequest):
    """Unified chat endpoint with 3-way intent classification."""
    user_bg = _get_user_background(req.aid)
    history_dicts = [{"role": m.role, "content": m.content} for m in req.conversation_history]

    result = _classify_and_respond(
        user_input=req.user_input,
        user_background=user_bg,
        conversation_history=history_dicts,
        current_query=req.current_query,
        past_queries=req.past_queries,
        prior_inputs=req.prior_inputs,
        search_results=req.search_results or None,
        search_phase=req.search_phase,
    )

    # Dedup check for search queries
    if result["action"] == "search":
        existing_norm = {normalize_query_text(q) for q in req.past_queries}
        attempts = 0
        while normalize_query_text(result["query"]) in existing_norm and attempts < 2:
            result = _classify_and_respond(
                user_input=req.user_input + "\nAvoid repeating any previous queries.",
                user_background=user_bg,
                conversation_history=history_dicts,
                current_query=req.current_query,
                past_queries=req.past_queries,
                prior_inputs=req.prior_inputs,
                search_results=req.search_results or None,
                search_phase=req.search_phase,
            )
            if result["action"] != "search":
                break
            attempts += 1

    return result


class SearchRequest(BaseModel):
    aid: str
    query: str
    top_k: int = 100


@app.post("/api/search")
async def search(req: SearchRequest):
    candidates = _retrieve_candidates(req.query, req.aid, req.top_k)
    results = []
    for author_id, score in candidates:
        details = _get_author_details(author_id)
        results.append(
            {
                "author_id": author_id,
                "retrieval_score": float(score),
                "name": details["name"],
                "affiliation": details["affiliation"],
            }
        )
    return {"candidates": results, "total": len(results)}


class RerankRequest(BaseModel):
    aid: str
    query: str
    candidates: list[dict]  # [{author_id, retrieval_score}]


@app.post("/api/rerank")
async def rerank(req: RerankRequest):
    """Rerank candidates via LLM. Returns Server-Sent Events for progressive updates."""
    user_bg = _get_user_background(req.aid)
    candidate_tuples = [(c["author_id"], c["retrieval_score"]) for c in req.candidates]

    async def event_generator():
        MAX_CONCURRENT = 5
        BATCH_SIZE = 2
        batches = [
            candidate_tuples[i : i + BATCH_SIZE]
            for i in range(0, len(candidate_tuples), BATCH_SIZE)
        ]
        total = len(batches)
        done = 0
        all_results: list[dict] = []

        # Process in waves of MAX_CONCURRENT
        import asyncio
        loop = asyncio.get_running_loop()

        for wave_start in range(0, total, MAX_CONCURRENT):
            wave = batches[wave_start : wave_start + MAX_CONCURRENT]
            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as executor:
                tasks = [
                    loop.run_in_executor(executor, _rerank_batch, batch, req.query, user_bg, len(batch))
                    for batch in wave
                ]
                for coro in asyncio.as_completed(tasks):
                    try:
                        batch_results = await coro
                        all_results.extend(batch_results)
                        done += 1
                        # Enrich each result with graph info
                        for r in batch_results:
                            aid = r["author_id"]
                            details = _get_author_details(aid)
                            r["name"] = details["name"]
                            r["affiliation"] = details["affiliation"]
                            r["papers"] = details["papers"]
                            r["hops"] = _get_hops(req.aid, aid)
                            r["mutual_coauthors"] = _get_mutual_coauthors(req.aid, aid)
                        yield {
                            "event": "batch",
                            "data": json.dumps(
                                {
                                    "results": batch_results,
                                    "progress": {"done": done, "total": total},
                                }
                            ),
                        }
                    except Exception as e:
                        yield {
                            "event": "error",
                            "data": json.dumps({"error": str(e)}),
                        }

        # Final sorted results
        all_results.sort(key=lambda x: x.get("score", 0), reverse=True)
        yield {
            "event": "complete",
            "data": json.dumps({"results": all_results}),
        }

    return EventSourceResponse(event_generator())


@app.get("/api/graph-path/{aid}/{collaborator_id}")
async def graph_path(aid: str, collaborator_id: str):
    path = _get_shortest_path(aid, collaborator_id)
    return {"path": path, "hops": len(path) - 1 if path else -1}


@app.get("/api/author/{aid}/details")
async def author_details(aid: str):
    details = _get_author_details(aid)
    if details["name"] == "Unknown":
        raise HTTPException(status_code=404, detail="Author not found")
    return details


# ---------- health ----------
@app.get("/api/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
