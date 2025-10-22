import os
import sys
import json
import time
import concurrent.futures
from typing import List, Tuple
import re

import numpy as np
import streamlit as st
from streamlit_agraph import agraph, Node, Edge, Config
import networkx as nx

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_loader import (
    load_author_nodes,
    load_knowledge_graph_nx,
    load_embeddings_and_index,
    load_specter_model,
    load_publication_counts,
)
from utils.retriever import Retriever


# Page config
st.set_page_config(layout="wide", initial_sidebar_state="collapsed")


# Fixed model names
MODEL_NAME_EXPERTISE = "gpt-4.1"
MODEL_NAME_RERANKING = "gpt-4.1-mini"

# Progressive loading (show UI only on first render to avoid flicker on first input)
status = st.empty()
show_bootstrap_ui = not st.session_state.get("cm4ai_bootstrap_done", False)
if show_bootstrap_ui:
    status.info("Loading resources…")

author_nodes = load_author_nodes()
author_n_publications = load_publication_counts()
author_knowledge_graph_nx = load_knowledge_graph_nx()
author_ids, faiss_index = load_embeddings_and_index()
tokenizer, model = load_specter_model()  # may be (None, None)

if faiss_index is None or not author_ids:
    st.error("Search index not available. Please provide local data under `cm4ai_demo/data`. See README.")
    st.stop()

# Ensure encoder is available
if tokenizer is None or model is None:
    st.error("Embedding model not available (SPECTER failed to load). Please configure S3 or allow HuggingFace download and restart.")
    st.stop()

retriever = Retriever(author_ids, faiss_index)
status.empty()
if show_bootstrap_ui:
    st.session_state["cm4ai_bootstrap_done"] = True


@st.cache_resource
def get_openai_client():
    from openai import OpenAI
    key = st.secrets.get("openai_key")
    if not key:
        st.error("OpenAI API key missing in secrets. Set 'openai_key' in .streamlit/secrets.toml.")
        st.stop()
    return OpenAI(api_key=key)


def normalize_query_text(text: str) -> str:
    try:
        s = str(text or "")
        s = re.sub(r"\[/?QUERY\]", "", s, flags=re.IGNORECASE)
        s = s.replace("**", "")
        s = " ".join(s.split()).strip().lower()
        return s
    except Exception:
        return (text or "").strip().lower()


def model_encode(query_text: str) -> np.ndarray:
    if tokenizer is None or model is None:
        st.error("Embedding model not available (SPECTER failed to load). Please configure S3 or allow HuggingFace download and restart.")
        st.stop()
    import torch
    inputs = tokenizer([str(query_text)], padding=True, truncation=True,
                       return_tensors="pt", return_token_type_ids=False, max_length=512)
    with torch.no_grad():
        output = model(**inputs)
        embeddings = output.last_hidden_state[:, 0, :].cpu().detach().numpy().astype(np.float32)
    return embeddings


# One-time warmup to avoid first-interaction lag
if "cm4ai_warmup_done" not in st.session_state:
    with st.spinner("Warming up resources…"):
        try:
            _ = get_openai_client()
        except Exception:
            pass
        try:
            _ = model_encode("warmup")
        except Exception:
            pass
    st.session_state["cm4ai_warmup_done"] = True


def get_user_background(user_id: str) -> str:
    user = author_nodes.get(user_id, {})
    features = user.get("features", {})
    name = features.get("FullName", user.get("title", "Unknown"))
    affiliation = features.get("Affiliation", "Unknown")
    papers = features.get("Top Cited or Most Recent Papers", [])
    bg = f"Name: {name}\nAffiliation: {affiliation}\nTop Cited or Most Recent Papers:\n"
    for p in papers:
        title = p.get("Title", "Untitled")
        venue = p.get("Venue", "")
        year = p.get("PubYear", "")
        cited = p.get("CitedCount", 0)
        bg += f"- {title} ({venue}, {year}) - Cited {cited} times\n"
    return bg


def get_user_name(user_id: str) -> str:
    user = author_nodes.get(user_id, {})
    features = user.get("features", {})
    return features.get("FullName", user.get("title", "Researcher"))


def try_generate_query(user_input: str, user_background: str, model_name: str, past_queries: List[str] = None) -> Tuple[str, str, str]:
    """Return (query, explanation, full_response) using strict tagged output from the LLM."""
    client = get_openai_client()
    system_message = (
    "Speak in a 'you' voice, as if you are directly talking to the researcher. "
    "Your task is to analyze the researcher's input to identify a specific expertise gap. "
    "Generate a short, specific expertise query focused on that gap for retrieval in a BERT-based vector database. "
    "Focus on the researcher's input, consider the researcher's previous publications and their input, "
    "and give the most accurate, content-based query possible."
    "BERT-based retrieval cannot understand negation or exclusion (e.g., 'not', 'without', 'exclude'), and some words like 'recent', 'exploration', 'expertise', are not helpful for the semantic-based retrieval. "
    "If the researcher gives a negative or exclusive query (e.g., 'medicine but not AI'), "
    "rephrase it into a positive form that naturally avoids the undesired topic "
    "(e.g., 'traditional clinical medicine'). "
    "Your query should thus always be phrased in a positive, descriptive way that conveys the intended scope clearly. "
    "If the researcher already specifies or implies a query, preserve it with minimal proofreading and only rephrase for clarity and non-negativity. "
    "Most importantly, the user's input is top priority, modify your query to align with the user's most recent input as much as possible. e.g., if the user ask if we can just serach 'medicine', do it even if it is not ideal. Instruction following is top priority. "
    "STRICT OUTPUT FORMAT: Output exactly two sections in order: [QUERY] the final query text with the expertise gap in bold [/QUERY]\n"
    "[JUSTIFICATION] In first person (use 'I'), one concise sentence explaining why this query best matches the user's needs, focusing on the bolded gap. Refer the researcher as 'you'. [/JUSTIFICATION]"
)
    past_queries = past_queries or []
    if past_queries:
        uniq = []
        seen = set()
        for q in reversed(past_queries):
            nq = normalize_query_text(q)
            if nq and nq not in seen:
                uniq.append(q)
                seen.add(nq)
        uniq = list(reversed(uniq[-5:]))
        past_block = "\nPreviously searched queries (do NOT repeat these):\n- " + "\n- ".join(uniq) + "\n\n"
    else:
        past_block = ""
    user_message = (
        f"The RESEARCHER's BACKGROUND:\n{user_background}\n\n"
        + past_block +
        f"The researcher's input (can be a gap, a need, or a general description of their research interests, or can be totaly unrelated):\n{user_input}\n\n"
        "Follow the STRICT OUTPUT FORMAT. Output exactly [QUERY]...[/QUERY] and [JUSTIFICATION]...[/JUSTIFICATION] and nothing else."
    )
    # Include prior raw user inputs as separate user-role messages for consistency
    messages_list = [
        {"role": "system", "content": system_message}
    ]
    try:
        prior_inputs = st.session_state.get("user_inputs", [])
        # send only the most recent 20 to control token usage
        for txt in prior_inputs[-20:]:
            if txt:
                messages_list.append({"role": "user", "content": str(txt)})
    except Exception:
        pass
    messages_list.append({"role": "user", "content": user_message})

    response = client.chat.completions.create(
        model=model_name,
        messages=messages_list,
    )
    full_response = response.choices[0].message.content or ""
    # Parse [JUSTIFICATION]/[EXPLANATION] and [QUERY] sections
    explanation = ""
    query = full_response.strip()
    try:
        q_match = re.search(r"\[QUERY\](.*?)(\[/QUERY\]|$)", full_response, re.DOTALL | re.IGNORECASE)
        if q_match:
            query = q_match.group(1).strip()
        j_match = re.search(r"\[JUSTIFICATION\](.*?)(\[/JUSTIFICATION\]|$)", full_response, re.DOTALL | re.IGNORECASE)
        if j_match:
            explanation = j_match.group(1).strip()
        else:
            # Backward compatibility if a model returns [EXPLANATION]
            e_match = re.search(r"\[EXPLANATION\](.*?)(\[/EXPLANATION\]|$)", full_response, re.DOTALL | re.IGNORECASE)
            if e_match:
                explanation = e_match.group(1).strip()
    except Exception:
        pass
    return query, (explanation or ""), full_response


def llm_is_confirmation(user_text: str, query: str, model_name: str) -> bool:
    """Use LLM to judge if the user approves proceeding. Returns True for approve/yes."""
    client = get_openai_client()
    try:
        system_message = (
            "You are a strict binary intent classifier for confirmation, judge if the user decides to proceed with the query. "
            "Output exactly one word: YES or NO. "
            "Return YES if the user approves/proceeds/affirms (including typos or multilingual equivalents). "
            "Return NO if they reject, ask to refine/change, ask questions, or anything uncertain, or if they are not sure about the query. only return yes if you are very sure about the user's intent."
        )
        user_message = f"Current query: {query}. User message: {user_text}\nAnswer with only YES or NO."
        messages_list = [
            {"role": "system", "content": system_message}
        ]
        try:
            prior_inputs = st.session_state.get("user_inputs", [])
            for txt in prior_inputs[-20:]:
                if txt:
                    messages_list.append({"role": "user", "content": str(txt)})
        except Exception:
            pass
        messages_list.append({"role": "user", "content": user_message})
        response = client.chat.completions.create(
            model=model_name,
            messages=messages_list,
            max_tokens=2,
        )
        ans = (response.choices[0].message.content or "").strip().upper()
        return ans.startswith("Y")
    except Exception:
        return False


def get_authors_within_n_hops_with_distances(user_id: str, max_distance: int = 7) -> dict:
    visited = {user_id: 0}
    from collections import deque
    q = deque([(user_id, 0)])
    while q:
        node, depth = q.popleft()
        if depth >= max_distance:
            continue
        for nb in author_knowledge_graph_nx.neighbors(node) if node in author_knowledge_graph_nx else []:
            if nb not in visited or depth + 1 < visited[nb]:
                visited[nb] = depth + 1
                q.append((nb, depth + 1))
    visited.pop(user_id, None)
    return visited


def retrieve_candidates(query_text: str, user_id: str, top_k: int = 50) -> List[Tuple[str, float]]:
    q_emb = model_encode(query_text)
    D, I = retriever.index.search(q_emb, 5000)
    original_indices = np.array(retriever.doc_lookup)[I].tolist()[0]
    scores = D[0]
    # Deduplicate by base id (strip variant suffix _idx)
    best_by_id = {}
    for key, score in zip(original_indices, scores):
        base_id = str(key).split("_")[0]
        if base_id not in best_by_id or score > best_by_id[base_id]:
            best_by_id[base_id] = float(score)
    # Network-aware weighting + publication count alignment with CM4AI
    hop_info = get_authors_within_n_hops_with_distances(user_id, max_distance=7)
    exclude = {aid for aid, dist in hop_info.items() if dist < 2}
    exclude.add(user_id)
    weighted = []
    for aid, sim in best_by_id.items():
        if aid in exclude:
            continue
        dist = hop_info.get(aid, 7)
        dist_weight = 1.0 / float(dist ** 2) if dist > 0 else 0.0
        pub_weight = 1.0
        try:
            n_pubs = int(author_n_publications.get(aid, 0))
            if n_pubs == 1:
                pub_weight = 0.05
        except Exception:
            pub_weight = 1.0
        weighted_score = sim * dist_weight * pub_weight if dist_weight > 0 else 0.0
        weighted.append((aid, weighted_score))
    items = sorted(weighted, key=lambda x: x[1], reverse=True)[:top_k]
    return items


def get_node_title(node: dict) -> str:
    return node.get("title", node.get("features", {}).get("FullName", "Unknown"))


def get_author_details(author_id: str) -> dict:
    info = author_nodes.get(author_id, {})
    features = info.get('features', {})
    return {
        'name': features.get('FullName', info.get('title', 'Unknown')),
        'affiliation': features.get('Affiliation', 'Unknown'),
        'Top Cited or Most Recent Papers': features.get('Top Cited or Most Recent Papers', []),
    }


def get_hops_between_authors(author_id: str, collaborator_id: str) -> int:
    try:
        return int(nx.shortest_path_length(author_knowledge_graph_nx, source=author_id, target=collaborator_id))
    except Exception:
        return -1


def get_mutual_co_authors(author_id: str, collaborator_id: str, n_co_authors: int = 3) -> List[str]:
    try:
        a_neighbors = set(str(x) for x in author_knowledge_graph_nx.neighbors(author_id))
        b_neighbors = set(str(x) for x in author_knowledge_graph_nx.neighbors(collaborator_id))
        mutual = a_neighbors.intersection(b_neighbors)
        ranked = []
        for mid in mutual:
            feats = author_nodes.get(mid, {}).get('features', {})
            name = feats.get('FullName', 'Unknown')
            hidx = feats.get('H-index', 0)
            ranked.append((name, hidx))
        ranked.sort(key=lambda x: x[1], reverse=True)
        return [name for name, _ in ranked[:n_co_authors]]
    except Exception:
        return []


def create_local_knowledge_graph(selected_author_id: str, top_collaborators: List[str]):
    nodes, edges = [], []
    valid_nodes = {n for n in author_knowledge_graph_nx if n in author_nodes}
    if selected_author_id in valid_nodes:
        sel_name = author_nodes[selected_author_id]['features'].get('FullName', 'Unknown') if selected_author_id in author_nodes else selected_author_id
        nodes.append(Node(id=selected_author_id, label=sel_name, size=40, color="lightblue"))
    else:
        return nodes, edges
    for cid in top_collaborators:
        if cid and cid in valid_nodes:
            try:
                sub = author_knowledge_graph_nx.subgraph(valid_nodes)
                path = nx.shortest_path(sub, source=selected_author_id, target=cid)
            except Exception:
                continue
            for nid in path:
                if nid in author_nodes and not any(n.id == nid for n in nodes):
                    nm = author_nodes[nid]['features'].get('FullName', 'Unknown')
                    color = (
                        "lightblue" if nid == selected_author_id else
                        "lightgreen" if nid == cid else
                        "orange" if (nid in set(author_knowledge_graph_nx.neighbors(selected_author_id)) and nid in set(author_knowledge_graph_nx.neighbors(cid))) else
                        "yellow" if nid in set(author_knowledge_graph_nx.neighbors(selected_author_id)) else
                        "purple"
                    )
                    size = 30 if nid == cid else 20
                    nodes.append(Node(id=nid, label=nm, size=size, color=color))
            for i in range(len(path) - 1):
                source_node = path[i]
                target_node = path[i + 1]
                hop_distance_to_collaborator = len(path) - 1
                edge_label = "Co-Author" if (hop_distance_to_collaborator == 2 and
                                              source_node in set(author_knowledge_graph_nx.neighbors(selected_author_id)) and
                                              target_node in set(author_knowledge_graph_nx.neighbors(cid))) else "Connection"
                edges.append(Edge(source=source_node, target=target_node, label=edge_label))
    return nodes, edges


def visualize_knowledge_graph(selected_author_id: str, top_collaborators: List[str]):
    nodes, edges = create_local_knowledge_graph(selected_author_id, top_collaborators)
    config = Config(width=400, height=300, directed=False, nodeHighlightBehavior=False, highlightColor="#F7A7A6", collapsible=False, physics=False)
    return agraph(nodes=nodes, edges=edges, config=config)


def rerank_authors_with_llm_batch(candidates: List[Tuple[str, float]], query: str, user_background: str, model_name: str, batch_size: int = 5) -> List[Tuple[str, float, str]]:
    """Use LLM to rerank candidates; returns list of (author_id, score, justification, cleaned_institution)."""
    client = get_openai_client()

    batch = candidates[0:batch_size]
    batch_authors = []
    for author_id, _ in batch:
        details = get_author_details(author_id)
        author_info = f"Name: {details['name']}\nAffiliation: {details['affiliation']}\nPapers:"
        for paper in details['Top Cited or Most Recent Papers']:
            title = paper.get('Title', 'Untitled')
            venue = paper.get('Venue', '')
            year = paper.get('PubYear', '')
            cited_count = paper.get('CitedCount', 0)
            author_info += f"\n- {title} ({venue}, {year}) - Cited {cited_count} times"
        batch_authors.append((author_id, author_info))

    prompt = f"""
    RESEARCHER's NEEDS (Most important): {query}. Yes, the candidates must meet the needs in every aspect in the query ' {query}'.
    \n\n\n
For each candidate, provide:

1. A numerical score (integer) from 50 to 100. A score of 50 means not relevant at all, and 100 means perfectly relevant. Avoid using numbers ending in 5 or 0. Only give a score above 90 when you feel it very suitable for the reseracher's needs interms of every aspect of the needs. e.g., if the user mention it should be located in US, give a score above 90 only if the candidate is from US. You can give a score above 80 if some aspects match the needs but not all.
2. A brief but specific justification that use specific detail to highlight how the candidate can benefit or can not benefit the current researcher by mitigating the identified expertise gap. (one sentence). Give evidence. If there are severe points that the candidate do not meet the needs, say it in the justification. Everything should be focused on address the provided or not provided expertise gap and nothing else. Use markdown bold to highlight the relevant expertise gap. Always use the candidate's name in stead of "the candidate" or "the researcher". If can not tell the gender, use "they".
3. INSTITUTION: The candidate's institution in the format "Institution Name (try to include the department name if possible), Country" only. Do not include departments, cities, states, postal codes, or extra punctuation. If uncertain, best-effort guess.

    CANDIDATE 1's name:
    SCORE: [score]
    JUSTIFICATION: [justification]
    INSTITUTION: [Institution Name, Country]
    
    CANDIDATE 2's name:
    SCORE: [score]
    JUSTIFICATION: [justification]
    INSTITUTION: [Institution Name, Country]
    ...

    The researcher's RESEARCHER BACKGROUND:
    {user_background}
    
    EVALUATE THE FOLLOWING Candidates COLLABORATORS:
    """
    for idx, (_, info) in enumerate(batch_authors):
        prompt += f"\n\nCANDIDATE {idx + 1}:\n{info}\n"

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": f"system", "content": f"You are an academic collaboration expert specialized in evaluating potential research partnerships. Speak in a 'you' voice, as if you are directly addressing the researcher. The researcher's needs: {prompt}"},
        ],
    )

    full = response.choices[0].message.content or ""
    results: List[Tuple[str, float, str, str]] = []
    current_score: float = 0.0
    current_just: str = ""
    current_inst: str = ""
    current_idx = -1
    for line in full.split('\n'):
        text = line.strip()
        if text.startswith('CANDIDATE'):
            if current_idx >= 0 and current_idx < len(batch_authors):
                results.append((batch_authors[current_idx][0], current_score, current_just, current_inst))
            current_idx += 1
            current_score, current_just, current_inst = 0.0, "", ""
        elif text.upper().startswith('SCORE:'):
            try:
                current_score = float(text.split(':', 1)[1].strip())
            except Exception:
                current_score = 50.0
        elif text.upper().startswith('JUSTIFICATION:'):
            current_just = text.split(':', 1)[1].strip()
        elif text.upper().startswith('INSTITUTION:'):
            current_inst = text.split(':', 1)[1].strip()
    if current_idx >= 0 and current_idx < len(batch_authors):
        results.append((batch_authors[current_idx][0], current_score, current_just, current_inst))
    return results


# --- Session state ---
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "I am a scientific teaming assistant. How can I help you find collaborators based on your teaming needs?"}
    ]
if "user_inputs" not in st.session_state:
    st.session_state["user_inputs"] = []
if "search_query" not in st.session_state:
    st.session_state["search_query"] = ""
if "aid" not in st.session_state:
    # Default user/author id for testing
    st.session_state["aid"] = "1000000010"
if "query_confirmed" not in st.session_state:
    st.session_state["query_confirmed"] = False
if "initial_results" not in st.session_state:
    st.session_state["initial_results"] = []
if "reranked_results" not in st.session_state:
    st.session_state["reranked_results"] = []
if "search_complete" not in st.session_state:
    st.session_state["search_complete"] = False
if "reranking_complete" not in st.session_state:
    st.session_state["reranking_complete"] = False
if "reranking_start" not in st.session_state:
    st.session_state["reranking_start"] = False
if "reranked_candidates_number" not in st.session_state:
    st.session_state["reranked_candidates_number"] = 0
if "rerank_total_batches" not in st.session_state:
    st.session_state["rerank_total_batches"] = 0
if "rerank_batches_done" not in st.session_state:
    st.session_state["rerank_batches_done"] = 0
if "pending_generate_query" not in st.session_state:
    st.session_state["pending_generate_query"] = False
if "pending_prompt" not in st.session_state:
    st.session_state["pending_prompt"] = ""
if "executed_queries" not in st.session_state:
    st.session_state["executed_queries"] = []
if "query_to_results" not in st.session_state:
    st.session_state["query_to_results"] = {}
if "proposed_queries" not in st.session_state:
    st.session_state["proposed_queries"] = []
# --- Early processing (pre-layout) to avoid duplicate layout on first input ---
if st.session_state.get("pending_generate_query", False):
    placeholder = st.empty()
    placeholder.info("Drafting a query based on your needs…")
    try:
        user_id = st.session_state.get('aid', "1000000010")
        user_bg = get_user_background(user_id)
        base_prompt = st.session_state.get("pending_prompt", "")
        if st.session_state.get("search_query") and not st.session_state.get("search_complete", False):
            improved = f"Original query: {st.session_state['search_query']}\nUser feedback (Most important, follow it as much as possible): {base_prompt}"
            past_qs = (st.session_state.get("executed_queries", [])
                       + st.session_state.get("proposed_queries", []))
            query, justification, _ = try_generate_query(improved, user_bg, MODEL_NAME_EXPERTISE, past_qs)
        else:
            past_qs = (st.session_state.get("executed_queries", [])
                       + st.session_state.get("proposed_queries", []))
            query, justification, _ = try_generate_query(base_prompt, user_bg, MODEL_NAME_EXPERTISE, past_qs)

        # Ensure the proposed query is not a duplicate of prior proposed/executed queries
        try:
            existing_norm = {normalize_query_text(q) for q in past_qs}
            attempts = 0
            while normalize_query_text(query) in existing_norm and attempts < 2:
                reinforce = (
                    (f"Original query: {st.session_state.get('search_query','')}\n" if st.session_state.get('search_query') else "") +
                    f"User feedback (Most important, follow it as much as possible): {base_prompt}\n" +
                    "Avoid repeating any previous queries. Propose a distinct angle."
                )
                query, justification, _ = try_generate_query(reinforce, user_bg, MODEL_NAME_EXPERTISE, past_qs)
                attempts += 1
        except Exception:
            pass
        st.session_state["search_query"] = query
        # Track proposed queries to avoid repeats on refinement
        try:
            if query:
                # Append only if not a normalized duplicate
                proposed_norm = [normalize_query_text(q) for q in st.session_state.get("proposed_queries", [])]
                if normalize_query_text(query) not in proposed_norm:
                    st.session_state["proposed_queries"].append(query)
                    if len(st.session_state["proposed_queries"]) > 30:
                        st.session_state["proposed_queries"] = st.session_state["proposed_queries"][-30:]
        except Exception:
            pass
        st.session_state["query_confirmed"] = False
        st.session_state["initial_results"] = []
        st.session_state["reranked_results"] = []
        st.session_state["search_complete"] = False
        st.session_state["reranking_complete"] = False
        st.session_state["reranking_start"] = False
        st.session_state["reranked_candidates_number"] = 0
        st.session_state["rerank_total_batches"] = 0
        st.session_state["rerank_batches_done"] = 0
        msg_content = f"Based on your request, I propose this search query:\n\n**{query}**"
        try:
            if justification:
                msg_content += f"\n\n{justification}"
        except Exception:
            pass
        msg_content += "\n\nWould you like me to proceed with this query, or would you like to refine it further? Respond with 'Proceed' or provide feedback to improve the query."
        st.session_state.messages.append({
            "role": "assistant",
            "content": msg_content
        })
    except Exception as e:
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"⚠️ Could not prepare search: {e}"
        })
    finally:
        st.session_state["pending_generate_query"] = False
        st.session_state["pending_prompt"] = ""
        placeholder.empty()
    st.rerun()

# --- URL params (aid) ---
def _get_query_params():
    try:
        return dict(st.query_params)
    except Exception:
        try:
            return st.experimental_get_query_params() or {}
        except Exception:
            return {}

qp = _get_query_params()
aid_from_qs = None
if isinstance(qp, dict):
    # st.query_params returns Mapping[str, str], experimental_get_query_params returns Mapping[str, List[str]]
    if 'aid' in qp:
        aid_val = qp['aid']
        if isinstance(aid_val, list):
            aid_from_qs = aid_val[0]
        else:
            aid_from_qs = aid_val
if aid_from_qs:
    st.session_state['aid'] = str(aid_from_qs)


# --- Layout ---
col_left, col_right = st.columns([1, 1])

with col_left:
    user_id_for_title = st.session_state.get('aid', "1000000010")
    display_name = get_user_name(user_id_for_title)
    st.title(f"Hello {display_name}! Let me know what are your teaming needs!")
    st.page_link("https://cm4aikg.vercel.app/", label="Click to Explore CM4AI Knowledge Graph", icon="🌎")
    # Chat transcript
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).write(msg["content"])

with col_right:
    if st.session_state.get("search_query"):
        st.write("## Top Recommended Collaborators")
        st.write(f"*Based on query: {st.session_state.get('search_query', '')}*")
        # Precompute hop distances from the current user to all reachable authors once per render
        current_user_id_for_right = st.session_state.get('aid', "1000000010")
        hop_info_display = get_authors_within_n_hops_with_distances(current_user_id_for_right, max_distance=7)
        # Inline reranking progress bar under the header
        if (
            st.session_state.get("search_complete", False)
            and st.session_state.get("reranking_start", False)
            and not st.session_state.get("reranking_complete", False)
        ):
            total_candidates = len(st.session_state.get("initial_results", []))
            done_candidates = st.session_state.get("reranked_candidates_number", 0)
            if total_candidates > 0:
                pct = int(100 * done_candidates / total_candidates)
                st.write(f"Reranking… {done_candidates}/{total_candidates}")
                st.progress(min(pct, 100))

        reranked_list = st.session_state.get("reranked_results", [])
        initial_list = st.session_state.get("initial_results", [])

        # Always render the full initial candidate list. As reranking results arrive,
        # enrich each item's score/justification/institution in place.
        if initial_list:
            reranked_map = {}
            for item in reranked_list:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    rid = item[0]
                    score = float(item[1])
                    justification = item[2] if len(item) > 2 else ""
                    cleaned_inst = item[3] if len(item) > 3 else ""
                    reranked_map[rid] = (score, justification, cleaned_inst)

            # Build base score lookup and compute an ordered list:
            base_score_map = {rid: base for rid, base in initial_list}
            # Reranked first (by LLM score desc), then remaining (original retrieval order)
            reranked_order = sorted(
                [(rid, reranked_map[rid][0]) for rid, _ in initial_list if rid in reranked_map],
                key=lambda x: x[1],
                reverse=True,
            )
            not_reranked_order = [rid for rid, _ in initial_list if rid not in reranked_map]
            ordered_ids = [rid for rid, _ in reranked_order] + not_reranked_order

            # Filter out candidates with score below 60
            filtered_ordered_ids = []
            for rid in ordered_ids:
                if rid in reranked_map:
                    score = reranked_map[rid][0]
                    if score >= 65:  # Only show candidates with score >= 60
                        filtered_ordered_ids.append(rid)
                else:
                    # For non-reranked candidates, show them (they don't have LLM scores yet)
                    filtered_ordered_ids.append(rid)

            for i, rid in enumerate(filtered_ordered_ids):
                details = get_author_details(rid)

                # Prefer cleaned institution if available and valid; otherwise fallback to stored affiliation
                aff_fallback = details.get('affiliation', 'Unknown')
                cleaned_inst = ""
                if rid in reranked_map:
                    cleaned_inst = reranked_map[rid][2]
                aff_display = aff_fallback
                try:
                    inst_trim = (cleaned_inst or "").strip()
                    if inst_trim and ("," in inst_trim) and (inst_trim.count(",") <= 2) and (len(inst_trim) <= 100):
                        aff_display = inst_trim
                except Exception:
                    aff_display = aff_fallback

                if rid in reranked_map:
                    score = reranked_map[rid][0]
                    header = f"{i+1}. {details['name']} ({aff_display}) - Score: {int(score)}"
                else:
                    base_score = base_score_map.get(rid, 0)
                    header = f"{i+1}. {details['name']} ({aff_display}) - Retrieval score: {int(base_score)}"

                with st.expander(header, expanded=i < 3):
                    if rid in reranked_map:
                        justification = reranked_map[rid][1]
                        if justification:
                            st.write("**Why this match may work:**")
                            st.write(justification)
                    else:
                        st.write("Reranking pending…")

                    mutual = get_mutual_co_authors(current_user_id_for_right, rid)
                    mutual_str = ", ".join(mutual) if mutual else "No mutual co-authors found."
                    st.write(f"**Mutual Co-Authors:** {mutual_str}")
                    hops = hop_info_display.get(rid, -1)
                    st.write(f"**Distance within the Co-Authorship Network:** {hops}")
                    if st.button('show shortest path', key=f"show_path_{i}"):
                        visualize_knowledge_graph(current_user_id_for_right, [rid])
                    st.write("**Selected Publications:**")
                    n = 0
                    for paper in author_nodes.get(rid, {}).get("features", {}).get('Top Cited or Most Recent Papers', []):
                        n += 1
                        if n % 2 == 0:
                            continue
                        title_p = paper.get('Title', 'Untitled')
                        venue = paper.get('Venue', '')
                        year = paper.get('PubYear', '')
                        cited = paper.get('CitedCount', 0)
                        st.write(f"- {title_p} ({venue}, {year}) - Cited {cited} times")
        else:
            st.write("## Results will appear here")
            st.write("*Describe your needs on the left. I will propose a query to confirm, then search and rank results.*")


with col_left:
    # State machine similar to demo flow
    # Keep left-column clean (no progress indicators here)
    if (
        st.session_state.get("search_query")
        and st.session_state.get("query_confirmed", False)
        and not st.session_state.get("search_complete", False)
    ):
        # Initial retrieval (inline notice, no overlay)
        try:
            current_user_id = st.session_state.get('aid', "1000000010")
            # Inline notice suppressed (right-panel progress covers user feedback)
            normalized_q = normalize_query_text(st.session_state["search_query"])
            cached = st.session_state.get("query_to_results", {}).get(normalized_q)
            if cached is not None:
                st.session_state["initial_results"] = cached
                st.session_state["search_complete"] = True
                # Ensure executed queries history includes this query
                try:
                    executed_norm = [normalize_query_text(q) for q in st.session_state.get("executed_queries", [])]
                    if normalized_q not in executed_norm:
                        st.session_state["executed_queries"].append(st.session_state["search_query"])  # store original format
                        if len(st.session_state["executed_queries"]) > 20:
                            st.session_state["executed_queries"] = st.session_state["executed_queries"][-20:]
                except Exception:
                    pass
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "🔁 This query was searched before. Reusing previous results to save time."
                })
            else:
                st.session_state["initial_results"] = retrieve_candidates(st.session_state["search_query"], current_user_id, top_k=100)
                st.session_state["search_complete"] = True
                # Update executed queries memory and cache
                try:
                    st.session_state["query_to_results"][normalized_q] = list(st.session_state["initial_results"])  # shallow copy
                    executed_norm = [normalize_query_text(q) for q in st.session_state.get("executed_queries", [])]
                    if normalized_q not in executed_norm:
                        st.session_state["executed_queries"].append(st.session_state["search_query"])  # store original format
                        # keep last 20
                        if len(st.session_state["executed_queries"]) > 20:
                            st.session_state["executed_queries"] = st.session_state["executed_queries"][-20:]
                except Exception:
                    pass
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "✅ Initial search complete! Found potential collaborators based on your query."
                })
        except Exception as e:
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"⚠️ Error during search: {e}"
            })
            st.session_state["query_confirmed"] = False
        finally:
            pass
        st.rerun()

    elif (
        st.session_state.get("search_complete", False)
        and not st.session_state.get("reranking_complete", False)
        and not st.session_state.get("reranking_start", False)
    ):
        st.session_state.messages.append({
            "role": "assistant",
            "content": "🧠 Now analyzing candidates to find the best matches..."
        })
        st.session_state["reranking_start"] = True
        st.rerun()
    elif (
        st.session_state.get("search_complete", False)
        and st.session_state.get("reranking_start", False)
        and not st.session_state.get("reranking_complete", False)
    ):
        # Parallel mini-batch reranking like CM4AI, with inline progress bar (no chat spam)
        user_id = st.session_state.get('aid', "1000000010")
        user_bg = get_user_background(user_id)
        remaining = st.session_state["initial_results"][st.session_state["reranked_candidates_number"]:]
        search_query_local = st.session_state.get("search_query", "")
        if remaining:
            MAX_CONCURRENT_BATCHES = 5
            ITEMS_PER_BATCH = 2
            def chunk_list(lst, chunk_size):
                return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]
            mini_batches = chunk_list(remaining[:MAX_CONCURRENT_BATCHES * ITEMS_PER_BATCH], ITEMS_PER_BATCH)
            st.session_state["rerank_total_batches"] = len(mini_batches)
            st.session_state["rerank_batches_done"] = 0
            # Progress is displayed only once in the right panel under the query header
            def process_mini_batch(mini_batch):
                return rerank_authors_with_llm_batch(
                    mini_batch,
                    search_query_local,
                    user_bg,
                    MODEL_NAME_RERANKING,
                    batch_size=len(mini_batch)
                )
            batch_results: List[Tuple[str, float, str]] = []
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(process_mini_batch, b) for b in mini_batches]
                for fut in concurrent.futures.as_completed(futures):
                    try:
                        res = fut.result()
                        batch_results.extend(res)
                        st.session_state["rerank_batches_done"] += 1
                        # Right panel progress is updated on rerun; no additional progress bars here
                    except Exception as e:
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"⚠️ Reranking error: {e}"
                        })
            st.session_state["reranked_results"].extend([(aid, float(score), just, inst) for (aid, score, just, inst) in batch_results])
            st.session_state["reranked_candidates_number"] += len(batch_results)
            st.session_state["reranked_results"] = sorted(st.session_state["reranked_results"], key=lambda x: x[1], reverse=True)
            if st.session_state["reranked_candidates_number"] >= len(st.session_state["initial_results"]):
                st.session_state["reranking_complete"] = True
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "🏁 Reranking complete. Please review the ranked collaborators on the right. Would you like to run another query or refine this one?"
                })
            # No-op: single global progress bar lives in the right panel
        else:
            st.session_state["reranking_complete"] = True
            st.session_state.messages.append({
                "role": "assistant",
                "content": "🏁 Reranking complete. Please review the ranked collaborators on the right. Would you like to run another query or refine this one?"
            })
        st.rerun()

    # Chat input handling
    if prompt := st.chat_input("Describe your research interests and collaboration needs..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Persist raw user inputs in order for expertise (query generation) agent consistency
        try:
            st.session_state["user_inputs"].append(prompt)
            if len(st.session_state["user_inputs"]) > 200:
                st.session_state["user_inputs"] = st.session_state["user_inputs"][-200:]
        except Exception:
            pass

        # If a query exists but not confirmed, ask LLM to judge confirmation intent
        if (
            st.session_state.get("search_query")
            and not st.session_state.get("search_complete", False)
            and not st.session_state.get("query_confirmed", False)
        ):
            # Use the current proposed query text for confirmation classification
            is_confirm = llm_is_confirmation(prompt, st.session_state.get("search_query", ""), MODEL_NAME_EXPERTISE)
            if is_confirm:
                st.session_state["query_confirmed"] = True
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "Query confirmed. Proceeding with search..."
                })
                time.sleep(0.2)
                st.rerun()
            else:
                # User did not confirm; trigger refinement flow to generate a new query
                # Provide the original query and explicit "do not repeat" guidance
                st.session_state["pending_prompt"] = (
                    f"Please refine the query to better match my intent.\n"
                    f"Original query: {st.session_state.get('search_query','')}\n"
                    f"User feedback (Most important, follow it as much as possible): {prompt}\n"
                )
                st.session_state["pending_generate_query"] = True
                st.rerun()

        elif st.session_state.get("search_complete", False):
            # New search cycle: keep last context, but start fresh query flow
            st.session_state["query_confirmed"] = False
            st.session_state["initial_results"] = []
            st.session_state["reranked_results"] = []
            st.session_state["search_complete"] = False
            st.session_state["reranking_complete"] = False
            st.session_state["reranking_start"] = False
            st.session_state["reranked_candidates_number"] = 0
            st.session_state["rerank_total_batches"] = 0
            st.session_state["rerank_batches_done"] = 0
            try:
                user_id = st.session_state.get('aid', "1000000010")
                user_bg = get_user_background(user_id)
                past_qs = (st.session_state.get("executed_queries", [])
                           + st.session_state.get("proposed_queries", []))
                query, justification, _ = try_generate_query(prompt, user_bg, MODEL_NAME_EXPERTISE, past_qs)
                st.session_state["search_query"] = query
                # Track proposed queries to avoid repeats on subsequent refinements
                try:
                    if query:
                        st.session_state["proposed_queries"].append(query)
                        if len(st.session_state["proposed_queries"]) > 30:
                            st.session_state["proposed_queries"] = st.session_state["proposed_queries"][-30:]
                except Exception:
                    pass
                msg_content = f"Based on your request, I propose this search query:\n\n**{query}**"
                try:
                    if justification:
                        msg_content += f"\n\n{justification}"
                except Exception:
                    pass
                msg_content += "\n\nWould you like me to proceed with this query, or would you like to refine it further? Respond with 'Proceed' or provide feedback to improve the query."
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": msg_content
                })
            except Exception as e:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"⚠️ Could not prepare search: {e}"
                })
            st.rerun()

        else:
            # First-time or refinement flow (inline notice, keep layout visible)
            # Defer first-time draft to pre-layout step to avoid duplicate render
            st.session_state["pending_prompt"] = prompt
            st.session_state["pending_generate_query"] = True
            st.rerun()
