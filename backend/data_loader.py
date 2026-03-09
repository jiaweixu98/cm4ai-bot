"""Data loading utilities for FastAPI backend.

Loads data from local paths and cache only (no S3).
"""

import os
import json
import pickle
import logging

import numpy as np
import networkx as nx

logger = logging.getLogger(__name__)

# ---------- paths & config ----------
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.environ.get("CACHE_DIR", os.path.join(_PROJECT_ROOT, "tmp", "matrix_cache"))
os.makedirs(CACHE_DIR, exist_ok=True)

LOCAL_DATA_DIR = os.environ.get(
    "LOCAL_DATA_DIR",
    os.path.join(_PROJECT_ROOT, "data"),
)


def _load_json_data(filename: str, source_filename: str) -> dict:
    cache_path = f"{CACHE_DIR}/{filename}"
    local_path = os.path.join(LOCAL_DATA_DIR, source_filename)
    for candidate in [local_path, cache_path]:
        if os.path.exists(candidate):
            with open(candidate, "r") as f:
                return json.load(f)
    logger.warning("Missing local %s and cache %s.", local_path, cache_path)
    return {}


# ---------- singletons ----------

_author_nodes: dict | None = None
_publication_counts: dict | None = None
_knowledge_graph_nx: nx.Graph | None = None
_author_ids: list | None = None
_faiss_index = None
_tokenizer = None
_model = None
_resources_loaded = False


def load_author_nodes() -> dict:
    global _author_nodes
    if _author_nodes is not None:
        return _author_nodes
    for candidate in [
        os.path.join(LOCAL_DATA_DIR, "updated_author_nodes_with_papers.json"),
        f"{CACHE_DIR}/author_nodes.json",
    ]:
        if os.path.exists(candidate):
            with open(candidate, "r") as f:
                _author_nodes = json.load(f)
                return _author_nodes
    logger.warning("Missing local author metadata. Returning empty dict.")
    _author_nodes = _load_json_data("author_nodes.json", "updated_author_nodes_with_papers.json")
    return _author_nodes


def load_publication_counts() -> dict:
    global _publication_counts
    if _publication_counts is not None:
        return _publication_counts
    local_counts = os.path.join(LOCAL_DATA_DIR, "author_n_publications.json")
    if os.path.exists(local_counts):
        try:
            with open(local_counts, "r") as f:
                data = json.load(f)
                _publication_counts = {str(k): int(v) for k, v in data.items()}
                return _publication_counts
        except Exception:
            pass
    nodes = load_author_nodes() or {}
    counts = {}
    for aid, node in nodes.items():
        try:
            pubs = node.get("features", {}).get("Top Cited or Most Recent Papers", []) or []
            counts[str(aid)] = len(pubs)
        except Exception:
            counts[str(aid)] = 0
    _publication_counts = counts
    return _publication_counts


def _load_knowledge_graph_raw() -> dict:
    local_graph = os.path.join(LOCAL_DATA_DIR, "author_knowledge_graph_2024.json")
    if os.path.exists(local_graph):
        with open(local_graph, "r") as f:
            return json.load(f)
    logger.info("Local knowledge graph not found. Falling back to cache.")
    return _load_json_data("knowledge_graph.json", "author_knowledge_graph_2024.json")


def load_knowledge_graph_nx() -> nx.Graph:
    global _knowledge_graph_nx
    if _knowledge_graph_nx is not None:
        return _knowledge_graph_nx
    graph = _load_knowledge_graph_raw()
    g = nx.Graph()
    for node, neighbors in graph.items():
        for nb in neighbors:
            g.add_edge(node, nb)
    _knowledge_graph_nx = g
    return _knowledge_graph_nx


# ---------- embeddings / FAISS ----------

def _extract_ids_and_embs(loaded, possible_ids_files: list[str]):
    if isinstance(loaded, dict):
        if loaded and all(isinstance(v, dict) for v in loaded.values()):
            author_ids, embeddings_list = [], []
            for aid, variants in loaded.items():
                for vidx, emb in variants.items():
                    author_ids.append(f"{aid}_{vidx}")
                    embeddings_list.append(np.asarray(emb, dtype=np.float32))
            return author_ids, np.asarray(embeddings_list, dtype=np.float32)
        if loaded and all(isinstance(v, (list, tuple, np.ndarray)) for v in loaded.values()):
            author_ids = [str(k) for k in loaded.keys()]
            embeddings_list = [np.asarray(v, dtype=np.float32) for v in loaded.values()]
            return author_ids, np.asarray(embeddings_list, dtype=np.float32)
        if "author_ids" in loaded and "embeddings" in loaded:
            return [str(x) for x in loaded["author_ids"]], np.asarray(loaded["embeddings"], dtype=np.float32)
    if isinstance(loaded, (list, tuple)) and len(loaded) == 2 and isinstance(loaded[1], (list, tuple, np.ndarray)):
        return [str(x) for x in loaded[0]], np.asarray(loaded[1], dtype=np.float32)
    if isinstance(loaded, (list, tuple)) and loaded and all(isinstance(x, (list, tuple)) and len(x) == 2 for x in loaded):
        return [str(x[0]) for x in loaded], np.asarray([x[1] for x in loaded], dtype=np.float32)
    if isinstance(loaded, np.ndarray):
        embs = loaded.astype(np.float32)
        for fname in possible_ids_files:
            if os.path.exists(fname):
                try:
                    if fname.endswith(".pkl"):
                        with open(fname, "rb") as f:
                            ids = pickle.load(f)
                    elif fname.endswith(".npy"):
                        ids = np.load(fname, allow_pickle=True).tolist()
                    elif fname.endswith(".json"):
                        with open(fname, "r") as f:
                            ids = json.load(f)
                    else:
                        continue
                    author_ids = [str(x) for x in ids]
                    if len(author_ids) == embs.shape[0]:
                        return author_ids, embs
                except Exception:
                    continue
        return [str(i) for i in range(embs.shape[0])], embs
    raise ValueError("Unsupported embeddings format.")


def load_embeddings_and_index():
    global _author_ids, _faiss_index
    if _faiss_index is not None and _author_ids is not None:
        return _author_ids, _faiss_index

    import faiss

    ids_path = f"{CACHE_DIR}/author_ids.pkl"
    index_path = f"{CACHE_DIR}/faiss_index.bin"

    # 0) Local ready-made
    local_ids = os.path.join(LOCAL_DATA_DIR, "author_ids.pkl")
    local_index = os.path.join(LOCAL_DATA_DIR, "faiss_index.bin")
    if os.path.exists(local_ids) and os.path.exists(local_index):
        logger.info("Loading FAISS index from local data…")
        _faiss_index = faiss.read_index(local_index)
        with open(local_ids, "rb") as f:
            _author_ids = pickle.load(f)
        return _author_ids, _faiss_index

    # 1) Cached
    if os.path.exists(index_path) and os.path.exists(ids_path):
        logger.info("Loading FAISS index from cache…")
        _faiss_index = faiss.read_index(index_path)
        with open(ids_path, "rb") as f:
            _author_ids = pickle.load(f)
        return _author_ids, _faiss_index

    # 2) Build from local embeddings
    try:
        cand_embs = [
            os.path.join(LOCAL_DATA_DIR, "author_embeddings.pkl"),
            os.path.join(LOCAL_DATA_DIR, "author_embeddings.npy"),
        ]
        loaded = None
        for p in cand_embs:
            if os.path.exists(p):
                with open(p, "rb") as f:
                    loaded = pickle.load(f) if p.endswith(".pkl") else np.load(f, allow_pickle=True)
                break
        if loaded is not None:
            possible_ids_files = [
                os.path.join(LOCAL_DATA_DIR, n)
                for n in ["author_ids.pkl", "author_ids.npy", "ids.pkl", "ids.npy", "author_ids.json"]
            ]
            aids, embs = _extract_ids_and_embs(loaded, possible_ids_files)
            if embs.size == 0:
                raise ValueError("Local embeddings are empty.")
            dim = embs.shape[1]
            index = faiss.IndexFlatL2(dim)
            index.add(embs)
            faiss.write_index(index, index_path)
            with open(ids_path, "wb") as f:
                pickle.dump(aids, f)
            _author_ids, _faiss_index = aids, index
            return _author_ids, _faiss_index
    except Exception as e:
        logger.warning("Local embeddings build failed: %s", e)

    logger.error("No embeddings found locally.")
    return [], None


# ---------- SPECTER model ----------


def load_specter_model():
    import time as _time
    global _tokenizer, _model
    if _tokenizer is not None and _model is not None:
        return _tokenizer, _model
    try:
        logger.info("Loading SPECTER2 model (this takes 30-60s on CPU)…")
        t0 = _time.time()
        import torch
        from transformers import AutoTokenizer
        from adapters import AutoAdapterModel

        base_path = os.path.join(CACHE_DIR, "specter2_base")
        adapter_path = os.path.join(CACHE_DIR, "specter2_adapter")
        base_config = os.path.join(base_path, "config.json")
        adapter_config = os.path.join(adapter_path, "adapter_config.json")

        # 1) Local cache
        if os.path.exists(base_config):
            logger.info("  Loading tokenizer…")
            tok = AutoTokenizer.from_pretrained(base_path, local_files_only=True)
            logger.info("  Loading base model…")
            mdl = AutoAdapterModel.from_pretrained(base_path, local_files_only=True)
            if os.path.exists(adapter_config):
                logger.info("  Loading adapter (if this hangs, try restarting)…")
                mdl.load_adapter(adapter_path, load_as="adhoc_query", set_active=True)
            logger.info("  Moving to CPU + eval mode…")
            _tokenizer, _model = tok, mdl.to("cpu").eval()
            logger.info("  SPECTER2 loaded in %.1fs ✅", _time.time() - t0)
            return _tokenizer, _model

        # 2) HuggingFace
        os.makedirs(base_path, exist_ok=True)
        try:
            tok = AutoTokenizer.from_pretrained("allenai/specter2", cache_dir=base_path)
            mdl = AutoAdapterModel.from_pretrained("allenai/specter2", cache_dir=base_path)
        except Exception:
            from huggingface_hub import snapshot_download

            snapshot_download(repo_id="allenai/specter2", local_dir=base_path, local_dir_use_symlinks=False)
            tok = AutoTokenizer.from_pretrained(base_path, local_files_only=True)
            mdl = AutoAdapterModel.from_pretrained(base_path, local_files_only=True)
        try:
            mdl.load_adapter("allenai/specter2_adhoc_query", load_as="adhoc_query", set_active=True)
        except Exception:
            try:
                from huggingface_hub import snapshot_download

                os.makedirs(adapter_path, exist_ok=True)
                snapshot_download(repo_id="allenai/specter2_adhoc_query", local_dir=adapter_path, local_dir_use_symlinks=False)
                if os.path.exists(adapter_config):
                    mdl.load_adapter(adapter_path, load_as="adhoc_query", set_active=True)
            except Exception:
                pass
        _tokenizer, _model = tok, mdl.to("cpu").eval()
        return _tokenizer, _model
    except Exception as e:
        logger.error("Failed to load SPECTER model: %s", e)
        return None, None


# ---------- convenience ----------

def load_all():
    """Pre-load every resource. Call once at startup."""
    import time as _time
    global _resources_loaded
    if _resources_loaded:
        return
    t0 = _time.time()
    logger.info("Loading all resources…")
    logger.info("[1/5] Author nodes…")
    load_author_nodes()
    logger.info("[2/5] Publication counts…")
    load_publication_counts()
    logger.info("[3/5] Knowledge graph…")
    load_knowledge_graph_nx()
    logger.info("[4/5] FAISS index…")
    load_embeddings_and_index()
    logger.info("[5/5] SPECTER model…")
    load_specter_model()
    _resources_loaded = True
    logger.info("All resources loaded in %.1fs.", _time.time() - t0)
