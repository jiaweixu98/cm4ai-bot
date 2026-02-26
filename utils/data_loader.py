import os, json, pickle
import torch  # explicit import to force CPU below
import streamlit as st
import numpy as np
import boto3
from botocore.exceptions import NoCredentialsError
from botocore import UNSIGNED
from botocore.client import Config
import networkx as nx

# Align cache directory with CM4AI demo
CACHE_DIR = "./tmp/matrix_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
LOCAL_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')

# S3 configuration (secrets-overridable)
S3_BUCKET_NAME = st.secrets.get('s3_bucket', 'streamlit-teaming')
S3_REGION = st.secrets.get('s3_region', 'us-east-2')
S3_DATA_PREFIX = st.secrets.get('s3_data_prefix', '')
S3_BASE_MODEL_PREFIX = st.secrets.get('s3_model_base_prefix', '')
S3_ADAPTER_PREFIX = st.secrets.get('s3_model_adapter_prefix', '')

def _generate_prefix_variants(prefix: str) -> list[str]:
    """Return common prefix variants to tolerate small misconfigurations.

    Examples:
      "cm4ai/data/" -> ["cm4ai/data/", "cm4ai/data", "cm4ai/", ""]
    """
    try:
        p = (prefix or "").strip()
        variants = []
        # as-is
        variants.append(p if p.endswith('/') else (p + '/' if p else ''))
        # trim trailing slash
        if p:
            variants.append(p.rstrip('/'))
        # parent (drop last segment)
        if '/' in p.strip('/'):
            parent = p.strip('/').split('/')[:-1]
            variants.append(('/'.join(parent) + '/') if parent else '')
        # empty
        variants.append('')
        # de-duplicate while preserving order
        seen = set()
        uniq = []
        for v in variants:
            if v not in seen:
                uniq.append(v)
                seen.add(v)
        return uniq
    except Exception:
        return [prefix or '', '']

def _s3_try_download_first(s3_client, bucket: str, candidate_keys: list[str], dest_path: str) -> str | None:
    """Try candidate keys in order; return the first working key or None.
    Shows compact diagnostics in the Streamlit UI if all fail.
    """
    errors = []
    tried = []
    for key in candidate_keys:
        if not key:
            continue
        try:
            s3_client.download_file(bucket, key, dest_path)
            return key
        except Exception as e:
            tried.append(key)
            errors.append(str(e))
            continue
    if tried:
        # Limit noise while surfacing actionable info
        st.error(
            "S3 download failed. Bucket '" + str(bucket) + "'. Tried keys: " + ", ".join(tried[:6]) + ("..." if len(tried) > 6 else "")
        )
    return None

@st.cache_resource
def get_s3_client():
    access_key = st.secrets.get("aws_access_key_id", None)
    secret_key = st.secrets.get("aws_secret_access_key", None)
    if access_key and secret_key:
        return boto3.client(
            's3',
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=S3_REGION
        )
    # Anonymous/public access (for publicly readable buckets)
    return boto3.client('s3', config=Config(signature_version=UNSIGNED), region_name=S3_REGION)

@st.cache_resource
def s3_enabled() -> bool:
    # Consider S3 enabled if we at least have a bucket configured; client may use anonymous access
    return bool(S3_BUCKET_NAME)

@st.cache_resource
def load_json_data(filename, s3_key):
    cache_path = f"{CACHE_DIR}/{filename}"
    # 1) Prefer local file if exists
    local_path = os.path.join(LOCAL_DATA_DIR, s3_key.split('/')[-1])
    for candidate in [local_path, cache_path]:
        if os.path.exists(candidate):
            with open(candidate, 'r') as f:
                return json.load(f)
    # 2) Download from S3 to cache if not found locally and S3 is enabled
    if not s3_enabled():
        st.warning(f"Missing local {filename} and S3 is not configured.")
        return {}
    s3 = get_s3_client()
    # Build tolerant candidate keys
    base = os.path.basename(s3_key)
    variants = _generate_prefix_variants(S3_DATA_PREFIX)
    candidate_keys = []
    # Explicit s3_key first
    candidate_keys.append(s3_key)
    # Prefix variants with basename
    for pv in variants:
        if pv:
            candidate_keys.append((pv if pv.endswith('/') else pv + '/') + base)
        else:
            candidate_keys.append(base)
    # De-duplicate
    seen = set()
    candidate_keys = [k for k in candidate_keys if not (k in seen or seen.add(k))]
    with st.spinner(f"Loading {filename} (one-time operation)..."):
        used_key = _s3_try_download_first(s3, S3_BUCKET_NAME, candidate_keys, cache_path)
        if used_key:
            try:
                with open(cache_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                st.error(f"Loaded {filename} from s3://{S3_BUCKET_NAME}/{used_key} but failed to parse JSON: {e}")
                return {}
        else:
            st.error(
                f"Unable to fetch {filename} from bucket '{S3_BUCKET_NAME}'. Ensure one of these paths exists: "
                + ", ".join(candidate_keys[:6]) + ("..." if len(candidate_keys) > 6 else "")
            )
            return {}


def _extract_ids_and_embs_from_loaded(loaded, possible_ids_files: list[str]):
    """Return (author_ids: list[str], embeddings: np.ndarray[float32]) from various formats."""
    # Case A: dict
    if isinstance(loaded, dict):
        # dict-of-dict: id -> {variant_idx: vector}
        if loaded and all(isinstance(v, dict) for v in loaded.values()):
            author_ids, embeddings_list = [], []
            for aid, variants in loaded.items():
                for vidx, emb in variants.items():
                    author_ids.append(f"{aid}_{vidx}")
                    embeddings_list.append(np.asarray(emb, dtype=np.float32))
            return author_ids, np.asarray(embeddings_list, dtype=np.float32)
        # dict of id -> vector
        if loaded and all(isinstance(v, (list, tuple, np.ndarray)) for v in loaded.values()):
            author_ids = [str(k) for k in loaded.keys()]
            embeddings_list = [np.asarray(v, dtype=np.float32) for v in loaded.values()]
            return author_ids, np.asarray(embeddings_list, dtype=np.float32)
        # dict with keys
        if 'author_ids' in loaded and 'embeddings' in loaded:
            author_ids = [str(x) for x in loaded['author_ids']]
            embs = np.asarray(loaded['embeddings'], dtype=np.float32)
            return author_ids, embs
    # Case B: tuple/list: (ids, embs)
    if isinstance(loaded, (list, tuple)) and len(loaded) == 2 and isinstance(loaded[1], (list, tuple, np.ndarray)):
        author_ids = [str(x) for x in loaded[0]]
        embs = np.asarray(loaded[1], dtype=np.float32)
        return author_ids, embs
    # Case C: list of pairs
    if isinstance(loaded, (list, tuple)) and loaded and all(isinstance(x, (list, tuple)) and len(x) == 2 for x in loaded):
        author_ids = [str(x[0]) for x in loaded]
        embs = np.asarray([x[1] for x in loaded], dtype=np.float32)
        return author_ids, embs
    # Case D: bare ndarray, try find ids in companion files
    if isinstance(loaded, np.ndarray):
        embs = loaded.astype(np.float32)
        # look for ids files
        for fname in possible_ids_files:
            if os.path.exists(fname):
                try:
                    if fname.endswith('.pkl'):
                        with open(fname, 'rb') as f:
                            ids = pickle.load(f)
                    elif fname.endswith('.npy'):
                        ids = np.load(fname, allow_pickle=True).tolist()
                    elif fname.endswith('.json'):
                        with open(fname, 'r') as f:
                            ids = json.load(f)
                    else:
                        continue
                    author_ids = [str(x) for x in ids]
                    if len(author_ids) == embs.shape[0]:
                        return author_ids, embs
                except Exception:
                    continue
        # fallback: enumerate
        author_ids = [str(i) for i in range(embs.shape[0])]
        return author_ids, embs
    raise ValueError("Unsupported embeddings format. Provide dict, (ids, embs), list of pairs, or ndarray with ids file.")


@st.cache_resource
def load_embeddings_and_index():
    ids_path = f"{CACHE_DIR}/author_ids.pkl"
    index_path = f"{CACHE_DIR}/faiss_index.bin"

    progress_text = st.empty()
    progress_bar = st.progress(0)

    # 0) Prefer local ready-made ids/index from cm4ai_demo/data
    local_ids = os.path.join(LOCAL_DATA_DIR, 'author_ids.pkl')
    local_index = os.path.join(LOCAL_DATA_DIR, 'faiss_index.bin')
    if os.path.exists(local_ids) and os.path.exists(local_index):
        progress_text.text("Loading author search index from local data...")
        progress_bar.progress(50)
        import faiss
        index = faiss.read_index(local_index)
        with open(local_ids, 'rb') as f:
            author_ids = pickle.load(f)
        progress_bar.progress(100)
        progress_text.empty()
        progress_bar.empty()
        return author_ids, index

    # 1) Cached
    if os.path.exists(index_path) and os.path.exists(ids_path):
        progress_text.text("Loading author search index from cache...")
        progress_bar.progress(50)
        import faiss
        index = faiss.read_index(index_path)
        with open(ids_path, 'rb') as f:
            author_ids = pickle.load(f)
        progress_bar.progress(100)
        progress_text.empty()
        progress_bar.empty()
        return author_ids, index

    # 2) Local build from author_embeddings.* (robust formats)
    try:
        progress_text.text("Building index from local author embeddings...")
        progress_bar.progress(25)
        # Try multiple possible local embeddings file names
        cand_embs = [
            os.path.join(LOCAL_DATA_DIR, 'author_embeddings.pkl'),
            os.path.join(LOCAL_DATA_DIR, 'author_embeddings.npy'),
        ]
        loaded = None
        for p in cand_embs:
            if os.path.exists(p):
                with open(p, 'rb') if p.endswith('.pkl') else open(p, 'rb') as f:
                    loaded = pickle.load(f) if p.endswith('.pkl') else np.load(f, allow_pickle=True)
                break
        if loaded is not None:
            possible_ids_files = [
                os.path.join(LOCAL_DATA_DIR, 'author_ids.pkl'),
                os.path.join(LOCAL_DATA_DIR, 'author_ids.npy'),
                os.path.join(LOCAL_DATA_DIR, 'ids.pkl'),
                os.path.join(LOCAL_DATA_DIR, 'ids.npy'),
                os.path.join(LOCAL_DATA_DIR, 'author_ids.json'),
            ]
            author_ids, embs = _extract_ids_and_embs_from_loaded(loaded, possible_ids_files)
            if embs.size == 0:
                raise ValueError("Local embeddings are empty.")
            import faiss
            dim = embs.shape[1]
            index = faiss.IndexFlatL2(dim)
            index.add(embs)
            progress_bar.progress(85)
            # write cache
            faiss.write_index(index, index_path)
            with open(ids_path, 'wb') as f:
                pickle.dump(author_ids, f)
            progress_bar.progress(100)
            progress_text.empty()
            progress_bar.empty()
            return author_ids, index
    except Exception as e:
        progress_text.error(f"Local embeddings build failed: {e}")
        progress_bar.empty()
        # continue to S3 if enabled

    # 3) S3 ready-made ids/index (only if S3 enabled)
    if s3_enabled():
        s3 = get_s3_client()
        progress_text.text("Downloading author ids/index (one-time operation)...")
        progress_bar.progress(25)
        # Build tolerant candidate keys for ids and index
        variants = _generate_prefix_variants(S3_DATA_PREFIX)
        ids_candidates = []
        idx_candidates = []
        # add explicit default
        ids_candidates.append(f"{S3_DATA_PREFIX}author_ids.pkl")
        idx_candidates.append(f"{S3_DATA_PREFIX}faiss_index.bin")
        base_pairs = [("author_ids.pkl", ids_candidates), ("faiss_index.bin", idx_candidates)]
        for base_name, coll in base_pairs:
            for pv in variants:
                if pv:
                    coll.append((pv if pv.endswith('/') else pv + '/') + base_name)
                else:
                    coll.append(base_name)
        # de-duplicate
        def _dedup(lst):
            seen = set()
            out = []
            for k in lst:
                if k and k not in seen:
                    out.append(k)
                    seen.add(k)
            return out
        ids_candidates = _dedup(ids_candidates)
        idx_candidates = _dedup(idx_candidates)
        used_ids_key = _s3_try_download_first(s3, S3_BUCKET_NAME, ids_candidates, ids_path)
        used_idx_key = _s3_try_download_first(s3, S3_BUCKET_NAME, idx_candidates, index_path)
        if used_ids_key and used_idx_key:
            progress_bar.progress(75)
            import faiss
            index = faiss.read_index(index_path)
            with open(ids_path, 'rb') as f:
                author_ids = pickle.load(f)
            progress_bar.progress(100)
            progress_text.empty()
            progress_bar.empty()
            return author_ids, index

        # 4) S3 build from author_embeddings (robust)
        progress_text.text("Building index from S3 author embeddings...")
        progress_bar.progress(25)
        emb_variants = _generate_prefix_variants(S3_DATA_PREFIX)
        emb_candidates = [f"{S3_DATA_PREFIX}author_embeddings.pkl"]
        for pv in emb_variants:
            if pv:
                emb_candidates.append((pv if pv.endswith('/') else pv + '/') + 'author_embeddings.pkl')
            else:
                emb_candidates.append('author_embeddings.pkl')
        emb_candidates = list(dict.fromkeys([k for k in emb_candidates if k]))
        emb_cache = f"{CACHE_DIR}/author_embeddings.pkl"
        used_emb_key = _s3_try_download_first(s3, S3_BUCKET_NAME, emb_candidates, emb_cache)
        if used_emb_key:
            try:
                with open(emb_cache, 'rb') as f:
                    loaded = pickle.load(f)
                possible_ids_files = [
                    os.path.join(CACHE_DIR, 'author_ids.pkl'),
                    os.path.join(CACHE_DIR, 'author_ids.npy'),
                ]
                author_ids, embs = _extract_ids_and_embs_from_loaded(loaded, possible_ids_files)
                import faiss
                dim = embs.shape[1]
                index = faiss.IndexFlatL2(dim)
                index.add(embs)
                # Save cache for future
                faiss.write_index(index, index_path)
                with open(ids_path, 'wb') as f:
                    pickle.dump(author_ids, f)
                progress_bar.progress(100)
                progress_text.empty()
                progress_bar.empty()
                return author_ids, index
            except Exception as e:
                progress_text.error(f"S3 embeddings build failed after downloading {used_emb_key}: {e}")
                progress_bar.empty()

    # If we reached here, nothing worked
    st.error("No local index/embeddings found and S3 not available. Provide local author_ids.pkl+faiss_index.bin or author_embeddings (supported formats) in cm4ai_demo/data.")
    return [], None

# CM4AI-named loaders
@st.cache_resource
def load_author_nodes():
    # Prefer local cm4ai_demo/data file name first
    for candidate in [
        os.path.join(LOCAL_DATA_DIR, 'updated_author_nodes_with_papers.json'),
        f"{CACHE_DIR}/author_nodes.json",
    ]:
        if os.path.exists(candidate):
            with open(candidate, 'r') as f:
                return json.load(f)
    # Else download using CM4AI key if enabled
    if not s3_enabled():
        st.warning("Missing local author metadata and S3 is not configured.")
        return {}
    return load_json_data('author_nodes.json', f"{S3_DATA_PREFIX}updated_author_nodes_with_papers.json")

@st.cache_resource
def load_publication_counts() -> dict:
    """Return mapping author_id -> number of publications.
    Prefers a local precomputed file if present; otherwise derives counts from author_nodes.
    """
    # Prefer local precomputed counts if available
    local_counts = os.path.join(LOCAL_DATA_DIR, 'author_n_publications.json')
    if os.path.exists(local_counts):
        try:
            with open(local_counts, 'r') as f:
                data = json.load(f)
                # Ensure keys are strings
                return {str(k): int(v) for k, v in data.items()}
        except Exception:
            pass

    # Fallback: derive from author_nodes
    nodes = load_author_nodes() or {}
    counts = {}
    for aid, node in nodes.items():
        try:
            features = node.get('features', {})
            pubs = features.get('Top Cited or Most Recent Papers', []) or []
            counts[str(aid)] = int(len(pubs))
        except Exception:
            counts[str(aid)] = 0
    return counts

@st.cache_resource
def load_knowledge_graph():
    # Prefer local
    local_graph = os.path.join(LOCAL_DATA_DIR, 'author_knowledge_graph_2024.json')
    if os.path.exists(local_graph):
        with open(local_graph, 'r') as f:
            return json.load(f)
    if not s3_enabled():
        st.warning("Missing local knowledge graph and S3 is not configured.")
        return {}
    return load_json_data('knowledge_graph.json', f"{S3_DATA_PREFIX}author_knowledge_graph_2024.json")

@st.cache_resource
def load_knowledge_graph_nx():
    graph = load_knowledge_graph()
    g = nx.Graph()
    for node, neighbors in graph.items():
        for nb in neighbors:
            g.add_edge(node, nb)
    return g

# --- Model loading aligned with CM4AI + HF fallback ---

def download_directory_from_s3(bucket, s3_prefix, local_directory, s3_client):
    try:
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket, Prefix=s3_prefix)
        for page in pages:
            if "Contents" in page:
                for obj in page["Contents"]:
                    s3_key = obj["Key"]
                    if s3_key.endswith('/'):
                        continue
                    relative_path = os.path.relpath(s3_key, s3_prefix)
                    local_path = os.path.join(local_directory, relative_path)
                    os.makedirs(os.path.dirname(local_path), exist_ok=True)
                    s3_client.download_file(bucket, s3_key, local_path)
        return True
    except Exception:
        return False

def _model_to_cpu(tok, mdl):
    """Force model onto CPU and switch to eval mode (avoids MPS segfault on Apple Silicon)."""
    if mdl is not None:
        mdl = mdl.to('cpu').eval()
    return tok, mdl


@st.cache_resource
def load_specter_model():
    try:
        from transformers import AutoTokenizer
        from adapters import AutoAdapterModel
        base_path = os.path.join(CACHE_DIR, "specter2_base")
        adapter_path = os.path.join(CACHE_DIR, "specter2_adapter")
        base_config = os.path.join(base_path, "config.json")
        adapter_config = os.path.join(adapter_path, "adapter_config.json")

        # 1) Local cache
        if os.path.exists(base_config):
            tok = AutoTokenizer.from_pretrained(base_path, local_files_only=True)
            mdl = AutoAdapterModel.from_pretrained(base_path, local_files_only=True)
            if os.path.exists(adapter_config):
                mdl.load_adapter(adapter_path, load_as="adhoc_query", set_active=True)
            return _model_to_cpu(tok, mdl)

        # 2) S3
        if s3_enabled():
            s3 = get_s3_client()
            ok = download_directory_from_s3(S3_BUCKET_NAME, S3_BASE_MODEL_PREFIX, base_path, s3)
            if ok and os.path.exists(base_config):
                if not os.path.exists(adapter_config):
                    download_directory_from_s3(S3_BUCKET_NAME, S3_ADAPTER_PREFIX, adapter_path, s3)
                tok = AutoTokenizer.from_pretrained(base_path, local_files_only=True)
                mdl = AutoAdapterModel.from_pretrained(base_path, local_files_only=True)
                if os.path.exists(adapter_config):
                    mdl.load_adapter(adapter_path, load_as="adhoc_query", set_active=True)
                return _model_to_cpu(tok, mdl)

        # 3) HuggingFace fallback
        os.makedirs(base_path, exist_ok=True)
        try:
            # First try normal transformers download into our cache dir
            tok = AutoTokenizer.from_pretrained("allenai/specter2", cache_dir=base_path, local_files_only=False)
            mdl = AutoAdapterModel.from_pretrained("allenai/specter2", cache_dir=base_path, local_files_only=False)
        except Exception:
            # If that fails (e.g., .no_exist), use huggingface_hub to materialize files into base_path
            try:
                from huggingface_hub import snapshot_download
                snapshot_download(
                    repo_id="allenai/specter2",
                    local_dir=base_path,
                    local_dir_use_symlinks=False,
                )
                tok = AutoTokenizer.from_pretrained(base_path, local_files_only=True)
                mdl = AutoAdapterModel.from_pretrained(base_path, local_files_only=True)
            except Exception:
                return None, None
        # Try to load adapter from HF, fall back gracefully
        try:
            mdl.load_adapter("allenai/specter2_adhoc_query", load_as="adhoc_query", set_active=True)
        except Exception:
            # Try to snapshot adapter locally then load from path
            try:
                from huggingface_hub import snapshot_download
                os.makedirs(adapter_path, exist_ok=True)
                snapshot_download(
                    repo_id="allenai/specter2_adhoc_query",
                    local_dir=adapter_path,
                    local_dir_use_symlinks=False,
                )
                if os.path.exists(adapter_config):
                    mdl.load_adapter(adapter_path, load_as="adhoc_query", set_active=True)
            except Exception:
                pass
        return _model_to_cpu(tok, mdl)
    except Exception:
        return None, None

# Backward-compatible aliases
load_resource_nodes = load_author_nodes
load_embedder = load_specter_model
