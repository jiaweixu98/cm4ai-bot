"""Per-paper SPECTER2 vectors from the Cheaha export.

The vectors are the proximity (document) encoding, one row per OpenAlex work.
Questions are encoded with the adhoc-query adapter already used for people
search, which is the adapter meant to be compared with these documents.

The float matrix stays memory-mapped. A flat L2 index is written once next to
the snapshot and reused. When available memory is under 3 GB the index is left
unloaded and callers keep the keyword paper lookup.
"""

import json
import logging
import os
import threading
import time

import numpy as np

logger = logging.getLogger(__name__)

# People stay in author-vector order. On 256 self-described expertise queries,
# ranking by the two closest papers raised recall@10 (0.137 vs 0.109) but not
# recall@5 (0.062 vs 0.078) or MRR (0.061 vs 0.068), and the shortlist shows
# about five people. The author-vector path remains in use, and is also the
# fallback when this is false or the paper index is missing.
RANK_BY_PAPER = False
PAPER_RANK_PER_PERSON = 1
PAPER_SEARCH_DEPTH = 8000
SIMILAR_PAPERS = 24
SIMILAR_NEIGHBORS = 40
SIMILAR_LIMIT = 8
_MIN_AVAILABLE_KB = 3 * 1024 * 1024
_QUERY_CACHE = 8

_lock = threading.Lock()
_index = None
_missing = False
_memory_logged = False
_source_logged = False
_retry_at = 0.0
_RETRY_SECONDS = 300


def _memory_ok() -> bool:
    try:
        with open("/proc/meminfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) >= _MIN_AVAILABLE_KB
    except OSError:
        return True
    return True


def _vector_dir() -> str | None:
    """Where the vectors are: the directory the snapshot manifest declares, or an explicit
    PAPER_VECTOR_DIR. With neither there are no vectors (fail closed); no older release
    or home-directory location is ever searched."""
    from data_loader import LOCAL_DATA_DIR

    manifest_path = os.path.join(LOCAL_DATA_DIR, 'snapshot_manifest.json')
    candidates = []
    if os.path.isfile(manifest_path):
        with open(manifest_path, encoding='utf-8') as handle:
            declared = json.load(handle).get('paper_vector_source')
        if declared:
            candidates.append(os.path.realpath(os.path.join(LOCAL_DATA_DIR, declared)))
    configured = os.environ.get("PAPER_VECTOR_DIR", "").strip()
    if configured:
        candidates.append(configured)
    for directory in candidates:
        if (os.path.isfile(os.path.join(directory, "paper_embeddings.npy"))
                and os.path.isfile(os.path.join(directory, "paper_embedding_ids.json"))):
            return directory
    global _source_logged
    if not _source_logged:
        _source_logged = True
        logger.warning("Paper vectors unavailable: no manifest-declared source or PAPER_VECTOR_DIR with "
                       "paper_embeddings.npy (%s)", ", ".join(candidates) or "none declared")
    return None


def _unit(vector: np.ndarray) -> np.ndarray:
    query = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(query))
    if norm > 0:
        query = query / norm
    return query


class PaperVectors:
    def __init__(self, matrix, ids: list[str], index, by_author: dict[int, np.ndarray], row_authors: list[list[int]]):
        self.matrix = matrix
        self.ids = ids
        self.index = index
        self.by_author = by_author
        self.row_authors = row_authors
        self.row_of = {work_id: i for i, work_id in enumerate(ids)}
        self._queries: dict[str, np.ndarray] = {}

    @classmethod
    def open(cls) -> "PaperVectors | None":
        directory = _vector_dir()
        if directory is None:
            logger.info("Paper vectors: not in the snapshot")
            return None
        if not _memory_ok():
            return None

        import faiss
        from data_loader import LOCAL_DATA_DIR

        ids = json.load(open(os.path.join(directory, "paper_embedding_ids.json"), encoding="utf-8"))
        ids = [str(work_id) for work_id in ids]
        matrix = np.load(os.path.join(directory, "paper_embeddings.npy"), mmap_mode="r")
        if matrix.ndim != 2 or matrix.shape[0] != len(ids):
            raise ValueError(f"paper vector rows {matrix.shape} do not match {len(ids)} ids")

        index_path = os.path.join(LOCAL_DATA_DIR, "paper_faiss_index.bin")
        meta_path = os.path.join(LOCAL_DATA_DIR, "paper_faiss_index.json")
        source = os.path.join(directory, "paper_embeddings.npy")
        stamp = {"rows": len(ids), "dim": int(matrix.shape[1]), "bytes": os.path.getsize(source)}
        manifest_path = os.path.join(LOCAL_DATA_DIR, 'snapshot_manifest.json')
        if os.path.isfile(manifest_path):
            manifest = json.load(open(manifest_path, encoding='utf-8'))
            stamp.update(snapshot_version=manifest.get('snapshot_version'),
                         ids_sha256=manifest.get('paper_vector_ids_sha256'),
                         vectors_sha256=manifest.get('paper_vectors_sha256'))
        index = None
        if os.path.isfile(index_path) and os.path.isfile(meta_path):
            try:
                saved = json.load(open(meta_path, encoding="utf-8"))
                if all(saved.get(key) == value for key,value in stamp.items() if key != 'snapshot_version'):
                    index = faiss.read_index(index_path)
                    if index.ntotal != len(ids):
                        index = None
            except Exception as exc:
                logger.warning("Paper index reload failed, rebuilding: %s", exc)
                index = None
        if index is None:
            logger.info("Building paper FAISS index (%s rows)", f"{len(ids):,}")
            index = faiss.IndexFlatL2(int(matrix.shape[1]))
            for start in range(0, len(ids), 8192):
                chunk = np.ascontiguousarray(matrix[start:start + 8192], dtype=np.float32)
                index.add(chunk)
            faiss.write_index(index, index_path)
            with open(meta_path, "w", encoding="utf-8") as handle:
                json.dump(stamp, handle)
            logger.info("Paper FAISS index written to %s", index_path)

        by_author, row_authors = _author_rows(ids)
        logger.info("Paper vectors: %s papers", f"{len(ids):,}")
        return cls(matrix, ids, index, by_author, row_authors)

    def embed(self, text: str) -> np.ndarray:
        key = " ".join(str(text or "").split())
        cached = self._queries.get(key)
        if cached is not None:
            return cached
        import torch
        from data_loader import load_specter_model

        tokenizer, model = load_specter_model()
        if tokenizer is None or model is None:
            raise RuntimeError("SPECTER model is not available")
        inputs = tokenizer(
            [key],
            padding=True,
            truncation=True,
            return_tensors="pt",
            return_token_type_ids=False,
            max_length=512,
        )
        with torch.no_grad():
            vector = model(**inputs).last_hidden_state[:, 0, :].cpu().numpy().astype(np.float32)[0]
        vector = _unit(vector)
        if len(self._queries) >= _QUERY_CACHE:
            self._queries.pop(next(iter(self._queries)))
        self._queries[key] = vector
        return vector

    def matching_papers(self, author_id, question: str, library, limit: int = 2) -> list[dict]:
        """This person's catalog papers closest to the question, strongest first."""
        rows = self.by_author.get(int(author_id))
        if rows is None or len(rows) == 0 or limit < 1 or not str(question or "").strip():
            return []
        visible = library.visible_work_ids(author_id)
        rows = np.asarray([r for r in rows if self.ids[int(r)] in visible], dtype=np.int64)
        if len(rows) == 0:
            return []
        scores = np.asarray(self.matrix[rows], dtype=np.float32) @ self.embed(question)
        order = np.argsort(-scores)[:limit]
        found = []
        for slot in order:
            row = library.work(self.ids[int(rows[int(slot)])])
            if row and row.get("title"):
                found.append(row)
        return found

    def author_similarity(self, query_vector: np.ndarray, per_person: int = PAPER_RANK_PER_PERSON,
                          top_papers: int = PAPER_SEARCH_DEPTH) -> dict[int, float]:
        """Author id to similarity in (0, 1], from their closest papers to the query.

        Similarity is 1/(1+L2), the same shape as author-vector retrieval. With
        per_person > 1 the score is the mean of that many closest papers.
        """
        query = np.ascontiguousarray(_unit(query_vector), dtype=np.float32).reshape(1, -1)
        depth = min(top_papers, self.index.ntotal)
        distances, indices = self.index.search(query, depth)
        best: dict[int, list[float]] = {}
        keep = max(1, per_person)
        from data_loader import load_paper_library
        library = load_paper_library()
        for distance, row in zip(distances[0], indices[0]):
            if row < 0:
                continue
            similarity = 1.0 / (1.0 + max(float(distance), 0.0))
            for author_id in self.row_authors[int(row)]:
                if library is not None and not library.owns(author_id, self.ids[int(row)]):
                    continue
                bucket = best.setdefault(author_id, [])
                if len(bucket) < keep:
                    bucket.append(similarity)
        return {author_id: float(sum(values) / len(values)) for author_id, values in best.items()}

    def similar_work(self, author_id: int, library, exclude: set[int], limit: int = SIMILAR_LIMIT) -> list[dict]:
        """People with no shared papers, each with the closest paper pair.

        The focal person's own papers and anyone in exclude (themselves, recorded
        coauthors, and anyone they have already written with) are dropped.
        """
        selected = self._recent_rows(int(author_id), library, SIMILAR_PAPERS)
        if len(selected) == 0:
            return []
        queries = np.ascontiguousarray(self.matrix[selected], dtype=np.float32)
        distances, indices = self.index.search(queries, min(SIMILAR_NEIGHBORS, self.index.ntotal))
        focal = int(author_id)
        pairs = []
        seen = set()
        for query_slot, (dist_row, hit_row) in enumerate(zip(distances, indices)):
            query_row = int(selected[query_slot])
            for distance, hit in zip(dist_row, hit_row):
                hit = int(hit)
                if hit < 0 or hit == query_row:
                    continue
                authors = [a for a in self.row_authors[hit] if library.owns(a,self.ids[hit])]
                if focal in authors:
                    continue
                distance = float(distance)
                for other in authors:
                    if other in exclude or other == focal or (other, hit) in seen:
                        continue
                    seen.add((other, hit))
                    pairs.append((distance, other, hit, query_row))
        # One person and one of their papers per row, closest pairs first, so the
        # list does not repeat the same nearby paper under every coauthor's name.
        pairs.sort(key=lambda item: (item[0], item[1]))
        from paper_library import title_key

        used_people, used_papers, used_titles = set(), set(), set()
        people = []
        for _distance, other, hit, query_row in pairs:
            if other in used_people or hit in used_papers:
                continue
            theirs = library.work(self.ids[hit])
            yours = library.work(self.ids[query_row])
            if not theirs or not yours or not theirs.get("title") or not yours.get("title"):
                used_papers.add(hit)
                continue
            their_key = title_key(theirs["title"])
            your_key = title_key(yours["title"])
            # A preprint and its published copy can share a title under two work ids.
            if not their_key or their_key in used_titles or their_key == your_key:
                used_papers.add(hit)
                continue
            used_people.add(other)
            used_papers.add(hit)
            used_titles.add(their_key)
            people.append({
                "author_id": other,
                "their_title": theirs["title"],
                "their_year": theirs.get("year"),
                "your_title": yours["title"],
                "your_year": yours.get("year"),
                "their_work_id": str(theirs.get("work_id") or self.ids[hit]),
                "your_work_id": str(yours.get("work_id") or self.ids[query_row]),
            })
            if len(people) >= limit:
                break
        return people

    def _recent_rows(self, author_id: int, library, cap: int) -> np.ndarray:
        rows = self.by_author.get(author_id)
        if rows is None or len(rows) == 0:
            return np.array([], dtype=np.int32)
        visible = library.visible_work_ids(author_id)
        rows = np.asarray([r for r in rows if self.ids[int(r)] in visible], dtype=np.int32)
        if len(rows) <= cap:
            return rows
        years = _years(library, [self.ids[int(row)] for row in rows])
        order = sorted(rows, key=lambda row: (-(years.get(self.ids[int(row)]) or 0), self.ids[int(row)]))
        return np.asarray(order[:cap], dtype=np.int32)


def _years(library, work_ids: list[str]) -> dict[str, int]:
    found = {}
    db = library._db()
    for start in range(0, len(work_ids), 400):
        chunk = work_ids[start:start + 400]
        marks = ",".join("?" * len(chunk))
        for work_id, year in db.execute(f"SELECT work_id, year FROM papers WHERE work_id IN ({marks})", chunk):
            if year is not None:
                found[work_id] = int(year)
    return found


def _author_rows(ids: list[str]) -> tuple[dict[int, np.ndarray], list[list[int]]]:
    from data_loader import load_paper_library

    library = load_paper_library()
    by_author: dict[int, list[int]] = {}
    row_authors: list[list[int]] = [[] for _ in ids]
    if library is None:
        return {}, row_authors
    row_of = {work_id: i for i, work_id in enumerate(ids)}
    for work_id, author_id in library._db().execute("SELECT work_id, author_id FROM paper_authors"):
        row = row_of.get(work_id)
        if row is None:
            continue
        author_id = int(author_id)
        row_authors[row].append(author_id)
        by_author.setdefault(author_id, []).append(row)
    return {author_id: np.asarray(rows, dtype=np.int32) for author_id, rows in by_author.items()}, row_authors


def load_paper_index() -> PaperVectors | None:
    """The shared paper index, or None when the vectors are absent or could not load.

    A missing source is permanent for the process. Low memory or a failed load is
    retried after a few minutes, so one tight moment at first use does not turn
    paper evidence off until the next restart.
    """
    global _index, _missing, _retry_at
    if _index is not None or _missing:
        return _index
    with _lock:
        if _index is not None or _missing or time.monotonic() < _retry_at:
            return _index
        if _vector_dir() is None:
            _missing = True
            return None
        if not _memory_ok():
            global _memory_logged
            if not _memory_logged:
                _memory_logged = True
                logger.warning("Paper vectors skipped; available memory is under 3 GB (retrying later)")
            _retry_at = time.monotonic() + _RETRY_SECONDS
            return None
        try:
            _index = PaperVectors.open()
        except Exception as exc:
            logger.warning("Paper vectors failed to load (retrying later): %s", exc)
            _index = None
        if _index is None:
            _retry_at = time.monotonic() + _RETRY_SECONDS
    return _index
