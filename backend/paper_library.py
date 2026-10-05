"""Read-only access to the snapshot paper table (papers.sqlite).

The table holds every paper in the TKG export that has a title: work id, title,
abstract, year, venue, DOI, PMID, citation count, primary topic, and the catalog
people listed as its authors. A full-text index covers titles and abstracts.
"""

import math
import os
import re
import sqlite3
import logging
import threading
import time
import urllib.parse
from tkg_publications import decisions_path
from collections import OrderedDict

logger = logging.getLogger(__name__)

PAPERS_DB = "papers.sqlite"

_SEARCH_STOP = set(
    "a an and or of for in on to the with from by at as into via is are was were be been what which who how "
    "why when where does do did can could should would about any some this that these those there their "
    "recent latest new current approach approaches paper papers work works study studies research literature "
    "review reviews give list find show me my our us please using use based method methods".split())
# Words that describe the person being sought rather than the science.
_PEOPLE_STOP = _SEARCH_STOP | set(
    "i im am want need looking someone somebody person people researcher researchers expert experts mentor mentors "
    "mentoring collaborator collaborators collaborate collaboration partner team teams help learn learning "
    "guidance advice build building working worked works interested interest experience experienced strong "
    "skills skill background area areas field fields topic topics project projects idea ideas".split())
# "learning" is a stop word only in the people sense; keep it when it is part of the science.
_KEEP_WITH = {"learning": {"machine", "deep", "federated", "reinforcement", "representation", "transfer",
                           "contrastive", "supervised", "self", "active", "statistical", "multitask"}}
# Titles shared by unrelated papers (editorials, replies, dataset records) are only
# trusted when they are long enough to be specific.
_MIN_TITLE_WORDS = 4
_QUERY_CACHE_SIZE = 64


def title_key(value) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(value or "").casefold()))


def doi_key(value) -> str:
    doi = str(value or "").strip().lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "https://dx.doi.org/", "doi:"):
        doi = doi.removeprefix(prefix)
    return doi.rstrip(".")


def work_key(value) -> str:
    return str(value or "").strip().rsplit("/", 1)[-1].upper()


class PaperLibrary:
    _decisions_warned = False

    def __init__(self, path: str):
        self.path = path
        self._local = threading.local()
        manifest = os.path.join(os.path.dirname(path), 'snapshot_manifest.json')
        import json
        self.snapshot_version = ''
        if os.path.isfile(manifest):
            with open(manifest, encoding='utf-8') as handle:
                self.snapshot_version = json.load(handle).get('snapshot_version', '')

    @classmethod
    def open(cls, data_dir: str) -> "PaperLibrary | None":
        path = os.path.join(data_dir, PAPERS_DB)
        return cls(path) if os.path.exists(path) else None

    def _db(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute('CREATE TEMP VIEW visible_paper_authors AS SELECT * FROM main.paper_authors')
            self._local.conn = conn
        # A Graph review may create the decision store after this process starts.
        if (self.snapshot_version and not getattr(self._local, 'attached_decisions', False)
                and time.monotonic() >= getattr(self._local, 'decisions_retry_at', 0)):
            path = decisions_path()
            if os.path.isfile(path):
                self._attach_decisions(conn, path)
        return conn

    def _attach_decisions(self, conn: sqlite3.Connection, path: str) -> None:
        """Hide links that people marked "not my papers". Read-only; a missing, locked or
        differently shaped decision store is logged once and treated as no decisions."""
        attached = False
        version = self.snapshot_version.replace("'", "''")
        try:
            conn.execute('ATTACH DATABASE ? AS corrections', ('file:' + urllib.parse.quote(path) + '?mode=ro',))
            attached = True
            conn.execute('SELECT snapshot_version, author_id, work_id FROM corrections.excluded_links LIMIT 1').fetchall()
            conn.execute('DROP VIEW visible_paper_authors')
            conn.execute(f'''CREATE TEMP VIEW visible_paper_authors AS SELECT pa.* FROM main.paper_authors pa
                WHERE NOT EXISTS (SELECT 1 FROM corrections.excluded_links x
                WHERE x.snapshot_version='{version}' AND x.author_id=pa.author_id AND x.work_id=pa.work_id)''')
            self._local.attached_decisions = True
        except sqlite3.Error as exc:
            if attached:
                try:
                    conn.execute('DETACH DATABASE corrections')
                except sqlite3.Error:
                    pass
            self._local.decisions_retry_at = time.monotonic() + 60
            if not PaperLibrary._decisions_warned:
                PaperLibrary._decisions_warned = True
                logger.warning('Publication decisions unavailable (%s); showing all catalog links', exc)

    def owns(self, author_id, work_id) -> bool:
        return self._db().execute('SELECT 1 FROM visible_paper_authors WHERE author_id=? AND work_id=?',
                                  (int(author_id), work_key(work_id))).fetchone() is not None

    def visible_work_ids(self, author_id) -> set[str]:
        """The work ids this person is shown on, in one query (corrections applied)."""
        return {r[0] for r in self._db().execute('SELECT work_id FROM visible_paper_authors WHERE author_id=?',
                                                 (int(author_id),))}

    def canonical_author(self, author_id) -> str:
        if not str(author_id).isdigit(): return str(author_id)
        if not hasattr(self, '_aliases'):
            try: self._aliases = {int(a):str(b) for a,b in self._db().execute('SELECT old_id,author_id FROM author_aliases')}
            except sqlite3.OperationalError: self._aliases = {}
        return self._aliases.get(int(author_id),str(author_id))

    def author_works(self, author_id) -> list[dict]:
        return [dict(r) for r in self._db().execute('''SELECT p.*
            FROM visible_paper_authors pa JOIN papers p USING(work_id) WHERE pa.author_id=?
            ORDER BY p.year DESC,p.cited_by DESC,p.work_id''', (int(author_id),))]

    def _one(self, where: str, value) -> dict | None:
        row = self._db().execute(f"SELECT rowid, * FROM papers WHERE {where} = ? LIMIT 1", (value,)).fetchone()
        return dict(row) if row else None

    def work(self, work_id) -> dict | None:
        key = work_key(work_id)
        return self._one("work_id", key) if re.fullmatch(r"W\d{4,12}", key) else None

    def by_doi(self, doi) -> dict | None:
        key = doi_key(doi)
        return self._one("doi_key", key) if key.startswith("10.") else None

    def by_title(self, title, year=None) -> dict | None:
        """The one paper with exactly this title (and year, when known). None when the
        title is short or shared by several papers, so a generic title never attaches
        the wrong paper."""
        key = title_key(title)
        if len(key.split()) < _MIN_TITLE_WORDS:
            return None
        rows = [dict(r) for r in self._db().execute(
            "SELECT rowid, * FROM papers WHERE title_key = ? LIMIT 20", (key,)).fetchall()]
        year = int(year) if str(year or "").strip().isdigit() else None
        if year is not None:
            rows = [r for r in rows if r.get("year") is None or abs(int(r["year"]) - year) <= 1]
        return rows[0] if len(rows) == 1 else None

    # ---------- query-to-paper matching ----------

    @staticmethod
    def query_terms(query: str) -> list[str]:
        words = re.findall(r"[a-z0-9][a-z0-9-]*", str(query or "").casefold())
        present = set(words)
        terms = []
        for word in words:
            keep = word in _KEEP_WITH and present & _KEEP_WITH[word]
            if len(word) < 2 or (word in _PEOPLE_STOP and not keep) or word in terms:
                continue
            terms.append(word)
        return terms[:16]

    def score_query(self, query: str) -> dict[int, float]:
        """BM25 over every paper's title and abstract: {rowid: score}, higher is better."""
        terms = self.query_terms(query)
        key = " ".join(terms)
        if not key:
            return {}
        cache = getattr(self._local, "scores", None)
        if cache is None:
            cache = self._local.scores = OrderedDict()
        if key in cache:
            cache.move_to_end(key)
            return cache[key]
        expression = " OR ".join('"' + t.replace('"', "") + '"' for t in terms)
        rows = self._db().execute(
            "SELECT rowid, bm25(papers_fts, 4.0, 1.0) FROM papers_fts WHERE papers_fts MATCH ?",
            (expression,)).fetchall()
        scores = {int(rowid): -float(rank) for rowid, rank in rows}
        cache[key] = scores
        if len(cache) > _QUERY_CACHE_SIZE:
            cache.popitem(last=False)
        return scores

    def author_rowids(self, author_id) -> list[int]:
        return [r[0] for r in self._db().execute(
            "SELECT p.rowid FROM visible_paper_authors pa JOIN papers p ON p.work_id = pa.work_id WHERE pa.author_id = ?",
            (int(author_id),)).fetchall()]

    def shared_years(self, author_id: int) -> dict[int, int]:
        """Latest shared publication year with each catalog coauthor on a joint paper."""
        if not str(author_id).isdigit():
            return {}
        rows = self._db().execute(
            """
            SELECT pa2.author_id AS author_id, MAX(p.year) AS latest_year
            FROM visible_paper_authors pa1
            JOIN visible_paper_authors pa2
              ON pa2.work_id = pa1.work_id AND pa2.author_id != pa1.author_id
            JOIN papers p ON p.work_id = pa1.work_id
            WHERE pa1.author_id = ?
            GROUP BY pa2.author_id
            """,
            (int(author_id),),
        ).fetchall()
        return {
            int(row["author_id"]): int(row["latest_year"])
            for row in rows
            if row["latest_year"] is not None
        }

    def titles(self, author_id: int, limit: int = 24) -> list[dict]:
        """This author's catalog papers, newest first. Duplicate titles are dropped."""
        if not str(author_id).isdigit():
            return []
        cap = max(1, min(int(limit), 24))
        rows = self._db().execute(
            """
            SELECT p.title AS title, p.year AS year
            FROM visible_paper_authors pa
            JOIN papers p ON p.work_id = pa.work_id
            WHERE pa.author_id = ?
            ORDER BY p.year DESC, p.title
            LIMIT ?
            """,
            (int(author_id), cap * 3),
        ).fetchall()
        found = []
        seen = set()
        for row in rows:
            title = str(row["title"] or "").strip()
            key = title.casefold()
            if not title or key in seen:
                continue
            seen.add(key)
            year_text = str(row["year"] or "")
            found.append({"title": title, "year": int(year_text) if year_text.isdigit() else None})
            if len(found) >= cap:
                break
        return found

    def term_weight(self, term: str) -> float:
        """Inverse document frequency of a term over the library, so rare, specific
        words count for more than common ones."""
        cache = getattr(self._local, "idf", None)
        if cache is None:
            cache = self._local.idf = {}
        if term not in cache:
            db = self._db()
            if not hasattr(self._local, "total"):
                self._local.total = db.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
            df = db.execute("SELECT COUNT(*) FROM papers_fts WHERE papers_fts MATCH ?",
                            ('"' + term.replace('"', "") + '"',)).fetchone()[0]
            cache[term] = math.log((self._local.total + 1) / (df + 1))
        return cache[term]

    @staticmethod
    def _covered(terms: list[str], text: str) -> list[str]:
        words = set(re.findall(r"[a-z0-9][a-z0-9-]*", text.casefold()))
        found = []
        for term in terms:
            stem = term[:max(4, len(term) - 2)] if len(term) > 5 else term
            if any(w.startswith(stem) for w in words):
                found.append(term)
        return found

    def matched_papers(self, author_id, query: str, limit: int = 2) -> list[dict]:
        """This person's papers whose title and abstract best match the query, strongest
        first. A paper must share at least two of the query's key terms (one when the
        query has a single term) carrying at least half of the query's term weight, so
        common words alone never make a match; [] when none does."""
        if not str(author_id or "").isdigit():
            return []
        terms = self.query_terms(query)
        scores = self.score_query(query) if terms else {}
        if not scores:
            return []
        hits = sorted(((scores[r], r) for r in self.author_rowids(author_id) if r in scores), reverse=True)
        need = min(2, len(terms))
        weights = {t: self.term_weight(t) for t in terms}
        total = sum(weights.values()) or 1.0
        found, seen = [], set()
        for _, rowid in hits[:limit * 8]:
            row = dict(self._db().execute("SELECT rowid, * FROM papers WHERE rowid = ?", (rowid,)).fetchone())
            covered = self._covered(terms, f"{row['title']} {row.get('abstract') or ''}")
            if (row["title_key"] in seen or len(covered) < need
                    or sum(weights[t] for t in covered) < 0.5 * total):
                continue
            seen.add(row["title_key"])
            found.append(row)
            if len(found) >= limit:
                break
        return found

    def people_scores(self, query: str, top_papers: int = 3000, per_person: int = 3,
                      exclude_rowids: set[int] | None = None) -> dict[int, float]:
        """People ranked by their best-matching papers: the sum of each person's top
        per_person BM25 paper scores among the top_papers matches."""
        scores = self.score_query(query)
        if not scores:
            return {}
        top = sorted(((r, s) for r, s in scores.items() if not exclude_rowids or r not in exclude_rowids),
                     key=lambda item: -item[1])[:top_papers]
        by_person: dict[int, list[float]] = {}
        db = self._db()
        for start in range(0, len(top), 500):
            chunk = top[start:start + 500]
            marks = ",".join("?" * len(chunk))
            for rowid, author_id in db.execute(
                    f"SELECT p.rowid, pa.author_id FROM papers p JOIN visible_paper_authors pa ON pa.work_id = p.work_id "
                    f"WHERE p.rowid IN ({marks})", [r for r, _ in chunk]):
                by_person.setdefault(int(author_id), []).append(scores[rowid])
        return {aid: sum(sorted(values, reverse=True)[:per_person]) for aid, values in by_person.items()}

    def authors(self, work_id) -> list[tuple[int, int | None]]:
        rows = self._db().execute(
            "SELECT author_id, position FROM visible_paper_authors WHERE work_id = ? ORDER BY position IS NULL, position",
            (work_key(work_id),)).fetchall()
        return [(int(r["author_id"]), r["position"]) for r in rows]

    def search(self, query: str, from_year: int | None = None, limit: int = 8) -> list[dict]:
        """Title and abstract matches, strongest first: the words as a phrase, then
        every word, then any word. Each row's "match" says which of these found it."""
        words = [w for w in re.findall(r"[a-z0-9][a-z0-9-]*", str(query or "").casefold())
                 if len(w) >= 2 and w not in _SEARCH_STOP][:12]
        if not words:
            return []
        terms = ['"' + w.replace('"', "") + '"' for w in words]
        stages = [("all_words", " AND ".join(terms))]
        if 2 <= len(terms) <= 4:
            stages.insert(0, ("phrase", '"' + " ".join(w.replace('"', "") for w in words) + '"'))
        if len(terms) > 1:
            stages.append(("any_word", " OR ".join(terms)))
        year_clause, params = ("AND p.year >= ?", [from_year]) if from_year else ("", [])
        found: dict[str, dict] = {}
        for match, expression in stages:
            if len(found) >= limit or (match == "any_word" and len(found) >= min(3, limit)):
                break
            rows = self._db().execute(
                f"""SELECT p.rowid, p.*, bm25(papers_fts, 4.0, 1.0) AS rank
                    FROM papers_fts JOIN papers p ON p.rowid = papers_fts.rowid
                    WHERE papers_fts MATCH ? {year_clause}
                    ORDER BY rank LIMIT ?""",
                [expression, *params, limit]).fetchall()
            for row in rows:
                found.setdefault(row["work_id"], {**dict(row), "match": match})
        return list(found.values())[:limit]
