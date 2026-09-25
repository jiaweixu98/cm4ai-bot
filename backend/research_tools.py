"""Runtime-independent, read-only research tools. Never loads an index on import."""

from dataclasses import dataclass, field
import hashlib
import json
import os
import re
import time
from typing import Callable
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request

OPENALEX_BASE = "https://api.openalex.org"
OPENALEX_URL = f"{OPENALEX_BASE}/works"
OPENALEX_FIELDS = "id,display_name,publication_year,doi,abstract_inverted_index,authorships,cited_by_count"
PAPER_FIELDS = (OPENALEX_FIELDS + ",primary_location,type,topics,referenced_works_count")
AUTHOR_FIELDS = ("id,display_name,orcid,works_count,cited_by_count,summary_stats,"
                 "last_known_institutions,affiliations,topics,counts_by_year")
# Each OpenAlex request is billed, so a turn gets a small fixed allowance.
OPENALEX_MAX_REQUESTS = 14
# Repeated questions within a few hours reuse responses instead of spending the daily budget.
OPENALEX_CACHE_SECONDS = 6 * 3600
OPENALEX_CACHE_ENTRIES = 500
_OPENALEX_CACHE: dict[str, tuple[float, dict]] = {}
_DOI = re.compile(r"(10\.\d{4,9}/[^\s\"<>]+)", re.I)
_WORK_ID = re.compile(r"\b(W\d{4,12})\b")
ABSTRACT_CHARS = 1500
RRF_K = 60


def paper_title(paper) -> str:
    if isinstance(paper, str):
        return paper.strip()
    return str(paper.get("Title") or paper.get("title") or "").strip()


def _normalize_title(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(value or "").casefold()))


_STOP = set("a an and or of for in on to the with from by at as into via using based study studies "
            "research model models method methods approach development analysis data new".split())


def _title_tokens(value: str) -> set[str]:
    tokens = set()
    for word in re.findall(r"[a-z0-9]+", str(value or "").casefold()):
        if len(word) < 3 or word in _STOP:
            continue
        tokens.add(word[:-1] if len(word) > 4 and word.endswith("s") else word)
    return tokens


def _abstract_text(inverted: dict | None) -> str:
    if not inverted:
        return ""
    positions = [(pos, word) for word, spots in inverted.items() for pos in spots]
    text = " ".join(word for _, word in sorted(positions))
    return text if len(text) <= ABSTRACT_CHARS else text[:ABSTRACT_CHARS].rsplit(" ", 1)[0] + " …"


def openalex_enabled() -> bool:
    return os.environ.get("MATRIX_OPENALEX", "on").strip().lower() not in {"off", "0", "false", "no"}


@dataclass
class ResearchTools:
    lookup: Callable
    details: Callable
    search: Callable
    path: Callable | None = None
    openalex: bool = False
    people: dict = field(default_factory=dict)
    evidence: dict = field(default_factory=dict)
    result_people: list | None = None
    result_query: str = ""
    calls: list = field(default_factory=list)
    max_calls: int = 20
    openalex_requests: int = 0
    self_id: str = ""
    candidates_reviewed: int = 0
    _niche_reviewed: set = field(default_factory=set)

    def topic_coverage(self, author_id: str, topics: list[str]) -> dict:
        """Count a person's listed papers whose titles contain a topic's key words."""
        raw = self.details(str(author_id)) or {}
        titles = [_title_tokens(paper_title(p)) for p in raw.get("papers", []) if paper_title(p)]
        per_topic, matched = [], set()
        for topic in topics[:6]:
            words = _title_tokens(topic)
            if not words:
                continue
            need = min(2, len(words)) if len(words) <= 4 else max(2, len(words) // 3)
            hits = {i for i, t in enumerate(titles) if len(words & t) >= need}
            matched |= hits
            per_topic.append({"topic": topic, "papers": len(hits), "short": len(words) <= 4})
        return {"listed_papers": len(titles), "matching_papers": len(matched), "per_topic": per_topic}

    def _charge(self, name: str):
        if len(self.calls) >= self.max_calls:
            raise ValueError("Research tool budget reached. Answer using the evidence already read.")
        self.calls.append(name)

    def _remember(self, person: dict):
        known = self.people.get(person["author_id"], {})
        self.people[person["author_id"]] = {**known, **{k: v for k, v in person.items() if v not in (None, "")}}

    def _person(self, author_id: str, question: str = "", limit: int = 10) -> dict | None:
        author_id = str(author_id).strip()
        if not author_id.isdigit() or len(author_id) > 20:
            return None
        raw = self.details(author_id)
        if not raw or not raw.get("name") or raw["name"] in {"Unknown", "Researcher"}:
            return None
        terms = set(re.findall(r"\w+", question.casefold()))
        papers = [p for p in raw.get("papers", []) if paper_title(p)]
        # A lexical ordering of the available titles, not a scientific fit score.
        papers = sorted(papers, key=lambda p: -len(terms & set(re.findall(r"\w+", paper_title(p).casefold()))))
        records = []
        seen = set()
        for paper in papers:
            if len(records) >= limit:
                break
            title = paper_title(paper)
            if title.casefold() in seen:
                continue
            seen.add(title.casefold())
            eid = "paper:" + hashlib.sha256(f"{author_id}:{title}".encode()).hexdigest()[:20]
            record = self.evidence.get(eid) or {"evidence_id": eid, "author_id": author_id, "title": title,
                                                "source": "local_catalog", "text_level": "title"}
            if isinstance(paper, dict):
                for target, keys in {"year": ("PubYear", "year"), "url": ("url", "URL")}.items():
                    value = next((paper[k] for k in keys if paper.get(k)), None)
                    if value and target not in record:
                        record[target] = str(value)
                doi = str(paper.get("DOI") or paper.get("doi") or "").strip()
                pmid = str(paper.get("PMID") or paper.get("pmid") or "").strip()
                if "url" not in record and (doi or pmid.isdigit()):
                    record["url"] = (f"https://doi.org/{doi}" if doi
                                     else f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/")
            self.evidence[eid] = record
            records.append(record)
        affiliation = raw.get("affiliation", "") if raw.get("affiliation") not in {None, "Unknown"} else ""
        profile_eid = f"profile:{author_id}"
        self.evidence[profile_eid] = {"evidence_id": profile_eid, "author_id": author_id,
                                      "source": "catalog_profile", "text_level": "metadata",
                                      "title": f"MATRIX catalog profile: {raw['name']}"
                                               + (f", {affiliation}" if affiliation else "")}
        years = [int(y) for p in raw.get("papers", []) if isinstance(p, dict)
                 for y in [str(p.get("PubYear") or p.get("year") or "")] if y.isdigit()]
        latest = str(raw.get("recent_year") or "") or (str(max(years)) if years else "")
        person = {"author_id": author_id, "name": raw["name"], "affiliation": affiliation,
                  "profile_evidence_id": profile_eid, "latest_year": latest, "papers": records}
        self._remember({k: v for k, v in person.items() if k != "papers"})
        return person

    def resolve_person(self, name: str) -> dict:
        self._charge("resolve_person")
        name = name.strip()[:200]
        rows = self.lookup(name, 8) if len(name) >= 2 else []
        for row in rows:
            self._remember({**row, "author_id": str(row["author_id"])})
        return {"status": "not_found" if not rows else "resolved" if len(rows) == 1 else "ambiguous",
                "people": rows}

    def read_person_evidence(self, author_id: str, question: str) -> dict:
        self._charge("read_person_evidence")
        person = self._person(author_id, question[:2000])
        return {"status": "ok" if person else "not_found", "person": person}

    def _fuse(self, facets: list[str], scope: str, exclude_ids: list[str]) -> list[dict]:
        # Existing retrieval scores are not comparable across independently phrased
        # facets. Fuse ranks instead and use the stable catalog ID for ties.
        fused: dict[str, dict] = {}
        for facet in facets:
            for rank, row in enumerate(self.search(facet, scope, exclude_ids[:25])[:16], start=1):
                author_id = str(row.get("author_id", ""))
                if not author_id or author_id in exclude_ids or author_id == self.self_id:
                    continue
                item = fused.setdefault(author_id, {"row": row, "rrf_score": 0.0,
                                                     "facet_ranks": {}, "matched_requirements": []})
                item["rrf_score"] += 1.0 / (RRF_K + rank)
                item["facet_ranks"][facet] = rank
                item["matched_requirements"].append(facet)
        return sorted(fused.values(), key=lambda item: (-item["rrf_score"], str(item["row"]["author_id"])))

    def search_people(self, question: str, requirements: list[str], scope: str,
                      exclude_ids: list[str]) -> dict:
        self._charge("search_people")
        if scope not in {"all", "bridge2ai"}:
            raise ValueError("Scope must be all or bridge2ai")
        question = question.strip()[:1000]
        facets = []
        for value in requirements[:3]:
            facet = " ".join(str(value).strip().split())[:300]
            if facet and facet.casefold() not in {item.casefold() for item in facets}:
                facets.append(facet)
        if not facets and question:
            facets = [question[:500]]
        if not facets:
            raise ValueError("Describe the research need")

        ranked = self._fuse(facets, scope, exclude_ids)
        people = []
        for item in ranked[:8]:
            row = item["row"]
            person = self._person(str(row["author_id"]), " ".join(facets), 3)
            if person:
                member = bool(row.get("is_bridge2ai_member"))
                self.people[person["author_id"]]["is_bridge2ai_member"] = member
                people.append({**person, "retrieval_score": round(item["rrf_score"], 8),
                               "facet_ranks": item["facet_ranks"],
                               "matched_requirements": item["matched_requirements"],
                               "is_bridge2ai_member": member})
        for person in people:
            person["topics"] = facets
            person["title_coverage"] = self.topic_coverage(person["author_id"], facets)
        self.candidates_reviewed = len(ranked)
        self.result_people, self.result_query = people, question or "; ".join(facets)
        return {"status": "ok" if people else "empty", "question": question,
                "requirements": facets, "scope": scope, "ranking": "rrf-v1", "people": people}

    def get_connection(self, from_id: str, to_id: str) -> dict:
        self._charge("get_connection")
        if self.path is None:
            raise ValueError("Coauthor connections are unavailable")
        ends = [self._person(from_id, limit=0), self._person(to_id, limit=0)]
        if not all(ends):
            return {"status": "not_found", "path": []}
        ids = [ends[0]["author_id"], ends[1]["author_id"]]
        if ids[0] == ids[1]:
            raise ValueError("Choose two different people")
        path = [{"author_id": str(n["id"]), "name": n["name"]} for n in self.path(*ids)]
        for node in path:
            if node["author_id"] not in self.people:
                self._remember(node)
        names = [self.people.get(i, {}).get("name", "") for i in ids]
        eid = "path:" + hashlib.sha256(":".join(ids).encode()).hexdigest()[:20]
        self.evidence[eid] = {"evidence_id": eid, "author_id": None, "source": "coauthor_graph",
                              "text_level": "metadata",
                              "title": ("Coauthor path: " + " → ".join(n["name"] for n in path)) if path
                              else f"No recorded coauthor path between {names[0]} and {names[1]}"}
        return {"status": "connected" if path else "no_recorded_path", "evidence_id": eid,
                "coauthor_hops": max(len(path) - 1, 0), "path": path}

    def read_context(self, profile_id: str, selected_ids: list[str], displayed_ids: list[str]) -> dict:
        self._charge("read_context")
        def read(ids):
            return [p for aid in ids[:8] if (p := self._person(aid, limit=3))]
        return {"profile_context": self._person(profile_id, limit=3),
                "selected_people": read(selected_ids), "displayed_people_in_order": read(displayed_ids)}

    # ---------- OpenAlex (approved external source) ----------

    def _openalex_raw(self, path: str, params: dict, select: str | None = OPENALEX_FIELDS) -> dict:
        if not self.openalex:
            raise ValueError("External literature lookup is not enabled")
        params = dict(params)
        if select:
            params["select"] = select
        cache_key = f"{path}?{urllib_parse.urlencode(sorted(params.items()))}"
        cached = _OPENALEX_CACHE.get(cache_key)
        if cached and time.monotonic() - cached[0] < OPENALEX_CACHE_SECONDS:
            return cached[1]
        if self.openalex_requests >= OPENALEX_MAX_REQUESTS:
            raise ValueError("Literature lookup allowance for this turn is used up")
        self.openalex_requests += 1
        payload = self._openalex_fetch(path, params)
        if len(_OPENALEX_CACHE) >= OPENALEX_CACHE_ENTRIES:
            _OPENALEX_CACHE.pop(next(iter(_OPENALEX_CACHE)))
        _OPENALEX_CACHE[cache_key] = (time.monotonic(), payload)
        return payload

    def _openalex_fetch(self, path: str, params: dict) -> dict:
        if key := os.environ.get("OPENALEX_API_KEY", "").strip():
            params["api_key"] = key
        if mailto := os.environ.get("OPENALEX_MAILTO", "").strip():
            params["mailto"] = mailto
        url = f"{OPENALEX_BASE}/{path}?{urllib_parse.urlencode(params)}"
        try:
            with urllib_request.urlopen(urllib_request.Request(url, headers={"Accept": "application/json"}),
                                        timeout=8.0) as response:
                return json.load(response)
        except urllib_error.HTTPError as exc:
            if exc.code == 404:
                return {}
            if exc.code == 429:
                raise ValueError("Publication counts are unavailable; answer from catalog evidence "
                                 "without counts and without mentioning this") from None
            raise ValueError("OpenAlex is unavailable right now; continue with catalog evidence") from None
        except (urllib_error.URLError, TimeoutError, ValueError):
            raise ValueError("OpenAlex is unavailable right now; continue with catalog evidence") from None

    def _openalex_get(self, params: dict) -> list[dict]:
        return self._openalex_raw("works", params).get("results") or []

    def read_abstracts(self, evidence_ids: list[str]) -> dict:
        self._charge("read_abstracts")
        results = {}
        for eid in list(dict.fromkeys(evidence_ids))[:5]:
            record = self.evidence.get(eid)
            if not record:
                results[eid] = {"status": "unknown_evidence_id"}
                continue
            if record.get("abstract"):
                results[eid] = {"status": "ok", "title": record["title"], "abstract": record["abstract"]}
                continue
            wanted = _normalize_title(record["title"])
            query = " ".join(wanted.split()[:30])
            try:
                found = self._openalex_get({"filter": f"title.search:{query}", "per-page": 3})
            except ValueError as exc:
                results[eid] = {"status": "unavailable", "reason": str(exc)}
                break
            match = next((w for w in found if _normalize_title(w.get("display_name")) == wanted), None)
            abstract = _abstract_text(match.get("abstract_inverted_index")) if match else ""
            if not abstract:
                results[eid] = {"status": "no_abstract_available", "title": record["title"]}
                continue
            record.update(abstract=abstract, text_level="abstract", openalex_id=match["id"])
            if match.get("doi") and not record.get("url"):
                record["url"] = match["doi"]
            results[eid] = {"status": "ok", "title": record["title"], "abstract": abstract}
        return {"abstracts": results}

    def search_literature(self, query: str, from_year: int | None = None) -> dict:
        self._charge("search_literature")
        words = re.findall(r"[\w-]+", query)[:20]
        if not words:
            raise ValueError("Give short topic keywords")
        params = {"search": " ".join(words), "per-page": 8}
        if from_year and 1900 < from_year < 2100:
            params["filter"] = f"from_publication_date:{from_year}-01-01"
        works = [self._public(self._work_record(work)) for work in self._openalex_get(params)
                 if str(work.get("display_name") or "").strip()]
        return {"status": "ok" if works else "empty", "works": works}

    @staticmethod
    def _public(record: dict) -> dict:
        return {k: v for k, v in record.items() if k not in {"author_id", "openalex_id"}}

    def _work_record(self, work: dict) -> dict:
        eid = "work:" + hashlib.sha256(str(work["id"]).encode()).hexdigest()[:20]
        authors = [a["author"]["display_name"] for a in work.get("authorships", [])[:6]
                   if a.get("author", {}).get("display_name")]
        record = {"evidence_id": eid, "author_id": None, "title": str(work.get("display_name") or "").strip(),
                  "year": str(work.get("publication_year") or ""), "url": work.get("doi") or work["id"],
                  "authors": authors, "source": "openalex", "openalex_id": work["id"],
                  "cited_by_count": work.get("cited_by_count"),
                  "abstract": _abstract_text(work.get("abstract_inverted_index"))}
        record["text_level"] = "abstract" if record["abstract"] else "title"
        self.evidence[eid] = {**self.evidence.get(eid, {}), **record}
        return self.evidence[eid]

    # ---------- Author and paper checks ----------

    def _match_openalex_author(self, candidates: list[dict], local: dict | None) -> tuple[dict | None, str]:
        """Link a catalog person to an OpenAlex author only on a shared publication
        or a unique institution match; never on the name alone when several exist."""
        if not candidates:
            return None, "no_external_record"
        short = {c["id"].rsplit("/", 1)[-1]: c for c in candidates}
        for paper in (local or {}).get("papers", [])[:2]:
            wanted = _normalize_title(paper["title"]).split()[:10]
            if len(wanted) < 3:
                continue
            works = self._openalex_raw("works", {
                "filter": f"authorships.author.id:{'|'.join(short)},title.search:{' '.join(wanted)}",
                "per-page": 5}, "id,display_name,authorships").get("results") or []
            for work in works:
                if _normalize_title(work.get("display_name")).split()[:10] != wanted:
                    continue
                for authorship in work.get("authorships", []):
                    aid = str(authorship.get("author", {}).get("id", "")).rsplit("/", 1)[-1]
                    if aid in short:
                        return short[aid], "shared_publication"
        if local and local.get("affiliation"):
            stop = {"university", "of", "the", "and", "school", "medicine", "medical", "center", "institute", "college"}
            wanted = set(re.findall(r"[a-z]{4,}", local["affiliation"].casefold())) - stop
            hits = [c for c in candidates
                    if wanted & (set(re.findall(r"[a-z]{4,}", " ".join(
                        i.get("display_name", "") for i in (c.get("last_known_institutions") or [])
                        + [a.get("institution", {}) for a in (c.get("affiliations") or [])]).casefold())) - stop)]
            if len(hits) == 1:
                return hits[0], "name_and_institution"
        if not local and len(candidates) == 1:
            return candidates[0], "single_name_match"
        return None, "ambiguous"

    def get_author_info(self, author_id: str = "", name: str = "") -> dict:
        self._charge("get_author_info")
        author_id, name = str(author_id or "").strip(), " ".join(str(name or "").split())[:200]
        local = self._person(author_id, name, 5) if author_id else None
        if not local and name:
            rows = self.lookup(name, 8) if len(name) >= 2 else []
            for row in rows:
                self._remember({**row, "author_id": str(row["author_id"])})
            if len(rows) > 1:
                return {"status": "ambiguous", "people": rows,
                        "note": "Several catalog people share this name; ask which one or pass author_id."}
            if rows:
                local = self._person(str(rows[0]["author_id"]), "", 5)
        query_name = (local or {}).get("name") or name
        if not query_name:
            raise ValueError("Give a catalog author_id or a name")
        result = {"status": "ok" if local else "not_in_catalog",
                  "catalog": None if not local else {
                      "author_id": local["author_id"], "name": local["name"],
                      "affiliation": local.get("affiliation", ""),
                      "is_bridge2ai_member": bool(self.people.get(local["author_id"], {}).get("is_bridge2ai_member")),
                      "catalog_papers": [{"evidence_id": p["evidence_id"], "title": p["title"], "year": p.get("year", "")}
                                         for p in local["papers"]]},
                  "openalex": None}
        if not self.openalex:
            result["openalex_status"] = "disabled"
            return result
        try:
            candidates = self._openalex_raw("authors", {"search": query_name, "per-page": 5},
                                            AUTHOR_FIELDS).get("results") or []
            match, basis = self._match_openalex_author(candidates, local)
        except ValueError as exc:
            result["openalex_status"] = str(exc)
            return result
        if not match:
            result["openalex_status"] = basis
            if basis == "ambiguous":
                result["openalex_candidates"] = [{
                    "name": c.get("display_name"), "works_count": c.get("works_count"),
                    "institutions": [i.get("display_name") for i in (c.get("last_known_institutions") or [])][:2]}
                    for c in candidates[:5]]
            if not local:
                result["status"] = "not_found" if basis == "no_external_record" else "ambiguous"
            return result
        stats = match.get("summary_stats") or {}
        eid = "author:" + hashlib.sha256(match["id"].encode()).hexdigest()[:20]
        display = match.get("display_name") or query_name
        self.evidence[eid] = {"evidence_id": eid, "author_id": (local or {}).get("author_id"),
                              "title": f"OpenAlex author profile: {display}", "url": match["id"],
                              "source": "openalex_author", "text_level": "metadata", "openalex_id": match["id"]}
        years = sorted((row for row in match.get("counts_by_year") or [] if row.get("year")),
                       key=lambda row: row["year"], reverse=True)[:6]
        result["openalex_status"] = "linked"
        result["openalex"] = {
            "evidence_id": eid, "identity_basis": basis, "display_name": display,
            "openalex_id": match["id"], "orcid": match.get("orcid"),
            "works_count": match.get("works_count"), "cited_by_count": match.get("cited_by_count"),
            "h_index": stats.get("h_index"), "i10_index": stats.get("i10_index"),
            "institutions": [i.get("display_name") for i in (match.get("last_known_institutions") or [])][:3],
            "top_topics": [{"topic": t.get("display_name"), "works": t.get("count")}
                           for t in (match.get("topics") or [])[:6]],
            "works_per_year": {str(row["year"]): row.get("works_count", 0) for row in years}}
        return result

    def get_paper_info(self, identifier: str) -> dict:
        self._charge("get_paper_info")
        identifier = str(identifier or "").strip()[:500]
        if not identifier:
            raise ValueError("Give a DOI, OpenAlex work ID, evidence ID or title")
        known = self.evidence.get(identifier)
        work = None
        if known and known.get("openalex_id"):
            work = self._openalex_raw(f"works/{known['openalex_id'].rsplit('/', 1)[-1]}", {}, PAPER_FIELDS)
        elif doi := _DOI.search(identifier if not known else known.get("url", "")):
            work = self._openalex_raw(f"works/doi:{doi.group(1).rstrip('.').lower()}", {}, PAPER_FIELDS)
        elif (wid := _WORK_ID.search(identifier)) and not known:
            work = self._openalex_raw(f"works/{wid.group(1)}", {}, PAPER_FIELDS)
        match_basis = "identifier"
        if not work:
            title = known["title"] if known else identifier
            wanted = _normalize_title(title)
            if len(wanted.split()) < 3:
                return {"status": "not_found", "reason": "Give a fuller title, DOI or OpenAlex ID"}
            found = self._openalex_raw("works", {"filter": f"title.search:{' '.join(wanted.split()[:30])}",
                                                 "per-page": 5}, PAPER_FIELDS).get("results") or []
            work = next((w for w in found if _normalize_title(w.get("display_name")) == wanted), None)
            match_basis = "exact_title"
            if not work and found and not known:
                work, match_basis = found[0], "closest_title"
        if not work or not work.get("id"):
            return {"status": "not_found"}
        record = self._work_record(work)
        if known and match_basis in {"identifier", "exact_title"} and known is not record:
            # Enrich the catalog record the user already sees instead of a duplicate.
            known.update(abstract=record["abstract"] or known.get("abstract", ""), openalex_id=work["id"],
                         text_level="abstract" if record["abstract"] or known.get("abstract") else known["text_level"])
            self.evidence.pop(record["evidence_id"], None)
            record = {**record, **known}
        title_key = _normalize_title(work.get("display_name"))
        authors, catalog_people = [], []
        for authorship in (work.get("authorships") or [])[:12]:
            person = authorship.get("author") or {}
            name = person.get("display_name")
            if not name:
                continue
            authors.append({"name": name, "position": authorship.get("author_position"),
                            "orcid": person.get("orcid"),
                            "institutions": [i.get("display_name") for i in authorship.get("institutions") or []][:2]})
            for row in self.lookup(name, 3)[:2]:
                raw = self.details(str(row["author_id"])) or {}
                if any(_normalize_title(paper_title(p)) == title_key for p in raw.get("papers", [])):
                    self._remember({**row, "author_id": str(row["author_id"])})
                    catalog_people.append({"author_id": str(row["author_id"]), "name": row["name"],
                                           "affiliation": row.get("affiliation", ""),
                                           "link_basis": "paper_listed_in_catalog_profile"})
                    break
        venue = ((work.get("primary_location") or {}).get("source") or {}).get("display_name")
        return {"status": "ok", "match_basis": match_basis, "evidence_id": record["evidence_id"],
                "title": record["title"], "year": record.get("year"), "venue": venue, "type": work.get("type"),
                "doi": work.get("doi"), "cited_by_count": work.get("cited_by_count"),
                "references_count": work.get("referenced_works_count"),
                "topics": [t.get("display_name") for t in (work.get("topics") or [])[:4]],
                "authors": authors, "catalog_people": catalog_people,
                "abstract": record.get("abstract") or "", "text_level": record.get("text_level", "title")}

    # ---------- Team assembly ----------

    def assemble_team(self, goal: str, roles: list[str], scope: str, exclude_ids: list[str]) -> dict:
        self._charge("assemble_team")
        if scope not in {"all", "bridge2ai"}:
            raise ValueError("Scope must be all or bridge2ai")
        cleaned = []
        for value in roles[:5]:
            role = " ".join(str(value).split())[:300]
            if role and role.casefold() not in {r.casefold() for r in cleaned}:
                cleaned.append(role)
        if len(cleaned) < 2:
            raise ValueError("Give 2 to 5 distinct roles the team needs")
        excluded = [str(e) for e in exclude_ids[:25]]
        fused = {role: self._fuse([role], scope, excluded) for role in cleaned}
        role_ranks = {role: [str(item["row"]["author_id"]) for item in items] for role, items in fused.items()}
        for items in fused.values():
            for item in items:
                aid = str(item["row"]["author_id"])
                if aid in self.people or self._person(aid, limit=0):
                    self.people[aid]["is_bridge2ai_member"] = bool(item["row"].get("is_bridge2ai_member"))
        chosen, slots = set(), []
        for role in cleaned:
            ranked = [aid for aid in role_ranks[role] if aid not in chosen]
            pick = next((aid for aid in ranked if self._person(aid, role, 3)), None)
            if pick:
                chosen.add(pick)
            slots.append({"role": role, "member_id": pick,
                          "alternate_ids": [aid for aid in ranked if aid != pick][:2]})
        members, coverage = [], []
        for slot in slots:
            aid = slot["member_id"]
            alternates = [p for a in slot["alternate_ids"] if (p := self._person(a, slot["role"], 2))]
            for alt in alternates:
                alt.pop("papers", None)
            if not aid:
                coverage.append({"role": slot["role"], "status": "no_candidate_found", "alternates": alternates})
                continue
            person = self._person(aid, slot["role"], 3)
            also = [role for role in cleaned if role != slot["role"] and aid in role_ranks[role][:16]]
            member = {**person, "team_role": slot["role"], "topics": [slot["role"]],
                      "role_rank": role_ranks[slot["role"]].index(aid) + 1,
                      "also_matches_roles": also,
                      "is_bridge2ai_member": bool(self.people.get(aid, {}).get("is_bridge2ai_member"))}
            members.append(member)
            coverage.append({"role": slot["role"], "status": "filled", "member": person["name"],
                             "author_id": aid, "alternates": alternates})
        self.candidates_reviewed = len({str(i["row"]["author_id"]) for items in fused.values() for i in items})
        self.result_people, self.result_query = members, goal.strip()[:200] or "; ".join(cleaned)
        return {"status": "ok" if members else "empty", "goal": goal[:500], "ranking": "greedy-role-order-v1",
                "members_in_role_order": members, "coverage": coverage}

    # ---------- Niche analysis ----------

    def analyze_niche(self, concepts: list[str], from_year: int | None = None) -> dict:
        """Publication counts for concepts and their intersection, plus catalog people near it."""
        self._charge("analyze_niche")
        import datetime
        this_year = datetime.date.today().year
        start = from_year if from_year and 1950 < from_year < this_year else this_year - 10
        terms = []
        for value in concepts[:3]:
            term = " ".join(re.findall(r"[\w-]+", str(value)))[:80]
            if term and term.casefold() not in {t.casefold() for t in terms}:
                terms.append(term)
        if not terms:
            raise ValueError("Give 1 to 3 short concept phrases")
        date = f"from_publication_date:{start}-01-01"
        filters = [f"title_and_abstract.search:{term}" for term in terms]
        joint = ",".join(filters + [date])
        local = []
        nearby = self._fuse([" ".join(terms)], "all", [])
        self._niche_reviewed |= {str(item["row"]["author_id"]) for item in nearby}
        self.candidates_reviewed = len(self._niche_reviewed)
        for item in nearby[:6]:
            person = self._person(str(item["row"]["author_id"]), " ".join(terms), 2)
            if person:
                member = bool(item["row"].get("is_bridge2ai_member"))
                self.people[person["author_id"]]["is_bridge2ai_member"] = member
                local.append({**person, "is_bridge2ai_member": member,
                              "title_coverage": self.topic_coverage(person["author_id"], terms)})
        # Niche researchers become card candidates, labelled with the niche they sit near.
        # Someone near several niches gets no single niche label.
        self.result_people = self.result_people or []
        known = {str(p.get("author_id")): p for p in self.result_people}
        for person in local:
            if str(person["author_id"]) in known:
                earlier = known[str(person["author_id"])]
                earlier["team_role"] = ""
                earlier["topics"] = list(dict.fromkeys(earlier.get("topics", []) + terms))
            else:
                self.result_people.append({**person, "team_role": " + ".join(terms), "topics": terms})
        try:
            groups = self._openalex_raw("works", {"filter": joint, "group_by": "publication_year"},
                                        None).get("group_by") or []
            recent_filter = ",".join(filters + [f"from_publication_date:{max(start, this_year - 3)}-01-01"])
            top = self._openalex_raw("works", {"filter": recent_filter, "sort": "cited_by_count:desc",
                                               "per-page": 4}).get("results") or []
        except ValueError as exc:
            # Catalog researchers near the intersection are still useful without counts.
            return {"status": "partial", "reason": str(exc), "concepts": terms,
                    "layout": "Present this niche like the others: bold title, Why it fits you, and "
                              "its researchers as cards; no Evidence or Recent work lines, no remarks "
                              "on crowding or verification.",
                    "catalog_researchers_near_niche": local}
        by_year = {int(g["key"]): g["count"] for g in groups if str(g.get("key", "")).isdigit()}
        by_year = {year: by_year.get(year, 0) for year in range(start, this_year + 1)}
        total = sum(by_year.values())
        recent = sum(by_year.get(y, 0) for y in range(this_year - 3, this_year))
        prior = sum(by_year.get(y, 0) for y in range(this_year - 6, this_year - 3))
        representative = [self._public(self._work_record(w)) for w in top if w.get("display_name")]
        for work in representative:
            work.pop("abstract", None)
        label = " + ".join(terms)
        eid = "niche:" + hashlib.sha256(f"{label}:{start}".encode()).hexdigest()[:20]
        query = urllib_parse.quote(joint, safe=":,")
        self.evidence[eid] = {"evidence_id": eid, "author_id": None, "source": "openalex_counts",
                              "text_level": "metadata", "year": f"{start}–{this_year}",
                              "title": f"OpenAlex publication counts for {label}",
                              "url": f"https://openalex.org/works?filter={query}"}
        return {"status": "ok", "evidence_id": eid, "concepts": terms, "window": f"{start}-{this_year}",
                "count_basis": "OpenAlex works whose title or abstract match every concept",
                "joint_works": total,
                "works_by_year": {str(y): c for y, c in by_year.items()},
                "last_3_complete_years": recent, "previous_3_years": prior,
                "growth_ratio": round(recent / prior, 2) if prior else None,
                "most_cited_recent_joint_works": representative, "catalog_researchers_near_niche": local}
