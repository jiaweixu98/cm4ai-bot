"""Build the MATRIX data snapshot from a TKG export.

Inputs, from one export (the October layout; older exports use 02_data and
03_cheaha for the table and vector folders):

    <export>/01_deployed_data/author_metadata.json
    <export>/01_deployed_data/author_collab_dataset.json
    <export>/01_deployed_data/tkg_ebd_89k_dataset.json, layout_manifest.json
    <export>/01_deployed_data/tkg_merge_map_*.csv
    <export>/05_database_tables/papers.csv, authorships.csv, collaborator_papers.csv,
        paper_topics.csv, author_affiliations.csv, paper_affiliations.csv
    <export>/04_cheaha/author_embeddings.npy, author_embedding_ids.json
    <export>/04_cheaha/paper_embeddings.npy, paper_embedding_ids.json

Writes the files data_loader.py reads, papers.sqlite, and the checksummed graph/
folder into --out (a new directory), and optionally the per-person paper shards
the graph app serves from --graph-papers-out.

    backend/.venv/bin/python backend/scripts/build_tkg_snapshot.py \\
        --export ~/box_upload_20261002/box_upload_20261002 \\
        --out data/tkg-20261002 --snapshot-version tkg-20261002.3

Check a built snapshot against the graph metadata it will serve beside
(exits non-zero on any mismatch):

    backend/.venv/bin/python backend/scripts/build_tkg_snapshot.py \\
        --check data/tkg-20261002 \\
        --graph-metadata data/tkg-20261002/graph/author_metadata.json

paper_faiss_index.bin (and its .json stamp) is not built or checksummed here. It
is a cache that paper_vectors.py writes on first use from the vectors, and it is
rebuilt whenever its stamp (row count, size and the vector and id checksums that
are in this manifest) no longer matches.
"""

import argparse
import csv
import glob
import hashlib
import json
import os
import pickle
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paper_library import PAPERS_DB, doi_key, title_key  # noqa: E402
from tkg_publications import displayable_affiliations  # noqa: E402
from tkg_release import publication_metadata, write_graph_release  # noqa: E402

RECENT_N = 6
CITED_N = 6
GRAPH_PAPER_SHARDS = 256


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def node_id(openalex_id: str) -> int:
    return int(str(openalex_id).strip().lstrip("A"))


def load_merge_map(deployed_dir: str, metadata: list) -> dict:
    merge = {}
    for path in sorted(glob.glob(os.path.join(deployed_dir, "tkg_merge_map_*.csv"))):
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                merge[int(row["old_node_id"])] = int(row["new_node_id"])
    for row in metadata:
        for old in row.get("MergedFrom") or []:
            merge.setdefault(int(old), int(row["id"]))
    return merge


def to_int(value):
    """An integer from CSV text that may be "2020", "2020.0" or empty."""
    text = str(value or "").strip()
    return int(float(text)) if text else None


def load_papers(path: str) -> dict:
    papers = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            title = (row.get("title") or "").strip()
            if not title or row.get("non_publication"):
                continue
            doi = (row.get("doi") or "").strip()
            if doi and not doi.startswith("http"):
                doi = f"https://doi.org/{doi}"
            pmid = (row.get("pmid") or "").strip()
            year = row.get("publication_year") or ""
            cited = row.get("cited_by_count") or ""
            papers[row["openalex_id"]] = {
                "title": title,
                "year": to_int(year),
                "cited_by": to_int(cited) or 0,
                "venue": (row.get("venue") or "").strip() or None,
                "doi": doi or None,
                "pmid": pmid if pmid.isdigit() else None,
            }
    return papers


def load_author_papers(path: str, merge: dict, papers: dict) -> dict:
    by_author = defaultdict(set)
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pid = row["paper_id"]
            if pid not in papers:
                continue
            aid = node_id(row["author_id"])
            by_author[merge.get(aid, aid)].add(pid)
    return by_author


def add_collaborator_papers(tables, merge, papers, author_papers):
    """Retain collaborators' own selected work, excluding known non-publication IDs."""
    with open(os.path.join(tables, "papers.csv"), newline="", encoding="utf-8") as f:
        blocked = {r["openalex_id"] for r in csv.DictReader(f) if r.get("non_publication")}
    with open(os.path.join(tables, "collaborator_papers.csv"), newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pid = row["openalex_id"]
            if pid in blocked or not (row.get("title") or "").strip():
                continue
            doi = (row.get("doi") or "").strip()
            if doi and not doi.startswith("http"): doi = f"https://doi.org/{doi}"
            papers.setdefault(pid, {"title": row["title"].strip(),
                "year": to_int(row.get("publication_year")),
                "cited_by": to_int(row.get("cited_by_count")) or 0, "venue": None,
                "doi": doi or None, "pmid": row.get("pmid") or None})
            aid = node_id(row["author_id"])
            author_papers[merge.get(aid, aid)].add(pid)
    return blocked


def select_papers(paper_ids, papers: dict) -> list:
    rows = [(pid, papers[pid]) for pid in paper_ids]
    recent = sorted(rows, key=lambda r: (-(r[1]["year"] or 0), -r[1]["cited_by"], r[0]))
    cited = sorted(rows, key=lambda r: (-r[1]["cited_by"], -(r[1]["year"] or 0), r[0]))
    picked, seen = [], set()
    for pool, limit in ((recent, RECENT_N), (cited, CITED_N)):
        taken = 0
        for pid, p in pool:
            if taken >= limit:
                break
            if pid in seen:
                continue
            seen.add(pid)
            picked.append((pid, p))
            taken += 1
    return picked


def paper_url(p: dict):
    if p["doi"]:
        return p["doi"]
    if p["pmid"]:
        return f"https://pubmed.ncbi.nlm.nih.gov/{p['pmid']}/"
    return None


def build_nodes(people: list, author_papers: dict, papers: dict):
    nodes, graph_papers = {}, {}
    for row in people:
        nid = int(row["id"])
        selected = select_papers(author_papers.get(nid, ()), papers)
        matrix_papers = [
            {
                "Title": p["title"],
                "PubYear": p["year"],
                "CitedCount": p["cited_by"],
                "Venue": p["venue"] or "",
                "DOI": p["doi"],
                "PMID": p["pmid"],
                "OpenAlexWork": pid,
            }
            for pid, p in selected
        ]
        graph_papers[nid] = [
            {
                "title": p["title"],
                "journal": p["venue"],
                "cited_by": p["cited_by"],
                "year": p["year"],
                "doi": p["doi"],
                "pmid": p["pmid"],
                "url": paper_url(p),
                "work_id": pid,
            }
            for pid, p in selected
        ]
        name = (row.get("FullName") or "").strip() or "Unknown"
        features = {
            "AID": str(nid),
            "FullName": name,
            "Affiliation": str(row.get("Affiliation") or ""),
            "BeginYear": row.get("BeginYear"),
            "RecentYear": row.get("RecentYear"),
            "PaperNum": int(row.get("PaperNum") or 0),
            "IsAuthor": True,
            "Bridge2AISeedAuthor": row.get("color_category") == 2,
            "ORCID": (row.get("orcid") or "").strip() or None,
            "OpenAlexId": row.get("openalex_id"),
            "topics": list(row.get("topics") or []),
            "fields": list(row.get("fields") or []),
            "mesh": list(row.get("mesh") or []),
            "affiliations": displayable_affiliations(row),
            "Top Cited or Most Recent Papers": matrix_papers,
        }
        nodes[str(nid)] = {"features": features, "title": name}
    return nodes, graph_papers


def build_graph(collab_path: str, merge: dict, keep: set) -> dict:
    with open(collab_path, encoding="utf-8") as f:
        raw = json.load(f)
    graph = {}
    for src, neighbors in raw.items():
        s = merge.get(int(src), int(src))
        if s not in keep:
            continue
        out = graph.setdefault(str(s), set())
        for nb in neighbors:
            n = merge.get(int(nb), int(nb))
            if n in keep and n != s:
                out.add(str(n))
    return {k: sorted(v, key=int) for k, v in graph.items()}


def build_indexes(emb_dir: str, keep: set, core: set, out_dir: str) -> None:
    import faiss

    ids = json.load(open(os.path.join(emb_dir, "author_embedding_ids.json"), encoding="utf-8"))
    X = np.load(os.path.join(emb_dir, "author_embeddings.npy"), mmap_mode="r")
    if len(ids) != X.shape[0]:
        raise SystemExit(f"vector count {X.shape[0]} != id count {len(ids)}")
    rows = [i for i, nid in enumerate(ids) if int(nid) in keep]
    author_ids = [str(ids[i]) for i in rows]
    core_rows = [i for i in rows if int(ids[i]) in core]
    core_ids = [str(ids[i]) for i in core_rows]

    for name, sel, sel_ids in (("faiss_index.bin", rows, author_ids),
                               ("faiss_core_index.bin", core_rows, core_ids)):
        mat = np.ascontiguousarray(X[sel], dtype=np.float32)
        index = faiss.IndexFlatL2(mat.shape[1])
        index.add(mat)
        faiss.write_index(index, os.path.join(out_dir, name))
        ids_name = "author_ids.pkl" if name == "faiss_index.bin" else "core_ids.pkl"
        with open(os.path.join(out_dir, ids_name), "wb") as f:
            pickle.dump(sel_ids, f)
        log(f"  {name}: {len(sel_ids):,} vectors, dim {mat.shape[1]}")
        del mat, index


def build_paper_db(tables: str, merge: dict, keep: set, out_dir: str) -> dict:
    """papers.sqlite: every titled paper in the export, its catalog authors, and a
    title/abstract full-text index, so the research tools answer paper questions
    from the export before calling OpenAlex."""
    import sqlite3

    csv.field_size_limit(sys.maxsize)
    path = os.path.join(out_dir, PAPERS_DB)
    if os.path.exists(path + ".tmp"):
        os.remove(path + ".tmp")
    db = sqlite3.connect(path + ".tmp")
    db.executescript("""
        PRAGMA journal_mode = OFF;
        PRAGMA synchronous = OFF;
        CREATE TABLE papers (work_id TEXT PRIMARY KEY, title TEXT NOT NULL, abstract TEXT, year INTEGER,
                             venue TEXT, doi TEXT, doi_key TEXT, pmid TEXT, cited_by INTEGER,
                             primary_topic TEXT, primary_field TEXT, title_key TEXT);
        CREATE TABLE paper_authors (work_id TEXT NOT NULL, author_id INTEGER NOT NULL, position INTEGER,
                                    PRIMARY KEY (work_id, author_id));
    """)

    def number(value):
        return int(float(value)) if str(value or "").strip() else None

    def paper_row(row, venue=None):
        doi = (row.get("doi") or "").strip()
        pmid = (row.get("pmid") or "").strip()
        return (row["openalex_id"], row["title"].strip(), (row.get("abstract") or "").strip() or None,
                number(row.get("publication_year")), venue, f"https://doi.org/{doi_key(doi)}" if doi else None,
                doi_key(doi) or None, pmid if pmid.isdigit() else None, number(row.get("cited_by_count")) or 0,
                title_key(row["title"]))

    insert = ("INSERT INTO papers (work_id, title, abstract, year, venue, doi, doi_key, pmid, cited_by, title_key) "
              "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)")
    blocked = set()
    with open(os.path.join(tables, "papers.csv"), newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("non_publication"):
                blocked.add(r["openalex_id"])
            elif (r.get("title") or "").strip():
                db.execute(insert, paper_row(r, (r.get("venue") or "").strip() or None))

    def add_author(rows):
        for r in rows:
            aid = node_id(r["author_id"])
            aid = merge.get(aid, aid)
            if aid in keep:
                yield r.get("paper_id") or r["openalex_id"], aid, number(r.get("author_position"))

    with open(os.path.join(tables, "authorships.csv"), newline="", encoding="utf-8") as f:
        db.executemany("INSERT OR IGNORE INTO paper_authors VALUES (?, ?, ?)", add_author(csv.DictReader(f)))
    # Collaborators' papers carry their own text; they fill papers and abstracts papers.csv lacks.
    with open(os.path.join(tables, "collaborator_papers.csv"), newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["openalex_id"] in blocked or not (r.get("title") or "").strip():
                continue
            db.execute(insert.replace("INSERT", "INSERT OR IGNORE"), paper_row(r))
            if (r.get("abstract") or "").strip():
                db.execute("UPDATE papers SET abstract = ? WHERE work_id = ? AND abstract IS NULL",
                           (r["abstract"].strip(), r["openalex_id"]))
            db.executemany("INSERT OR IGNORE INTO paper_authors VALUES (?, ?, ?)", add_author([r]))
    with open(os.path.join(tables, "paper_topics.csv"), newline="", encoding="utf-8") as f:
        db.executemany("UPDATE papers SET primary_topic = ?, primary_field = ? WHERE work_id = ?",
                       ((r["topic_name"], r["field_name"], r["paper_id"])
                        for r in csv.DictReader(f) if r.get("is_primary") == "t"))
    db.executescript("""
        DELETE FROM paper_authors WHERE work_id NOT IN (SELECT work_id FROM papers);
        CREATE INDEX papers_doi ON papers (doi_key);
        CREATE INDEX papers_title ON papers (title_key);
        CREATE INDEX paper_authors_author ON paper_authors (author_id);
        CREATE VIRTUAL TABLE papers_fts USING fts5(title, abstract, content='papers', content_rowid='rowid',
                                                   tokenize='porter unicode61');
        INSERT INTO papers_fts (papers_fts) VALUES ('rebuild');
    """)
    db.commit()
    stats = {
        "papers": db.execute("SELECT COUNT(*) FROM papers").fetchone()[0],
        "with_abstract": db.execute("SELECT COUNT(*) FROM papers WHERE abstract IS NOT NULL").fetchone()[0],
        "author_links": db.execute("SELECT COUNT(*) FROM paper_authors").fetchone()[0],
    }
    db.execute("VACUUM")
    db.close()
    os.replace(path + ".tmp", path)
    log(f"  {PAPERS_DB}: {stats['papers']:,} papers, {stats['with_abstract']:,} with abstracts, "
        f"{stats['author_links']:,} catalog author links")
    return stats


def write_graph_paper_shards(graph_papers: dict, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    shards = defaultdict(dict)
    for nid, rows in graph_papers.items():
        if rows:
            shards[nid % GRAPH_PAPER_SHARDS][str(nid)] = rows
    for shard in range(GRAPH_PAPER_SHARDS):
        path = os.path.join(out_dir, f"{shard:03d}.json")
        with open(path + ".tmp", "w", encoding="utf-8") as f:
            json.dump(shards.get(shard, {}), f, ensure_ascii=False, separators=(",", ":"))
        os.replace(path + ".tmp", path)
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"shards": GRAPH_PAPER_SHARDS, "key": "node_id % shards",
                   "people": sum(1 for r in graph_papers.values() if r)}, f, indent=2)


def dump_json(obj, path: str) -> None:
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, separators=(",", ":"))
    os.replace(path + ".tmp", path)


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


SNAPSHOT_FILES = (
    "updated_author_nodes_with_papers.json",
    "author_n_publications.json",
    "author_knowledge_graph_2024.json",
    "faiss_index.bin",
    "author_ids.pkl",
    "faiss_core_index.bin",
    "core_ids.pkl",
    PAPERS_DB,
)


def check_snapshot(snapshot: str, graph_metadata: str) -> int:
    import faiss

    problems = []
    missing = [n for n in SNAPSHOT_FILES if not os.path.exists(os.path.join(snapshot, n))]
    if missing:
        print(f"FAIL missing files: {missing}")
        return 1

    metadata = json.load(open(graph_metadata, encoding="utf-8"))
    graph_ids = {int(r["id"]) for r in metadata}
    graph_core = {int(r["id"]) for r in metadata if r.get("color_category") == 2}
    merged_away = {int(old) for r in metadata for old in (r.get("MergedFrom") or [])}
    del metadata

    nodes = json.load(open(os.path.join(snapshot, "updated_author_nodes_with_papers.json"), encoding="utf-8"))
    node_ids = {int(k) for k in nodes}
    snapshot_core = {int(k) for k, v in nodes.items() if v["features"].get("Bridge2AISeedAuthor")}
    del nodes

    if node_ids - graph_ids:
        problems.append(f"{len(node_ids - graph_ids)} snapshot people are not in the graph")
    if node_ids & merged_away:
        problems.append(f"{len(node_ids & merged_away)} snapshot people are merged-away ids")
    if not snapshot_core <= graph_core:
        problems.append(f"{len(snapshot_core - graph_core)} snapshot core people are not core in the graph")
    if len(graph_core - snapshot_core) > len(graph_core) * 0.02:
        problems.append(f"{len(graph_core - snapshot_core)} graph core people are missing from the snapshot")

    for index_name, ids_name, expected in (("faiss_index.bin", "author_ids.pkl", node_ids),
                                           ("faiss_core_index.bin", "core_ids.pkl", snapshot_core)):
        ids = pickle.load(open(os.path.join(snapshot, ids_name), "rb"))
        ntotal = faiss.read_index(os.path.join(snapshot, index_name)).ntotal
        if ntotal != len(ids):
            problems.append(f"{index_name} has {ntotal} vectors but {ids_name} has {len(ids)} ids")
        if {int(i) for i in ids} != expected:
            problems.append(f"{ids_name} does not match the snapshot people")

    graph = json.load(open(os.path.join(snapshot, "author_knowledge_graph_2024.json"), encoding="utf-8"))
    stray = sum(1 for k, v in graph.items() for n in [k, *v] if not isinstance(n, str) or int(n) not in node_ids)
    if stray:
        problems.append(f"{stray} coauthor graph entries point outside the snapshot")
    del graph

    import sqlite3
    db = sqlite3.connect(f"file:{os.path.join(snapshot, PAPERS_DB)}?mode=ro", uri=True)
    if not db.execute("SELECT COUNT(*) FROM papers").fetchone()[0]:
        problems.append(f"{PAPERS_DB} has no papers")
    linked = {r[0] for r in db.execute("SELECT DISTINCT author_id FROM paper_authors")}
    if linked - node_ids:
        problems.append(f"{len(linked - node_ids)} {PAPERS_DB} authors are not snapshot people")
    db.close()

    manifest_path = os.path.join(snapshot, "snapshot_manifest.json")
    recorded = json.load(open(manifest_path, encoding="utf-8")).get("sha256", {}) if os.path.exists(manifest_path) else {}
    for name, digest in recorded.items():
        if sha256(os.path.join(snapshot, name)) != digest:
            problems.append(f"{name} does not match its manifest checksum")

    for p in problems:
        print(f"FAIL {p}")
    if not problems:
        print(f"OK {len(node_ids):,} people ({len(snapshot_core)} core) all present in {graph_metadata}")
    return 1 if problems else 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--export")
    ap.add_argument("--out")
    ap.add_argument("--graph-papers-out")
    ap.add_argument("--check", metavar="SNAPSHOT_DIR")
    ap.add_argument("--graph-metadata")
    ap.add_argument("--snapshot-version", help="Release identifier, e.g. tkg-20261002")
    args = ap.parse_args()

    if args.check:
        if not args.graph_metadata:
            ap.error("--check needs --graph-metadata")
        sys.exit(check_snapshot(args.check, args.graph_metadata))
    if not (args.export and args.out):
        ap.error("--export and --out are required to build")

    deployed = os.path.join(args.export, "01_deployed_data")
    tables = os.path.join(args.export, "05_database_tables" if os.path.isdir(os.path.join(args.export, "05_database_tables")) else "02_data")
    cheaha = os.path.join(args.export, "04_cheaha" if os.path.isdir(os.path.join(args.export, "04_cheaha")) else "03_cheaha")
    if os.path.exists(args.out):
        ap.error("--out must be a new directory; existing snapshots are never overwritten")
    os.makedirs(args.out)
    version = args.snapshot_version or os.path.basename(os.path.abspath(args.out))

    log("reading author_metadata.json")
    metadata = json.load(open(os.path.join(deployed, "author_metadata.json"), encoding="utf-8"))
    people = [r for r in metadata if r.get("is_author")]
    merge = load_merge_map(deployed, metadata)
    log(f"  {len(people):,} people, {len(merge):,} merged-away ids")

    log("reading papers.csv and authorships.csv")
    papers = load_papers(os.path.join(tables, "papers.csv"))
    author_papers = load_author_papers(os.path.join(tables, "authorships.csv"), merge, papers)
    blocked = add_collaborator_papers(tables, merge, papers, author_papers)
    log(f"  {len(papers):,} titled papers, {len(author_papers):,} authors with papers")

    nodes, graph_papers = build_nodes(people, author_papers, papers)

    # A person with no papers has no vector and nothing to ground a card on.
    keep = {int(k) for k, v in nodes.items() if v["features"]["Top Cited or Most Recent Papers"]}
    dropped = [v["title"] for k, v in nodes.items() if int(k) not in keep]
    nodes = {k: v for k, v in nodes.items() if int(k) in keep}
    core = {int(k) for k, v in nodes.items() if v["features"]["Bridge2AISeedAuthor"]}
    log(f"  {len(nodes):,} MATRIX people ({len(core)} core); left out with no papers: {dropped}")

    dump_json(nodes, os.path.join(args.out, "updated_author_nodes_with_papers.json"))
    dump_json({k: v["features"]["PaperNum"] for k, v in nodes.items()},
              os.path.join(args.out, "author_n_publications.json"))
    del papers

    log("building coauthor graph")
    graph = build_graph(os.path.join(deployed, "author_collab_dataset.json"), merge, keep)
    dump_json(graph, os.path.join(args.out, "author_knowledge_graph_2024.json"))
    graph_people = len(graph)
    log(f"  {graph_people:,} people with coauthors, {sum(len(v) for v in graph.values()):,} directed edges")
    del graph

    log("building FAISS indexes")
    build_indexes(cheaha, keep, core, args.out)

    log(f"building {PAPERS_DB}")
    paper_db = build_paper_db(tables, merge, keep, args.out)
    if os.path.basename(tables) == "05_database_tables":
        paper_db["affiliation_groups"] = publication_metadata(tables, merge, keep,
                                                            os.path.join(args.out, PAPERS_DB), metadata)
        for row in metadata:
            if str(row['id']) in nodes:
                nodes[str(row['id'])]['features']['affiliations'] = displayable_affiliations(row)
        dump_json(nodes, os.path.join(args.out, "updated_author_nodes_with_papers.json"))
        graph_release = write_graph_release(deployed, cheaha, args.out, metadata, merge, version)
    else:
        graph_release = None
    del metadata

    if args.graph_papers_out:
        log(f"writing graph paper shards to {args.graph_papers_out}")
        write_graph_paper_shards(graph_papers, args.graph_papers_out)

    with open(os.path.join(args.out, "snapshot_manifest.json"), "w", encoding="utf-8") as f:
        json.dump({
            "export": os.path.abspath(args.export),
            "snapshot_version": version,
            "paper_vector_source": os.path.relpath(cheaha, args.out),
            "paper_vectors_sha256": sha256(os.path.join(cheaha, "paper_embeddings.npy")),
            "paper_vector_ids_sha256": sha256(os.path.join(cheaha, "paper_embedding_ids.json")),
            "excluded_non_publication_ids": len(blocked),
            "graph": graph_release,
            "people": len(nodes),
            "core": len(core),
            "papers_per_person": f"{RECENT_N} most recent + {CITED_N} most cited",
            "merged_away_ids": len(merge),
            "coauthor_people": graph_people,
            "left_out_without_papers": dropped,
            "paper_db": paper_db,
            "sha256": {name: sha256(os.path.join(args.out, name)) for name in SNAPSHOT_FILES},
        }, f, indent=2)
    log("done")


if __name__ == "__main__":
    main()
