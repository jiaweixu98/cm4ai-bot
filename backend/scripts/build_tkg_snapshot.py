"""Build the MATRIX data snapshot from a TKG export.

Inputs are the three graph files the bridge app serves, the Cheaha author
vectors, and the database CSV export, all from the same export:

    <export>/01_deployed_data/author_metadata.json
    <export>/01_deployed_data/author_collab_dataset.json
    <export>/01_deployed_data/tkg_merge_map_*.csv
    <export>/02_data/papers.csv
    <export>/02_data/authorships.csv
    <export>/03_cheaha/author_embeddings.npy
    <export>/03_cheaha/author_embedding_ids.json

Writes the files data_loader.py reads into --out, and optionally the
per-person paper shards the graph app serves from --graph-papers-out.

    backend/.venv/bin/python backend/scripts/build_tkg_snapshot.py \\
        --export ~/box_upload_20260927/box_upload_20260927 \\
        --out data/tkg-20260927 \\
        --graph-papers-out ../bridge2aikg/work/data/author_papers

Check a built snapshot against the graph metadata it will serve beside
(exits non-zero on any mismatch):

    backend/.venv/bin/python backend/scripts/build_tkg_snapshot.py \\
        --check data/tkg-20260927 \\
        --graph-metadata ../bridge2aikg/work/data/author_metadata.json
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

RECENT_N = 6
CITED_N = 6
GRAPH_PAPER_SHARDS = 256
HIDDEN_AFFILIATION_SOURCE = "openalex_low_conf"


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


def load_papers(path: str) -> dict:
    papers = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            title = (row.get("title") or "").strip()
            if not title:
                continue
            doi = (row.get("doi") or "").strip()
            if doi and not doi.startswith("http"):
                doi = f"https://doi.org/{doi}"
            pmid = (row.get("pmid") or "").strip()
            year = row.get("publication_year") or ""
            cited = row.get("cited_by_count") or ""
            papers[row["openalex_id"]] = {
                "title": title,
                "year": int(float(year)) if year else None,
                "cited_by": int(float(cited)) if cited else 0,
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


def displayable_affiliations(row: dict) -> list:
    return [
        a for a in (row.get("affiliations") or [])
        if a.get("source") != HIDDEN_AFFILIATION_SOURCE and (a.get("institution") or "").strip()
    ]


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
            }
            for _, p in selected
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
        del mat


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
    args = ap.parse_args()

    if args.check:
        if not args.graph_metadata:
            ap.error("--check needs --graph-metadata")
        sys.exit(check_snapshot(args.check, args.graph_metadata))
    if not (args.export and args.out):
        ap.error("--export and --out are required to build")

    deployed = os.path.join(args.export, "01_deployed_data")
    tables = os.path.join(args.export, "02_data")
    cheaha = os.path.join(args.export, "03_cheaha")
    os.makedirs(args.out, exist_ok=True)

    log("reading author_metadata.json")
    metadata = json.load(open(os.path.join(deployed, "author_metadata.json"), encoding="utf-8"))
    people = [r for r in metadata if r.get("is_author")]
    merge = load_merge_map(deployed, metadata)
    log(f"  {len(people):,} people, {len(merge):,} merged-away ids")

    log("reading papers.csv and authorships.csv")
    papers = load_papers(os.path.join(tables, "papers.csv"))
    author_papers = load_author_papers(os.path.join(tables, "authorships.csv"), merge, papers)
    log(f"  {len(papers):,} titled papers, {len(author_papers):,} authors with papers")

    nodes, graph_papers = build_nodes(people, author_papers, papers)
    del metadata

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

    if args.graph_papers_out:
        log(f"writing graph paper shards to {args.graph_papers_out}")
        write_graph_paper_shards(graph_papers, args.graph_papers_out)

    with open(os.path.join(args.out, "snapshot_manifest.json"), "w", encoding="utf-8") as f:
        json.dump({
            "export": os.path.abspath(args.export),
            "people": len(nodes),
            "core": len(core),
            "papers_per_person": f"{RECENT_N} most recent + {CITED_N} most cited",
            "merged_away_ids": len(merge),
            "coauthor_people": graph_people,
            "left_out_without_papers": dropped,
            "sha256": {name: sha256(os.path.join(args.out, name)) for name in SNAPSHOT_FILES},
        }, f, indent=2)
    log("done")


if __name__ == "__main__":
    main()
