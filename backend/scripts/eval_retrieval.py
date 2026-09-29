"""Measure people search on labelled queries: the current author-vector ranking
against keyword (BM25) people ranking from papers.sqlite and their fusion.

Two query sets, both from the snapshot and the export, no external calls:

  held_out   titles of recent papers by Bridge2AI core members. The paper itself is
             left out of the keyword index; the author vectors still include it, so
             the comparison favours the current ranking.
  expertise  the self-described expertise of consortium members; the target is the
             member. Only the expertise text and ORCID columns are read.

Usage (from backend/):
  LOCAL_DATA_DIR=../data/tkg-20260927 .venv/bin/python scripts/eval_retrieval.py \
      --export /home/shaked/box_upload_20260927/box_upload_20260927 --out /tmp/eval_retrieval.json
"""

import argparse
import csv
import json
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_loader import (  # noqa: E402
    LOCAL_DATA_DIR,
    load_core_index,
    load_embeddings_and_index,
    load_paper_library,
    load_publication_counts,
    load_specter_model,
)
from retriever import Retriever  # noqa: E402

RRF_K = 60
DEPTH = 200


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def encode(texts, batch=16):
    import torch

    tok, model = load_specter_model()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            enc = tok(texts[i:i + batch], padding=True, truncation=True, return_tensors="pt",
                      return_token_type_ids=False, max_length=512)
            out.append(model(**enc).last_hidden_state[:, 0, :].numpy().astype(np.float32))
    return np.vstack(out)


def vector_ranking(vec, scope, core, pub_counts):
    if scope == "core":
        ids, index = load_core_index()
        hits = Retriever(ids, index).search(vec, len(ids))
        return [int(str(a).split("_")[0]) for a, _ in hits]
    ids, index = load_embeddings_and_index()
    best = {}
    for key, dist in Retriever(ids, index).search(vec, 5000):
        aid = str(key).split("_")[0]
        best[aid] = min(best.get(aid, 9.0), float(dist))
    scored = [(aid, (1.0 / (1.0 + d)) * (0.05 if int(pub_counts.get(aid, 0)) == 1 else 1.0))
              for aid, d in best.items()]
    return [int(a) for a, _ in sorted(scored, key=lambda x: -x[1])[:DEPTH]]


def keyword_ranking(library, query, scope, core, exclude=None):
    scores = library.people_scores(query, exclude_rowids=exclude)
    ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
    return [a for a, _ in ranked if scope != "core" or a in core][:DEPTH]


def fuse(*rankings):
    fused = {}
    for ranking in rankings:
        for rank, aid in enumerate(ranking, start=1):
            fused[aid] = fused.get(aid, 0.0) + 1.0 / (RRF_K + rank)
    return [a for a, _ in sorted(fused.items(), key=lambda x: (-x[1], x[0]))][:DEPTH]


def metrics(results):
    out = {}
    for method in results[0]["ranks"]:
        ranks = [r["ranks"][method] for r in results]
        out[method] = {
            "recall@5": round(sum(1 for k in ranks if k and k <= 5) / len(ranks), 3),
            "recall@10": round(sum(1 for k in ranks if k and k <= 10) / len(ranks), 3),
            "mrr": round(sum(1.0 / k for k in ranks if k) / len(ranks), 3),
        }
    return out


def first_hit(ranking, targets):
    return next((i for i, aid in enumerate(ranking, start=1) if aid in targets), None)


def held_out_queries(library, core, n, seed):
    db = library._db()
    rows = db.execute(
        "SELECT p.rowid, p.title, GROUP_CONCAT(pa.author_id) FROM papers p "
        "JOIN paper_authors pa ON pa.work_id = p.work_id WHERE p.year >= 2025 GROUP BY p.rowid").fetchall()
    picked = []
    for rowid, title, authors in rows:
        people = {int(a) for a in authors.split(",")}
        if len(str(title).split()) >= 6 and people & core:
            picked.append({"rowid": rowid, "query": title, "core_targets": people & core, "all_targets": people})
    random.Random(seed).shuffle(picked)
    return picked[:n]


def expertise_queries(export, core):
    orcid_to_id = {}
    with open(os.path.join(export, "02_data", "authors.csv"), newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        head = next(reader)
        oid, orc = head.index("openalex_id"), head.index("orcid")
        for row in reader:
            aid = int(row[oid].lstrip("A")) if row[oid].lstrip("A").isdigit() else None
            if aid in core and row[orc].strip():
                orcid_to_id[row[orc].strip().rsplit("/", 1)[-1]] = aid
    queries = []
    with open(os.path.join(export, "02_data", "consortium_roster_419.csv"), newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        head = next(reader)
        exp, orc = head.index("expertise"), head.index("orcid")
        for row in reader:
            text = " ".join(row[exp].split())
            aid = orcid_to_id.get(row[orc].strip().rsplit("/", 1)[-1])
            if aid and len(text.split()) >= 3:
                queries.append({"query": text[:500], "core_targets": {aid}, "all_targets": {aid}})
    return queries


def run(name, queries, library, core, pub_counts, scopes):
    log(f"{name}: encoding {len(queries)} queries")
    vectors = encode([q["query"] for q in queries])
    report = {}
    for scope in scopes:
        results = []
        for q, vec in zip(queries, vectors):
            targets = q["core_targets"] if scope == "core" else q["all_targets"]
            exclude = {q["rowid"]} if "rowid" in q else None
            vec_rank = vector_ranking(vec, scope, core, pub_counts)
            kw_rank = keyword_ranking(library, q["query"], scope, core, exclude)
            results.append({"ranks": {
                "current_vector": first_hit(vec_rank, targets),
                "keyword": first_hit(kw_rank, targets),
                "hybrid": first_hit(fuse(vec_rank, kw_rank), targets),
            }})
        report[scope] = {"queries": len(results), **metrics(results)}
        log(f"{name}/{scope}: {json.dumps(report[scope])}")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True)
    ap.add_argument("--papers", type=int, default=300)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", default="/tmp/eval_retrieval.json")
    args = ap.parse_args()

    import torch

    torch.set_num_threads(args.threads)
    log(f"snapshot {LOCAL_DATA_DIR}")
    library = load_paper_library()
    if library is None:
        raise SystemExit("papers.sqlite is missing from the snapshot")
    core = {int(a) for a in load_core_index()[0]}
    pub_counts = load_publication_counts()
    load_embeddings_and_index()
    load_specter_model()

    report = {
        "snapshot": LOCAL_DATA_DIR,
        "held_out": run("held_out", held_out_queries(library, core, args.papers, args.seed),
                        library, core, pub_counts, ["core", "all"]),
        "expertise": run("expertise", expertise_queries(args.export, core), library, core, pub_counts, ["core"]),
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
