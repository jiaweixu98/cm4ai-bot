"""Build release-specific graph and publication metadata without changing the export."""

import csv
import hashlib
import json
import os
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from tkg_publications import ASSERTED, STATE_RANK, affiliation_state, displayable_affiliations, institution_key


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def repair_layout(layout, original_ids, merge):
    """Translate the GPU's original row indices through author aliases to current rows."""
    ids = [int(x) for x in layout["ids"]]
    original_ids = [int(x) for x in original_ids]
    expected = [nid for nid in original_ids if nid not in merge]
    if ids != expected:
        raise ValueError("Layout IDs do not match the embedding IDs after documented merges")
    row_of = {nid: i for i, nid in enumerate(ids)}
    neighbors = layout["neighbors"]["neighbors"]
    repaired = {}
    for row, source in enumerate(ids):
        out, seen = [], set()
        for index in neighbors.get(str(row), []):
            if not isinstance(index, int) or not 0 <= index < len(original_ids):
                raise ValueError(f"Invalid original neighbor index: {index}")
            nid = original_ids[index]
            nid = merge.get(nid, nid)
            if nid != source and nid in row_of and nid not in seen:
                seen.add(nid)
                out.append(row_of[nid])
        repaired[str(row)] = out
    layout["neighbors"]["neighbors"] = repaired
    return layout


def enrich_affiliation(affiliation, matches):
    # An institution can have several jobs across different periods. Never put
    # today's title on an earlier history row merely because the name matches.
    exact = [a for a in matches if a.get('start_year') == affiliation.get('start_year')
             and a.get('end_year') == affiliation.get('end_year')]
    same_source = [a for a in exact if a.get('source') == affiliation.get('source')
                   and a.get('verification') == affiliation.get('verification')]
    pool = same_source or exact
    titles = list(dict.fromkeys(a['role_title'] for a in pool if a.get('role_title')))
    if titles: affiliation['role_title'] = '; '.join(titles)
    rors = {a['ror_id'] for a in (pool or matches) if a.get('ror_id')}
    if len(rors) == 1: affiliation['ror_id'] = rors.pop()


def publication_metadata(tables, merge, keep, database, metadata):
    """Store institution groups for every retained person, with summaries for members."""
    history = defaultdict(list)
    with open(os.path.join(tables, "author_affiliations.csv"), newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            nid = int(row["author_id"].lstrip("A"))
            nid = merge.get(nid, nid)
            if nid not in keep or row.get("excluded_reason") or row.get("source") == "openalex_low_conf":
                continue
            history[nid].append({
                "institution": row["institution"], "ror_id": row.get("ror_id") or "",
                "source": row.get("source") or "", "verification": row.get("verification") or "",
                "role_title": row.get("role_title") or "",
                "start_year": int(row["start_year"]) if row.get("start_year") else None,
                "end_year": int(row["end_year"]) if row.get("end_year") else None,
            })
    db = sqlite3.connect(database)
    db.executescript("""
        CREATE TABLE IF NOT EXISTS publication_groups (
            author_id INTEGER NOT NULL, group_id TEXT NOT NULL, institution TEXT,
            ror_id TEXT, status TEXT NOT NULL, PRIMARY KEY (author_id, group_id));
        CREATE TABLE IF NOT EXISTS publication_affiliations (
            author_id INTEGER NOT NULL, work_id TEXT NOT NULL, group_id TEXT NOT NULL,
            PRIMARY KEY (author_id, work_id, group_id));
        CREATE INDEX IF NOT EXISTS publication_affiliations_work ON publication_affiliations(work_id, author_id);
        CREATE TABLE IF NOT EXISTS author_aliases (old_id INTEGER PRIMARY KEY, author_id INTEGER NOT NULL);
        DELETE FROM publication_groups;
        DELETE FROM publication_affiliations;
        DELETE FROM author_aliases;
    """)
    if 'affiliation_status' not in {row[1] for row in db.execute('PRAGMA table_info(paper_authors)')}:
        db.execute("ALTER TABLE paper_authors ADD COLUMN affiliation_status TEXT NOT NULL DEFAULT 'unknown'")
    db.executemany("INSERT INTO author_aliases VALUES (?, ?)", merge.items())
    links = {(int(a), w) for a, w in db.execute("SELECT author_id, work_id FROM paper_authors")}
    groups = {}
    with open(os.path.join(tables, "paper_affiliations.csv"), newline="", encoding="utf-8") as f:
        batch = []
        for row in csv.DictReader(f):
            aid = int(row["author_id"].lstrip("A")); aid = merge.get(aid, aid)
            work = row["paper_id"]
            if (aid, work) not in links:
                continue
            name, ror = row.get("institution") or "", row.get("ror_id") or ""
            key = institution_key(name, ror)
            groups.setdefault((aid, key), (name or None, ror, affiliation_state(name, ror, history[aid])))
            batch.append((aid, work, key))
            if len(batch) >= 10000:
                db.executemany("INSERT OR IGNORE INTO publication_affiliations VALUES (?, ?, ?)", batch)
                batch.clear()
        db.executemany("INSERT OR IGNORE INTO publication_affiliations VALUES (?, ?, ?)", batch)
    db.executemany("INSERT INTO publication_groups VALUES (?, ?, ?, ?, ?)",
                   ((a, key, *values) for (a, key), values in groups.items()))
    db.executescript("""
        INSERT OR IGNORE INTO publication_groups
            SELECT author_id, '__none__', NULL, '', 'unknown' FROM paper_authors;
        INSERT OR IGNORE INTO publication_affiliations
            SELECT pa.author_id, pa.work_id, '__none__' FROM paper_authors pa
            WHERE NOT EXISTS (SELECT 1 FROM publication_affiliations af
                              WHERE af.author_id = pa.author_id AND af.work_id = pa.work_id);
        UPDATE paper_authors SET affiliation_status = COALESCE((
            SELECT g.status FROM publication_affiliations af
            JOIN publication_groups g ON g.author_id = af.author_id AND g.group_id = af.group_id
            WHERE af.author_id = paper_authors.author_id AND af.work_id = paper_authors.work_id
            ORDER BY CASE g.status WHEN 'confirmed' THEN 4 WHEN 'in_history' THEN 3
                     WHEN 'unrecognised' THEN 2 ELSE 1 END DESC LIMIT 1), 'unknown');
    """)
    for row in metadata:
        aid = int(row["id"])
        if aid not in keep:
            continue
        # Keep the export's presentation ordering, enriching the matched history rows.
        for affiliation in row.get("affiliations") or []:
            matches = [a for a in history[aid] if a["institution"] == affiliation.get("institution")]
            if matches:
                enrich_affiliation(affiliation,matches)
        if row.get("color_category") != 2:
            continue
        group_rows = db.execute("""
            SELECT g.group_id, g.institution, g.ror_id, g.status, COUNT(*) AS n,
                   MIN(p.year), MAX(p.year)
            FROM publication_groups g JOIN publication_affiliations af
              ON af.author_id = g.author_id AND af.group_id = g.group_id
            JOIN papers p ON p.work_id = af.work_id WHERE g.author_id = ?
            GROUP BY g.group_id ORDER BY n DESC, g.group_id
        """, (aid,)).fetchall()
        row["affiliation_groups"] = [dict(zip(
            ("group_id", "institution", "ror_id", "status", "paper_count", "first_year", "last_year"), g))
            for g in group_rows]
        counts = dict.fromkeys(STATE_RANK, 0)
        counts.update(db.execute("SELECT affiliation_status, COUNT(*) FROM paper_authors WHERE author_id = ? "
                                 "GROUP BY affiliation_status", (aid,)))
        counts["total"] = sum(counts.values())
        row["paper_verification"] = counts
    db.commit()
    group_count = db.execute("SELECT COUNT(*) FROM publication_groups").fetchone()[0]
    db.close()
    return group_count


def write_graph_release(deployed, cheaha, out_dir, metadata, merge, version):
    graph = Path(out_dir) / "graph"
    graph.mkdir()
    layout = json.load(open(os.path.join(deployed, "tkg_ebd_89k_dataset.json"), encoding="utf-8"))
    original = json.load(open(os.path.join(cheaha, "author_embedding_ids.json"), encoding="utf-8"))
    repair_layout(layout, original, merge)
    for name, data in (("author_metadata.json", metadata), ("tkg_ebd_89k_dataset.json", layout)):
        with (graph / name).open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
    manifest = json.load(open(os.path.join(deployed, "layout_manifest.json"), encoding="utf-8"))
    layout_ids = set(layout["ids"])
    manifest.update(snapshot_version=version, n_points=len(layout["ids"]),
                    n_authors=sum(bool(r.get("is_author")) and r["id"] in layout_ids for r in metadata),
                    neighbor_index_space="layout_ids", merged_away_ids=len(merge))
    with (graph / "layout_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    (graph / "author_collab_dataset.json").symlink_to(os.path.relpath(
        os.path.join(deployed, "author_collab_dataset.json"), graph))
    record = {"snapshot_version": version, "papers_db": "../papers.sqlite",
              "sha256": {name: digest(graph / name) for name in
                         ("author_metadata.json", "tkg_ebd_89k_dataset.json", "layout_manifest.json", "author_collab_dataset.json")}}
    with (graph / "snapshot_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    return record


def refresh_release(snapshot, version):
    """Refresh derived affiliation labels only after validating the prepared release."""
    snapshot = Path(snapshot).resolve()
    if Path('/home/ubuntu') in snapshot.parents:
        raise ValueError('Local metadata refresh does not operate on production')
    manifest = json.loads((snapshot/'snapshot_manifest.json').read_text())
    for name, expected in manifest['sha256'].items():
        if digest(snapshot/name) != expected: raise ValueError(f'Existing checksum mismatch: {name}')
    source = Path(manifest['export'])
    metadata = json.loads((source/'01_deployed_data/author_metadata.json').read_text())
    nodes = json.loads((snapshot/'updated_author_nodes_with_papers.json').read_text())
    keep = {int(a) for a in nodes}
    merge = {int(old):int(r['id']) for r in metadata for old in r.get('MergedFrom',[])}
    manifest['paper_db']['affiliation_groups'] = publication_metadata(source/'05_database_tables',merge,keep,snapshot/'papers.sqlite',metadata)
    for row in metadata:
        if str(row['id']) in nodes:
            nodes[str(row['id'])]['features']['affiliations'] = displayable_affiliations(row)
    (snapshot/'updated_author_nodes_with_papers.json').write_text(json.dumps(nodes,ensure_ascii=False,separators=(',',':')))
    graph=snapshot/'graph'
    (graph/'author_metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,separators=(',',':')))
    layout_manifest=json.loads((graph/'layout_manifest.json').read_text())
    layout_manifest['snapshot_version']=version
    (graph/'layout_manifest.json').write_text(json.dumps(layout_manifest,indent=2))
    graph_manifest=json.loads((graph/'snapshot_manifest.json').read_text())
    graph_manifest['snapshot_version']=version
    graph_manifest['sha256']={name:digest(graph/name) for name in graph_manifest['sha256']}
    (graph/'snapshot_manifest.json').write_text(json.dumps(graph_manifest,indent=2))
    manifest.update(snapshot_version=version,graph=graph_manifest,
                    sha256={name:digest(snapshot/name) for name in manifest['sha256']})
    (snapshot/'snapshot_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(f'Refreshed prepared local release {version}; activate both pointers before restarting services')


if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser(description='Refresh a checksum-verified prepared local release while both apps are stopped')
    parser.add_argument('--refresh',required=True); parser.add_argument('--snapshot-version',required=True)
    args=parser.parse_args(); refresh_release(args.refresh,args.snapshot_version)
