"""Read-only consistency checks for a prepared October release and its local APIs."""
import argparse
import csv
import hashlib
import json
import sqlite3
import urllib.request
from pathlib import Path


def digest(path):
    result=hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda:handle.read(1<<20),b''): result.update(block)
    return result.hexdigest()


def check(snapshot, running=False):
    snapshot=Path(snapshot).resolve()
    manifest=json.loads((snapshot/'snapshot_manifest.json').read_text())
    graph=snapshot/'graph'
    graph_manifest=json.loads((graph/'snapshot_manifest.json').read_text())
    assert manifest['snapshot_version']==graph_manifest['snapshot_version']
    for base,record in ((snapshot,manifest),(graph,graph_manifest)):
        for name,expected in record['sha256'].items(): assert digest(base/name)==expected,name
    layout=json.loads((graph/'tkg_ebd_89k_dataset.json').read_text())
    ids=layout['ids']; neighbors=layout['neighbors']['neighbors']
    assert len(ids)==len(set(ids))==len(layout['points'])==len(neighbors)
    for row, values in neighbors.items():
        assert len(values)==len(set(values)) and int(row) not in values
        assert all(0<=value<len(ids) for value in values)
    db=sqlite3.connect(f'file:{snapshot}/papers.sqlite?mode=ro',uri=True)
    tables=Path(manifest['export'])/'05_database_tables'
    removed=[]
    with (tables/'papers.csv').open(newline='') as handle:
        for row in csv.DictReader(handle):
            if row['non_publication']:removed.append(row['openalex_id'])
    for work in removed: assert db.execute('SELECT 1 FROM papers WHERE work_id=?',(work,)).fetchone() is None,work
    assert not db.execute('SELECT 1 FROM paper_authors WHERE author_id IN (SELECT old_id FROM author_aliases) LIMIT 1').fetchone()
    print(f"PASS {manifest['snapshot_version']}: checksums, {len(ids):,} valid map rows, {len(removed):,} known non-publications absent, aliases resolved")
    if running:
        records=[]
        for url in ('http://127.0.0.1:4173/api/data/version','http://127.0.0.1:8100/api/health','http://127.0.0.1:4173/data/snapshot_manifest.json'):
            with urllib.request.urlopen(url,timeout=15) as response:records.append(json.load(response))
        assert all(r['snapshot_version']==manifest['snapshot_version'] for r in records)
        print('PASS running Graph API, static graph and MATRIX version parity')
    db.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('snapshot'); parser.add_argument('--running',action='store_true')
    args=parser.parse_args(); check(args.snapshot,args.running)
