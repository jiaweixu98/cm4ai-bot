"""Validate and activate a prepared local release without editing environment files."""
import argparse
import hashlib
import json
import os
from pathlib import Path


def activate(snapshot, matrix_root, graph_root):
    snapshot, matrix_root, graph_root = map(lambda p:Path(p).resolve(),(snapshot,matrix_root,graph_root))
    if any(Path('/home/ubuntu') in p.parents for p in (snapshot,matrix_root,graph_root)):
        raise ValueError('This helper is for isolated local activation, not production deployment')
    manifest = json.loads((snapshot/'snapshot_manifest.json').read_text())
    graph_manifest = json.loads((snapshot/'graph/snapshot_manifest.json').read_text())
    version = manifest['snapshot_version']
    if not version or version != graph_manifest['snapshot_version']:
        raise ValueError('Graph and MATRIX release versions differ')
    for base, record in ((snapshot,manifest),(snapshot/'graph',graph_manifest)):
        for name, expected in record['sha256'].items():
            digest=hashlib.sha256()
            with (base/name).open('rb') as handle:
                for block in iter(lambda:handle.read(1<<20),b''):digest.update(block)
            if digest.hexdigest()!=expected: raise ValueError(f'Checksum mismatch: {name}')
    # Each pointer is atomically replaced; activate while the two local services
    # are stopped, then restart both and check their version endpoints.
    for base,target in ((matrix_root,snapshot),(graph_root,snapshot/'graph')):
        if not base.is_dir():raise ValueError('The existing data roots must exist')
        temporary=base/'active-snapshot.json.tmp'
        temporary.write_text(json.dumps({'path':os.path.relpath(target,base),'snapshot_version':version})+'\n')
        temporary.replace(base/'active-snapshot.json')
    print(f'Activated local release {version}; restart Graph and MATRIX and check version parity')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot',required=True)
    parser.add_argument('--matrix-data-root',required=True)
    parser.add_argument('--graph-data-root',required=True)
    args=parser.parse_args()
    activate(args.snapshot,args.matrix_data_root,args.graph_data_root)
