"""Refresh in-memory author retrieval after a profile decision; never mutate source indexes."""
import json
import math
import weakref
from pathlib import Path
import numpy as np

_matrix = None
_row_of = None
_index_states = weakref.WeakKeyDictionary()


def revision_and_authors(library):
    db = library._db()
    if not getattr(library._local, 'attached_decisions', False):
        return 0, []
    revision = db.execute('SELECT COALESCE(MAX(revision),0) FROM corrections.decision_events').fetchone()[0]
    authors = [int(r[0]) for r in db.execute('SELECT DISTINCT author_id FROM corrections.decisions')]
    return int(revision), authors


def retained_vector(library, author_id, matrix, row_of):
    vector = np.zeros(matrix.shape[1],dtype=np.float32)
    used = 0
    for work, position in library._db().execute('SELECT work_id,position FROM visible_paper_authors WHERE author_id=?',(author_id,)):
        row = row_of.get(work)
        if row is None: continue
        weight = 1/math.sqrt(position) if position and position > 0 else 0.5
        vector += weight * matrix[row]
        used += 1
    norm = float(np.linalg.norm(vector))
    return vector/norm if norm else vector, used


def sync_index(ids, index, library, data_directory):
    if library is None or index is None: return
    from retriever import INDEX_LOCK
    with INDEX_LOCK:
        revision, authors = revision_and_authors(library)
        state = _index_states.setdefault(index, {'revision':0,'originals':{}})
        if state['revision'] == revision: return
        global _matrix, _row_of
        if _matrix is None:
            manifest = json.loads((Path(data_directory)/'snapshot_manifest.json').read_text())
            source = Path(data_directory)/manifest['paper_vector_source']
            _matrix = np.load(source/'paper_embeddings.npy',mmap_mode='r')
            _row_of = {work:i for i,work in enumerate(json.loads((source/'paper_embedding_ids.json').read_text()))}
        import faiss
        rows = {int(aid):i for i,aid in enumerate(ids)}
        contents = faiss.rev_swig_ptr(index.get_xb(),index.ntotal*index.d).reshape(index.ntotal,index.d)
        originals = state['originals']
        for aid in authors:
            row = rows.get(aid)
            if row is None: continue
            originals.setdefault(aid,contents[row].copy())
            excluded = library._db().execute('SELECT 1 FROM corrections.excluded_links WHERE snapshot_version=? AND author_id=? LIMIT 1',(library.snapshot_version,aid)).fetchone()
            if excluded:
                vector, _ = retained_vector(library,aid,_matrix,_row_of)
                contents[row] = vector
            else:
                contents[row] = originals[aid]
        state['revision'] = revision


def sync_graph(graph, library):
    if library is None: return graph
    from retriever import INDEX_LOCK
    with INDEX_LOCK:
        revision, authors = revision_and_authors(library)
        if graph.graph.get('publication_revision',0) == revision: return graph
        for aid in authors:
            key = str(aid)
            raw = {str(r[0]) for r in library._db().execute('''SELECT DISTINCT b.author_id FROM paper_authors a
                JOIN paper_authors b ON b.work_id=a.work_id AND b.author_id!=a.author_id WHERE a.author_id=?''',(aid,))}
            valid = {str(r[0]) for r in library._db().execute('''SELECT DISTINCT b.author_id FROM visible_paper_authors a
                JOIN visible_paper_authors b ON b.work_id=a.work_id AND b.author_id!=a.author_id WHERE a.author_id=?''',(aid,))}
            for other in list(graph.neighbors(key)) if key in graph else []:
                if other in raw and other not in valid: graph.remove_edge(key,other)
            for other in valid:
                if other in graph: graph.add_edge(key,other)
        graph.graph['publication_revision'] = revision
    return graph
