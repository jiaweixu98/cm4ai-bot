import numpy as np
import faiss
import threading
INDEX_LOCK = threading.RLock()


class Retriever:
    """Retriever class using FAISS for efficient similarity search."""

    def __init__(self, ids, index):
        self.doc_lookup = ids
        self.index = index

    def search(self, query_embed, topk: int = 5000):
        # Author vectors are unit length; a unit query keeps L2 distances in [0, 4]
        # and lets blended query/team vectors weigh both parts as intended.
        query = np.ascontiguousarray(query_embed, dtype=np.float32).reshape(1, -1)
        norm = float(np.linalg.norm(query))
        if norm > 0:
            query = query / norm
        with INDEX_LOCK:
            D, I = self.index.search(query, min(topk, self.index.ntotal))
        original_indices = np.array(self.doc_lookup)[I].tolist()[0]
        return list(zip(original_indices, D[0]))
