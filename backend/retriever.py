import numpy as np
import faiss


class Retriever:
    """Retriever class using FAISS for efficient similarity search."""

    def __init__(self, ids, index):
        self.doc_lookup = ids
        self.index = index

    def search(self, query_embed, topk: int = 5000):
        D, I = self.index.search(query_embed, topk)
        original_indices = np.array(self.doc_lookup)[I].tolist()[0]
        return list(zip(original_indices, D[0]))
