import numpy as np
import faiss

class Retriever:
    """Retriever class using FAISS for efficient similarity search."""

    def __init__(self, ids, index):
        self.doc_lookup = ids
        self.index = index
        self.use_gpu = False

    def reset(self, embeds):
        self.init_index_and_add(embeds)

    def init_index_and_add(self, embeds):
        dim = embeds.shape[1]
        self._initialize_faiss_index(dim)
        self.index.add(embeds)
        if self.use_gpu:
            self._move_index_to_gpu()

    def _initialize_faiss_index(self, dim: int):
        self.index = faiss.IndexFlatIP(dim)

    def _move_index_to_gpu(self):
        pass

    def search_single(self, query_embed, topk: int = 10):
        if self.index is None:
            raise ValueError("Index is not initialized")
        D, I = self.index.search(query_embed, topk)
        original_indices = np.array(self.doc_lookup)[I].tolist()[0]
        return list(zip(original_indices, D[0]))
