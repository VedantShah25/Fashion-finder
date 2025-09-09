# utils/similarity.py
import faiss
import numpy as np

class SimilaritySearch:
    def __init__(self, embeddings):
        """
        embeddings: list of numpy arrays (dataset embeddings)
        """
        self.dimension = len(embeddings[0])
        # Normalize embeddings for cosine similarity
        embeddings = np.array([emb / np.linalg.norm(emb) for emb in embeddings]).astype("float32")

        # Use IndexFlatIP for cosine similarity (inner product on normalized vectors)
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings)

    def search(self, query_embedding, top_k=5):
        """
        query_embedding: numpy array
        Returns top_k most similar items and their distances
        """
        # Normalize query embedding
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        query_embedding = np.array([query_embedding]).astype("float32")

        # Inner product search
        distances, indices = self.index.search(query_embedding, top_k)
        unique_indices = list(dict.fromkeys(indices[0]))  # keeps order, removes dups
        unique_distances = distances[0][: len(unique_indices)]
        return unique_distances, unique_indices
