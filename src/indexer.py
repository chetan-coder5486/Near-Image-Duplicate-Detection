import faiss
import numpy as np
import pickle
import os


class Indexer:
    def __init__(self,
                 index_path="data/processed/faiss_index.bin",
                 metadata_path="data/processed/metadata.pkl"):
        """
        Manages the FAISS vector index and filename mappings.
        """
        self.dimension = 512  # SSCD output size
        self.index_path = index_path
        self.metadata_path = metadata_path

        # ID -> Filename mapping
        self.metadata = []

        # Ensure directory exists
        os.makedirs(os.path.dirname(index_path), exist_ok=True)

        # FAISS Index (Exact Search, Inner Product)
        self.index = faiss.IndexFlatIP(self.dimension)

        # Load existing data if available
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            self.load()

    def add_vectors(self, vectors: np.ndarray, filenames: list):
        """
        Adds vectors to the index and updates metadata.
        Args:
            vectors: (N, 512) float32, normalized
            filenames: list of N strings
        """
        if len(vectors) != len(filenames):
            raise ValueError("Vectors and filenames count mismatch.")

        self.index.add(vectors.astype("float32"))
        self.metadata.extend(filenames)

        print(f"✅ Added {len(vectors)} vectors. Total index size: {self.index.ntotal}")

    def search(self, query_vector: np.ndarray, k=1):
        """
        Search nearest neighbors.
        Returns:
            [{'filename': str, 'score': float, 'id': int}]
        """
        if self.index.ntotal == 0:
            return []

        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        D, I = self.index.search(query_vector.astype("float32"), k)

        results = []
        for score, idx in zip(D[0], I[0]):
            if idx != -1:
                results.append({
                    "filename": self.metadata[idx],
                    "score": float(score),
                    "id": int(idx)
                })

        return results

    def save(self):
        """Save FAISS index + metadata"""
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, "wb") as f:
            pickle.dump(self.metadata, f)
        print("💾 Index and metadata saved.")

    def load(self):
        """Load FAISS index + metadata"""
        print("Loading existing index...")
        self.index = faiss.read_index(self.index_path)
        with open(self.metadata_path, "rb") as f:
            self.metadata = pickle.load(f)
        print(f"Loaded {len(self.metadata)} records.")
