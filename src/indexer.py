import faiss
import numpy as np
import pickle
import os

class Indexer:
    def __init__(self, index_path="data/processed/faiss_index.bin", metadata_path="data/processed/metadata.pkl"):
        """
        Manages the FAISS vector index and filename mappings.
        """
        self.dimension = 512  # SSCD output size
        self.index_path = index_path
        self.metadata_path = metadata_path
        
        # This list acts as our "Phonebook" mapping ID -> Filename
        self.metadata = []

        # Ensure processed directory exists
        os.makedirs(os.path.dirname(index_path), exist_ok=True)

        # Initialize FAISS Index
        # IndexFlatIP = Exact Search using Inner Product.
        # Since SSCD vectors are normalized, Inner Product == Cosine Similarity.
        self.index = faiss.IndexFlatIP(self.dimension)
        
        # Load existing index if available
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            self.load()

    def add_vectors(self, vectors: np.ndarray, filenames: list):
        """
        Adds vectors to the index and updates metadata.
        Args:
            vectors: Numpy array of shape (N, 512). MUST BE NORMALIZED (float32).
            filenames: List of N strings corresponding to the vectors.
        """
        if len(vectors)!= len(filenames):
            raise ValueError("Error: Number of vectors must match number of filenames.")

        # 1. Add to FAISS (It automatically assigns sequential IDs: 0, 1, 2...)
        # FAISS strictly requires float32
        self.index.add(vectors.astype('float32'))
        
        # 2. Add to Metadata (To track what ID 0 actually is)
        self.metadata.extend(filenames)
        
        print(f"✅ Added {len(vectors)} vectors. Total index size: {self.index.ntotal}")

    def search(self, query_vector: np.ndarray, k=1):
        """
        Searches for the k nearest neighbors.
        Args:
            query_vector: Numpy array of shape (1, 512).
            k: Number of results to return.
        Returns:
            List of dictionaries: [{'filename': 'abc.jpg', 'score': 0.95, 'id': 5},...]
        """
        # FAISS search expects float32
        # D = Distances (Scores), I = Indices (IDs)
        # D and I are 2D arrays of shape (num_queries, k)
        D, I = self.index.search(query_vector.astype('float32'), k)
        
        results = []
        # Loop through results (D[0] and I[0] for the first query)
        for score, idx in zip(D[0], I[0]):
            if idx != -1:  # -1 means no match found
                results.append({
                    "filename": self.metadata[idx],
                    "score": float(score),
                    "id": int(idx)
                })
        return results

    def save(self):
        """Persist index and metadata to disk."""
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)
        print("💾 Index and metadata saved to disk.")

    def load(self):
        """Load index and metadata from disk."""
        print("Loading existing index...")
        self.index = faiss.read_index(self.index_path)
        with open(self.metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)
        print(f"Loaded {len(self.metadata)} records.")