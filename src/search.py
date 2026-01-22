import faiss
import numpy as np
import pickle
import os
from typing import List, Tuple
from pathlib import Path


class VectorStore:
    """FAISS-based vector store for efficient similarity search"""
    
    def __init__(self, index_path: str = "data/faiss_index.idx", 
                 metadata_path: str = "data/metadata.pkl"):
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.index = None
        self.metadata = []
        self.loaded = False
        
    def build_index(self, embeddings: np.ndarray, metadata: List[dict]):
        """Build FAISS index from embeddings
        
        Args:
            embeddings: numpy array of shape (N, embedding_dim)
            metadata: list of dicts with image info {path, filename, etc}
        """
        if len(embeddings) == 0:
            raise ValueError("No embeddings provided")
            
        # Ensure embeddings are float32 (required by FAISS)
        embeddings = embeddings.astype(np.float32)
        
        # Create index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)  # L2 distance (Euclidean)
        self.index = faiss.IndexIDMap(self.index)
        
        # Add vectors with IDs
        ids = np.arange(len(embeddings), dtype=np.int64)
        self.index.add_with_ids(embeddings, ids)
        self.metadata = metadata
        
        # Save index and metadata
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)
    
    def load_index(self):
        """Load FAISS index and metadata from disk"""
        if not os.path.exists(self.index_path):
            raise FileNotFoundError(f"Index not found at {self.index_path}")
            
        self.index = faiss.read_index(self.index_path)
        with open(self.metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)
        self.loaded = True
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[int, float, dict]]:
        """Search for k nearest neighbors
        
        Args:
            query_embedding: numpy array of shape (embedding_dim,)
            k: number of neighbors to return
            
        Returns:
            List of tuples (index, distance, metadata)
        """
        if not self.loaded:
            self.load_index()
            
        # Reshape and ensure float32
        query = query_embedding.astype(np.float32).reshape(1, -1)
        
        # Search
        distances, indices = self.index.search(query, k)
        distances = distances[0]
        indices = indices[0]
        
        # Convert L2 distance to similarity score (0-1, higher is more similar)
        # L2 distance inversely correlates with similarity
        similarities = 1.0 / (1.0 + distances)
        
        results = []
        for idx, distance, similarity in zip(indices, distances, similarities):
            if idx >= 0:  # Valid result
                results.append({
                    'index': int(idx),
                    'distance': float(distance),
                    'similarity': float(similarity),
                    'metadata': self.metadata[idx]
                })
        
        return results
