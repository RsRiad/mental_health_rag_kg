"""
FAISS Vector Store Manager
Handles index creation, saving, loading, and similarity search.
"""
import pickle
import json
from pathlib import Path
from typing import List, Dict, Tuple
import faiss
import numpy as np
from rag.embedder import Embedder

class VectorStore:
    def __init__(self, index_dir: str = "data/faiss_index", config_path: str = "configs/config.yaml"):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.index_dir / "index.faiss"
        self.meta_path = self.index_dir / "index_meta.pkl"
        self.embedder = Embedder(config_path)
        self.index = None
        self.metadata = []

    def build_index(self, chunks_path: str):
        """Build FAISS index from chunked JSONL."""
        texts = []
        meta = []

        with open(chunks_path, "r", encoding="utf-8") as f:
            for line in f:
                chunk = json.loads(line.strip())
                texts.append(chunk["text"])
                meta.append(chunk)

        print(f"Encoding {len(texts)} chunks...")
        embeddings = self.embedder.encode(texts)

        self.dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(self.dim)  # Inner product for cosine similarity
        self.index.add(embeddings)
        self.metadata = meta

        self.save()
        print(f"FAISS index built with {self.index.ntotal} vectors (dim={self.dim}).")

    def save(self):
        """Save index and metadata."""
        if self.index is None:
            raise ValueError("No index to save.")
        faiss.write_index(self.index, str(self.index_path))
        with open(self.meta_path, "wb") as f:
            pickle.dump(self.metadata, f)
        print(f"Index saved to {self.index_dir}")

    def load(self):
        """Load index and metadata."""
        if not self.index_path.exists():
            raise FileNotFoundError(f"Index not found at {self.index_path}")
        self.index = faiss.read_index(str(self.index_path))
        with open(self.meta_path, "rb") as f:
            self.metadata = pickle.load(f)
        print(f"Loaded index with {self.index.ntotal} vectors.")

    def search(self, query: str, top_k: int = 5) -> Tuple[List[Dict], List[float]]:
        """
        Search index for query.
        Returns: (list of metadata dicts, list of scores)
        """
        if self.index is None:
            self.load()

        q_emb = self.embedder.encode_query(query).reshape(1, -1)
        scores, indices = self.index.search(q_emb, top_k)

        results = []
        result_scores = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            results.append(self.metadata[idx])
            result_scores.append(float(score))

        return results, result_scores
