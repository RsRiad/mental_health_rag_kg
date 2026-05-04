"""
Embedding Module
Wraps sentence-transformers for RAG document and query embedding.
"""
import numpy as np
from sentence_transformers import SentenceTransformer
import yaml

class Embedder:
    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        model_name = config["models"]["embedding"]
        self.model = SentenceTransformer(model_name)
        self.normalize = config["embeddings"]["normalize"]
        self.dim = config["embeddings"]["dimension"]

    def encode(self, texts: list, batch_size: int = 32) -> np.ndarray:
        """Encode list of texts to vectors."""
        embeddings = self.model.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)
        if self.normalize:
            embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10)
        return embeddings.astype("float32")

    def encode_query(self, text: str) -> np.ndarray:
        """Encode a single query."""
        emb = self.model.encode(text, convert_to_numpy=True)
        if self.normalize:
            emb = emb / (np.linalg.norm(emb) + 1e-10)
        return emb.astype("float32")
