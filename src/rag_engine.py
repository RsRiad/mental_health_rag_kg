"""
RAG Engine
Handles retrieval and LLM-based confidence scoring (R, G, C).
"""
import re
import yaml
from typing import List, Dict, Tuple
from src.llm_client import LLMClient
from rag.vector_store import VectorStore

class RAGEngine:
    def __init__(self, config_path: str = "configs/config.yaml", prompts_path: str = "configs/prompts.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        with open(prompts_path, "r") as f:
            self.prompts = yaml.safe_load(f)

        self.vector_store = VectorStore(config_path=config_path)
        self.llm = LLMClient(config_path)
        self.top_k = self.config["rag"]["top_k"]
        self.retrieval_threshold = self.config["rag"]["retrieval_threshold"]

    def retrieve(self, query: str) -> Tuple[List[Dict], List[float]]:
        """Retrieve relevant chunks from FAISS."""
        docs, scores = self.vector_store.search(query, top_k=self.top_k)
        # Filter by threshold
        filtered = [(d, s) for d, s in zip(docs, scores) if s >= self.retrieval_threshold]
        if not filtered:
            return docs[:2], scores[:2]  # Return top 2 even if below threshold
        return [x[0] for x in filtered], [x[1] for x in filtered]

    def generate_response(self, query: str, context_docs: List[Dict]) -> str:
        """Generate response using retrieved context."""
        context = "\n\n".join([f"[{i+1}] {doc['text']}" for i, doc in enumerate(context_docs)])
        prompt = self.prompts["rag_generation"].format(context=context, question=query)
        response = self.llm.generate(prompt, temperature=0.3, max_tokens=256)
        return response

    def calculate_retrieval_quality(self, query: str, docs: List[Dict], scores: List[float]) -> float:
        """
        R = Retrieval Information Quality
        Based on average similarity score and result count.
        """
        if not scores:
            return 0.0
        avg_score = sum(scores) / len(scores)
        # Normalize: assume score range 0-1 (cosine similarity)
        coverage = min(len(docs) / self.top_k, 1.0)
        R = avg_score * 0.7 + coverage * 0.3
        return min(R, 1.0)

    def calculate_grounding_score(self, response: str, docs: List[Dict]) -> float:
        """
        G = Grounding Score
        Measures how much response is grounded in retrieved context.
        """
        if not docs:
            return 0.0

        context_text = " ".join([d["text"] for d in docs]).lower()
        response_words = set(re.findall(r"\b\w+\b", response.lower()))

        if not response_words:
            return 0.0

        grounded_count = sum(1 for w in response_words if w in context_text and len(w) > 4)
        G = grounded_count / len(response_words)
        return min(G * 2.5, 1.0)  # Scale factor since not all words need to match

    def calculate_consistency_score(self, response: str, query: str) -> float:
        """
        C = Consistency Score
        Check if response is self-consistent and relevant to query.
        """
        # Simple heuristic: response should contain some query-related terms
        query_terms = set(re.findall(r"\b\w+\b", query.lower()))
        response_terms = set(re.findall(r"\b\w+\b", response.lower()))

        if not query_terms:
            return 0.5

        overlap = len(query_terms.intersection(response_terms))
        relevance = overlap / len(query_terms)

        # Check for contradictions (simple: negation patterns)
        contradiction_markers = ["however", "but", "although", "contrary"]
        has_contradiction = any(m in response.lower() for m in contradiction_markers)

        C = relevance * (0.8 if has_contradiction else 1.0)
        return min(C, 1.0)

    def calculate_confidence(self, query: str, docs: List[Dict], scores: List[float], response: str) -> Dict:
        """
        Calculate full LLM confidence score.
        Returns dict with R, G, C, and final LLM_Confidence.
        """
        weights = self.config["confidence"]["llm_weights"]
        R = self.calculate_retrieval_quality(query, docs, scores)
        G = self.calculate_grounding_score(response, docs)
        C = self.calculate_consistency_score(response, query)

        final = weights["R"] * R + weights["G"] * G + weights["C"] * C

        return {
            "R_retrieval": round(R, 3),
            "G_grounding": round(G, 3),
            "C_consistency": round(C, 3),
            "llm_confidence": round(final, 3)
        }

    def query(self, user_query: str) -> Dict:
        """
        Full RAG pipeline for a query.
        Returns dict with response, docs, scores, and confidence.
        """
        docs, scores = self.retrieve(user_query)
        response = self.generate_response(user_query, docs)
        conf = self.calculate_confidence(user_query, docs, scores, response)

        return {
            "source": "rag",
            "response": response,
            "retrieved_docs": docs,
            "retrieval_scores": scores,
            "confidence": conf
        }
