"""
Fusion Layer
Merges RAG and KG outputs using confidence-weighted fusion.
Detects hallucinations and decides final output.
"""
import re
import yaml
from typing import Dict, List
from src.llm_client import LLMClient

class FusionLayer:
    def __init__(self, config_path: str = "configs/config.yaml", prompts_path: str = "configs/prompts.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        with open(prompts_path, "r") as f:
            self.prompts = yaml.safe_load(f)

        self.threshold_high = self.config["confidence"]["fusion"]["threshold_high"]
        self.threshold_medium = self.config["confidence"]["fusion"]["threshold_medium"]
        self.threshold_low = self.config["confidence"]["fusion"]["threshold_low"]
        self.hallucination_threshold = self.config["confidence"]["fusion"]["hallucination_threshold"]
        self.llm = LLMClient(config_path)

    def detect_hallucination(self, rag_response: str, kg_facts: List[Dict]) -> float:
        """
        Detect hallucination by checking if RAG claims exist in KG facts.
        Returns hallucination score (0 = no hallucination, 1 = complete hallucination).
        """
        if not kg_facts or not rag_response:
            return 0.5  # Unknown if no facts to verify

        # Extract claims from RAG response (simplified: noun phrases)
        response_lower = rag_response.lower()

        # Build set of KG fact objects (symptoms, conditions)
        kg_entities = set()
        for fact in kg_facts:
            for key in ["subject", "object"]:
                val = fact.get(key, "").lower().replace("_", " ")
                if len(val) > 3:
                    kg_entities.add(val)

        if not kg_entities:
            return 0.5

        # Check overlap
        # Tokenize response into potential claims
        response_tokens = set(re.findall(r"\b\w{4,}\b", response_lower))

        # Count how many significant terms appear in KG
        matched = sum(1 for token in response_tokens if any(token in ent for ent in kg_entities))

        if not response_tokens:
            return 0.0

        coverage = matched / len(response_tokens)
        hallucination_score = 1.0 - coverage

        return min(hallucination_score, 1.0)

    # def fuse(self, rag_result: Dict, kg_result: Dict) -> Dict:
        """
        Fuse RAG and KG outputs.
        Returns final response, confidence, and metadata.
        """
        llm_conf = rag_result["confidence"]["llm_confidence"]
        kg_conf = kg_result["confidence"]["kg_confidence"]
        rag_resp = rag_result["response"]
        kg_resp = kg_result["response"]

        # Hallucination check
        kg_facts = kg_result.get("matches", [])
        hall_score = self.detect_hallucination(rag_resp, kg_facts)

        # Adjust LLM confidence if hallucination detected
        if hall_score > self.hallucination_threshold:
            llm_conf = llm_conf * (1.0 - hall_score)

        # Decision logic
        if llm_conf >= self.threshold_high and kg_conf >= self.threshold_high:
            # Both high confidence: merge responses
            final_conf = 0.6 * llm_conf + 0.4 * kg_conf
            final_response = self._merge_responses(rag_resp, kg_resp)
            strategy = "merged_high_confidence"

        elif kg_conf >= self.threshold_high:
            # KG is highly trustworthy
            final_conf = kg_conf
            final_response = kg_resp
            strategy = "kg_dominant"

        elif llm_conf >= self.threshold_high and hall_score < self.hallucination_threshold:
            # RAG is trustworthy and not hallucinating
            final_conf = llm_conf
            final_response = rag_resp
            strategy = "rag_dominant"

        elif llm_conf >= self.threshold_medium or kg_conf >= self.threshold_medium:
            # Medium confidence: cautious merged response
            final_conf = max(llm_conf, kg_conf)
            final_response = self._cautious_merge(rag_resp, kg_resp)
            strategy = "cautious_merged"

        else:
            # Low confidence: abstain
            final_conf = max(llm_conf, kg_conf)
            final_response = self.prompts["low_confidence_response"]
            strategy = "abstain"

        return {
            "final_response": final_response,
            "final_confidence": round(final_conf, 3),
            "strategy": strategy,
            "hallucination_score": round(hall_score, 3),
            "rag_confidence": llm_conf,
            "kg_confidence": kg_conf,
            "rag_response": rag_resp,
            "kg_response": kg_resp
        }
    def fuse(self, rag_result: Dict, kg_result: Dict) -> Dict:
    # ... existing code ...
    
    # Hallucination check — PASS rag_docs explicitly
        kg_facts = kg_result.get("matches", [])
        rag_docs = rag_result.get("retrieved_docs", [])
        hall_score = self.detect_hallucination(rag_resp, kg_facts, rag_docs)
    
    # NEW: Determine if hallucination was mitigated
        mitigation_status = self._determine_mitigation(
            hall_score, 
            llm_conf, 
            kg_conf,
            rag_result.get("retrieved_docs", [])
        )
    
    # ... rest of fusion logic ...
    
        return {
            "final_response": final_response,
            "final_confidence": round(final_conf, 3),
            "strategy": strategy,
            "hallucination_score": round(hall_score, 3),
            "mitigation_status": mitigation_status,  # ← NEW
            "rag_confidence": llm_conf,
            "kg_confidence": kg_conf,
            "rag_response": rag_resp,
            "kg_response": kg_resp
        }

def _determine_mitigation(self, hall_score: float, rag_conf: float, kg_conf: float, docs: list) -> str:
    """
    Classify how hallucination was handled.
    Returns: 'prevented', 'reduced', 'abstained', 'unmitigated', 'none'
    """
    if not docs:  # No retrieval happened
        return "none"
    
    if hall_score > 0.7 and rag_conf < 0.35 and kg_conf < 0.35:
        return "abstained"  # Rejected due to low confidence + high hallucination
    
    if hall_score > 0.7 and kg_conf >= 0.75:
        return "prevented"  # KG overrode hallucinating RAG
    
    if hall_score > 0.5 and (rag_conf < 0.5 or kg_conf > rag_conf):
        return "reduced"  # Confidence lowered or KG tempered it
    
    if hall_score < 0.3:
        return "none"  # No hallucination detected
    
    return "unmitigated"  # Hallucination present but not caught
    def _merge_responses(self, rag_resp: str, kg_resp: str) -> str:
        """Intelligently merge RAG and KG responses."""
        # If RAG is detailed and KG is factual, combine them
        if len(rag_resp) > len(kg_resp):
            return f"{rag_resp}\n\n[Verified against Knowledge Graph]"
        else:
            return kg_resp + "\n\n[Additional context from medical literature]"

    def _cautious_merge(self, rag_resp: str, kg_resp: str) -> str:
        """Cautious merge for medium confidence."""
        return (
            "Based on available information:\n"
            f"{kg_resp}\n\n"
            "Note: This answer has moderate confidence. Please verify with a professional."
        )

    def detect_hallucination(self, rag_response: str, kg_facts: List[Dict], rag_docs: List[Dict] = None) -> float:
        """
        Detect hallucination by checking RAG claims against KG facts AND retrieved docs.
        Returns: 0 = no hallucination, 1 = complete hallucination, 0.5 = uncertain
        """
        if not rag_response:
            return 0.5
    
        response_lower = rag_response.lower()
        response_tokens = set(re.findall(r"\b\w{4,}\b", response_lower))
    
        if not response_tokens:
            return 0.0  # Empty response = nothing to hallucinate
    
        # Layer 1: Check against KG facts (structured verification)
        kg_score = self._check_kg_grounding(rag_response, kg_facts)
    
        # Layer 2: Check against retrieved documents (context verification)
        doc_score = self._check_doc_grounding(rag_response, rag_docs) if rag_docs else 0.5
    
        # Combine: if KG is empty, rely more on docs; if both empty, uncertain
        if kg_facts and rag_docs:
            final_score = 0.6 * kg_score + 0.4 * doc_score
        elif kg_facts:
            final_score = kg_score
        elif rag_docs:
            final_score = doc_score
        else:
            final_score = 0.5  # Truly unknown — no verification sources
    
        return round(min(final_score, 1.0), 3)

    def _check_kg_grounding(self, response: str, kg_facts: List[Dict]) -> float:
        """Check response against KG facts. Returns hallucination score (0=grounded, 1=hallucinated)."""
        if not kg_facts:
            return 0.5  # Unknown — no KG facts
    
        # Build KG entity set
        kg_entities = set()
        for fact in kg_facts:
            for key in ["subject", "object"]:
                val = fact.get(key, "").lower().replace("_", " ")
                if len(val) > 3:
                    kg_entities.add(val)
    
        if not kg_entities:
            return 0.5
    
        # Check overlap
        response_tokens = set(re.findall(r"\b\w{4,}\b", response.lower()))
        matched = sum(1 for token in response_tokens 
                    if any(token in ent for ent in kg_entities))
    
        if not response_tokens:
            return 0.0
    
        coverage = matched / len(response_tokens)
        return 1.0 - coverage  # 0 = fully grounded, 1 = fully hallucinated

    def _check_doc_grounding(self, response: str, docs: List[Dict]) -> float:
        """Check response against retrieved documents. Returns hallucination score."""
        if not docs:
            return 0.5
    
        context_text = " ".join([d.get("text", "") for d in docs]).lower()
        response_words = set(re.findall(r"\b\w{4,}\b", response.lower()))
    
        if not response_words:
            return 0.0
    
        # Count words that appear in context
        grounded_count = sum(1 for w in response_words if w in context_text)
        coverage = grounded_count / len(response_words)
    
        # Scale: not all words need to match (names, stop words, etc.)
        hallucination = 1.0 - (coverage * 1.5)  # 1.5x boost since exact match is strict
        return max(0.0, min(hallucination, 1.0))
