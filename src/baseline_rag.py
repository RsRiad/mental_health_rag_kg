"""
Baseline RAG-only pipeline (no KG fusion) for comparison.
This isolates the hallucination mitigation effect of the KG.
"""
from src.rag_engine import RAGEngine
from src.safety_checker import SafetyChecker
from src.input_processor import InputProcessor
from src.llm_client import LLMClient
import yaml

class BaselineRAGPipeline:
    """RAG-only pipeline without KG fusion or hallucination detection."""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        self.safety = SafetyChecker(config_path)
        self.processor = InputProcessor(config_path)
        self.rag = RAGEngine(config_path)
        self.llm = LLMClient(config_path)
        
        # Load prompts
        with open("configs/prompts.yaml", "r") as f:
            self.prompts = yaml.safe_load(f)
    
    def run(self, user_input: str) -> dict:
        """Run RAG-only, no KG, no fusion."""
        # Safety check
        if not self.safety.is_safe(user_input):
            return {
                "rejected": True,
                "final_response": self.safety.get_rejection_response(),
                "source": "safety"
            }
        
        # Preprocessing
        preproc = self.processor.process(user_input)
        query = preproc["refined_query"]
        
        # RAG only
        rag_result = self.rag.query(query)
        
        # No fusion, no hallucination check — return raw RAG
        return {
            "rejected": False,
            "final_response": rag_result["response"],
            "source": "rag_only",
            "rag_confidence": rag_result["confidence"]["llm_confidence"],
            "retrieved_docs": rag_result["retrieved_docs"]
        }