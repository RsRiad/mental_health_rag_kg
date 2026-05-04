"""
Main Pipeline
Orchestrates: Safety -> Preprocessing -> RAG + KG (parallel) -> Fusion -> Output
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Dict

from src.safety_checker import SafetyChecker
from src.input_processor import InputProcessor
from src.rag_engine import RAGEngine
from src.kg_engine import KGEngine
from src.fusion_layer import FusionLayer

class MentalHealthPipeline:
    def __init__(self, config_path: str = "configs/config.yaml"):
        self.config_path = config_path
        self.safety = SafetyChecker(config_path)
        self.processor = InputProcessor(config_path)
        self.rag = RAGEngine(config_path)
        self.kg = KGEngine(config_path)
        self.fusion = FusionLayer(config_path)

        # Setup output logging
        self.output_dir = Path("outputs/responses")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self, user_input: str, verbose: bool = False) -> Dict:
        """
        Run full pipeline on user input.
        Returns structured result dict.
        """
        result = {
            "timestamp": datetime.utcnow().isoformat(),
            "user_input": user_input,
            "safety_passed": False,
            "rejected": False,
            "rejection_reason": None,
            "preprocessing": None,
            "rag_result": None,
            "kg_result": None,
            "fusion": None,
            "final_response": None,
            "final_confidence": 0.0
        }

        # STEP 1: Safety Check
        if not self.safety.is_safe(user_input):
            result["rejected"] = True
            result["rejection_reason"] = "safety"
            result["final_response"] = self.safety.get_rejection_response()
            self._log(result)
            if verbose:
                print("[SAFETY] Input rejected.")
            return result

        result["safety_passed"] = True
        if verbose:
            print("[SAFETY] Input is safe.")

        # STEP 2: Preprocessing
        preproc = self.processor.process(user_input)
        result["preprocessing"] = {
            "symptoms": preproc["symptoms"],
            "refined_query": preproc["refined_query"]
        }
        if verbose:
            print(f"[PREPROC] Symptoms: {preproc['symptoms']}")
            print(f"[PREPROC] Refined query: {preproc['refined_query']}")

        # STEP 3: Parallel RAG + KG
        query = preproc["refined_query"]
        symptoms = preproc["symptoms"]

        if verbose:
            print("[RAG] Retrieving from vector store...")
        rag_result = self.rag.query(query)
        result["rag_result"] = rag_result

        if verbose:
            print(f"[RAG] Confidence: {rag_result['confidence']}")
            print("[KG] Querying knowledge graph...")
        kg_result = self.kg.query_symptoms(symptoms)
        result["kg_result"] = kg_result

        if verbose:
            print(f"[KG] Confidence: {kg_result['confidence']}")

        # STEP 4: Fusion
        fusion_result = self.fusion.fuse(rag_result, kg_result)
        result["fusion"] = {
            "strategy": fusion_result["strategy"],
            "hallucination_score": fusion_result["hallucination_score"],
            "rag_confidence": fusion_result["rag_confidence"],
            "kg_confidence": fusion_result["kg_confidence"]
        }
        result["final_response"] = fusion_result["final_response"]
        result["final_confidence"] = fusion_result["final_confidence"]

        if verbose:
            print(f"[FUSION] Strategy: {fusion_result['strategy']}")
            print(f"[FUSION] Hallucination score: {fusion_result['hallucination_score']}")
            print(f"[FUSION] Final confidence: {fusion_result['final_confidence']}")

        self._log(result)
        return result

    def _log(self, result: Dict):
        """Append result to log file."""
        log_path = self.output_dir / "response_log.jsonl"
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    def chat(self):
        """Interactive chat loop."""
        print("=" * 60)
        print("Mental Health RAG+KG Assistant")
        print("Type 'exit' to quit, 'verbose' to toggle details.")
        print("=" * 60)

        verbose = False
        while True:
            user_input = input("\nYou: ").strip()
            if user_input.lower() == "exit":
                print("Goodbye.")
                break
            if user_input.lower() == "verbose":
                verbose = not verbose
                print(f"Verbose mode: {verbose}")
                continue
            if not user_input:
                continue

            result = self.run(user_input, verbose=verbose)
            print(f"\nBot: {result['final_response']}")
            if not result["rejected"]:
                print(f"[Confidence: {result['final_confidence']}]")
