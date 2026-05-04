"""
Input Processor Module
Handles: Tokenization, Embedding, Symptom Extraction, Query Refinement
"""
import re
import yaml
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
from src.llm_client import LLMClient

class InputProcessor:
    def __init__(self, config_path: str = "configs/config.yaml", prompts_path: str = "configs/prompts.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        with open(prompts_path, "r") as f:
            self.prompts = yaml.safe_load(f)

        self.embedder = SentenceTransformer(self.config["models"]["embedding"])
        self.llm = LLMClient(config_path)

        # Canonical symptom lexicon for fast extraction
        self.symptom_lexicon = [
            "anxiety", "depression", "insomnia", "fatigue", "hopelessness",
            "restlessness", "worry", "panic", "fear", "sadness", "low mood",
            "loss of interest", "concentration", "irritability", "anger",
            "hallucination", "delusion", "paranoia", "mania", "mood swings",
            "appetite", "weight", "sleep", "nightmare", "flashback",
            "avoidance", "hypervigilance", "guilt", "shame", "loneliness",
            "suicidal", "self-harm", "overwhelmed", "stress", "tension"
        ]

    def tokenize_embed(self, text: str) -> Tuple[List[int], List[float]]:
        """
        Tokenize and embed input text.
        Returns: (tokens as ints placeholder, embedding vector)
        """
        embedding = self.embedder.encode(text, convert_to_numpy=True).tolist()
        tokens = text.split()  # Simplified token representation
        return tokens, embedding

    def extract_symptoms(self, text: str, use_llm: bool = True) -> List[str]:
        """
        Extract mental health symptoms from text.
        Uses rule-based matching first, LLM fallback if enabled.
        """
        text_lower = text.lower()
        found = []

        # Rule-based extraction
        for symptom in self.symptom_lexicon:
            if symptom in text_lower:
                found.append(symptom)

        # LLM-based extraction for better coverage
        if use_llm and self.llm:
            prompt = self.prompts["symptom_extraction"].format(text=text)
            response = self.llm.generate(prompt, temperature=0.1, max_tokens=100)
            if response.lower() != "none":
                llm_symptoms = [s.strip().lower() for s in response.split(",") if len(s.strip()) > 2]
                found.extend(llm_symptoms)

        # Deduplicate and clean
        found = list(dict.fromkeys(found))
        return found[:self.config["kg"]["max_symptoms"]]

    def refine_query(self, user_input: str, symptoms: List[str]) -> str:
        """
        Refine user input into a medical search query.
        """
        if not self.llm:
            return user_input

        symptoms_str = ", ".join(symptoms) if symptoms else "none"
        prompt = self.prompts["query_refinement"].format(
            user_input=user_input,
            symptoms=symptoms_str
        )
        refined = self.llm.generate(prompt, temperature=0.2, max_tokens=100)
        return refined if refined else user_input

    def process(self, user_input: str) -> Dict:
        """
        Full preprocessing pipeline.
        Returns dict with: tokens, embedding, symptoms, refined_query
        """
        tokens, embedding = self.tokenize_embed(user_input)
        symptoms = self.extract_symptoms(user_input)
        refined = self.refine_query(user_input, symptoms)

        return {
            "original_input": user_input,
            "tokens": tokens,
            "embedding": embedding,
            "symptoms": symptoms,
            "refined_query": refined
        }
