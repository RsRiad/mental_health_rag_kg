"""
KG Engine
Handles graph queries and KG-based confidence scoring (K, M, G).
"""
import yaml
from typing import List, Dict
from rdflib import Graph
from kg.query import KGQueryEngine
from kg.builder import KGBuilder
from src.llm_client import LLMClient

class KGEngine:
    def __init__(self, config_path: str = "configs/config.yaml", prompts_path: str = "configs/prompts.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        with open(prompts_path, "r") as f:
            self.prompts = yaml.safe_load(f)

        self.graph = Graph()
        self.query_engine = KGQueryEngine(self.graph, self.config["kg"]["namespace"])
        self.llm = LLMClient(config_path)
        self._load_graphs()

    def _load_graphs(self):
        """Load base and dynamic KG files."""
        base_path = self.config["kg"]["base_ttl"]
        dynamic_path = self.config["kg"]["dynamic_ttl"]

        try:
            self.query_engine.load_graph(base_path)
        except Exception as e:
            print(f"[KG Engine] Could not load base KG: {e}")

        try:
            self.query_engine.load_graph(dynamic_path)
        except Exception as e:
            print(f"[KG Engine] Could not load dynamic KG: {e}")

    def query_symptoms(self, symptoms: List[str]) -> Dict:
        """
        Match symptoms to conditions and generate KG-based response.
        """
        matches = self.query_engine.match_symptoms_to_conditions(symptoms, top_k=3)

        if not matches:
            return {
                "source": "kg",
                "response": "No matching conditions found in the knowledge graph.",
                "matches": [],
                "confidence": {"K_known": 0.0, "M_alignment": 0.0, "G_graph": 0.0, "kg_confidence": 0.0}
            }

        # Build factual response from matches
        lines = ["Based on verified knowledge graph data:"]
        for m in matches:
            lines.append(f"- {m['condition']}: matches {m['match_count']}/{m['total_symptoms']} symptoms ({m['score']}%)")
        lines.append("\nThis is not a diagnosis. Consult a professional.")

        response = "\n".join(lines)

        # Calculate confidence
        conf = self.calculate_confidence(symptoms, matches)

        return {
            "source": "kg",
            "response": response,
            "matches": matches,
            "confidence": conf
        }

    def calculate_known_symptom_match(self, user_symptoms: List[str], matches: List[Dict]) -> float:
        """
        K = Known Symptoms Match
        Ratio of user symptoms that matched KG conditions.
        """
        if not user_symptoms:
            return 0.0

        matched = set()
        for m in matches:
            matched.update(m["matched_symptoms"])

        K = len(matched) / len(user_symptoms)
        return min(K, 1.0)

    def calculate_kg_alignment(self, matches: List[Dict]) -> float:
        """
        M = KG Alignment
        Measure of how well the matched conditions are represented.
        """
        if not matches:
            return 0.0

        # Higher score if top match has high coverage
        top_score = matches[0]["score"] / 100.0 if matches else 0
        avg_score = sum(m["score"] for m in matches) / (len(matches) * 100) if matches else 0

        M = top_score * 0.6 + avg_score * 0.4
        return min(M, 1.0)

    def calculate_graph_consistency(self, matches: List[Dict]) -> float:
        """
        G = Graph Consistency
        Check if the graph structure supports the matches.
        """
        if not matches:
            return 0.0

        # Simple heuristic: if matches exist, graph is consistent
        # Advanced: check for conflicting triples
        G = 1.0 if len(matches) > 0 else 0.0

        # Penalize if only one symptom matched per condition
        weak_matches = sum(1 for m in matches if m["match_count"] == 1)
        if weak_matches == len(matches):
            G *= 0.7

        return G

    def calculate_confidence(self, user_symptoms: List[str], matches: List[Dict]) -> Dict:
        """
        Calculate full KG confidence score.
        Returns dict with K, M, G, and final KG_Confidence.
        """
        weights = self.config["confidence"]["kg_weights"]
        K = self.calculate_known_symptom_match(user_symptoms, matches)
        M = self.calculate_kg_alignment(matches)
        G = self.calculate_graph_consistency(matches)

        final = weights["K"] * K + weights["M"] * M + weights["G"] * G

        return {
            "K_known": round(K, 3),
            "M_alignment": round(M, 3),
            "G_graph": round(G, 3),
            "kg_confidence": round(final, 3)
        }

    def get_facts(self) -> List[Dict]:
        """Get all facts from KG for hallucination checking."""
        return self.query_engine.get_all_facts()
