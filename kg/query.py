"""
KG Query Engine
SPARQL queries for symptom matching, condition lookup, and fact verification.
"""
from rdflib import Graph, Namespace
from typing import List, Dict, Tuple

class KGQueryEngine:
    def __init__(self, graph: Graph = None, namespace: str = "http://example.org/mentalhealth#"):
        self.graph = graph or Graph()
        self.MH = Namespace(namespace)

    def load_graph(self, ttl_path: str):
        """Load RDF graph from Turtle file."""
        self.graph.parse(ttl_path, format="turtle")

    def get_conditions(self) -> List[Tuple[str, str]]:
        """Get all conditions with labels."""
        q = """
        PREFIX mh: <http://example.org/mentalhealth#>
        SELECT ?cond ?label WHERE {
            ?cond a mh:Condition .
            ?cond mh:label ?label .
        }
        """
        return [(str(row[0]).split("#")[-1], str(row[1])) for row in self.graph.query(q)]

    def get_symptoms_of_condition(self, condition_name: str) -> List[str]:
        """Get symptoms for a specific condition."""
        q = f"""
        PREFIX mh: <http://example.org/mentalhealth#>
        SELECT ?symptomLabel WHERE {{
            ?cond a mh:Condition ;
                  mh:label "{condition_name}" ;
                  mh:associated_with ?sym .
            ?sym mh:label ?symptomLabel .
        }}
        """
        return [str(row[0]) for row in self.graph.query(q)]

    def match_symptoms_to_conditions(self, symptoms: List[str], top_k: int = 3) -> List[Dict]:
        """
        Match user symptoms to conditions in KG.
        Returns ranked list of {condition, matched_symptoms, total_symptoms, score}
        """
        conditions = self.get_conditions()
        results = []
        user_symptoms = set(s.lower() for s in symptoms)

        for cond_uri, cond_label in conditions:
            cond_symptoms = self.get_symptoms_of_condition(cond_label)
            cond_set = set(s.lower() for s in cond_symptoms)

            if not cond_set:
                continue

            overlap = user_symptoms.intersection(cond_set)
            if not overlap:
                continue

            score = (len(overlap) / len(cond_set)) * 100.0
            results.append({
                "condition": cond_label,
                "matched_symptoms": list(overlap),
                "total_symptoms": len(cond_set),
                "match_count": len(overlap),
                "score": round(score, 1)
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def verify_fact(self, subject: str, predicate: str, obj: str) -> bool:
        """Verify if a triple exists in the graph."""
        s = self.MH[subject]
        p = self.MH[predicate]
        o = self.MH[obj]
        return (s, p, o) in self.graph

    def get_all_facts(self) -> List[Dict]:
        """Export all triples as readable facts."""
        facts = []
        for s, p, o in self.graph:
            facts.append({
                "subject": str(s).split("#")[-1],
                "predicate": str(p).split("#")[-1],
                "object": str(o).split("#")[-1] if "#" in str(o) else str(o)
            })
        return facts
