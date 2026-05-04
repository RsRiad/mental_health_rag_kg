"""
Knowledge Graph Builder
Constructs RDF-based mental health KG from structured data and web sources.
"""
import re
from pathlib import Path
from datetime import datetime
from rdflib import Graph, Namespace, Literal, RDF, URIRef
import yaml

class KGBuilder:
    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.ns_uri = self.config["kg"]["namespace"]
        self.MH = Namespace(self.ns_uri)
        self.graph = Graph()
        self.graph.bind("mh", self.MH)

        self.base_path = Path(self.config["kg"]["base_ttl"])
        self.dynamic_path = Path(self.config["kg"]["dynamic_ttl"])
        self.base_path.parent.mkdir(parents=True, exist_ok=True)

    def _sanitize(self, text: str) -> str:
        """Create URI-safe string."""
        text = text.lower().strip()
        text = re.sub(r"[^a-z0-9]+", "_", text)
        text = re.sub(r"_+", "_", text)
        return text.strip("_").title()

    def add_condition(self, condition: str, symptoms: list, source: str = "manual"):
        """Add a condition and its symptoms to the graph."""
        cond_id = self._sanitize(condition)
        cond_uri = self.MH[cond_id]

        self.graph.add((cond_uri, RDF.type, self.MH.Condition))
        self.graph.add((cond_uri, self.MH.label, Literal(condition)))
        self.graph.add((cond_uri, self.MH.source, Literal(source)))
        self.graph.add((cond_uri, self.MH.last_updated, Literal(datetime.utcnow().isoformat())))

        for sym in symptoms:
            sym_id = self._sanitize(sym)
            sym_uri = self.MH[sym_id]
            self.graph.add((sym_uri, RDF.type, self.MH.Symptom))
            self.graph.add((sym_uri, self.MH.label, Literal(sym.lower())))
            self.graph.add((cond_uri, self.MH.associated_with, sym_uri))

    def add_relationship(self, entity1: str, entity2: str, relation: str = "related_to"):
        """Add generic relationship between two entities."""
        e1 = self.MH[self._sanitize(entity1)]
        e2 = self.MH[self._sanitize(entity2)]
        pred = self.MH[relation]
        self.graph.add((e1, pred, e2))

    def build_base_kg(self):
        """Build initial seed KG with common mental health conditions."""
        seed_data = {
            "Anxiety": ["restlessness", "muscle tension", "worry", "insomnia", "fatigue"],
            "Depression": ["low mood", "loss of interest", "fatigue", "sleep problems", "hopelessness"],
            "OCD": ["intrusive thoughts", "compulsive behaviors", "rituals", "anxiety"],
            "Schizophrenia": ["hallucinations", "delusions", "disorganized thinking", "social withdrawal"],
            "PTSD": ["flashbacks", "nightmares", "avoidance", "hypervigilance", "emotional numbness"],
            "Bipolar Disorder": ["mood swings", "mania", "depression", "impulsivity", "sleep changes"],
        }

        for condition, symptoms in seed_data.items():
            self.add_condition(condition, symptoms, source="seed")

        self.save(self.base_path)
        print(f"Base KG saved with {len(self.graph)} triples to {self.base_path}")

    def save(self, path: Path = None):
        """Serialize graph to Turtle."""
        path = path or self.dynamic_path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.graph.serialize(destination=str(path), format="turtle")

    def load(self, path: Path = None):
        """Load graph from Turtle."""
        path = path or self.base_path
        if path.exists():
            self.graph.parse(str(path), format="turtle")
            print(f"Loaded KG from {path} with {len(self.graph)} triples.")
        else:
            print(f"No existing KG at {path}. Starting fresh.")
