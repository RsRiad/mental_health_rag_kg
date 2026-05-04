"""
Document Processor
Chunks documents for RAG ingestion.
"""
import json
import re
from pathlib import Path
from typing import List, Dict
import yaml

class DocumentProcessor:
    def __init__(self, config_path: str = None):
        # with open(config_path, "r") as f:
        #     self.config = yaml.safe_load(f)
        if config_path is None:
            project_root = Path(__file__).parent.parent
            config_path = project_root / "configs" / "config.yaml"
        
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        self.chunk_size = self.config["rag"]["chunk_size"]
        self.chunk_overlap = self.config["rag"]["chunk_overlap"]
        self.min_length = self.config["rag"]["min_chunk_length"]

    def clean_text(self, text: str) -> str:
        """Basic text cleaning."""
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^\w\s.,;!?-]", "", text)
        return text.strip()

    def chunk_text(self, text: str, source_meta: Dict = None) -> List[Dict]:
        """
        Split text into overlapping chunks.
        Returns list of {text, source, chunk_id}
        """
        text = self.clean_text(text)
        words = text.split()
        chunks = []

        if len(words) <= self.chunk_size:
            if len(text) >= self.min_length:
                chunks.append({
                    "text": text,
                    "source": source_meta or {},
                    "chunk_id": 0
                })
            return chunks

        start = 0
        chunk_id = 0
        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words)

            if len(chunk_text) >= self.min_length:
                chunks.append({
                    "text": chunk_text,
                    "source": source_meta or {},
                    "chunk_id": chunk_id
                })
                chunk_id += 1

            start += self.chunk_size - self.chunk_overlap
            if start >= len(words):
                break

        return chunks

    def process_jsonl(self, input_path: str, output_path: str):
        """Process a JSONL file of documents into chunked JSONL."""
        in_path = Path(input_path)
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        all_chunks = []
        with open(in_path, "r", encoding="utf-8") as f:
            for line in f:
                doc = json.loads(line.strip())
                text = doc.get("text", "")
                meta = {k: v for k, v in doc.items() if k != "text"}
                chunks = self.chunk_text(text, meta)
                all_chunks.extend(chunks)

        with open(out_path, "w", encoding="utf-8") as f:
            for chunk in all_chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

        print(f"Processed {in_path} -> {len(all_chunks)} chunks saved to {out_path}")
        return str(out_path)
