"""
PubMed Data Fetcher
Uses Biopython Entrez to fetch mental health articles.
"""
import os
import time
import json
from pathlib import Path
from typing import List, Dict
from Bio import Entrez
from dotenv import load_dotenv
load_dotenv()

class PubMedFetcher:
    def __init__(self, api_key: str = None, email: str = "user@example.com"):
        self.api_key = api_key or os.getenv("PUBMED_API_KEY", "")
        self.email = email
        Entrez.email = self.email
        if self.api_key:
            Entrez.api_key = self.api_key

    def search(self, query: str, max_results: int = 50) -> List[str]:
        """Search PubMed and return list of PMIDs."""
        try:
            handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results, sort="relevance")
            record = Entrez.read(handle)
            handle.close()
            return record.get("IdList", [])
        except Exception as e:
            print(f"[PubMed Search Error] {e}")
            return []

    def fetch_abstracts(self, pmids: List[str]) -> List[Dict[str, str]]:
        """Fetch abstracts for given PMIDs."""
        if not pmids:
            return []

        abstracts = []
        try:
            handle = Entrez.efetch(db="pubmed", id=",".join(pmids), rettype="abstract", retmode="xml")
            records = Entrez.read(handle)
            handle.close()

            for article in records.get("PubmedArticle", []):
                try:
                    medline = article["MedlineCitation"]
                    article_data = medline["Article"]

                    pmid = str(medline.get("PMID", ""))
                    title = article_data.get("ArticleTitle", "")

                    abstract_text = ""
                    if "Abstract" in article_data and "AbstractText" in article_data["Abstract"]:
                        abstract_parts = article_data["Abstract"]["AbstractText"]
                        if isinstance(abstract_parts, list):
                            abstract_text = " ".join([str(p) for p in abstract_parts])
                        else:
                            abstract_text = str(abstract_parts)

                    if abstract_text and len(abstract_text) > 50:
                        abstracts.append({
                            "pmid": pmid,
                            "title": title,
                            "text": abstract_text,
                            "source": "pubmed"
                        })
                except Exception as e:
                    continue
        except Exception as e:
            print(f"[PubMed Fetch Error] {e}")

        return abstracts

    def fetch_and_save(self, query: str, output_dir: str, max_results: int = 50) -> str:
        """Search, fetch, and save to JSONL."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        print(f"Searching PubMed for: {query}")
        pmids = self.search(query, max_results)
        print(f"Found {len(pmids)} articles.")

        if not pmids:
            return ""

        articles = self.fetch_abstracts(pmids)
        print(f"Fetched {len(articles)} abstracts with text.")

        file_path = out_path / "pubmed_articles.jsonl"
        with open(file_path, "w", encoding="utf-8") as f:
            for art in articles:
                f.write(json.dumps(art, ensure_ascii=False) + "\n")

        return str(file_path)
