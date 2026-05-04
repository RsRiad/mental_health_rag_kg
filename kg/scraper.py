"""
KG Web Scraper
Scrapes trusted medical sources to dynamically expand KG.
"""
import re
import requests
from bs4 import BeautifulSoup
from typing import List, Dict

class MedicalScraper:
    def __init__(self):
        self.headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        self.session = requests.Session()

    def fetch_wikipedia(self, condition: str) -> str:
        """Fetch Wikipedia article text."""
        name = condition.replace(" ", "_")
        url = f"https://en.wikipedia.org/wiki/{name}"
        try:
            r = self.session.get(url, headers=self.headers, timeout=10)
            if r.status_code == 200:
                soup = BeautifulSoup(r.text, "lxml")
                paragraphs = soup.find_all("p")
                return " ".join(p.get_text() for p in paragraphs[:10])
        except Exception as e:
            print(f"[Wiki Error] {e}")
        return ""

    def fetch_medlineplus(self, condition: str) -> str:
        """Fetch MedlinePlus summary."""
        try:
            query = condition.replace(" ", "+")
            search_url = f"https://medlineplus.gov/search/?q={query}"
            r = self.session.get(search_url, headers=self.headers, timeout=10)
            if r.status_code != 200:
                return ""

            soup = BeautifulSoup(r.text, "lxml")
            result = soup.find("a", href=True)
            if not result:
                return ""

            page_url = "https://medlineplus.gov" + result["href"]
            r2 = self.session.get(page_url, headers=self.headers, timeout=10)
            soup2 = BeautifulSoup(r2.text, "lxml")
            paragraphs = soup2.find_all("p")
            return " ".join(p.get_text() for p in paragraphs[:10])
        except Exception as e:
            print(f"[Medline Error] {e}")
        return ""

    def extract_symptoms_from_text(self, text: str) -> List[str]:
        """Extract symptom keywords from scraped text."""
        symptom_keywords = [
            "anxiety", "depression", "insomnia", "fatigue", "restlessness",
            "worry", "panic", "fear", "sadness", "hopelessness", "mood swings",
            "hallucinations", "delusions", "intrusive thoughts", "compulsive",
            "flashbacks", "nightmares", "avoidance", "hypervigilance",
            "loss of interest", "concentration", "appetite", "weight",
            "sleep problems", "irritability", "agitation", "social withdrawal"
        ]

        text_lower = text.lower()
        found = []
        for kw in symptom_keywords:
            if kw in text_lower and kw not in found:
                found.append(kw)
        return found[:15]

    def fetch_condition_data(self, condition: str) -> Dict[str, List[str]]:
        """Fetch and extract symptoms from multiple sources."""
        wiki_text = self.fetch_wikipedia(condition)
        medline_text = self.fetch_medlineplus(condition)

        wiki_symptoms = self.extract_symptoms_from_text(wiki_text)
        medline_symptoms = self.extract_symptoms_from_text(medline_text)

        # Merge and deduplicate
        all_symptoms = list(dict.fromkeys(wiki_symptoms + medline_symptoms))

        return {
            "condition": condition,
            "symptoms": all_symptoms,
            "sources": {
                "wikipedia": len(wiki_text) > 0,
                "medlineplus": len(medline_text) > 0
            }
        }
