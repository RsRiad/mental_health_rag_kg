"""
LLM Client for SambaNova API (OpenAI-compatible)
Supports Meta-Llama-3.3-70B-Instruct and fallback models.
"""
import os
import requests
import yaml
from typing import Dict, List, Optional
from pathlib import Path

class LLMClient:
    def __init__(self, config_path: str = "configs/config.yaml"):
        self.config = self._load_config(config_path)
        self.api_key = os.getenv("SAMBANOVA_API_KEY", "")
        self.base_url = self.config["api"]["sambanova_url"]
        self.model = self.config["api"].get("sambanova_model", "Meta-Llama-3.3-70B-Instruct")
        self.max_tokens = self.config["api"]["max_tokens"]
        self.temperature = self.config["api"]["temperature"]
        self.top_p = self.config["api"]["top_p"]

        if not self.api_key:
            raise ValueError("SAMBANOVA_API_KEY not found in environment variables.")

    def _load_config(self, path: str) -> dict:
        with open(path, "r") as f:
            return yaml.safe_load(f)

    def chat(self, messages: List[Dict[str, str]], max_tokens: Optional[int] = None, temperature: Optional[float] = None) -> str:
        """Send chat completion request to SambaNova."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": temperature if temperature is not None else self.temperature,
            "top_p": self.top_p
        }
        try:
            response = requests.post(self.base_url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"[LLM ERROR] {e}")
            return ""

    def generate(self, prompt: str, system_prompt: str = "You are a helpful assistant.", **kwargs) -> str:
        """Simple prompt-based generation."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        return self.chat(messages, **kwargs)

    def classify(self, prompt: str, system_prompt: str = "You are a classifier.") -> str:
        """Low-temperature classification."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        return self.chat(messages, temperature=0.0, max_tokens=10)
