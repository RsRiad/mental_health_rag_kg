"""
Safety Checker Module
Performs keyword-based fast filtering + LLM-based semantic safety check.
"""
import os
import yaml
from typing import Dict
from src.llm_client import LLMClient

class SafetyChecker:
    def __init__(self, config_path: str = "configs/config.yaml", prompts_path: str = "configs/prompts.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        with open(prompts_path, "r") as f:
            self.prompts = yaml.safe_load(f)

        self.danger_keywords = [kw.lower() for kw in self.config["safety"]["danger_keywords"]]
        self.use_llm = self.config["safety"]["llm_safety_check"]
        self.llm = LLMClient(config_path) if self.use_llm else None

    def keyword_check(self, text: str) -> bool:
        """Fast keyword-based check. Returns True if unsafe."""
        text_lower = text.lower()
        return any(kw in text_lower for kw in self.danger_keywords)

    def llm_check(self, text: str) -> bool:
        """LLM-based semantic safety check. Returns True if unsafe."""
        if not self.llm:
            return False
        prompt = self.prompts["safety_check"].format(user_input=text)
        response = self.llm.classify(prompt, system_prompt="You are a safety classifier.")
        return "UNSAFE" in response.upper()

    def is_safe(self, text: str) -> bool:
        """
        Returns True if input is SAFE, False if UNSAFE.
        Priority: keyword check first (fast), then LLM check (slow but accurate).
        """
        if self.keyword_check(text):
            return False
        if self.use_llm and self.llm_check(text):
            return False
        return True

    def get_rejection_response(self) -> str:
        return self.prompts["rejection_response"]
