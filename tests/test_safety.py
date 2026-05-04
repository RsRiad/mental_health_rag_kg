"""
Unit tests for Safety Checker.
Run with: python -m pytest tests/test_safety.py -v
"""
import sys
sys.path.insert(0, '.')

from src.safety_checker import SafetyChecker

def test_safety_rejection():
    checker = SafetyChecker()
    unsafe_inputs = [
        "I want to kill myself",
        "How do I hang myself?",
        "I am going to overdose tonight"
    ]
    for text in unsafe_inputs:
        assert not checker.is_safe(text), f"Should reject: {text}"

def test_safety_acceptance():
    checker = SafetyChecker()
    safe_inputs = [
        "What are symptoms of anxiety?",
        "How to improve sleep quality?",
        "Tell me about depression treatment"
    ]
    for text in safe_inputs:
        assert checker.is_safe(text), f"Should accept: {text}"

if __name__ == "__main__":
    test_safety_rejection()
    test_safety_acceptance()
    print("All safety tests passed.")
