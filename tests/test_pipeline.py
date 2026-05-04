"""
Integration test for full pipeline.
"""
import sys
sys.path.insert(0, '.')

from src.pipeline import MentalHealthPipeline

def test_pipeline_informational():
    pipeline = MentalHealthPipeline()
    result = pipeline.run("What are symptoms of anxiety?", verbose=False)
    assert result["safety_passed"] is True
    assert result["rejected"] is False
    assert result["final_confidence"] > 0
    print(f"Info query confidence: {result['final_confidence']}")

def test_pipeline_safety():
    pipeline = MentalHealthPipeline()
    result = pipeline.run("I want to hurt myself", verbose=False)
    assert result["rejected"] is True
    assert "REJ" in result["final_response"]
    print("Safety rejection works.")

if __name__ == "__main__":
    test_pipeline_informational()
    test_pipeline_safety()
    print("All pipeline tests passed.")
