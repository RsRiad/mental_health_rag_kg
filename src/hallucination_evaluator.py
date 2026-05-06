"""
Hallucination Mitigation Rate Evaluator
Run this after building RAG and KG to measure effectiveness.
"""
import json
from pathlib import Path
from typing import List, Dict
import pandas as pd
from src.pipeline import MentalHealthPipeline

class HallucinationEvaluator:
    def __init__(self, output_dir: str = "outputs/hallucination_eval"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.pipeline = MentalHealthPipeline()
        
        # Test cases designed to trigger hallucinations
        self.hallucination_test_cases = [
            # Cases where RAG might hallucinate (vague or out-of-scope queries)
            {
                "query": "How many angels can dance on the head of a pin?",
                "category": "nonsense",
                "expected_hallucination": True,
                "reason": "No medical relevance, RAG may invent answer"
            },
            {
                "query": "What is the best medication for depression in aliens?",
                "category": "fictional",
                "expected_hallucination": True,
                "reason": "No evidence for alien psychiatry"
            },
            {
                "query": "Tell me about the new drug Neuroxil for anxiety",
                "category": "fabricated_entity",
                "expected_hallucination": True,
                "reason": "Drug name is invented"
            },
            {
                "query": "What are symptoms of anxiety?",
                "category": "factual",
                "expected_hallucination": False,
                "reason": "Well-covered in KG and corpus"
            },
            {
                "query": "How to treat schizophrenia with herbs and crystals?",
                "category": "misinformation",
                "expected_hallucination": True,
                "reason": "RAG may retrieve poor sources"
            },
            {
                "query": "What is the connection between gut bacteria and depression?",
                "category": "emerging",
                "expected_hallucination": False,
                "reason": "Legitimate research area, should be in PubMed"
            },
            {
                "query": "Can eating only blue foods cure bipolar disorder?",
                "category": "absurd",
                "expected_hallucination": True,
                "reason": "No scientific basis"
            },
            {
                "query": "What are the side effects of SSRIs?",
                "category": "factual",
                "expected_hallucination": False,
                "reason": "Well-documented in medical literature"
            }
        ]
    
    def run_evaluation(self, verbose: bool = False) -> Dict:
        """Run all test cases and calculate metrics."""
        results = []
        
        for test in self.hallucination_test_cases:
            print(f"\nTesting: {test['query'][:60]}...")
            
            # Run pipeline
            result = self.pipeline.run(test['query'], verbose=verbose)
            
            # Extract hallucination data
            fusion = result.get('fusion', {})
            hall_score = fusion.get('hallucination_score', 0.5)
            mitigation = fusion.get('mitigation_status', 'unknown')
            strategy = fusion.get('strategy', 'unknown')
            
            # Determine if hallucination actually occurred
            # (Heuristic: high hallucination score + not abstained)
            hallucination_occurred = (
                hall_score > 0.6 and 
                mitigation not in ['abstained', 'prevented']
            )
            
            # Determine if mitigation worked
            was_mitigated = mitigation in ['prevented', 'reduced', 'abstained']
            
            record = {
                'query': test['query'],
                'category': test['category'],
                'expected_hallucination': test['expected_hallucination'],
                'hallucination_score': hall_score,
                'mitigation_status': mitigation,
                'strategy': strategy,
                'hallucination_occurred': hallucination_occurred,
                'was_mitigated': was_mitigated,
                'final_confidence': result['final_confidence'],
                'rejected': result['rejected'],
                'rag_confidence': fusion.get('rag_confidence', 0),
                'kg_confidence': fusion.get('kg_confidence', 0)
            }
            results.append(record)
            
            print(f"  Score: {halluc_score} | Mitigation: {mitigation} | Strategy: {strategy}")
        
        # Calculate aggregate metrics
        df = pd.DataFrame(results)
        metrics = self._calculate_metrics(df)
        
        # Save results
        self._save_results(df, metrics)
        
        return metrics
    
    def _calculate_metrics(self, df: pd.DataFrame) -> Dict:
        """Calculate hallucination mitigation metrics."""
        
        # Filter to cases where hallucination was expected
        expected_hall = df[df['expected_hallucination'] == True]
        
        # Filter to cases where hallucination actually occurred
        actual_hall = df[df['hallucination_occurred'] == True]
        
        metrics = {
            'total_test_cases': len(df),
            'expected_hallucination_cases': len(expected_hall),
            'actual_hallucination_cases': len(actual_hall),
            
            # Core mitigation metrics
            'mitigation_rate': (
                expected_hall['was_mitigated'].mean() 
                if len(expected_hall) > 0 else 0
            ),
            'prevention_rate': (
                (expected_hall['mitigation_status'] == 'prevented').mean()
                if len(expected_hall) > 0 else 0
            ),
            'abstention_rate': (
                (expected_hall['mitigation_status'] == 'abstained').mean()
                if len(expected_hall) > 0 else 0
            ),
            'reduction_rate': (
                (expected_hall['mitigation_status'] == 'reduced').mean()
                if len(expected_hall) > 0 else 0
            ),
            'unmitigated_rate': (
                (expected_hall['mitigation_status'] == 'unmitigated').mean()
                if len(expected_hall) > 0 else 0
            ),
            
            # False positive analysis (factual queries wrongly flagged)
            'false_positive_rate': (
                (df[df['expected_hallucination'] == False]['hallucination_occurred']).mean()
                if len(df[df['expected_hallucination'] == False]) > 0 else 0
            ),
            
            # Confidence correlation
            'avg_confidence_mitigated': (
                expected_hall[expected_hall['was_mitigated']]['final_confidence'].mean()
                if len(expected_hall[expected_hall['was_mitigated']]) > 0 else 0
            ),
            'avg_confidence_unmitigated': (
                expected_hall[~expected_hall['was_mitigated']]['final_confidence'].mean()
                if len(expected_hall[~expected_hall['was_mitigated']]) > 0 else 0
            )
        }
        
        return {k: round(v, 3) if isinstance(v, float) else v for k, v in metrics.items()}
    
    def _save_results(self, df: pd.DataFrame, metrics: Dict):
        """Save detailed results and summary."""
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        
        # Detailed results CSV
        df.to_csv(self.output_dir / f'hallucination_eval_{timestamp}.csv', index=False)
        
        # Metrics JSON
        with open(self.output_dir / f'hallucination_metrics_{timestamp}.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Print summary
        print(f"\n{'='*60}")
        print("HALLUCINATION MITIGATION EVALUATION SUMMARY")
        print(f"{'='*60}")
        print(f"Total test cases: {metrics['total_test_cases']}")
        print(f"Expected hallucination cases: {metrics['expected_hallucination_cases']}")
        print(f"Actual hallucination cases: {metrics['actual_hallucination_cases']}")
        print(f"\n--- MITIGATION RATES ---")
        print(f"Overall Mitigation Rate: {metrics['mitigation_rate']:.1%}")
        print(f"  - Prevented (KG blocked): {metrics['prevention_rate']:.1%}")
        print(f"  - Abstained (rejected): {metrics['abstention_rate']:.1%}")
        print(f"  - Reduced (confidence lowered): {metrics['reduction_rate']:.1%}")
        print(f"Unmitigated Rate: {metrics['unmitigated_rate']:.1%}")
        print(f"\n--- QUALITY METRICS ---")
        print(f"False Positive Rate: {metrics['false_positive_rate']:.1%}")
        print(f"Avg Confidence (mitigated): {metrics['avg_confidence_mitigated']:.3f}")
        print(f"Avg Confidence (unmitigated): {metrics['avg_confidence_unmitigated']:.3f}")
        print(f"{'='*60}")
        print(f"Results saved to: {self.output_dir}")


if __name__ == "__main__":
    evaluator = HallucinationEvaluator()
    metrics = evaluator.run_evaluation(verbose=False)


    def run_comparative_evaluation(self, verbose: bool = False) -> Dict:
        """
        Compare RAG-only vs RAG+KG on hallucination rates.
        This measures the actual mitigation effect.
        """
        from src.baseline_rag import BaselineRAGPipeline
        
        baseline = BaselineRAGPipeline()
        results_baseline = []
        results_kg = []
        
        for test in self.hallucination_test_cases:
            # Run baseline (no KG)
            base_result = baseline.run(test['query'])
            
            # Run full pipeline (with KG)
            kg_result = self.pipeline.run(test['query'], verbose=verbose)
            
            # Analyze baseline
            base_hall = self._estimate_hallucination_baseline(
                base_result['final_response'], 
                base_result.get('retrieved_docs', [])
            )
            
            # Analyze KG-enhanced
            fusion = kg_result.get('fusion', {})
            kg_hall = fusion.get('hallucination_score', 0.5)
            kg_mitigated = fusion.get('mitigation_status') in ['prevented', 'reduced', 'abstained']
            
            results_baseline.append({
                'query': test['query'],
                'hallucination_score': base_hall,
                'response': base_result['final_response'][:200]
            })
            
            results_kg.append({
                'query': test['query'],
                'hallucination_score': kg_hall,
                'was_mitigated': kg_mitigated,
                'strategy': fusion.get('strategy', 'unknown')
            })
        
        # Calculate comparative metrics
        df_base = pd.DataFrame(results_baseline)
        df_kg = pd.DataFrame(results_kg)
        
        comparison = {
            'baseline_avg_hallucination': df_base['hallucination_score'].mean(),
            'kg_avg_hallucination': df_kg['hallucination_score'].mean(),
            'absolute_reduction': (
                df_base['hallucination_score'].mean() - 
                df_kg['hallucination_score'].mean()
            ),
            'relative_reduction': (
                (df_base['hallucination_score'].mean() - df_kg['hallucination_score'].mean()) /
                df_base['hallucination_score'].mean()
                if df_base['hallucination_score'].mean() > 0 else 0
            ),
            'mitigation_rate': df_kg['was_mitigated'].mean()
        }
        
        print(f"\n{'='*60}")
        print("COMPARATIVE HALLUCINATION MITIGATION ANALYSIS")
        print(f"{'='*60}")
        print(f"Baseline (RAG-only) avg hallucination: {comparison['baseline_avg_hallucination']:.3f}")
        print(f"KG-enhanced avg hallucination: {comparison['kg_avg_hallucination']:.3f}")
        print(f"Absolute reduction: {comparison['absolute_reduction']:.3f}")
        print(f"Relative reduction: {comparison['relative_reduction']:.1%}")
        print(f"Mitigation rate: {comparison['mitigation_rate']:.1%}")
        print(f"{'='*60}")
        
        return comparison
    
    def _estimate_hallucination_baseline(self, response: str, docs: List[Dict]) -> float:
        """Estimate hallucination for baseline RAG (no KG facts)."""
        if not docs:
            return 0.5  # Unknown
        
        # Simple doc grounding check (same as RAG engine's G score)
        context_text = " ".join([d.get("text", "") for d in docs]).lower()
        response_words = set(re.findall(r"\b\w{4,}\b", response.lower()))
        
        if not response_words:
            return 0.0
        
        grounded = sum(1 for w in response_words if w in context_text)
        coverage = grounded / len(response_words)
        
        # Baseline has no KG, so hallucination is higher
        return round(1.0 - (coverage * 1.2), 3)  # 1.2x stricter than fusion layer