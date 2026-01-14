"""
Quick test run with 30 samples per task/language.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from experiments import DomainGeneralizationExperiment

if __name__ == "__main__":
    print("="*60)
    print("Running Quick Test with 30 samples per task/language")
    print("="*60)
    print()

    exp = DomainGeneralizationExperiment(
        languages=['ro', 'de', 'fr'],  # All 3 languages
        tasks=['xnli', 'xquad', 'pawsx'],  # All 3 tasks
        k=3,  # Top-3 retrieval
        n_samples=30,  # 30 samples per task/language
        output_dir='results'
    )

    results = exp.run_all_experiments()

    print("\n" + "="*60)
    print("Quick test completed!")
    print("Results saved to: results/experiment_results.json")
    print("="*60)
