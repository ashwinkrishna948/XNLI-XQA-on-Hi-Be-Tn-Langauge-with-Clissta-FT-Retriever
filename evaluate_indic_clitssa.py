"""
Evaluate fine-tuned CLiTSSA retrievers on downstream tasks.

Compares:
1. Base multilingual Sentence-BERT (baseline)
2. Task-specific CLiTSSA models
3. Integrated CLiTSSA* model

Following the paper's evaluation methodology.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from sentence_transformers import SentenceTransformer, util
from indic_data_loader import get_indic_loader
from typing import List, Tuple, Dict
import json
import random
from tqdm import tqdm
import numpy as np
import argparse


class CLiTSSAEvaluator:
    """Evaluator for CLiTSSA retrievers on downstream X-ICL tasks"""

    def __init__(self, model_path: str, model_name: str = "model"):
        """
        Args:
            model_path: Path to the retriever model
            model_name: Name for logging
        """
        self.model = SentenceTransformer(model_path)
        self.model_name = model_name
        print(f"Loaded model: {model_name} from {model_path}")

    def retrieve_examples(self, query: str, examples: List[str],
                         k: int = 3) -> Tuple[List[int], List[float]]:
        """
        Retrieve top-k most similar examples for a query.

        Args:
            query: Query text
            examples: List of example texts
            k: Number of examples to retrieve

        Returns:
            Tuple of (indices, similarity_scores)
        """
        # Encode query and examples
        query_emb = self.model.encode(query, convert_to_tensor=True)
        example_embs = self.model.encode(examples, convert_to_tensor=True,
                                        show_progress_bar=False)

        # Compute similarities
        similarities = util.cos_sim(query_emb, example_embs)[0]

        # Get top-k indices
        top_k_idx = similarities.topk(k).indices.cpu().numpy()
        top_k_scores = similarities.topk(k).values.cpu().numpy()

        return top_k_idx.tolist(), top_k_scores.tolist()

    def evaluate_retrieval_quality(self, language: str, task: str,
                                   n_samples: int = 100, k: int = 3) -> Dict:
        """
        Evaluate retrieval quality.

        Metrics:
        - Mean retrieval score
        - Std retrieval score
        - Top-1 accuracy (if labels available)

        Args:
            language: Language code
            task: Task name
            n_samples: Number of test samples
            k: Number of examples to retrieve

        Returns:
            Dictionary with evaluation metrics
        """
        print(f"\n  Evaluating {self.model_name} on {language}/{task}...")

        # Load test data
        loader = get_indic_loader(task)
        queries_test, _ = loader.load_data(language, split='validation', n_samples=n_samples)
        _, examples_en = loader.load_data('en', split='train', n_samples=500)

        # Retrieve for each query
        all_scores = []

        for query in tqdm(queries_test, desc=f"  Retrieving {language}/{task}"):
            indices, scores = self.retrieve_examples(query, examples_en, k=k)
            all_scores.extend(scores)

        # Compute metrics
        metrics = {
            'model': self.model_name,
            'language': language,
            'task': task,
            'n_samples': len(queries_test),
            'k': k,
            'mean_score': float(np.mean(all_scores)),
            'std_score': float(np.std(all_scores)),
            'min_score': float(np.min(all_scores)),
            'max_score': float(np.max(all_scores)),
        }

        print(f"    Mean score: {metrics['mean_score']:.4f} ± {metrics['std_score']:.4f}")

        return metrics


def compare_retrievers(languages: List[str],
                      tasks: List[str],
                      models_dir: str = 'models/indic_clitssa',
                      base_model: str = 'sentence-transformers/distiluse-base-multilingual-cased-v1',
                      n_samples: int = 100,
                      k: int = 3,
                      output_file: str = 'results/indic_clitssa_evaluation.json'):
    """
    Compare different retriever models.

    Args:
        languages: List of languages to test
        tasks: List of tasks to test
        models_dir: Directory containing trained models
        base_model: Base model name (for baseline)
        n_samples: Number of test samples
        k: Number of examples to retrieve
        output_file: Output file for results
    """
    print("="*60)
    print("CLiTSSA Retriever Evaluation - Indian Languages")
    print("="*60)
    print(f"Languages: {languages}")
    print(f"Tasks: {tasks}")
    print(f"Test samples: {n_samples}")
    print(f"Retrieval k: {k}")
    print()

    results = {
        'config': {
            'languages': languages,
            'tasks': tasks,
            'n_samples': n_samples,
            'k': k
        },
        'evaluations': []
    }

    # 1. Evaluate base model
    print("\n" + "="*60)
    print("1. Evaluating Base Model (Baseline)")
    print("="*60)

    base_evaluator = CLiTSSAEvaluator(base_model, "base_model")

    for lang in languages:
        for task in tasks:
            metrics = base_evaluator.evaluate_retrieval_quality(lang, task, n_samples, k)
            results['evaluations'].append(metrics)

    # 2. Evaluate task-specific models
    print("\n" + "="*60)
    print("2. Evaluating Task-Specific CLiTSSA Models")
    print("="*60)

    for lang in languages:
        for task in tasks:
            model_name = f"clitssa_{lang}_{task}"
            model_path = Path(models_dir) / model_name / "best_model"

            if model_path.exists():
                evaluator = CLiTSSAEvaluator(str(model_path), model_name)
                metrics = evaluator.evaluate_retrieval_quality(lang, task, n_samples, k)
                results['evaluations'].append(metrics)
            else:
                print(f"  ⚠ Model not found: {model_path}")

    # 3. Evaluate integrated model
    print("\n" + "="*60)
    print("3. Evaluating Integrated CLiTSSA* Model")
    print("="*60)

    integrated_path = Path(models_dir) / "clitssa_integrated" / "best_model"
    if integrated_path.exists():
        integrated_evaluator = CLiTSSAEvaluator(str(integrated_path), "clitssa_integrated")

        for lang in languages:
            for task in tasks:
                metrics = integrated_evaluator.evaluate_retrieval_quality(lang, task, n_samples, k)
                results['evaluations'].append(metrics)
    else:
        print(f"  ⚠ Integrated model not found: {integrated_path}")

    # Save results
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n\n✓ Results saved to: {output_file}")

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    # Group by model type
    model_types = {}
    for eval_result in results['evaluations']:
        model = eval_result['model']
        lang = eval_result['language']
        task = eval_result['task']
        score = eval_result['mean_score']

        key = f"{lang}/{task}"
        if key not in model_types:
            model_types[key] = {}

        model_types[key][model] = score

    # Print comparison table
    print("\nRetrieval Quality (Mean Similarity Score):")
    print("-" * 80)
    print(f"{'Task':<15} {'Base Model':<12} {'CLiTSSA':<12} {'CLiTSSA*':<12} {'Improvement':<12}")
    print("-" * 80)

    for key in sorted(model_types.keys()):
        scores = model_types[key]
        base_score = scores.get('base_model', 0)

        # Find task-specific CLiTSSA score
        task_specific = [v for k, v in scores.items() if k.startswith('clitssa_') and 'integrated' not in k]
        clitssa_score = task_specific[0] if task_specific else 0

        integrated_score = scores.get('clitssa_integrated', 0)

        # Calculate best improvement
        improvements = []
        if clitssa_score > 0:
            improvements.append((clitssa_score - base_score) / base_score * 100)
        if integrated_score > 0:
            improvements.append((integrated_score - base_score) / base_score * 100)

        best_improvement = max(improvements) if improvements else 0

        print(f"{key:<15} {base_score:.4f}      {clitssa_score:.4f}      "
              f"{integrated_score:.4f}      {best_improvement:+.2f}%")

    print("-" * 80)

    # Overall statistics
    all_base_scores = [e['mean_score'] for e in results['evaluations'] if e['model'] == 'base_model']
    all_clitssa_scores = [e['mean_score'] for e in results['evaluations']
                         if e['model'].startswith('clitssa_') and 'integrated' not in e['model']]

    if all_base_scores and all_clitssa_scores:
        avg_base = np.mean(all_base_scores)
        avg_clitssa = np.mean(all_clitssa_scores)
        overall_improvement = (avg_clitssa - avg_base) / avg_base * 100

        print(f"\nOverall Average:")
        print(f"  Base Model:  {avg_base:.4f}")
        print(f"  CLiTSSA:     {avg_clitssa:.4f}")
        print(f"  Improvement: {overall_improvement:+.2f}%")

    print("\n" + "="*60)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate CLiTSSA retrievers for Indic languages")
    parser.add_argument('--languages', nargs='+', default=['hi', 'bn', 'ta'],
                       help='Languages to evaluate')
    parser.add_argument('--tasks', nargs='+', default=['xnli', 'qa'],
                       help='Tasks to evaluate')
    parser.add_argument('--models_dir', default='models/indic_clitssa',
                       help='Directory containing trained models')
    parser.add_argument('--base_model',
                       default='sentence-transformers/distiluse-base-multilingual-cased-v1',
                       help='Base model for baseline')
    parser.add_argument('--n_samples', type=int, default=100,
                       help='Number of test samples')
    parser.add_argument('--k', type=int, default=3,
                       help='Number of examples to retrieve')
    parser.add_argument('--output', default='results/indic_clitssa_evaluation.json',
                       help='Output file for results')

    args = parser.parse_args()

    results = compare_retrievers(
        languages=args.languages,
        tasks=args.tasks,
        models_dir=args.models_dir,
        base_model=args.base_model,
        n_samples=args.n_samples,
        k=args.k,
        output_file=args.output
    )
