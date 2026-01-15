"""
Main experiment runner for Domain Generalization Test.

Tests whether in-language similarity superiority generalizes
beyond temporal reasoning to other NLP tasks.
"""

import json
import os
from pathlib import Path
from typing import Dict, List
import numpy as np
from tqdm import tqdm

from retrieval import CrossLingualSimilarityRetriever, InLanguageSimilarityRetriever
from data_loader import XNLILoader, XQuADLoader, PAWSXLoader, format_for_retrieval


class DomainGeneralizationExperiment:
    """Run domain generalization experiment across multiple tasks."""

    def __init__(
        self,
        languages: List[str] = ["ro", "de", "fr"],
        tasks: List[str] = ["xnli", "xquad", "pawsx"],
        k: int = 3,
        n_samples: int = 100,
        output_dir: str = "results"
    ):
        """
        Initialize experiment.

        Args:
            languages: List of low-resource language codes
            tasks: List of task names
            k: Number of examples to retrieve
            n_samples: Number of test samples per task/language
            output_dir: Directory to save results
        """
        self.languages = languages
        self.tasks = tasks
        self.k = k
        self.n_samples = n_samples
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize retrievers
        print("Initializing retrievers...")
        self.cross_lingual_retriever = CrossLingualSimilarityRetriever()
        self.in_language_retriever = InLanguageSimilarityRetriever()

        # Initialize data loaders
        self.loaders = {
            "xnli": XNLILoader(),
            "xquad": XQuADLoader(),
            "pawsx": PAWSXLoader()
        }

    def load_task_data(self, task: str, language: str) -> tuple:
        """
        Load data for a specific task and language.

        Args:
            task: Task name
            language: Language code

        Returns:
            Tuple of (queries, examples, query_texts, example_texts)
        """
        loader = self.loaders[task]

        if task == "xnli":
            queries, examples = loader.load(language, split="test", n_samples=self.n_samples)
        elif task == "xquad":
            queries, examples = loader.load(language, n_samples=self.n_samples)
        elif task == "pawsx":
            queries, examples = loader.load(language, split="test", n_samples=self.n_samples)

        # Format for retrieval
        query_texts = format_for_retrieval(queries, task)
        example_texts = format_for_retrieval(examples, task)

        return queries, examples, query_texts, example_texts

    def compute_retrieval_overlap(
        self,
        cross_indices: List[int],
        in_lang_indices: List[int]
    ) -> float:
        """
        Compute overlap between retrievals.

        Args:
            cross_indices: Indices from cross-lingual retrieval
            in_lang_indices: Indices from in-language retrieval

        Returns:
            Overlap ratio (0-1)
        """
        overlap = len(set(cross_indices) & set(in_lang_indices))
        return overlap / len(cross_indices)

    def run_experiment(self, task: str, language: str) -> Dict:
        """
        Run experiment for a single task-language pair.

        Args:
            task: Task name
            language: Language code

        Returns:
            Dictionary with results
        """
        print(f"\n{'='*60}")
        print(f"Task: {task.upper()} | Language: {language}")
        print(f"{'='*60}")

        # Load data
        queries, examples, query_texts, example_texts = self.load_task_data(task, language)

        print(f"Loaded {len(queries)} queries and {len(examples)} examples")

        # Run retrievals
        results = {
            "task": task,
            "language": language,
            "n_queries": len(queries),
            "n_examples": len(examples),
            "k": self.k,
            "cross_lingual_scores": [],
            "in_language_scores": [],
            "overlaps": [],
            "score_differences": []
        }

        # Process each query
        for i, query_text in enumerate(tqdm(query_texts, desc=f"Processing {task}/{language}")):
            # Cross-lingual retrieval
            cross_indices, cross_scores = self.cross_lingual_retriever.retrieve(
                query_text,
                example_texts,
                k=self.k
            )

            # In-language retrieval
            in_lang_indices, in_lang_scores = self.in_language_retriever.retrieve(
                query_text,
                example_texts,
                query_lang=language,
                k=self.k
            )

            # Compute metrics
            overlap = self.compute_retrieval_overlap(cross_indices, in_lang_indices)
            avg_cross_score = np.mean(cross_scores)
            avg_in_lang_score = np.mean(in_lang_scores)
            score_diff = avg_in_lang_score - avg_cross_score

            results["cross_lingual_scores"].append(float(avg_cross_score))
            results["in_language_scores"].append(float(avg_in_lang_score))
            results["overlaps"].append(float(overlap))
            results["score_differences"].append(float(score_diff))

        # Compute aggregated statistics
        results["mean_cross_lingual_score"] = float(np.mean(results["cross_lingual_scores"]))
        results["mean_in_language_score"] = float(np.mean(results["in_language_scores"]))
        results["mean_overlap"] = float(np.mean(results["overlaps"]))
        results["mean_score_difference"] = float(np.mean(results["score_differences"]))
        results["std_score_difference"] = float(np.std(results["score_differences"]))

        # Determine advantage
        results["in_language_advantage"] = results["mean_score_difference"] > 0

        print(f"\nResults:")
        print(f"  Mean Cross-lingual Score: {results['mean_cross_lingual_score']:.4f}")
        print(f"  Mean In-language Score: {results['mean_in_language_score']:.4f}")
        print(f"  Mean Score Difference: {results['mean_score_difference']:.4f}")
        print(f"  In-language Advantage: {results['in_language_advantage']}")

        return results

    def run_all_experiments(self) -> Dict:
        """
        Run experiments across all tasks and languages.

        Returns:
            Dictionary with all results
        """
        all_results = {
            "config": {
                "languages": self.languages,
                "tasks": self.tasks,
                "k": self.k,
                "n_samples": self.n_samples
            },
            "results": []
        }

        # Run each task-language combination
        for task in self.tasks:
            for language in self.languages:
                try:
                    result = self.run_experiment(task, language)
                    all_results["results"].append(result)
                except Exception as e:
                    print(f"Error processing {task}/{language}: {e}")
                    continue

        # Save results
        output_file = self.output_dir / "experiment_results.json"
        with open(output_file, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to: {output_file}")

        # Print summary
        self.print_summary(all_results)

        return all_results

    def print_summary(self, results: Dict):
        """Print experiment summary."""
        print(f"\n{'='*60}")
        print("EXPERIMENT SUMMARY")
        print(f"{'='*60}")

        for result in results["results"]:
            task = result["task"]
            lang = result["language"]
            diff = result["mean_score_difference"]
            advantage = "✓" if result["in_language_advantage"] else "✗"

            print(f"{task.upper():8} | {lang.upper():2} | Score Diff: {diff:+.4f} | In-lang Advantage: {advantage}")

        # Overall statistics
        all_diffs = [r["mean_score_difference"] for r in results["results"]]
        n_advantages = sum(1 for r in results["results"] if r["in_language_advantage"])

        print(f"\n{'='*60}")
        print(f"Overall Mean Difference: {np.mean(all_diffs):+.4f}")
        print(f"In-language Advantage: {n_advantages}/{len(results['results'])} cases")
        print(f"{'='*60}")


if __name__ == "__main__":
    # Run experiment
    exp = DomainGeneralizationExperiment(
        languages=["ro", "de", "fr"],
        tasks=["xnli", "xquad", "pawsx"],
        k=3,
        n_samples=100  # Start with 100 samples for quick testing
    )

    results = exp.run_all_experiments()
