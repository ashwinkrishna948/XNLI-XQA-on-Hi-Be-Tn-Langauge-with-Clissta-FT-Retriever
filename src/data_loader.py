"""
Data loading utilities for multilingual NLP datasets.

Supports:
1. XNLI - Natural Language Inference
2. XQuAD - Extractive Question Answering
3. PAWS-X - Paraphrase Identification
"""

from datasets import load_dataset
from typing import List, Dict, Tuple
import random


class MultilingualDataLoader:
    """Base class for loading multilingual datasets."""

    def __init__(self, seed: int = 42):
        """
        Initialize data loader.

        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        random.seed(seed)

    def sample_examples(
        self,
        data: List[Dict],
        n_examples: int,
        stratify_key: str = None
    ) -> List[Dict]:
        """
        Sample examples from dataset.

        Args:
            data: List of data examples
            n_examples: Number of examples to sample
            stratify_key: Optional key for stratified sampling

        Returns:
            List of sampled examples
        """
        if stratify_key:
            # Stratified sampling
            categories = {}
            for example in data:
                cat = example[stratify_key]
                if cat not in categories:
                    categories[cat] = []
                categories[cat].append(example)

            # Sample equally from each category
            samples_per_cat = n_examples // len(categories)
            sampled = []
            for cat_examples in categories.values():
                sampled.extend(random.sample(
                    cat_examples,
                    min(samples_per_cat, len(cat_examples))
                ))
            return sampled[:n_examples]
        else:
            # Random sampling
            return random.sample(data, min(n_examples, len(data)))


class XNLILoader(MultilingualDataLoader):
    """
    Load XNLI dataset for Natural Language Inference.

    Task: Given premise and hypothesis, predict entailment/contradiction/neutral
    """

    LABEL_MAP = {
        0: "entailment",
        1: "neutral",
        2: "contradiction"
    }

    def load(
        self,
        language: str,
        split: str = "test",
        n_samples: int = None
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Load XNLI dataset.

        Args:
            language: Language code ('en', 'fr', 'de', 'ro', etc.)
            split: Dataset split ('train', 'validation', 'test')
            n_samples: Number of samples to load (None = all)

        Returns:
            Tuple of (queries, examples) where:
            - queries: Low-resource language queries
            - examples: English examples (for retrieval)
        """
        print(f"Loading XNLI for {language} ({split})")

        # Load target language data
        dataset = load_dataset("xnli", language, split=split)

        # Convert to list of dicts
        data = []
        for item in dataset:
            data.append({
                "premise": item["premise"],
                "hypothesis": item["hypothesis"],
                "label": self.LABEL_MAP[item["label"]],
                "language": language
            })

        # Sample if requested
        if n_samples:
            data = self.sample_examples(data, n_samples, stratify_key="label")

        # If loading low-resource language, also load English examples
        if language != "en":
            en_dataset = load_dataset("xnli", "en", split=split)
            en_data = []
            for item in en_dataset:
                en_data.append({
                    "premise": item["premise"],
                    "hypothesis": item["hypothesis"],
                    "label": self.LABEL_MAP[item["label"]],
                    "language": "en"
                })

            # Sample same number of English examples
            if n_samples:
                en_data = self.sample_examples(en_data, n_samples, stratify_key="label")

            return data, en_data

        return data, data


class XQuADLoader(MultilingualDataLoader):
    """
    Load XQuAD dataset for Extractive Question Answering.

    Task: Given context and question, extract answer span
    """

    def load(
        self,
        language: str,
        n_samples: int = None
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Load XQuAD dataset.

        Args:
            language: Language code ('xquad.{lang}' format)
            n_samples: Number of samples to load (None = all)

        Returns:
            Tuple of (queries, examples)
        """
        print(f"Loading XQuAD for {language}")

        # Load target language data
        dataset = load_dataset("xquad", f"xquad.{language}", split="validation")

        # Convert to list of dicts
        data = []
        for item in dataset:
            data.append({
                "context": item["context"],
                "question": item["question"],
                "answers": item["answers"]["text"],
                "language": language
            })

        # Sample if requested
        if n_samples:
            data = random.sample(data, min(n_samples, len(data)))

        # If loading low-resource language, also load English examples
        if language != "en":
            en_dataset = load_dataset("xquad", "xquad.en", split="validation")
            en_data = []
            for item in en_dataset:
                en_data.append({
                    "context": item["context"],
                    "question": item["question"],
                    "answers": item["answers"]["text"],
                    "language": "en"
                })

            if n_samples:
                en_data = random.sample(en_data, min(n_samples, len(en_data)))

            return data, en_data

        return data, data


class PAWSXLoader(MultilingualDataLoader):
    """
    Load PAWS-X dataset for Paraphrase Identification.

    Task: Given two sentences, determine if they are paraphrases
    """

    LABEL_MAP = {
        0: "different",
        1: "paraphrase"
    }

    def load(
        self,
        language: str,
        split: str = "test",
        n_samples: int = None
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Load PAWS-X dataset.

        Args:
            language: Language code ('en', 'fr', 'de', etc.)
            split: Dataset split ('train', 'validation', 'test')
            n_samples: Number of samples to load (None = all)

        Returns:
            Tuple of (queries, examples)
        """
        print(f"Loading PAWS-X for {language} ({split})")

        # Load target language data
        dataset = load_dataset("paws-x", language, split=split)

        # Convert to list of dicts
        data = []
        for item in dataset:
            data.append({
                "sentence1": item["sentence1"],
                "sentence2": item["sentence2"],
                "label": self.LABEL_MAP[item["label"]],
                "language": language
            })

        # Sample if requested
        if n_samples:
            data = self.sample_examples(data, n_samples, stratify_key="label")

        # If loading low-resource language, also load English examples
        if language != "en":
            en_dataset = load_dataset("paws-x", "en", split=split)
            en_data = []
            for item in en_dataset:
                en_data.append({
                    "sentence1": item["sentence1"],
                    "sentence2": item["sentence2"],
                    "label": self.LABEL_MAP[item["label"]],
                    "language": "en"
                })

            if n_samples:
                en_data = self.sample_examples(en_data, n_samples, stratify_key="label")

            return data, en_data

        return data, data


def format_for_retrieval(data: List[Dict], task: str) -> List[str]:
    """
    Format dataset examples as text strings for retrieval.

    Args:
        data: List of data examples
        task: Task name ('xnli', 'xquad', 'pawsx')

    Returns:
        List of formatted text strings
    """
    formatted = []

    if task == "xnli":
        for item in data:
            text = f"Premise: {item['premise']}\nHypothesis: {item['hypothesis']}\nRelation: {item['label']}"
            formatted.append(text)

    elif task == "xquad":
        for item in data:
            text = f"Context: {item['context'][:200]}...\nQuestion: {item['question']}\nAnswer: {item['answers'][0]}"
            formatted.append(text)

    elif task == "pawsx":
        for item in data:
            text = f"Sentence 1: {item['sentence1']}\nSentence 2: {item['sentence2']}\nParaphrase: {item['label']}"
            formatted.append(text)

    return formatted
