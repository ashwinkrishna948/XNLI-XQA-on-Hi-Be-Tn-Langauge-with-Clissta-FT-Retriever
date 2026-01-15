"""
Retrieval strategies for cross-lingual in-context learning.

Implements:
1. Cross-lingual similarity (baseline from paper)
2. In-language similarity (paper's observation)
"""

import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Tuple, Dict
from deep_translator import GoogleTranslator
from tqdm import tqdm


class CrossLingualRetriever:
    """Base retriever using multilingual sentence embeddings."""

    def __init__(self, model_name: str = "distiluse-base-multilingual-cased-v1"):
        """
        Initialize retriever with multilingual sentence transformer.

        Args:
            model_name: HuggingFace model name for sentence embeddings
        """
        print(f"Loading model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        print(f"Model loaded on {self.device}")

    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Encode texts to dense embeddings.

        Args:
            texts: List of text strings
            batch_size: Batch size for encoding

        Returns:
            numpy array of embeddings (n_texts, embedding_dim)
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            device=self.device
        )
        return embeddings

    def compute_similarity(self, query_emb: np.ndarray, example_embs: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity between query and examples.

        Args:
            query_emb: Query embedding (embedding_dim,)
            example_embs: Example embeddings (n_examples, embedding_dim)

        Returns:
            Similarity scores (n_examples,)
        """
        # Normalize embeddings
        query_norm = query_emb / np.linalg.norm(query_emb)
        example_norms = example_embs / np.linalg.norm(example_embs, axis=1, keepdims=True)

        # Cosine similarity
        similarities = np.dot(example_norms, query_norm)
        return similarities

    def retrieve_top_k(
        self,
        query: str,
        examples: List[str],
        k: int = 3
    ) -> Tuple[List[int], List[float]]:
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
        query_emb = self.encode([query])[0]
        example_embs = self.encode(examples)

        # Compute similarities
        similarities = self.compute_similarity(query_emb, example_embs)

        # Get top-k indices
        top_k_indices = np.argsort(similarities)[::-1][:k]
        top_k_scores = similarities[top_k_indices]

        return top_k_indices.tolist(), top_k_scores.tolist()


class CrossLingualSimilarityRetriever(CrossLingualRetriever):
    """
    Direct cross-lingual similarity (Baseline).

    Computes similarity directly between:
    - Low-resource query (e.g., Romanian)
    - High-resource examples (e.g., English)
    """

    def retrieve(
        self,
        query: str,
        examples: List[str],
        k: int = 3
    ) -> Tuple[List[int], List[float]]:
        """
        Direct cross-lingual retrieval.

        Args:
            query: Query in low-resource language
            examples: Examples in high-resource language (English)
            k: Number of examples to retrieve

        Returns:
            Tuple of (indices, similarity_scores)
        """
        return self.retrieve_top_k(query, examples, k)


class InLanguageSimilarityRetriever(CrossLingualRetriever):
    """
    In-language similarity (Paper's approach).

    Process:
    1. Translate English examples → Target language
    2. Compute similarity in target language space
    3. Return original English examples
    """

    def __init__(
        self,
        model_name: str = "distiluse-base-multilingual-cased-v1",
        cache_translations: bool = True
    ):
        super().__init__(model_name)
        self.cache_translations = cache_translations
        self.translation_cache: Dict[Tuple[str, str, str], str] = {}

    def translate_text(self, text: str, source_lang: str, target_lang: str) -> str:
        """
        Translate text using Google Translate.

        Args:
            text: Text to translate
            source_lang: Source language code
            target_lang: Target language code

        Returns:
            Translated text
        """
        cache_key = (text, source_lang, target_lang)

        # Check cache
        if self.cache_translations and cache_key in self.translation_cache:
            return self.translation_cache[cache_key]

        # Translate
        try:
            translator = GoogleTranslator(source=source_lang, target=target_lang)
            translated = translator.translate(text)
        except Exception as e:
            print(f"Translation error: {e}")
            translated = text  # Fallback to original

        # Cache result
        if self.cache_translations:
            self.translation_cache[cache_key] = translated

        return translated

    def translate_batch(
        self,
        texts: List[str],
        source_lang: str,
        target_lang: str,
        batch_size: int = 10
    ) -> List[str]:
        """
        Translate a batch of texts with progress bar.

        Args:
            texts: List of texts to translate
            source_lang: Source language code
            target_lang: Target language code
            batch_size: Number of texts to translate at once

        Returns:
            List of translated texts
        """
        translated = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Translating"):
            batch = texts[i:i+batch_size]
            batch_translated = [
                self.translate_text(text, source_lang, target_lang)
                for text in batch
            ]
            translated.extend(batch_translated)
        return translated

    def retrieve(
        self,
        query: str,
        examples: List[str],
        query_lang: str,
        example_lang: str = "en",
        k: int = 3
    ) -> Tuple[List[int], List[float]]:
        """
        In-language similarity retrieval.

        Args:
            query: Query in low-resource language
            examples: Examples in high-resource language (English)
            query_lang: Language code of query (e.g., 'ro', 'de', 'fr')
            example_lang: Language code of examples (default: 'en')
            k: Number of examples to retrieve

        Returns:
            Tuple of (indices, similarity_scores)
        """
        # Step 1: Translate examples to query language
        print(f"Translating {len(examples)} examples from {example_lang} to {query_lang}")
        translated_examples = self.translate_batch(examples, example_lang, query_lang)

        # Step 2: Compute similarity in query language space
        indices, scores = self.retrieve_top_k(query, translated_examples, k)

        # Step 3: Return indices (which map to original English examples)
        return indices, scores


def compare_retrievers(
    query: str,
    examples: List[str],
    query_lang: str,
    k: int = 3
) -> Dict[str, Tuple[List[int], List[float]]]:
    """
    Compare both retrieval strategies side-by-side.

    Args:
        query: Query in low-resource language
        examples: Examples in English
        query_lang: Language code of query
        k: Number of examples to retrieve

    Returns:
        Dictionary with results from both retrievers
    """
    # Cross-lingual similarity
    cross_lingual = CrossLingualSimilarityRetriever()
    cross_results = cross_lingual.retrieve(query, examples, k)

    # In-language similarity
    in_language = InLanguageSimilarityRetriever()
    in_lang_results = in_language.retrieve(query, examples, query_lang, k=k)

    return {
        "cross_lingual": cross_results,
        "in_language": in_lang_results
    }
