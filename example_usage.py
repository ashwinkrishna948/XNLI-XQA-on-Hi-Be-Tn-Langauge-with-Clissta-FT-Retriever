#!/usr/bin/env python3
"""
Example usage of CLiTSSA models for Indian languages.

This script demonstrates various use cases:
1. Basic sentence encoding
2. Cross-lingual retrieval
3. In-context learning example selection
4. Semantic similarity search
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Tuple


def example_1_basic_encoding():
    """Example 1: Basic sentence encoding."""
    print("=" * 70)
    print("Example 1: Basic Sentence Encoding")
    print("=" * 70)

    # Load Hindi XNLI model
    model = SentenceTransformer('./models/clitssa_hi_xnli/best_model')

    # Hindi sentences
    sentences = [
        'भारत एक विशाल देश है',
        'दिल्ली भारत की राजधानी है',
        'मुझे फिल्में देखना पसंद है'
    ]

    # Encode
    embeddings = model.encode(sentences)

    print(f"\nEncoded {len(sentences)} sentences")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Embedding dimension: {embeddings.shape[1]}")

    # Show first few values
    print(f"\nFirst 10 values of first embedding:")
    print(embeddings[0][:10])
    print()


def example_2_cross_lingual_retrieval():
    """Example 2: Cross-lingual retrieval (Hindi query, English examples)."""
    print("=" * 70)
    print("Example 2: Cross-Lingual Retrieval")
    print("=" * 70)

    # Load model
    model = SentenceTransformer('./models/clitssa_hi_xnli/best_model')

    # Hindi query
    query = "भारत की राजधानी क्या है?"

    # English knowledge base
    examples = [
        "The capital of India is New Delhi, located in northern India.",
        "Mumbai is the financial capital and largest city in India.",
        "The Taj Mahal is located in Agra, Uttar Pradesh.",
        "India gained independence from British rule on August 15, 1947.",
        "New Delhi serves as the seat of the Government of India."
    ]

    print(f"\nQuery (Hindi): {query}")
    print(f"\nSearching through {len(examples)} English examples...")

    # Encode
    query_emb = model.encode(query)
    example_embs = model.encode(examples)

    # Calculate similarities
    similarities = np.dot(example_embs, query_emb)

    # Get top-3
    k = 3
    top_k_indices = np.argsort(similarities)[-k:][::-1]

    print(f"\nTop-{k} most relevant examples:")
    for rank, idx in enumerate(top_k_indices, 1):
        print(f"\n{rank}. Similarity: {similarities[idx]:.4f}")
        print(f"   Text: {examples[idx]}")
    print()


def example_3_bengali_qa():
    """Example 3: Bengali question answering retrieval."""
    print("=" * 70)
    print("Example 3: Bengali Question Answering")
    print("=" * 70)

    # Load Bengali QA model
    model = SentenceTransformer('./models/clitssa_bn_qa/best_model')

    # Bengali query
    query = "পৃথিবীর সবচেয়ে বড় মহাসাগর কোনটি?"

    # English QA examples (few-shot examples)
    qa_examples = [
        "Q: What is the largest ocean on Earth? A: The Pacific Ocean",
        "Q: What is the capital of France? A: Paris",
        "Q: How many continents are there? A: Seven continents",
        "Q: What is the deepest ocean? A: The Pacific Ocean",
        "Q: Where is Mount Everest located? A: Nepal and Tibet"
    ]

    print(f"\nQuery (Bengali): {query}")
    print("Translation: Which is the largest ocean in the world?")
    print(f"\nRetrieving from {len(qa_examples)} QA examples...")

    # Encode and retrieve
    query_emb = model.encode(query)
    example_embs = model.encode(qa_examples)
    similarities = np.dot(example_embs, query_emb)

    # Top-3 for in-context learning
    k = 3
    top_k_indices = np.argsort(similarities)[-k:][::-1]

    print(f"\nTop-{k} examples for in-context prompting:")
    for rank, idx in enumerate(top_k_indices, 1):
        print(f"\n{rank}. Similarity: {similarities[idx]:.4f}")
        print(f"   {qa_examples[idx]}")

    print("\n📝 Use these examples in your prompt for better LLM performance!")
    print()


def example_4_tamil_similarity():
    """Example 4: Tamil sentence similarity."""
    print("=" * 70)
    print("Example 4: Tamil Sentence Similarity")
    print("=" * 70)

    # Load Tamil XNLI model
    model = SentenceTransformer('./models/clitssa_ta_xnli/best_model')

    # Tamil sentence pairs
    pairs = [
        ("இந்தியா ஒரு பெரிய நாடு", "இந்தியா மிகப்பெரிய நாடு"),
        ("நான் சினிமா பார்க்க விரும்புகிறேன்", "எனக்கு திரைப்படங்கள் பிடிக்கும்"),
        ("சென்னை தமிழ்நாட்டின் தலைநகரம்", "மும்பை மகாராஷ்டிராவின் தலைநகரம்"),
    ]

    print("\nCalculating similarity for Tamil sentence pairs:")

    for sent1, sent2 in pairs:
        emb1 = model.encode(sent1)
        emb2 = model.encode(sent2)

        # Cosine similarity
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

        print(f"\n📝 Sentence 1: {sent1}")
        print(f"   Sentence 2: {sent2}")
        print(f"   Similarity: {similarity:.4f}")
    print()


def example_5_multilingual_comparison():
    """Example 5: Compare all three languages."""
    print("=" * 70)
    print("Example 5: Multilingual Comparison")
    print("=" * 70)

    print("\nLoading all three language models...")

    models = {
        'Hindi': SentenceTransformer('./models/clitssa_hi_xnli/best_model'),
        'Bengali': SentenceTransformer('./models/clitssa_bn_xnli/best_model'),
        'Tamil': SentenceTransformer('./models/clitssa_ta_xnli/best_model')
    }

    # Same semantic content in three languages
    sentences = {
        'Hindi': 'भारत एक लोकतांत्रिक देश है',
        'Bengali': 'ভারত একটি গণতান্ত্রিক দেশ',
        'Tamil': 'இந்தியா ஒரு ஜனநாயக நாடு'
    }

    # English reference
    english_ref = "India is a democratic country"

    print(f"\nEnglish reference: {english_ref}")
    print("\nSemantic similarity with English for each language:")

    # Get embeddings
    results = {}
    for lang, model in models.items():
        # Encode both sentences with the same model
        sent_emb = model.encode(sentences[lang])
        eng_emb = model.encode(english_ref)

        # Calculate similarity
        similarity = np.dot(sent_emb, eng_emb) / (
            np.linalg.norm(sent_emb) * np.linalg.norm(eng_emb)
        )

        results[lang] = similarity
        print(f"\n  {lang:10s}: {sentences[lang]}")
        print(f"  {'':10s}  Similarity: {similarity:.4f}")

    print("\n✓ All three models successfully encode their respective languages!")
    print()


def example_6_batch_processing():
    """Example 6: Efficient batch processing."""
    print("=" * 70)
    print("Example 6: Batch Processing for Scale")
    print("=" * 70)

    model = SentenceTransformer('./models/clitssa_hi_xnli/best_model')

    # Large batch of Hindi sentences
    sentences = [
        f'यह वाक्य संख्या {i} है' for i in range(100)
    ]

    print(f"\nEncoding {len(sentences)} sentences in batch...")

    # Batch encoding (much faster than one-by-one)
    import time
    start = time.time()
    embeddings = model.encode(sentences, batch_size=32, show_progress_bar=True)
    elapsed = time.time() - start

    print(f"\n✓ Encoded {len(sentences)} sentences in {elapsed:.2f} seconds")
    print(f"  Average: {elapsed/len(sentences)*1000:.2f} ms per sentence")
    print(f"  Throughput: {len(sentences)/elapsed:.1f} sentences/second")
    print()


def main():
    """Run all examples."""
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "CLiTSSA Models - Usage Examples" + " " * 21 + "║")
    print("╚" + "═" * 68 + "╝")
    print()

    examples = [
        ("Basic Encoding", example_1_basic_encoding),
        ("Cross-Lingual Retrieval", example_2_cross_lingual_retrieval),
        ("Bengali QA", example_3_bengali_qa),
        ("Tamil Similarity", example_4_tamil_similarity),
        ("Multilingual Comparison", example_5_multilingual_comparison),
        ("Batch Processing", example_6_batch_processing),
    ]

    print("Available examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")

    print("\nSelect example (1-6, or 'all' for all examples): ", end="")
    try:
        choice = input().strip().lower()

        if choice == 'all':
            for _, func in examples:
                func()
                input("\nPress Enter to continue to next example...")
        else:
            idx = int(choice) - 1
            if 0 <= idx < len(examples):
                examples[idx][1]()
            else:
                print("Invalid choice!")
    except KeyboardInterrupt:
        print("\n\nExited.")
    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure you have downloaded the models first:")
        print("  python download_model.py --all")


if __name__ == "__main__":
    main()
