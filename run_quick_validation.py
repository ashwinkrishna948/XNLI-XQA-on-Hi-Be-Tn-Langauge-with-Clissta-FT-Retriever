"""
Quick validation of CLiTSSA training pipeline for Indian languages.

Uses minimal samples to verify the pipeline works before full training.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

print("="*60)
print("CLiTSSA Pipeline Validation - Indian Languages")
print("="*60)
print()
print("This runs a quick validation with minimal samples to verify")
print("the entire pipeline works before running full training.")
print()

# Test 1: Data Loading
print("Test 1: Loading data...")
print("-" * 60)

from indic_data_loader import get_indic_loader

# Test XNLI for Hindi (native dataset available)
print("  Loading Hindi XNLI (10 samples)...")
xnli_loader = get_indic_loader('xnli')
queries_hi, examples_hi = xnli_loader.load_data('hi', split='validation', n_samples=10)
print(f"  ✓ Loaded {len(queries_hi)} Hindi queries")
print(f"    Sample: {queries_hi[0][:80]}...")

# Test QA for Hindi
print("\n  Loading Hindi QA (10 samples)...")
qa_loader = get_indic_loader('qa')
queries_qa, examples_qa = qa_loader.load_data('hi', split='validation', n_samples=10)
print(f"  ✓ Loaded {len(queries_qa)} Hindi QA queries")
print(f"    Sample: {queries_qa[0][:80]}...")

print("\n✓ Test 1 PASSED: Data loading works")

# Test 2: CLiTSSA Training Dataset Creation
print("\n\nTest 2: Creating CLiTSSA training dataset...")
print("-" * 60)

from clitssa_trainer import CLiTSSATrainer

trainer = CLiTSSATrainer(
    base_model_name='sentence-transformers/distiluse-base-multilingual-cased-v1',
    h=5,   # Reduced for quick test
    w=2    # Reduced for quick test
)

print("  Creating training dataset (10 Hindi queries + 10 English examples)...")
# Load English examples
_, examples_en = xnli_loader.load_data('en', split='validation', n_samples=10)

# Create training dataset
train_dataset = trainer.create_training_dataset(
    queries_low=queries_hi,
    examples_high=examples_en,
    low_lang='hi',
    high_lang='en'
)

print(f"  ✓ Created training dataset with {len(train_dataset)} pairs")
print(f"    (Expected: 10 queries × (5 top + 2 random) = ~70 pairs)")

print("\n✓ Test 2 PASSED: Training dataset creation works")

# Test 3: Model Training (1 epoch, small batch)
print("\n\nTest 3: Training CLiTSSA model (1 epoch, minimal samples)...")
print("-" * 60)
print("  This will take 2-3 minutes...")

output_dir = 'models/indic_clitssa/validation_test'

try:
    model = trainer.train(
        train_dataset=train_dataset,
        epochs=1,  # Just 1 epoch for validation
        batch_size=4,  # Small batch
        learning_rate=2e-5,
        output_dir=output_dir,
        warmup_steps=10
    )

    print(f"\n  ✓ Model trained successfully")
    print(f"    Saved to: {output_dir}/best_model")

    print("\n✓ Test 3 PASSED: Model training works")

except Exception as e:
    print(f"\n  ✗ Test 3 FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Model Loading and Inference
print("\n\nTest 4: Loading trained model and testing inference...")
print("-" * 60)

try:
    from sentence_transformers import SentenceTransformer

    model_path = f"{output_dir}/best_model"
    loaded_model = SentenceTransformer(model_path)

    # Test encoding
    test_texts = [
        "यह एक परीक्षण वाक्य है।",  # Hindi: This is a test sentence
        "This is a test sentence."    # English
    ]

    embeddings = loaded_model.encode(test_texts)
    print(f"  ✓ Model loaded and encoded {len(test_texts)} texts")
    print(f"    Embedding shape: {embeddings.shape}")

    # Compute similarity
    from sentence_transformers import util
    similarity = util.cos_sim(embeddings[0], embeddings[1])[0][0]
    print(f"    Hindi-English similarity: {similarity:.4f}")

    print("\n✓ Test 4 PASSED: Model loading and inference works")

except Exception as e:
    print(f"\n  ✗ Test 4 FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Summary
print("\n\n" + "="*60)
print("✓ ALL VALIDATION TESTS PASSED!")
print("="*60)
print()
print("The CLiTSSA training pipeline is working correctly.")
print("You can now proceed with full training using:")
print()
print("  python train_indic_clitssa.py --n_train 500 --epochs 3")
print()
print("Recommended configurations:")
print("  - Quick test:  --n_train 100 --epochs 2  (~2-3 hours)")
print("  - Full run:    --n_train 500 --epochs 3  (~8-10 hours)")
print("  - Paper scale: --n_train 1000 --epochs 3 (~15-18 hours)")
print()
print("="*60)
