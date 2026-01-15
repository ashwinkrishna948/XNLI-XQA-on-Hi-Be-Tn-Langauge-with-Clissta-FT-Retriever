"""
Train CLiTSSA retriever for Hindi, Bengali, and Tamil.

Following the paper's methodology:
1. Create parallel training dataset with in-language similarity scores
2. Fine-tune multilingual Sentence-BERT with CoSENT loss
3. Save task-specific and integrated models
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from clitssa_trainer import CLiTSSATrainer, TemporalParallelDataset
from indic_data_loader import get_indic_loader
import argparse
import json


def create_training_data_for_task(task: str, language: str, n_train_samples: int = 1000):
    """
    Create CLiTSSA training data for a specific task and language.

    Args:
        task: Task name ('xnli' or 'qa')
        language: Language code ('hi', 'bn', 'ta')
        n_train_samples: Number of training samples

    Returns:
        Tuple of (queries_low_resource, examples_english)
    """
    print(f"\n{'='*60}")
    print(f"Creating training data: {task} / {language}")
    print(f"{'='*60}\n")

    loader = get_indic_loader(task)

    # Load low-resource language data
    queries_low, _ = loader.load_data(language, split='train', n_samples=n_train_samples)

    # Load English examples (resource-rich)
    _, examples_en = loader.load_data('en', split='train', n_samples=n_train_samples)

    print(f"✓ Created training data:")
    print(f"  Queries ({language}): {len(queries_low)}")
    print(f"  Examples (en): {len(examples_en)}")

    return queries_low, examples_en


def train_task_specific_models(languages=['hi', 'bn', 'ta'],
                               tasks=['xnli', 'qa'],
                               n_train_samples=1000,
                               n_epochs=3,
                               batch_size=16,
                               output_dir='models/indic_clitssa'):
    """
    Train task-specific CLiTSSA models for each language-task combination.

    Following paper's approach: separate fine-tuned model for each task/language.
    """
    print(f"\n{'='*60}")
    print(f"Training Task-Specific CLiTSSA Models")
    print(f"{'='*60}")
    print(f"Languages: {languages}")
    print(f"Tasks: {tasks}")
    print(f"Training samples per task: {n_train_samples}")
    print(f"Epochs: {n_epochs}")
    print()

    results = {}

    for language in languages:
        for task in tasks:
            model_name = f"clitssa_{language}_{task}"
            print(f"\n{'#'*60}")
            print(f"Training: {model_name}")
            print(f"{'#'*60}\n")

            try:
                # Create training data
                queries_low, examples_en = create_training_data_for_task(
                    task, language, n_train_samples
                )

                # Initialize trainer
                trainer = CLiTSSATrainer(
                    base_model_name='sentence-transformers/distiluse-base-multilingual-cased-v1',
                    h=30,  # Top-30 similar examples (paper's setting)
                    w=10   # Random-10 examples (paper's setting)
                )

                # Create training dataset following CLiTSSA methodology
                print("\nCreating CLiTSSA training dataset...")
                train_dataset = trainer.create_training_dataset(
                    queries_low=queries_low,
                    examples_high=examples_en,
                    low_lang=language,
                    high_lang='en'
                )

                # Fine-tune
                print(f"\nFine-tuning {model_name}...")
                model_output_dir = f"{output_dir}/{model_name}"
                trainer.train(
                    train_dataset=train_dataset,
                    epochs=n_epochs,
                    batch_size=batch_size,
                    learning_rate=2e-5,
                    output_dir=model_output_dir,
                    warmup_steps=100
                )

                results[model_name] = {
                    'status': 'success',
                    'language': language,
                    'task': task,
                    'n_train_samples': len(train_dataset),
                    'model_path': f"{model_output_dir}/best_model"
                }

                print(f"\n✓ {model_name} training complete!")

            except Exception as e:
                print(f"\n✗ Error training {model_name}: {e}")
                import traceback
                traceback.print_exc()

                results[model_name] = {
                    'status': 'failed',
                    'error': str(e)
                }

    # Save training results
    results_file = f"{output_dir}/training_results.json"
    Path(results_file).parent.mkdir(parents=True, exist_ok=True)
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"{'='*60}")
    print(f"Results saved to: {results_file}")
    print(f"\nSummary:")
    for model_name, result in results.items():
        status = result['status']
        symbol = '✓' if status == 'success' else '✗'
        print(f"  {symbol} {model_name}: {status}")

    return results


def train_integrated_model(languages=['hi', 'bn', 'ta'],
                           tasks=['xnli', 'qa'],
                           n_train_samples=1000,
                           n_epochs=3,
                           batch_size=16,
                           output_dir='models/indic_clitssa'):
    """
    Train integrated CLiTSSA* model across all languages and tasks.

    Following paper: "Additionally, an integrated CLiTSSA retriever is fine-tuned
    across languages and temporal tasks."
    """
    print(f"\n{'='*60}")
    print(f"Training Integrated CLiTSSA* Model")
    print(f"{'='*60}")
    print(f"Languages: {languages}")
    print(f"Tasks: {tasks}")
    print(f"Training samples per task: {n_train_samples}")
    print()

    # Collect all training data
    all_queries_low = []
    all_examples_en = []

    for language in languages:
        for task in tasks:
            print(f"\nCollecting data: {language} / {task}...")

            queries_low, examples_en = create_training_data_for_task(
                task, language, n_train_samples
            )

            all_queries_low.extend(queries_low)
            all_examples_en.extend(examples_en)

    print(f"\n✓ Collected total training data:")
    print(f"  Total queries: {len(all_queries_low)}")
    print(f"  Total examples: {len(all_examples_en)}")

    # Note: For integrated model, we combine all queries but need to be careful
    # about language mixing. We'll use the first language for the combined dataset
    # This is a simplification - ideally you'd want a more sophisticated approach

    print("\n⚠ Note: Using simplified approach for integrated model")
    print("  Using mixed language queries with English examples")

    # Initialize trainer
    trainer = CLiTSSATrainer(
        base_model_name='sentence-transformers/distiluse-base-multilingual-cased-v1',
        h=30,
        w=10
    )

    # Create training dataset
    # For integrated model, we compute similarities in a mixed way
    print("\nCreating integrated training dataset...")

    # Sample to keep it manageable
    max_samples = min(len(all_queries_low), 5000)  # Cap at 5000 for practical reasons
    if len(all_queries_low) > max_samples:
        import random
        indices = random.sample(range(len(all_queries_low)), max_samples)
        all_queries_low = [all_queries_low[i] for i in indices]
        all_examples_en = [all_examples_en[i] for i in indices]

    print(f"  Using {len(all_queries_low)} samples for integrated model")

    # For integrated model, we can't use create_training_dataset directly
    # because queries are in mixed languages. We'll create a simpler version.

    from sentence_transformers import SentenceTransformer, util
    import torch

    print("  Encoding queries and examples...")
    base_model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v1')

    query_embeddings = base_model.encode(all_queries_low, show_progress_bar=True, convert_to_tensor=True)
    example_embeddings = base_model.encode(all_examples_en, show_progress_bar=True, convert_to_tensor=True)

    print("  Computing cross-similarities...")
    # For each query, find similar examples
    training_queries = []
    training_examples = []
    training_scores = []

    import random
    from tqdm import tqdm

    for i in tqdm(range(len(all_queries_low)), desc="Creating training pairs"):
        # Compute similarities
        similarities = util.cos_sim(query_embeddings[i:i+1], example_embeddings)[0]
        similarities_np = similarities.cpu().numpy()

        # Select top-h + random-w
        top_indices = similarities_np.argsort()[-30:][::-1]
        remaining = list(set(range(len(all_examples_en))) - set(top_indices))
        random_indices = random.sample(remaining, min(10, len(remaining)))

        for idx in list(top_indices) + random_indices:
            training_queries.append(all_queries_low[i])
            training_examples.append(all_examples_en[idx])
            training_scores.append(float(similarities_np[idx]))

    print(f"  Created {len(training_queries)} training pairs")

    # Create dataset
    from clitssa_trainer import TemporalParallelDataset
    train_dataset = TemporalParallelDataset(training_queries, training_examples, training_scores)

    # Train
    print("\nFine-tuning integrated model...")
    model_output_dir = f"{output_dir}/clitssa_integrated"
    trainer.train(
        train_dataset=train_dataset,
        epochs=n_epochs,
        batch_size=batch_size,
        learning_rate=2e-5,
        output_dir=model_output_dir,
        warmup_steps=100
    )

    print(f"\n✓ Integrated CLiTSSA* training complete!")
    print(f"  Model saved to: {model_output_dir}/best_model")

    return model_output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CLiTSSA retriever for Indic languages")
    parser.add_argument('--languages', nargs='+', default=['hi', 'bn', 'ta'],
                       help='Languages to train on')
    parser.add_argument('--tasks', nargs='+', default=['xnli', 'qa'],
                       help='Tasks to train on')
    parser.add_argument('--n_train', type=int, default=500,
                       help='Number of training samples per task (default: 500 for quick training)')
    parser.add_argument('--epochs', type=int, default=3,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for training')
    parser.add_argument('--output_dir', default='models/indic_clitssa',
                       help='Output directory for models')
    parser.add_argument('--mode', choices=['task_specific', 'integrated', 'both'],
                       default='both', help='Training mode')

    args = parser.parse_args()

    print("="*60)
    print("CLiTSSA Training for Indian Languages")
    print("="*60)
    print(f"Configuration:")
    print(f"  Languages: {args.languages}")
    print(f"  Tasks: {args.tasks}")
    print(f"  Training samples: {args.n_train} per task")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Mode: {args.mode}")
    print(f"  Output: {args.output_dir}")
    print("="*60)

    # Train models
    if args.mode in ['task_specific', 'both']:
        print("\n>>> Training task-specific models...")
        task_results = train_task_specific_models(
            languages=args.languages,
            tasks=args.tasks,
            n_train_samples=args.n_train,
            n_epochs=args.epochs,
            batch_size=args.batch_size,
            output_dir=args.output_dir
        )

    if args.mode in ['integrated', 'both']:
        print("\n>>> Training integrated model...")
        integrated_path = train_integrated_model(
            languages=args.languages,
            tasks=args.tasks,
            n_train_samples=args.n_train,
            n_epochs=args.epochs,
            batch_size=args.batch_size,
            output_dir=args.output_dir
        )

    print("\n" + "="*60)
    print("All training complete!")
    print("="*60)
