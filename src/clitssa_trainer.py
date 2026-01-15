"""
CLiTSSA Retriever Fine-tuning Implementation

Implements the Cross-Lingual Time-Sensitive Semantic Alignment (CLiTSSA)
fine-tuning methodology from the paper.

Based on Section "Cross-Lingual Time-Sensitive Semantic Alignment" (pages 4-5)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer, models, losses, util
from typing import List, Tuple, Dict
import numpy as np
from tqdm import tqdm
from deep_translator import GoogleTranslator
import random
import pickle
from pathlib import Path
import json


class CoSENTLoss(nn.Module):
    """
    CoSENT (Cosine Sentence) Loss implementation.

    From paper: "CoSENT loss generates a more robust training signal for
    optimizing the cosine value than the traditional cosine similarity loss function."

    Loss formula (Equation 4):
    L = log Σ(1 + exp(f(s)(q_a, q_b) - f(s)(q_y, q_z) + exp(...))

    where (q_a, q_b) and (q_y, q_z) are instance pairs within a batch,
    and similarity of (a,b) exceeds that of (y,z).
    """

    def __init__(self, temperature=0.05):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings1, embeddings2, labels):
        """
        Args:
            embeddings1: Embeddings for queries [batch_size, hidden_dim]
            embeddings2: Embeddings for examples [batch_size, hidden_dim]
            labels: Similarity scores [batch_size]
        """
        # Normalize embeddings
        embeddings1 = F.normalize(embeddings1, p=2, dim=1)
        embeddings2 = F.normalize(embeddings2, p=2, dim=1)

        # Compute cosine similarities
        similarities = torch.sum(embeddings1 * embeddings2, dim=1)  # [batch_size]

        # CoSENT loss computation
        # For each pair (i,j) where label[i] > label[j],
        # penalize if similarity[i] <= similarity[j]

        batch_size = similarities.size(0)
        labels = labels.view(-1, 1)
        similarities = similarities.view(-1, 1)

        # Create masks for pairs where label_i > label_j
        label_diff = labels - labels.t()  # [batch_size, batch_size]
        sim_diff = similarities - similarities.t()  # [batch_size, batch_size]

        # Only consider pairs where label_i > label_j
        mask = (label_diff > 0).float()

        # CoSENT loss: log(1 + sum(exp(sim_diff))) for valid pairs
        loss_matrix = torch.log(1 + torch.exp(-sim_diff / self.temperature)) * mask
        loss = loss_matrix.sum() / (mask.sum() + 1e-8)

        return loss


class TemporalParallelDataset(Dataset):
    """
    Dataset for CLiTSSA fine-tuning with parallel temporal queries and similarity scores.

    From paper: "A training dataset Dt is constructed comprising pairs of sentences
    alongside their associated similarity scores. Specifically, Dt consists of
    (q_u^l, q_v^r | f(s)_{u,v})"
    """

    def __init__(self, queries_low: List[str], queries_high: List[str],
                 similarity_scores: List[float]):
        """
        Args:
            queries_low: Low-resource language queries
            queries_high: High-resource language queries (English)
            similarity_scores: Expected in-language similarity scores
        """
        self.queries_low = queries_low
        self.queries_high = queries_high
        self.similarity_scores = similarity_scores

        assert len(queries_low) == len(queries_high) == len(similarity_scores)

    def __len__(self):
        return len(self.queries_low)

    def __getitem__(self, idx):
        return {
            'query_low': self.queries_low[idx],
            'query_high': self.queries_high[idx],
            'similarity': self.similarity_scores[idx]
        }


class CLiTSSATrainer:
    """
    Trainer for CLiTSSA retriever fine-tuning.

    Implements the methodology from Section "Cross-Lingual Time-Sensitive Semantic Alignment"
    """

    def __init__(self,
                 base_model_name: str = 'distiluse-base-multilingual-cased-v1',
                 h: int = 30,  # Top-h similar examples
                 w: int = 10,  # Random examples for diversity
                 temperature: float = 0.05,
                 device: str = None):
        """
        Args:
            base_model_name: Base multilingual Sentence-BERT model
            h: Number of top similar examples to select (paper uses 30)
            w: Number of random examples to select (paper uses 10)
            temperature: Temperature for CoSENT loss
            device: Device for training
        """
        self.h = h
        self.w = w
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        print(f"Initializing CLiTSSA Trainer...")
        print(f"Base model: {base_model_name}")
        print(f"Device: {self.device}")
        print(f"h (top examples): {h}, w (random examples): {w}")

        # Load base model
        self.model = SentenceTransformer(base_model_name)
        self.model.to(self.device)

        # Loss function
        self.loss_fn = CoSENTLoss(temperature=temperature)

        # Translation cache
        self.translation_cache = {}
        self.cache_file = Path('data/translation_cache_clitssa.pkl')
        self._load_translation_cache()

    def _load_translation_cache(self):
        """Load translation cache if exists"""
        if self.cache_file.exists():
            with open(self.cache_file, 'rb') as f:
                self.translation_cache = pickle.load(f)
            print(f"Loaded translation cache with {len(self.translation_cache)} entries")

    def _save_translation_cache(self):
        """Save translation cache"""
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_file, 'wb') as f:
            pickle.dump(self.translation_cache, f)
        print(f"Saved translation cache with {len(self.translation_cache)} entries")

    def translate_batch(self, texts: List[str], src_lang: str, tgt_lang: str) -> List[str]:
        """
        Translate a batch of texts with caching.

        From paper: "Employing a concise prompt prefix, 'Translate the following
        sentences from English to the {Target Language},' we employ the T5 model"

        Note: Using Google Translate instead of T5 for simplicity
        """
        translations = []
        to_translate = []
        to_translate_indices = []

        # Check cache
        for i, text in enumerate(texts):
            cache_key = f"{src_lang}_{tgt_lang}_{text}"
            if cache_key in self.translation_cache:
                translations.append(self.translation_cache[cache_key])
            else:
                translations.append(None)
                to_translate.append(text)
                to_translate_indices.append(i)

        # Translate uncached texts
        if to_translate:
            translator = GoogleTranslator(source=src_lang, target=tgt_lang)
            print(f"Translating {len(to_translate)} texts from {src_lang} to {tgt_lang}...")

            for text, idx in zip(tqdm(to_translate, desc="Translating"), to_translate_indices):
                try:
                    translated = translator.translate(text)
                    translations[idx] = translated
                    cache_key = f"{src_lang}_{tgt_lang}_{text}"
                    self.translation_cache[cache_key] = translated
                except Exception as e:
                    print(f"Translation error: {e}")
                    translations[idx] = text  # Fallback to original

            # Save cache periodically
            if len(to_translate) > 10:
                self._save_translation_cache()

        return translations

    def create_training_dataset(self,
                                queries_low: List[str],
                                examples_high: List[str],
                                low_lang: str,
                                high_lang: str = 'en') -> TemporalParallelDataset:
        """
        Create training dataset with parallel queries and similarity scores.

        From paper:
        "Here, we present a systematic approach to construct Dt:
        1. Delineate D'_r, a transformed resource-rich dataset in low-resource language
        2. Determine temporal semantic alignment scores f(s) among queries in D_l and D'_r
        3. Select top-h analogous examples from D'_r for each query in D_l
        4. Randomly select w examples from remaining dataset
        5. Replace D'_r with original D_r"

        Args:
            queries_low: Low-resource language queries
            examples_high: High-resource language examples (English)
            low_lang: Low-resource language code
            high_lang: High-resource language code

        Returns:
            Dataset with (query_low, query_high, similarity_score) tuples
        """
        print(f"\nCreating CLiTSSA training dataset...")
        print(f"Queries: {len(queries_low)}, Examples: {len(examples_high)}")

        # Step 1: Translate examples to low-resource language (D'_r)
        print(f"Step 1: Translating examples from {high_lang} to {low_lang}...")
        examples_translated = self.translate_batch(examples_high, high_lang, low_lang)

        # Step 2: Compute in-language similarity scores
        print("Step 2: Computing in-language similarity scores...")

        # Encode queries and translated examples
        query_embeddings = self.model.encode(queries_low, convert_to_tensor=True,
                                            show_progress_bar=True, device=self.device)
        example_embeddings = self.model.encode(examples_translated, convert_to_tensor=True,
                                               show_progress_bar=True, device=self.device)

        # Prepare training pairs
        training_queries_low = []
        training_queries_high = []
        training_scores = []

        print(f"Step 3-4: Selecting top-{self.h} + random-{self.w} examples per query...")

        for i, query in enumerate(tqdm(queries_low, desc="Creating training pairs")):
            # Compute similarities for this query
            similarities = util.cos_sim(query_embeddings[i:i+1], example_embeddings)[0]
            similarities_np = similarities.cpu().numpy()

            # Step 3: Select top-h similar examples
            top_indices = np.argsort(similarities_np)[-self.h:][::-1]

            # Step 4: Select random w examples from remaining
            remaining_indices = list(set(range(len(examples_high))) - set(top_indices))
            if len(remaining_indices) >= self.w:
                random_indices = random.sample(remaining_indices, self.w)
            else:
                random_indices = remaining_indices

            # Combine selected indices
            selected_indices = list(top_indices) + random_indices

            # Step 5: Create training pairs with original English examples
            for idx in selected_indices:
                training_queries_low.append(query)
                training_queries_high.append(examples_high[idx])
                training_scores.append(float(similarities_np[idx]))

        print(f"Created {len(training_queries_low)} training pairs")
        self._save_translation_cache()

        return TemporalParallelDataset(training_queries_low, training_queries_high,
                                      training_scores)

    def train(self,
              train_dataset: TemporalParallelDataset,
              epochs: int = 3,
              batch_size: int = 16,
              learning_rate: float = 2e-5,
              output_dir: str = 'models/clitssa',
              warmup_steps: int = 100):
        """
        Fine-tune the retriever using CoSENT loss.

        Args:
            train_dataset: Training dataset
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            output_dir: Directory to save the fine-tuned model
            warmup_steps: Number of warmup steps
        """
        print(f"\n{'='*60}")
        print(f"Training CLiTSSA Retriever")
        print(f"{'='*60}")
        print(f"Training samples: {len(train_dataset)}")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}")
        print(f"Learning rate: {learning_rate}")
        print(f"Output directory: {output_dir}")
        print()

        # Create data loader
        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                 shuffle=True, num_workers=0)

        # Optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)

        # Learning rate scheduler
        total_steps = len(train_loader) * epochs
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=learning_rate, total_steps=total_steps,
            pct_start=warmup_steps/total_steps
        )

        # Training loop
        self.model.train()
        global_step = 0
        best_loss = float('inf')

        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print("-" * 60)

            epoch_loss = 0
            progress_bar = tqdm(train_loader, desc=f"Training")

            for batch_idx, batch in enumerate(progress_bar):
                # Encode queries and examples
                queries_low = batch['query_low']
                queries_high = batch['query_high']
                similarity_scores = batch['similarity'].float().to(self.device)

                # Get embeddings with gradients using SentenceTransformer's tokenizer and forward pass
                # Tokenize
                features_low = self.model.tokenize(queries_low)
                features_low = {k: v.to(self.device) for k, v in features_low.items()}

                features_high = self.model.tokenize(queries_high)
                features_high = {k: v.to(self.device) for k, v in features_high.items()}

                # Forward pass through the model
                embeddings_low = self.model(features_low)['sentence_embedding']
                embeddings_high = self.model(features_high)['sentence_embedding']

                # Compute loss
                loss = self.loss_fn(embeddings_low, embeddings_high, similarity_scores)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                # Update metrics
                epoch_loss += loss.item()
                global_step += 1

                # Update progress bar
                avg_loss = epoch_loss / (batch_idx + 1)
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{avg_loss:.4f}',
                    'lr': f'{scheduler.get_last_lr()[0]:.2e}'
                })

            avg_epoch_loss = epoch_loss / len(train_loader)
            print(f"Epoch {epoch + 1} - Average Loss: {avg_epoch_loss:.4f}")

            # Save best model
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                output_path = Path(output_dir) / 'best_model'
                self.model.save(str(output_path))
                print(f"✓ Saved best model to {output_path}")

        # Save final model
        final_path = Path(output_dir) / 'final_model'
        self.model.save(str(final_path))
        print(f"\n✓ Training complete! Final model saved to {final_path}")

        return self.model

    def save_model(self, output_path: str):
        """Save the fine-tuned model"""
        self.model.save(output_path)
        print(f"Model saved to {output_path}")

    def load_model(self, model_path: str):
        """Load a fine-tuned model"""
        self.model = SentenceTransformer(model_path)
        self.model.to(self.device)
        print(f"Model loaded from {model_path}")
        return self.model


def create_clitssa_training_data(language: str,
                                 task: str,
                                 n_queries: int = 1000,
                                 data_dir: str = 'data') -> Tuple[List[str], List[str]]:
    """
    Create training data for CLiTSSA from mTEMPREASON dataset.

    Args:
        language: Low-resource language code (ro, de, fr)
        task: Temporal task (xnli, xquad, pawsx, or actual temporal tasks if available)
        n_queries: Number of queries to use for training
        data_dir: Data directory

    Returns:
        Tuple of (queries_low_resource, examples_english)
    """
    from data_loader import get_loader

    print(f"Creating CLiTSSA training data for {language}/{task}...")

    # Load data for the target language
    loader = get_loader(task)
    queries_low, examples_low = loader.load_data(language, 'train', n_samples=n_queries)

    # Load English examples (resource-rich)
    queries_en, examples_en = loader.load_data('en', 'train', n_samples=n_queries)

    return queries_low, examples_en


if __name__ == "__main__":
    # Example usage
    print("CLiTSSA Trainer Module")
    print("=" * 60)
    print("This module implements the CLiTSSA fine-tuning methodology.")
    print("Use create_clitssa_training_data() to prepare data, then")
    print("use CLiTSSATrainer.train() to fine-tune the retriever.")
