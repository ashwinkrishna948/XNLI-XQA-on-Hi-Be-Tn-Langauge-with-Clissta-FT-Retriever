"""
Data loaders for Hindi, Bengali, and Tamil NLP tasks.

Tasks:
1. Natural Language Inference (XNLI)
2. Question Answering (XQuAD/TyDiQA)
"""

from typing import List, Tuple, Dict
from datasets import load_dataset
from deep_translator import GoogleTranslator
import random
from tqdm import tqdm
import pickle
from pathlib import Path


class IndicDataLoader:
    """Base class for Indic language data loaders"""

    def __init__(self):
        self.cache_file = Path('data/indic_translation_cache.pkl')
        self.translation_cache = {}
        self._load_cache()

    def _load_cache(self):
        """Load translation cache"""
        if self.cache_file.exists():
            with open(self.cache_file, 'rb') as f:
                self.translation_cache = pickle.load(f)
            print(f"Loaded translation cache: {len(self.translation_cache)} entries")

    def _save_cache(self):
        """Save translation cache"""
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_file, 'wb') as f:
            pickle.dump(self.translation_cache, f)

    def translate_text(self, text: str, src_lang: str, tgt_lang: str) -> str:
        """Translate text with caching"""
        cache_key = f"{src_lang}_{tgt_lang}_{text}"

        if cache_key in self.translation_cache:
            return self.translation_cache[cache_key]

        try:
            translator = GoogleTranslator(source=src_lang, target=tgt_lang)
            translated = translator.translate(text)
            self.translation_cache[cache_key] = translated
            return translated
        except Exception as e:
            print(f"Translation error: {e}")
            return text  # Fallback to original

    def translate_batch(self, texts: List[str], src_lang: str, tgt_lang: str) -> List[str]:
        """Translate batch of texts"""
        translations = []
        needs_save = False

        for text in tqdm(texts, desc=f"Translating {src_lang}→{tgt_lang}"):
            translated = self.translate_text(text, src_lang, tgt_lang)
            translations.append(translated)
            if len(translations) % 100 == 0:
                self._save_cache()
                needs_save = False
            else:
                needs_save = True

        if needs_save:
            self._save_cache()

        return translations


class IndicXNLILoader(IndicDataLoader):
    """
    Loader for Natural Language Inference task.

    Languages: Hindi (hi), Bengali (bn), Tamil (ta)
    Source: XNLI dataset
    """

    def __init__(self):
        super().__init__()
        self.task_name = "xnli"

    def load_data(self, language: str, split: str = 'test',
                  n_samples: int = None) -> Tuple[List[str], List[str]]:
        """
        Load XNLI data for the specified language.

        Args:
            language: Language code (hi, bn, ta, en)
            split: Data split (train, validation, test)
            n_samples: Number of samples to load (None for all)

        Returns:
            Tuple of (queries, examples) where both are formatted strings
        """
        print(f"Loading XNLI for {language} ({split})...")

        # Map split names
        split_map = {'test': 'test', 'train': 'train', 'dev': 'validation',
                    'validation': 'validation'}
        hf_split = split_map.get(split, split)

        # Load dataset
        if language == 'hi':
            # Hindi is available in XNLI
            try:
                dataset = load_dataset('xnli', language, split=hf_split)
                # Sample if requested (CRITICAL: must sample for performance!)
                if n_samples and n_samples < len(dataset):
                    indices = random.sample(range(len(dataset)), n_samples)
                    dataset = dataset.select(indices)
                    print(f"  Sampled {n_samples} examples")
            except:
                # Fallback: translate from English
                print(f"  Hindi XNLI not found, translating from English...")
                dataset = load_dataset('xnli', 'en', split=hf_split)
                # Sample BEFORE translation
                if n_samples and n_samples < len(dataset):
                    indices = random.sample(range(len(dataset)), n_samples)
                    dataset = dataset.select(indices)
                dataset = self._translate_dataset(dataset, 'en', 'hi')
        elif language in ['bn', 'ta']:
            # Bengali and Tamil: translate from English
            print(f"  Translating XNLI from English to {language}...")
            dataset = load_dataset('xnli', 'en', split=hf_split)
            # Sample BEFORE translation (critical for performance!)
            if n_samples and n_samples < len(dataset):
                indices = random.sample(range(len(dataset)), n_samples)
                dataset = dataset.select(indices)
                print(f"  Sampled {n_samples} examples before translation")
            dataset = self._translate_dataset(dataset, 'en', language)
        elif language == 'en':
            dataset = load_dataset('xnli', 'en', split=hf_split)
            # Sample if requested
            if n_samples and n_samples < len(dataset):
                indices = random.sample(range(len(dataset)), n_samples)
                dataset = dataset.select(indices)
        else:
            raise ValueError(f"Unsupported language: {language}")

        # Note: Sampling already done above for translated datasets

        # Format data
        queries = []
        examples = []

        for item in dataset:
            # Format: "Premise: X | Hypothesis: Y"
            premise = item['premise']
            hypothesis = item['hypothesis']
            label = item['label']

            # Label mapping
            label_names = ['entailment', 'neutral', 'contradiction']
            label_name = label_names[label] if isinstance(label, int) else label

            query_text = f"Premise: {premise} | Hypothesis: {hypothesis}"
            example_text = f"Premise: {premise} | Hypothesis: {hypothesis} | Label: {label_name}"

            queries.append(query_text)
            examples.append(example_text)

        print(f"  Loaded {len(queries)} {language} XNLI samples")
        return queries, examples

    def _translate_dataset(self, dataset, src_lang: str, tgt_lang: str):
        """Translate dataset fields"""
        translated_premises = self.translate_batch(
            [item['premise'] for item in dataset], src_lang, tgt_lang
        )
        translated_hypotheses = self.translate_batch(
            [item['hypothesis'] for item in dataset], src_lang, tgt_lang
        )

        # Create new dataset with translations
        translated_data = []
        for i, item in enumerate(dataset):
            translated_data.append({
                'premise': translated_premises[i],
                'hypothesis': translated_hypotheses[i],
                'label': item['label']
            })

        # Convert to dataset format (simple list of dicts for now)
        return translated_data


class IndicQALoader(IndicDataLoader):
    """
    Loader for Question Answering task.

    Languages: Hindi (hi), Bengali (bn), Tamil (ta)
    Source: XQuAD, TyDiQA, or SQuAD translations
    """

    def __init__(self):
        super().__init__()
        self.task_name = "qa"

    def load_data(self, language: str, split: str = 'validation',
                  n_samples: int = None) -> Tuple[List[str], List[str]]:
        """
        Load QA data for the specified language.

        Args:
            language: Language code (hi, bn, ta, en)
            split: Data split (train, validation, test)
            n_samples: Number of samples to load

        Returns:
            Tuple of (queries, examples)
        """
        print(f"Loading QA data for {language}...")

        # Try XQuAD first (if available)
        if language in ['hi', 'en']:
            return self._load_xquad(language, split, n_samples)
        elif language == 'bn':
            # Try TyDiQA for Bengali
            return self._load_tydiqa(language, n_samples)
        elif language == 'ta':
            # Translate from English SQuAD for Tamil
            return self._load_translated_squad(language, n_samples)
        else:
            raise ValueError(f"Unsupported language: {language}")

    def _load_xquad(self, language: str, split: str, n_samples: int):
        """Load XQuAD dataset"""
        try:
            lang_map = {'hi': 'xquad.hi', 'en': 'xquad.en'}
            dataset = load_dataset('xquad', lang_map[language], split='validation')

            if n_samples and n_samples < len(dataset):
                indices = random.sample(range(len(dataset)), n_samples)
                dataset = dataset.select(indices)

            queries = []
            examples = []

            for item in dataset:
                context = item['context']
                question = item['question']
                answers = item['answers']['text'][0] if item['answers']['text'] else ""

                query_text = f"Context: {context} | Question: {question}"
                example_text = f"Context: {context} | Question: {question} | Answer: {answers}"

                queries.append(query_text)
                examples.append(example_text)

            print(f"  Loaded {len(queries)} XQuAD samples")
            return queries, examples

        except Exception as e:
            print(f"  Error loading XQuAD: {e}")
            return self._load_translated_squad(language, n_samples)

    def _load_tydiqa(self, language: str, n_samples: int):
        """Load TyDiQA dataset for Bengali"""
        try:
            dataset = load_dataset('tydiqa', 'secondary_task', split='validation')

            # Filter for Bengali
            dataset = dataset.filter(lambda x: x['id'].startswith('bengali'))

            if n_samples and n_samples < len(dataset):
                indices = random.sample(range(len(dataset)), n_samples)
                dataset = dataset.select(indices)

            queries = []
            examples = []

            for item in dataset:
                context = item['context']
                question = item['question']
                answers = item['answers']['text'][0] if item['answers']['text'] else ""

                query_text = f"Context: {context} | Question: {question}"
                example_text = f"Context: {context} | Question: {question} | Answer: {answers}"

                queries.append(query_text)
                examples.append(example_text)

            print(f"  Loaded {len(queries)} TyDiQA Bengali samples")
            return queries, examples

        except Exception as e:
            print(f"  Error loading TyDiQA: {e}")
            return self._load_translated_squad(language, n_samples)

    def _load_translated_squad(self, language: str, n_samples: int):
        """Load and translate SQuAD dataset"""
        print(f"  Translating SQuAD to {language}...")

        # Load English SQuAD
        dataset = load_dataset('squad', split='validation')

        if n_samples and n_samples < len(dataset):
            indices = random.sample(range(len(dataset)), n_samples)
            dataset = dataset.select(indices)

        # Translate
        contexts = [item['context'] for item in dataset]
        questions = [item['question'] for item in dataset]

        translated_contexts = self.translate_batch(contexts, 'en', language)
        translated_questions = self.translate_batch(questions, 'en', language)

        queries = []
        examples = []

        for i, item in enumerate(dataset):
            answer = item['answers']['text'][0] if item['answers']['text'] else ""
            # Note: We keep answer in English since translation might break span indices
            # In real use, you'd want to align the translated answer

            query_text = f"Context: {translated_contexts[i]} | Question: {translated_questions[i]}"
            example_text = f"Context: {translated_contexts[i]} | Question: {translated_questions[i]} | Answer: {answer}"

            queries.append(query_text)
            examples.append(example_text)

        print(f"  Loaded {len(queries)} translated SQuAD samples")
        return queries, examples


def get_indic_loader(task: str):
    """Get the appropriate data loader for a task"""
    loaders = {
        'xnli': IndicXNLILoader,
        'qa': IndicQALoader,
        'nli': IndicXNLILoader,
    }

    if task not in loaders:
        raise ValueError(f"Unknown task: {task}. Available: {list(loaders.keys())}")

    return loaders[task]()


if __name__ == "__main__":
    # Test data loaders
    print("="*60)
    print("Testing Indic Data Loaders")
    print("="*60)

    # Test XNLI
    print("\n1. Testing XNLI Loader...")
    xnli_loader = IndicXNLILoader()

    for lang in ['hi', 'bn', 'ta']:
        print(f"\n  Loading {lang} XNLI (5 samples)...")
        queries, examples = xnli_loader.load_data(lang, split='validation', n_samples=5)
        print(f"  Sample query: {queries[0][:100]}...")
        print(f"  Sample example: {examples[0][:100]}...")

    # Test QA
    print("\n2. Testing QA Loader...")
    qa_loader = IndicQALoader()

    for lang in ['hi', 'bn', 'ta']:
        print(f"\n  Loading {lang} QA (5 samples)...")
        queries, examples = qa_loader.load_data(lang, n_samples=5)
        print(f"  Sample query: {queries[0][:100]}...")
        print(f"  Sample example: {examples[0][:100]}...")

    print("\n" + "="*60)
    print("✓ Data loader tests complete!")
