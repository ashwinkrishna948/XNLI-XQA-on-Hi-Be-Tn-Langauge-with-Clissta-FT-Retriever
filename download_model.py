#!/usr/bin/env python3
"""
CLiTSSA Model Download Script
==============================

Download trained CLiTSSA models for Indian languages (Hindi, Bengali, Tamil).

Usage:
    python download_model.py --language hi --task xnli --output ./models
    python download_model.py --list
    python download_model.py --all --output ./models

Models:
    - Hindi XNLI: +79.2% improvement
    - Hindi QA: +48.6% improvement
    - Bengali XNLI: +81.8% improvement
    - Bengali QA: +45.9% improvement
    - Tamil XNLI: +75.2% improvement
    - Tamil QA: +80.1% improvement
"""

import argparse
import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Optional
import shutil

# Model metadata
MODELS = {
    "hi_xnli": {
        "name": "Hindi XNLI CLiTSSA",
        "language": "Hindi",
        "task": "Natural Language Inference (XNLI)",
        "language_code": "hi",
        "task_code": "xnli",
        "model_dir": "clitssa_hi_xnli",
        "size_mb": 514,
        "performance": {
            "base_score": 0.4302,
            "clitssa_score": 0.7711,
            "improvement": "+79.2%"
        },
        "description": "Fine-tuned for Hindi natural language inference tasks"
    },
    "hi_qa": {
        "name": "Hindi QA CLiTSSA",
        "language": "Hindi",
        "task": "Question Answering (XQuAD)",
        "language_code": "hi",
        "task_code": "qa",
        "model_dir": "clitssa_hi_qa",
        "size_mb": 514,
        "performance": {
            "base_score": 0.4079,
            "clitssa_score": 0.6060,
            "improvement": "+48.6%"
        },
        "description": "Fine-tuned for Hindi question answering tasks"
    },
    "bn_xnli": {
        "name": "Bengali XNLI CLiTSSA",
        "language": "Bengali",
        "task": "Natural Language Inference (XNLI)",
        "language_code": "bn",
        "task_code": "xnli",
        "model_dir": "clitssa_bn_xnli",
        "size_mb": 514,
        "performance": {
            "base_score": 0.3963,
            "clitssa_score": 0.7206,
            "improvement": "+81.8%"
        },
        "description": "Fine-tuned for Bengali natural language inference tasks"
    },
    "bn_qa": {
        "name": "Bengali QA CLiTSSA",
        "language": "Bengali",
        "task": "Question Answering (TyDiQA)",
        "language_code": "bn",
        "task_code": "qa",
        "model_dir": "clitssa_bn_qa",
        "size_mb": 514,
        "performance": {
            "base_score": 0.3737,
            "clitssa_score": 0.5452,
            "improvement": "+45.9%"
        },
        "description": "Fine-tuned for Bengali question answering tasks"
    },
    "ta_xnli": {
        "name": "Tamil XNLI CLiTSSA",
        "language": "Tamil",
        "task": "Natural Language Inference (XNLI)",
        "language_code": "ta",
        "task_code": "xnli",
        "model_dir": "clitssa_ta_xnli",
        "size_mb": 514,
        "performance": {
            "base_score": 0.3987,
            "clitssa_score": 0.6985,
            "improvement": "+75.2%"
        },
        "description": "Fine-tuned for Tamil natural language inference tasks"
    },
    "ta_qa": {
        "name": "Tamil QA CLiTSSA",
        "language": "Tamil",
        "task": "Question Answering (SQuAD-translated)",
        "language_code": "ta",
        "task_code": "qa",
        "model_dir": "clitssa_ta_qa",
        "size_mb": 514,
        "performance": {
            "base_score": 0.3947,
            "clitssa_score": 0.7110,
            "improvement": "+80.1%"
        },
        "description": "Fine-tuned for Tamil question answering tasks"
    }
}

# Configuration for model hosting
# Update these URLs when models are uploaded to GitHub releases or other hosting
MODEL_BASE_URL = "https://github.com/YOUR_USERNAME/YOUR_REPO/releases/download/v1.0"
LOCAL_MODEL_PATH = "models/indic_clitssa"


def print_banner():
    """Print script banner."""
    print("=" * 70)
    print("CLiTSSA Model Downloader - Indian Languages (Hindi, Bengali, Tamil)")
    print("=" * 70)
    print()


def list_models():
    """List all available models with their details."""
    print_banner()
    print("Available Models:")
    print()

    for model_id, info in MODELS.items():
        print(f"  {model_id}")
        print(f"    Name:        {info['name']}")
        print(f"    Language:    {info['language']} ({info['language_code']})")
        print(f"    Task:        {info['task']}")
        print(f"    Size:        ~{info['size_mb']} MB")
        print(f"    Performance: {info['performance']['base_score']:.4f} → "
              f"{info['performance']['clitssa_score']:.4f} "
              f"({info['performance']['improvement']})")
        print(f"    Description: {info['description']}")
        print()


def get_model_key(language: str, task: str) -> Optional[str]:
    """Get model key from language and task codes."""
    model_key = f"{language}_{task}"
    if model_key in MODELS:
        return model_key
    return None


def download_from_url(url: str, output_path: Path, model_name: str):
    """Download model from URL (for GitHub releases, HuggingFace, etc.)."""
    try:
        import requests
        from tqdm import tqdm
    except ImportError:
        print("Error: Required packages not installed.")
        print("Please install: pip install requests tqdm")
        sys.exit(1)

    print(f"Downloading {model_name}...")
    print(f"URL: {url}")

    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get('content-length', 0))

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=model_name) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

    print(f"✓ Downloaded to: {output_path}")


def copy_local_model(source_path: Path, output_path: Path, model_name: str):
    """Copy model from local directory."""
    if not source_path.exists():
        print(f"Error: Model not found at {source_path}")
        print(f"Please ensure models are available locally or use remote download mode.")
        sys.exit(1)

    print(f"Copying {model_name}...")
    print(f"From: {source_path}")
    print(f"To:   {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if source_path.is_dir():
        shutil.copytree(source_path, output_path, dirs_exist_ok=True)
    else:
        shutil.copy2(source_path, output_path)

    print(f"✓ Copied to: {output_path}")


def download_model(model_id: str, output_dir: str = "./models", mode: str = "local"):
    """Download a specific model."""
    if model_id not in MODELS:
        print(f"Error: Model '{model_id}' not found.")
        print(f"Available models: {', '.join(MODELS.keys())}")
        print(f"Use --list to see all models with details.")
        sys.exit(1)

    model_info = MODELS[model_id]
    output_path = Path(output_dir) / model_info['model_dir']

    print_banner()
    print(f"Model: {model_info['name']}")
    print(f"Task:  {model_info['task']}")
    print(f"Size:  ~{model_info['size_mb']} MB")
    print(f"Performance Improvement: {model_info['performance']['improvement']}")
    print()

    if mode == "remote":
        # Download from URL (GitHub releases, etc.)
        model_url = f"{MODEL_BASE_URL}/{model_info['model_dir']}.tar.gz"
        download_path = output_path.parent / f"{model_info['model_dir']}.tar.gz"
        download_from_url(model_url, download_path, model_info['name'])

        # Extract archive
        print("Extracting...")
        import tarfile
        with tarfile.open(download_path, 'r:gz') as tar:
            tar.extractall(output_path.parent)
        download_path.unlink()  # Remove archive
        print(f"✓ Extracted to: {output_path}")
    else:
        # Copy from local directory
        source_path = Path(LOCAL_MODEL_PATH) / model_info['model_dir'] / "best_model"
        copy_local_model(source_path, output_path / "best_model", model_info['name'])

    # Save model info
    info_path = output_path / "model_info.json"
    with open(info_path, 'w') as f:
        json.dump(model_info, f, indent=2)
    print(f"✓ Model info saved to: {info_path}")

    print()
    print("=" * 70)
    print("✓ Model download complete!")
    print("=" * 70)
    print()
    print("Usage example (Python):")
    print(f"""
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('{output_path / 'best_model'}')
embeddings = model.encode(['Your text here'])
""")


def download_all_models(output_dir: str = "./models", mode: str = "local"):
    """Download all available models."""
    print_banner()
    print(f"Downloading all {len(MODELS)} models to: {output_dir}")
    print()

    for i, model_id in enumerate(MODELS.keys(), 1):
        print(f"\n[{i}/{len(MODELS)}] Downloading {model_id}...")
        download_model(model_id, output_dir, mode)

    print()
    print("=" * 70)
    print(f"✓ All {len(MODELS)} models downloaded successfully!")
    print("=" * 70)


def interactive_mode(output_dir: str = "./models", mode: str = "local"):
    """Interactive model selection."""
    print_banner()
    print("Interactive Model Selection")
    print()

    # Select language
    languages = sorted(set(m['language_code'] for m in MODELS.values()))
    print("Available languages:")
    for i, lang in enumerate(languages, 1):
        lang_name = next(m['language'] for m in MODELS.values() if m['language_code'] == lang)
        print(f"  {i}. {lang_name} ({lang})")

    lang_choice = input(f"\nSelect language (1-{len(languages)}): ").strip()
    try:
        selected_lang = languages[int(lang_choice) - 1]
    except (ValueError, IndexError):
        print("Invalid selection.")
        sys.exit(1)

    # Select task
    tasks = sorted(set(m['task_code'] for m in MODELS.values()))
    print("\nAvailable tasks:")
    for i, task in enumerate(tasks, 1):
        task_name = next(m['task'] for m in MODELS.values() if m['task_code'] == task)
        print(f"  {i}. {task_name} ({task})")

    task_choice = input(f"\nSelect task (1-{len(tasks)}): ").strip()
    try:
        selected_task = tasks[int(task_choice) - 1]
    except (ValueError, IndexError):
        print("Invalid selection.")
        sys.exit(1)

    # Get model
    model_id = get_model_key(selected_lang, selected_task)
    if not model_id:
        print(f"Error: No model found for {selected_lang} + {selected_task}")
        sys.exit(1)

    print()
    download_model(model_id, output_dir, mode)


def main():
    parser = argparse.ArgumentParser(
        description="Download CLiTSSA models for Indian languages",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all available models
  python download_model.py --list

  # Download Hindi XNLI model
  python download_model.py --language hi --task xnli

  # Download Bengali QA model to specific directory
  python download_model.py --language bn --task qa --output ./my_models

  # Download all models
  python download_model.py --all

  # Interactive mode (no arguments)
  python download_model.py

Available languages: hi (Hindi), bn (Bengali), ta (Tamil)
Available tasks: xnli (Natural Language Inference), qa (Question Answering)
        """
    )

    parser.add_argument(
        '--list',
        action='store_true',
        help='List all available models'
    )

    parser.add_argument(
        '--language', '-l',
        choices=['hi', 'bn', 'ta'],
        help='Language code (hi=Hindi, bn=Bengali, ta=Tamil)'
    )

    parser.add_argument(
        '--task', '-t',
        choices=['xnli', 'qa'],
        help='Task code (xnli=Natural Language Inference, qa=Question Answering)'
    )

    parser.add_argument(
        '--output', '-o',
        default='./models',
        help='Output directory for models (default: ./models)'
    )

    parser.add_argument(
        '--all',
        action='store_true',
        help='Download all available models'
    )

    parser.add_argument(
        '--mode',
        choices=['local', 'remote'],
        default='local',
        help='Download mode: local (copy from local dir) or remote (download from URL)'
    )

    args = parser.parse_args()

    # Handle different modes
    if args.list:
        list_models()
    elif args.all:
        download_all_models(args.output, args.mode)
    elif args.language and args.task:
        model_id = get_model_key(args.language, args.task)
        if model_id:
            download_model(model_id, args.output, args.mode)
        else:
            print(f"Error: No model found for language={args.language}, task={args.task}")
            sys.exit(1)
    elif args.language or args.task:
        print("Error: Both --language and --task are required.")
        print("Use --list to see available models or run without arguments for interactive mode.")
        sys.exit(1)
    else:
        # Interactive mode
        interactive_mode(args.output, args.mode)


if __name__ == "__main__":
    main()
