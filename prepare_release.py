#!/usr/bin/env python3
"""
Prepare CLiTSSA models for GitHub release or sharing.

This script packages trained models into compressed archives ready for distribution.
"""

import tarfile
import hashlib
import json
from pathlib import Path
from typing import Dict
import shutil

MODELS_DIR = Path("models/indic_clitssa")
OUTPUT_DIR = Path("release")

MODELS = [
    "clitssa_hi_xnli",
    "clitssa_hi_qa",
    "clitssa_bn_xnli",
    "clitssa_bn_qa",
    "clitssa_ta_xnli",
    "clitssa_ta_qa"
]


def calculate_checksum(file_path: Path) -> str:
    """Calculate SHA256 checksum of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def create_archive(model_name: str, source_dir: Path, output_dir: Path) -> Dict:
    """Create compressed archive for a model."""
    print(f"Packaging {model_name}...")

    model_path = source_dir / model_name / "best_model"
    if not model_path.exists():
        print(f"  Warning: Model not found at {model_path}")
        return None

    # Create archive
    archive_name = f"{model_name}.tar.gz"
    archive_path = output_dir / archive_name

    with tarfile.open(archive_path, 'w:gz') as tar:
        tar.add(model_path, arcname=f"{model_name}/best_model")

    # Calculate checksum
    checksum = calculate_checksum(archive_path)
    size_mb = archive_path.stat().st_size / (1024 * 1024)

    print(f"  ✓ Created: {archive_path}")
    print(f"  Size: {size_mb:.2f} MB")
    print(f"  SHA256: {checksum}")

    return {
        "name": model_name,
        "archive": archive_name,
        "size_mb": round(size_mb, 2),
        "sha256": checksum
    }


def create_checksums_file(manifests: list, output_dir: Path):
    """Create checksums file for verification."""
    checksums_path = output_dir / "SHA256SUMS"

    with open(checksums_path, 'w') as f:
        for manifest in manifests:
            if manifest:
                f.write(f"{manifest['sha256']}  {manifest['archive']}\n")

    print(f"\n✓ Checksums written to: {checksums_path}")


def create_manifest(manifests: list, output_dir: Path):
    """Create JSON manifest with model information."""
    manifest_path = output_dir / "manifest.json"

    manifest_data = {
        "version": "1.0",
        "models": manifests,
        "total_models": len([m for m in manifests if m]),
        "total_size_mb": sum(m['size_mb'] for m in manifests if m)
    }

    with open(manifest_path, 'w') as f:
        json.dump(manifest_data, f, indent=2)

    print(f"✓ Manifest written to: {manifest_path}")


def create_release_notes(output_dir: Path):
    """Create release notes template."""
    notes = """# CLiTSSA Models for Indian Languages - v1.0

This release contains trained CLiTSSA (Cross-Lingual Time-Sensitive Semantic Alignment) models for Hindi, Bengali, and Tamil languages across two NLP tasks.

## Models Included

### Hindi Models
- **Hindi XNLI** (`clitssa_hi_xnli.tar.gz`) - Natural Language Inference (+79.2% improvement)
- **Hindi QA** (`clitssa_hi_qa.tar.gz`) - Question Answering (+48.6% improvement)

### Bengali Models
- **Bengali XNLI** (`clitssa_bn_xnli.tar.gz`) - Natural Language Inference (+81.8% improvement)
- **Bengali QA** (`clitssa_bn_qa.tar.gz`) - Question Answering (+45.9% improvement)

### Tamil Models
- **Tamil XNLI** (`clitssa_ta_xnli.tar.gz`) - Natural Language Inference (+75.2% improvement)
- **Tamil QA** (`clitssa_ta_qa.tar.gz`) - Question Answering (+80.1% improvement)

## Download

### Using the download script (recommended):
```bash
# Download specific model
python download_model.py --language hi --task xnli

# List all models
python download_model.py --list

# Download all models
python download_model.py --all
```

### Manual download:
Download the model archives from the release assets and extract:
```bash
tar -xzf clitssa_hi_xnli.tar.gz
```

## Verification

Verify downloaded files using SHA256 checksums:
```bash
sha256sum -c SHA256SUMS
```

## Usage

```python
from sentence_transformers import SentenceTransformer

# Load model
model = SentenceTransformer('clitssa_hi_xnli/best_model')

# Generate embeddings
embeddings = model.encode([
    'Your Hindi text here',
    'और यहाँ एक और वाक्य'
])

# Use for similarity search, retrieval, etc.
```

## Performance

All models show significant improvements over the base multilingual model:
- Average improvement: **+68.5%**
- XNLI tasks: 75-82% improvement
- QA tasks: 46-80% improvement

## Technical Details

- **Base Model:** distiluse-base-multilingual-cased-v1
- **Training Data:** 200 samples per language-task
- **Training Epochs:** 2
- **Loss Function:** CoSENT (Cosine Sentence)
- **Framework:** Sentence-Transformers

## Citation

If you use these models, please cite the original CLiTSSA paper and this work:

```bibtex
@article{clitssa2025,
  title={CLiTSSA: Cross-Lingual Time-Sensitive Semantic Alignment for Indian Languages},
  author={Your Name},
  year={2025}
}
```

## License

[Specify your license here]

## References

- Full evaluation report: See `COMPREHENSIVE_EVALUATION_REPORT.md`
- Original CLiTSSA paper: [Add link when available]
"""

    notes_path = output_dir / "RELEASE_NOTES.md"
    with open(notes_path, 'w') as f:
        f.write(notes)

    print(f"✓ Release notes written to: {notes_path}")


def main():
    print("=" * 70)
    print("CLiTSSA Model Release Preparation")
    print("=" * 70)
    print()

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    # Package each model
    manifests = []
    for model_name in MODELS:
        manifest = create_archive(model_name, MODELS_DIR, OUTPUT_DIR)
        manifests.append(manifest)
        print()

    # Create supporting files
    create_checksums_file(manifests, OUTPUT_DIR)
    create_manifest(manifests, OUTPUT_DIR)
    create_release_notes(OUTPUT_DIR)

    print()
    print("=" * 70)
    print("✓ Release preparation complete!")
    print("=" * 70)
    print()
    print(f"Total models: {len([m for m in manifests if m])}")
    print(f"Total size: {sum(m['size_mb'] for m in manifests if m):.2f} MB")
    print()
    print("Next steps:")
    print("1. Review files in:", OUTPUT_DIR)
    print("2. Update MODEL_BASE_URL in download_model.py with your GitHub release URL")
    print("3. Create GitHub release and upload files from release/")
    print("4. Test download script: python download_model.py --mode remote --list")


if __name__ == "__main__":
    main()
