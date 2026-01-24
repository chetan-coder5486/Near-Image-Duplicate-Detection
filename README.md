# Near-Image-Duplicate-Detection

A lightweight, two-stage near-duplicate image detection system combining fast perceptual hashing with deep learning embeddings for accurate and efficient duplicate detection at scale.

## Overview

This project implements a **two-stage funnel approach** to near-duplicate detection:

1. **Stage 1 - dHash Sieve**: Fast perceptual hashing using Difference Hash (dHash) to quickly filter obvious non-duplicates with O(1) comparisons
2. **Stage 2 - SSCD Verification**: Deep learning-based verification using Meta's Self-Supervised Copy Detection (SSCD) model with FAISS indexing for accurate similarity scoring

This architecture balances **speed** (hash-based filtering) with **accuracy** (neural network verification), making it suitable for real-world applications.

## Features

- 🚀 **Fast Filtering**: dHash sieve eliminates ~99% of candidates in milliseconds
- 🎯 **High Accuracy**: SSCD embeddings catch semantic duplicates (crops, filters, compression)
- 📊 **Scalable Search**: FAISS vector index enables efficient similarity search over millions of images
- 🖼️ **Multiple UIs**: Streamlit web app and FastAPI REST API
- 🔧 **Configurable Thresholds**: Tune sensitivity for your use case
- 🧪 **Evaluation Tools**: Scripts for threshold tuning and large-scale benchmarking

## Project Structure

```
Near-Image-Duplicate-Detection/
├── app.py                  # FastAPI REST API server
├── streamlit_app.py        # Streamlit web interface
├── main.py                 # CLI example usage
├── requirements.txt        # Python dependencies
│
├── src/                    # Core library
│   ├── config.py           # Configuration (paths, thresholds)
│   ├── pipeline.py         # DuplicateDetector orchestration
│   ├── sieves.py           # dHash computation & Hamming distance
│   ├── verifier.py         # SSCD model wrapper
│   ├── indexer.py          # FAISS index management
│   ├── build_index.py      # Script to build FAISS index
│   └── data_loader.py      # Data loading utilities
│
├── scripts/                # Utility scripts
│   ├── compare_pair.py     # Compare two images directly
│   ├── tune_thresholds.py  # Find optimal thresholds
│   └── evaluate_with_distractors.py  # Large-scale evaluation
│
├── data/                   # Data directory
│   ├── download_gldv2.py   # Download GLDv2 distractor images
│   ├── downoad_copydays.py # Download COPYDAYS benchmark
│   ├── generate_attacks.py # Generate synthetic augmentations
│   ├── processed/          # SSCD model & FAISS index
│   ├── raw/                # Raw image datasets
│   ├── synthetic_attacks/  # Generated test images
│   └── uploads/            # User uploads (runtime)
│
└── tests/                  # Unit tests
    ├── test_indexer.py
    ├── test_sieve.py
    └── test_verifier.py
```

## Installation

### Prerequisites
- Python 3.8+
- ~2GB disk space for model and sample data

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/Near-Image-Duplicate-Detection.git
cd Near-Image-Duplicate-Detection

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### Download SSCD Model

Place the SSCD TorchScript model at `data/processed/sscd.pt`. You can download it from [Meta's SSCD repository](https://github.com/facebookresearch/sscd-copy-detection).

## Quick Start

### 1. Build the FAISS Index

First, populate `data/raw/copydays/original/` with your reference images, then build the index:

```bash
python src/build_index.py
```

This extracts SSCD embeddings for all images and stores them in a FAISS index for fast retrieval.

### 2. Launch the Web UI (Streamlit)

```bash
streamlit run streamlit_app.py
```

Open the browser link, upload an image, and view:
- Top-K similar matches with similarity scores
- Duplicate/not-duplicate classification
- Visual comparison of query vs matches

### 3. Launch the REST API (FastAPI)

```bash
uvicorn app:app --reload
```

Then visit `http://localhost:8000` for the web interface or use the API:

```bash
curl -X POST "http://localhost:8000/api/detect" \
  -F "file=@your_image.jpg"
```

### 4. CLI Usage

```bash
python main.py
```

Or compare two specific images:

```bash
python scripts/compare_pair.py path/to/image1.jpg path/to/image2.jpg
```

## Configuration

Edit `src/config.py` to customize:

```python
# SSCD Model
SSCD_MODEL_PATH = "data/processed/sscd.pt"
SSCD_INPUT_SIZE = 288
SSCD_SIM_THRESHOLD = 0.2      # Similarity threshold for duplicates

# dHash Sieve
HASH_HAMMING_THRESHOLD = 15   # Max Hamming distance for sieve pass

# Data Locations
IMAGE_DIR = "data/raw/copydays/original"
UPLOAD_DIR = "data/uploads"

# Search Settings
TOP_K = 10                    # Number of results to return
```

## How It Works

### Detection Pipeline

```
Query Image
     │
     ▼
┌─────────────┐
│ Compute     │  Fast: ~1ms
│ dHash       │
└─────────────┘
     │
     ▼
┌─────────────┐
│ Sieve       │  Compare against hash DB
│ (Hamming)   │  Filter candidates with dist > threshold
└─────────────┘
     │
     ▼
┌─────────────┐
│ Extract     │  ~50ms per image
│ SSCD Embed  │
└─────────────┘
     │
     ▼
┌─────────────┐
│ FAISS       │  Cosine similarity search
│ Search      │  Returns top-K matches
└─────────────┘
     │
     ▼
  Results
```

### dHash (Difference Hash)

- Resizes image to 9x8 grayscale
- Computes horizontal gradient (each pixel vs right neighbor)
- Produces 64-bit hash
- Hamming distance measures similarity (lower = more similar)

### SSCD (Self-Supervised Copy Detection)

- Meta's state-of-the-art copy detection model
- Trained on augmented image pairs
- 512-dimensional embeddings
- Robust to crops, filters, compression, overlays

## Evaluation & Tuning

### Tune Thresholds

Find optimal thresholds for your dataset:

```bash
python scripts/tune_thresholds.py
```

This analyzes duplicate vs non-duplicate pairs and suggests threshold values.

### Large-Scale Evaluation

Test with distractor images to measure real-world performance:

```bash
# Download distractor images (~10,000 images)
python data/download_gldv2.py

# Generate synthetic attacks (augmented versions)
python data/generate_attacks.py

# Run evaluation
python scripts/evaluate_with_distractors.py
```

## API Reference

### FastAPI Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web interface |
| `/api/detect` | POST | Upload image, get duplicate detection results |
| `/preview?path=...` | GET | Preview an image by path |

### Python API

```python
from src.pipeline import DuplicateDetector, build_hash_db
from src.config import IMAGE_DIR

# Initialize detector
hash_db = build_hash_db(IMAGE_DIR)
detector = DuplicateDetector(image_dir=IMAGE_DIR, hash_db=hash_db)

# Detect duplicates
result = detector.detect("query_image.jpg", top_k=5)

# Result structure:
# {
#   "is_duplicate": bool,
#   "stage": "sieve" | "verifier",
#   "match": "path/to/match.jpg",
#   "sieve_matches": [...],
#   "verifier_matches": [...]
# }
```

## Dependencies

- **torch / torchvision**: Deep learning framework
- **faiss-cpu**: Vector similarity search
- **imagehash**: Perceptual hashing
- **Pillow**: Image processing
- **streamlit**: Web UI framework
- **FastAPI**: REST API framework
- **albumentations**: Image augmentation (for evaluation)
- **opencv-python**: Image processing utilities

## Testing

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_sieve.py -v
```

## Performance

| Stage | Time per Image | Purpose |
|-------|---------------|---------|
| dHash | ~1ms | Fast filtering |
| SSCD Embedding | ~50ms (CPU) | Feature extraction |
| FAISS Search | ~1ms | Similarity lookup |

Typical end-to-end latency: **50-100ms** per query on CPU.

## License

MIT License - See LICENSE file for details.

## Acknowledgments

- [SSCD](https://github.com/facebookresearch/sscd-copy-detection) - Meta's copy detection model
- [FAISS](https://github.com/facebookresearch/faiss) - Facebook AI Similarity Search
- [ImageHash](https://github.com/JohannesBuchner/imagehash) - Perceptual hashing library
- [COPYDAYS](https://lear.inrialpes.fr/~jegou/data.php) - Benchmark dataset