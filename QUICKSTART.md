# 🚀 Quick Start Guide - Image Duplicate Detection UI

## Installation

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## Setup

2. **Build the vector index (run once):**
```bash
python build_index.py
```

This scans all images in `data/data/raw/` and builds a FAISS index for fast similarity search.

**Note:** The script will:
- Find all images recursively in `data/data/raw/`
- Compute embeddings using the SSCD model
- Create `data/faiss_index.idx` (index file)
- Create `data/metadata.pkl` (image metadata)

## Running the App

3. **Start the Streamlit web app:**
```bash
streamlit run app.py
```

This will open in your browser at `http://localhost:8501`

## Usage

1. **Upload an image** using the file uploader on the left
2. **Configure search parameters** in the sidebar:
   - Number of duplicates to show (k)
   - Similarity threshold (0-1)
3. **View results** showing:
   - Similar images displayed as thumbnails
   - Similarity scores
   - Distance metrics
   - File information

## Features

✅ **Fast similarity search** using FAISS
✅ **Adjustable k-nearest neighbors** (1-20)
✅ **Configurable similarity threshold**
✅ **Visual preview** of results
✅ **Detailed metrics** (similarity, distance)
✅ **Results summary table**

## Project Structure

```
├── app.py                    # Streamlit web app
├── build_index.py           # Index builder script
├── src/
│   ├── verifier.py          # SSCD model wrapper
│   ├── search.py            # FAISS vector store
│   └── config.py            # Configuration
├── data/
│   ├── data/raw/            # Images to index
│   ├── faiss_index.idx      # Generated index
│   ├── metadata.pkl         # Generated metadata
│   └── processed/
│       └── sscd.pt          # SSCD model
└── requirements.txt
```

## Troubleshooting

**Q: "Vector index not found" error**
- Run `python build_index.py` first

**Q: No images found**
- Ensure images are in `data/data/raw/` directory
- Supported formats: jpg, jpeg, png, webp, bmp

**Q: Streamlit not found**
- Run `pip install streamlit` or reinstall requirements

## Configuration

Edit `src/config.py` to adjust:
- `SSCD_MODEL_PATH` - Path to model weights
- `SSCD_SIM_THRESHOLD` - Default similarity threshold
- `SSCD_INPUT_SIZE` - Image input size (288 or 224)
