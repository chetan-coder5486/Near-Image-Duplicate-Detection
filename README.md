# Near-Image-Duplicate-Detection

Lightweight two-stage near-duplicate detector: fast dHash sieve followed by SSCD + FAISS verification.

## Streamlit UI quickstart
1) Install deps: `pip install -r requirements.txt`
2) Build the FAISS index (uses images in `src.config.IMAGE_DIR`): `python src/build_index.py`
3) Launch the UI: `streamlit run streamlit_app.py`
4) Open the browser link, upload an image, and view top matches plus duplicate status.

Dataset/model paths are configurable in `src/config.py`.