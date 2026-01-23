import os
import sys
from pathlib import Path
import numpy as np
from tqdm import tqdm

# Ensure project root is on sys.path for module imports
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.verifier import SSCDVerifier
from src.indexer import Indexer
from src.config import SSCD_MODEL_PATH, IMAGE_DIR


# -------------------------
# Build FAISS Index
# -------------------------

def build_index():
    print(f"[INFO] IMAGE_DIR: {IMAGE_DIR}")
    print("[INFO] Loading SSCD model...")
    verifier = SSCDVerifier(SSCD_MODEL_PATH)

    print("[INFO] Initializing FAISS index...")
    indexer = Indexer()

    vectors = []
    filenames = []

    print("[INFO] Extracting embeddings from images...")

    for img_name in tqdm(os.listdir(IMAGE_DIR)):
        img_path = os.path.join(IMAGE_DIR, img_name)

        try:
            emb = verifier.get_embedding(img_path)
            vectors.append(emb)
            filenames.append(img_path)
        except Exception as e:
            print(f"[WARN] Skipping {img_name}: {e}")

    if len(vectors) == 0:
        print("[ERROR] No embeddings generated. Check dataset path.")
        return

    vectors = np.array(vectors).astype("float32")

    print("[INFO] Adding vectors to FAISS index...")
    indexer.add_vectors(vectors, filenames)

    print("[INFO] Saving index to disk...")
    indexer.save()

    print("✅ FAISS index successfully built!")


# -------------------------
# Run Script
# -------------------------

if __name__ == "__main__":
    build_index()
