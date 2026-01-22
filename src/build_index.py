import os
import numpy as np
from tqdm import tqdm

from src.verifier import SSCDVerifier
from src.indexer import Indexer
from src.config import SSCD_MODEL_PATH


# -------------------------
# Config
# -------------------------

IMAGE_DIR = "data/raw/copydays/original"


# -------------------------
# Build FAISS Index
# -------------------------

def build_index():
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
