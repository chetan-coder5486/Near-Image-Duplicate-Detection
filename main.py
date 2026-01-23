import os
from PIL import Image

from src.sieves import compute_dhash, hamming_distance
from src.verifier import SSCDVerifier
from src.indexer import Indexer
from src.config import HASH_HAMMING_THRESHOLD, SSCD_SIM_THRESHOLD, SSCD_MODEL_PATH


# -------------------------
# 1. Build Hash Database
# -------------------------

HASH_DB = {}  # {dhash: image_path}

def build_hash_db(image_folder: str):
    print("[INFO] Building Hash Database...")
    
    for img_name in os.listdir(image_folder):
        img_path = os.path.join(image_folder, img_name)

        try:
            img = Image.open(img_path).convert("RGB")
            dhash = compute_dhash(img)
            HASH_DB[dhash] = img_path
        except Exception as e:
            print(f"[WARN] Skipping {img_name}: {e}")

    print(f"[INFO] Hash DB size: {len(HASH_DB)}")


# -------------------------
# 2. Sieve Stage (dHash)
# -------------------------

def check_sieve(query_image: Image.Image):
    q_hash = compute_dhash(query_image)

    for db_hash, img_path in HASH_DB.items():
        dist = hamming_distance(q_hash, db_hash)

        if dist <= HASH_HAMMING_THRESHOLD:
            return True, img_path, dist

    return False, None, None


# -------------------------
# 3. Verifier Stage (SSCD + FAISS)
# -------------------------

verifier = SSCDVerifier(SSCD_MODEL_PATH)
indexer = Indexer()

def check_verifier(query_image_path: str):
    query_vec = verifier.get_embedding(query_image_path)

    results = indexer.search(query_vec, k=1)

    if len(results) == 0:
        return False, None, None

    best = results[0]
    #print("Top match:", best)
    return True, best["filename"], best["score"]


# -------------------------
# 4. Full Funnel Logic
# -------------------------

def detect_duplicate(image_path: str):
    query_image = Image.open(image_path).convert("RGB")

    # Stage 1: Sieve
    is_dup, match, dist = check_sieve(query_image)

    if is_dup:
        return {
            "status": "Duplicate (Stage 1 - Sieve)",
            "match": match,
            "hamming_distance": dist
        }

    # Stage 2: Verifier
    is_dup, match, sim = check_verifier(image_path)

    if is_dup and sim >= SSCD_SIM_THRESHOLD:
        return {
            "status": "Duplicate (Stage 2 - Verifier)",
            "match": match,
            "similarity": sim
        }

    return {
        "status": "Unique",
        "match": None
    }


# -------------------------
# 5. Run Test
# -------------------------

if __name__ == "__main__":
    # Build hash DB from original images
    build_hash_db("data/raw/copydays/original")

    test_image = "data/raw/copydays/strong/214401.jpg"

    result = detect_duplicate(test_image)
    print(result)
