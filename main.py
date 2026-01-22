import os
from PIL import Image

from src.sieves import compute_dhash, hamming_distance
from src.verifier import SSCDVerifier
from src.config import HASH_HAMMING_THRESHOLD, SSCD_SIM_THRESHOLD

# -------------------------
# 1. Build Hash Database
# -------------------------

HASH_DB = {}  # {image_path: dhash}

def build_hash_db(image_folder: str):
    print("[INFO] Building Hash Database...")
    for img_name in os.listdir(image_folder):
        img_path = os.path.join(image_folder, img_name)

        try:
            img = Image.open(img_path)
            HASH_DB[img_path] = compute_dhash(img)
        except Exception as e:
            print(f"[WARN] Skipping {img_name}: {e}")

    print(f"[INFO] Hash DB size: {len(HASH_DB)}")


# -------------------------
# 2. Sieve Stage (dHash)
# -------------------------

def check_sieve(query_image: Image.Image):
    q_hash = compute_dhash(query_image)

    for img_path, db_hash in HASH_DB.items():
        dist = hamming_distance(q_hash, db_hash)

        if dist <= HASH_HAMMING_THRESHOLD:
            return True, img_path, dist

    return False, None, None


# -------------------------
# 3. Verifier Stage (SSCD)
# -------------------------

verifier = SSCDVerifier()

def check_verifier(query_image: Image.Image):
    query_vec = verifier.get_embedding(query_image)

    # ⚠️ Placeholder until FAISS is integrated
    # Later: search(query_vec) → get best match + similarity

    return False, None, None


# -------------------------
# 4. Full Funnel Logic
# -------------------------

def detect_duplicate(image_path: str):
    query_image = Image.open(image_path)

    # Stage 1: Sieve
    is_dup, match, dist = check_sieve(query_image)

    if is_dup:
        return {
            "status": "Duplicate (Stage 1 - Sieve)",
            "match": match,
            "hamming_distance": dist
        }

    # Stage 2: Verifier
    is_dup, match, sim = check_verifier(query_image)

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
    # Build hash DB from originals
    build_hash_db("data/raw/copydays/original")

    # Test with a strong attack image
    test_image = "data/raw/copydays/strong/200000_1.png"

    result = detect_duplicate(test_image)
    print(result)
