"""
Threshold Tuning Script
Run this to find optimal thresholds for your dataset.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image
from tqdm import tqdm

from src.sieves import compute_dhash, hamming_distance
from src.verifier import SSCDVerifier
from src.config import SSCD_MODEL_PATH, IMAGE_DIR


def analyze_dataset():
    """
    Compute similarity scores between:
    1. Original vs Strong (modified) images → should be HIGH (duplicates)
    2. Random unrelated pairs → should be LOW (non-duplicates)
    """
    original_dir = Path("data/raw/copydays/original")
    strong_dir = Path("data/raw/copydays/strong")
    
    if not original_dir.exists() or not strong_dir.exists():
        print("❌ Copydays dataset not found. Run data/downoad_copydays.py first.")
        return
    
    print("Loading SSCD model...")
    verifier = SSCDVerifier(SSCD_MODEL_PATH)
    
    # Get matching pairs (same base ID)
    originals = {f.stem: f for f in original_dir.glob("*.jpg")}
    strongs = {f.stem: f for f in strong_dir.glob("*.jpg")}
    
    # Find pairs with same ID prefix (e.g., 207600 matches 207600, 207601, etc.)
    duplicate_scores_sscd = []
    duplicate_scores_hash = []
    non_duplicate_scores_sscd = []
    non_duplicate_scores_hash = []
    
    original_list = list(originals.items())[:50]  # Limit for speed
    
    print("\n📊 Analyzing duplicate pairs (original vs modified)...")
    for orig_id, orig_path in tqdm(original_list):
        # Find corresponding strong versions
        for strong_id, strong_path in strongs.items():
            if strong_id.startswith(orig_id[:4]):  # Same image group
                try:
                    # SSCD similarity
                    emb1 = verifier.get_embedding(str(orig_path))
                    emb2 = verifier.get_embedding(str(strong_path))
                    sim = float(emb1 @ emb2)
                    duplicate_scores_sscd.append(sim)
                    
                    # dHash distance
                    img1 = Image.open(orig_path).convert("RGB")
                    img2 = Image.open(strong_path).convert("RGB")
                    h1 = compute_dhash(img1)
                    h2 = compute_dhash(img2)
                    dist = hamming_distance(h1, h2)
                    duplicate_scores_hash.append(dist)
                except:
                    pass
                break
    
    print("\n📊 Analyzing non-duplicate pairs (random unrelated images)...")
    import random
    random.shuffle(original_list)
    for i in range(min(50, len(original_list) - 1)):
        try:
            path1 = original_list[i][1]
            path2 = original_list[i + 1][1]
            
            # SSCD similarity
            emb1 = verifier.get_embedding(str(path1))
            emb2 = verifier.get_embedding(str(path2))
            sim = float(emb1 @ emb2)
            non_duplicate_scores_sscd.append(sim)
            
            # dHash distance
            img1 = Image.open(path1).convert("RGB")
            img2 = Image.open(path2).convert("RGB")
            h1 = compute_dhash(img1)
            h2 = compute_dhash(img2)
            dist = hamming_distance(h1, h2)
            non_duplicate_scores_hash.append(dist)
        except:
            pass
    
    # Print results
    print("\n" + "="*60)
    print("📈 SSCD SIMILARITY SCORES (higher = more similar)")
    print("="*60)
    print(f"  Duplicates:     min={min(duplicate_scores_sscd):.3f}, max={max(duplicate_scores_sscd):.3f}, mean={np.mean(duplicate_scores_sscd):.3f}")
    print(f"  Non-duplicates: min={min(non_duplicate_scores_sscd):.3f}, max={max(non_duplicate_scores_sscd):.3f}, mean={np.mean(non_duplicate_scores_sscd):.3f}")
    
    # Find optimal threshold (maximize gap)
    dup_min = min(duplicate_scores_sscd)
    non_dup_max = max(non_duplicate_scores_sscd)
    suggested_sscd = (dup_min + non_dup_max) / 2
    
    print(f"\n  ✅ Suggested SSCD_SIM_THRESHOLD: {suggested_sscd:.2f}")
    print(f"     (Duplicates start at {dup_min:.3f}, non-duplicates max at {non_dup_max:.3f})")
    
    print("\n" + "="*60)
    print("📈 dHASH HAMMING DISTANCE (lower = more similar)")
    print("="*60)
    print(f"  Duplicates:     min={min(duplicate_scores_hash)}, max={max(duplicate_scores_hash)}, mean={np.mean(duplicate_scores_hash):.1f}")
    print(f"  Non-duplicates: min={min(non_duplicate_scores_hash)}, max={max(non_duplicate_scores_hash)}, mean={np.mean(non_duplicate_scores_hash):.1f}")
    
    dup_max_hash = max(duplicate_scores_hash)
    non_dup_min_hash = min(non_duplicate_scores_hash)
    suggested_hash = (dup_max_hash + non_dup_min_hash) // 2
    
    print(f"\n  ✅ Suggested HASH_HAMMING_THRESHOLD: {suggested_hash}")
    print(f"     (Duplicates max at {dup_max_hash}, non-duplicates min at {non_dup_min_hash})")
    
    print("\n" + "="*60)
    print("📝 RECOMMENDED CONFIG VALUES:")
    print("="*60)
    print(f"  SSCD_SIM_THRESHOLD = {suggested_sscd:.2f}")
    print(f"  HASH_HAMMING_THRESHOLD = {suggested_hash}")


if __name__ == "__main__":
    analyze_dataset()
