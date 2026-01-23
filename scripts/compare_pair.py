"""
Quick score checker - test individual image pairs
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image
from src.sieves import compute_dhash, hamming_distance
from src.verifier import SSCDVerifier
from src.config import SSCD_MODEL_PATH

def compare_images(path1: str, path2: str):
    print(f"\nComparing:")
    print(f"  Image 1: {path1}")
    print(f"  Image 2: {path2}")
    
    # dHash
    img1 = Image.open(path1).convert("RGB")
    img2 = Image.open(path2).convert("RGB")
    h1 = compute_dhash(img1)
    h2 = compute_dhash(img2)
    dist = hamming_distance(h1, h2)
    print(f"\n  dHash Distance: {dist} (threshold=5, lower=more similar)")
    
    # SSCD
    verifier = SSCDVerifier(SSCD_MODEL_PATH)
    emb1 = verifier.get_embedding(path1)
    emb2 = verifier.get_embedding(path2)
    sim = float(emb1 @ emb2)
    print(f"  SSCD Similarity: {sim:.4f} (threshold=0.65, higher=more similar)")
    
    print(f"\n  Verdict: {'DUPLICATE' if dist <= 5 or sim >= 0.65 else 'NOT DUPLICATE'}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) == 3:
        compare_images(sys.argv[1], sys.argv[2])
    else:
        print("Usage: python scripts/compare_pair.py <image1> <image2>")
        print("\nExample with copydays:")
        compare_images(
            "data/raw/copydays/original/207600.jpg",
            "data/raw/copydays/strong/207601.jpg"
        )
