"""
Large-Scale Evaluation with Distractors
=========================================
Tests if the duplicate detector can find the "needle" (original image)
in a "haystack" of 10,000+ distractor images.

This is the proper way to validate your system's real-world performance.
"""

import os
import sys
import glob
import shutil
from pathlib import Path
from tqdm import tqdm
import numpy as np

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.verifier import SSCDVerifier
from src.indexer import Indexer
from src.sieves import compute_dhash, hamming_distance
from src.config import SSCD_MODEL_PATH, SSCD_SIM_THRESHOLD, HASH_HAMMING_THRESHOLD
from PIL import Image

# --- CONFIGURATION ---
DISTRACTOR_DIR = "data/raw/distractors"
ORIGINAL_DIR = "data/raw/copydays/original"
ATTACK_DIR = "data/raw/copydays/strong"  # Modified versions of originals
SYNTHETIC_ATTACK_DIR = "data/synthetic_attacks"  # Your generated attacks

# Evaluation index paths (separate from production to avoid conflicts)
EVAL_INDEX_PATH = "data/eval/faiss_index.bin"
EVAL_METADATA_PATH = "data/eval/metadata.pkl"


class Evaluator:
    def __init__(self):
        print("🔧 Initializing Evaluator...")
        self.verifier = SSCDVerifier(SSCD_MODEL_PATH)
        
        # Create fresh index for evaluation
        eval_dir = Path(EVAL_INDEX_PATH).parent
        if eval_dir.exists():
            shutil.rmtree(eval_dir)
        eval_dir.mkdir(parents=True, exist_ok=True)
        
        self.indexer = Indexer(
            index_path=EVAL_INDEX_PATH,
            metadata_path=EVAL_METADATA_PATH
        )
        
        # Hash database for sieve evaluation
        self.hash_db = {}
    
    def add_images_to_index(self, image_paths: list, label: str):
        """Add a batch of images to the FAISS index and hash DB."""
        vectors = []
        filenames = []
        
        print(f"📥 Indexing {len(image_paths)} {label}...")
        for path in tqdm(image_paths, desc=label):
            try:
                # SSCD embedding
                emb = self.verifier.get_embedding(path)
                vectors.append(emb)
                filenames.append(path)
                
                # dHash for sieve
                img = Image.open(path).convert("RGB")
                dhash = compute_dhash(img)
                self.hash_db[dhash] = path
                
            except Exception as e:
                continue
        
        if vectors:
            vectors = np.array(vectors).astype("float32")
            self.indexer.add_vectors(vectors, filenames)
        
        return len(filenames)
    
    def query(self, image_path: str, top_k: int = 1):
        """Query the index and return the best match."""
        # Stage 1: Sieve check
        try:
            img = Image.open(image_path).convert("RGB")
            q_hash = compute_dhash(img)
            
            for db_hash, db_path in self.hash_db.items():
                dist = hamming_distance(q_hash, db_hash)
                if dist <= HASH_HAMMING_THRESHOLD:
                    return {
                        "stage": "sieve",
                        "match": db_path,
                        "score": dist
                    }
        except:
            pass
        
        # Stage 2: SSCD + FAISS
        try:
            query_vec = self.verifier.get_embedding(image_path)
            results = self.indexer.search(query_vec, k=top_k)
            
            if results and results[0]["score"] >= SSCD_SIM_THRESHOLD:
                return {
                    "stage": "verifier",
                    "match": results[0]["filename"],
                    "score": results[0]["score"]
                }
        except:
            pass
        
        return {"stage": "none", "match": None, "score": 0}


def get_original_id(filename: str) -> str:
    """
    Extract the image group ID from filename.
    Copydays naming: 207600.jpg (original) -> 207601.jpg (attack)
    First 4-5 digits identify the image group.
    """
    basename = Path(filename).stem
    # Remove any suffix like '_attack'
    basename = basename.replace("_attack", "")
    # Return first 4 characters as group ID
    return basename[:4]


def is_correct_match(query_path: str, match_path: str) -> bool:
    """Check if the matched image is the correct original."""
    if match_path is None:
        return False
    
    query_id = get_original_id(Path(query_path).name)
    match_id = get_original_id(Path(match_path).name)
    
    # Also check it's from originals, not another attack or distractor
    is_from_originals = "original" in match_path or ORIGINAL_DIR in match_path
    
    return query_id == match_id and is_from_originals


def run_evaluation():
    print("=" * 60)
    print("🚀 LARGE-SCALE DUPLICATE DETECTION EVALUATION")
    print("=" * 60)
    
    evaluator = Evaluator()
    
    # --- PHASE 1: Add Distractors (The Haystack) ---
    distractors = glob.glob(os.path.join(DISTRACTOR_DIR, "**", "*.jpg"), recursive=True)
    distractors += glob.glob(os.path.join(DISTRACTOR_DIR, "**", "*.jpeg"), recursive=True)
    
    if len(distractors) == 0:
        print(f"\n⚠️  No distractors found in {DISTRACTOR_DIR}")
        print("   Run: python data/download_gldv2.py")
        print("   Continuing without distractors...\n")
    else:
        num_distractors = min(len(distractors), 10000)  # Cap at 10k
        evaluator.add_images_to_index(distractors[:num_distractors], "distractors")
    
    # --- PHASE 2: Add Originals (The Needles) ---
    originals = glob.glob(os.path.join(ORIGINAL_DIR, "*.jpg"))
    if len(originals) == 0:
        print(f"\n❌ No originals found in {ORIGINAL_DIR}")
        print("   Run: python data/downoad_copydays.py")
        return
    
    evaluator.add_images_to_index(originals, "originals")
    
    # --- PHASE 3: Query with Attacks ---
    attacks = glob.glob(os.path.join(ATTACK_DIR, "*.jpg"))
    synthetic = glob.glob(os.path.join(SYNTHETIC_ATTACK_DIR, "*.jpg")) if Path(SYNTHETIC_ATTACK_DIR).exists() else []
    
    print(f"\n📊 Database size: {evaluator.indexer.index.ntotal} images")
    print(f"   - Distractors: {len(distractors[:10000]) if distractors else 0}")
    print(f"   - Originals: {len(originals)}")
    print(f"\n🔍 Test queries:")
    print(f"   - Copydays attacks: {len(attacks)}")
    print(f"   - Synthetic attacks: {len(synthetic)}")
    
    # Evaluate Copydays attacks
    results = {"correct": 0, "wrong_match": 0, "no_match": 0, "distractor_match": 0}
    
    print(f"\n📈 Evaluating Copydays attacks...")
    for attack_path in tqdm(attacks, desc="Querying"):
        result = evaluator.query(attack_path)
        
        if result["match"] is None:
            results["no_match"] += 1
        elif is_correct_match(attack_path, result["match"]):
            results["correct"] += 1
        elif "distractor" in result["match"].lower() or DISTRACTOR_DIR in result["match"]:
            results["distractor_match"] += 1
        else:
            results["wrong_match"] += 1
    
    total = len(attacks)
    
    print("\n" + "=" * 60)
    print("📊 RESULTS - COPYDAYS ATTACKS")
    print("=" * 60)
    print(f"  ✅ Correct matches:      {results['correct']:4d} / {total} ({100*results['correct']/total:.1f}%)")
    print(f"  ❌ Wrong original:       {results['wrong_match']:4d} / {total} ({100*results['wrong_match']/total:.1f}%)")
    print(f"  🎯 Distractor confusion: {results['distractor_match']:4d} / {total} ({100*results['distractor_match']/total:.1f}%)")
    print(f"  ⚪ No match found:       {results['no_match']:4d} / {total} ({100*results['no_match']/total:.1f}%)")
    
    precision = results['correct'] / (results['correct'] + results['wrong_match'] + results['distractor_match']) if (results['correct'] + results['wrong_match'] + results['distractor_match']) > 0 else 0
    recall = results['correct'] / total
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\n📈 METRICS:")
    print(f"  Precision: {precision:.2%}")
    print(f"  Recall:    {recall:.2%}")
    print(f"  F1 Score:  {f1:.2%}")
    
    # Evaluate synthetic attacks if available
    if synthetic:
        print(f"\n📈 Evaluating Synthetic attacks...")
        syn_results = {"correct": 0, "wrong_match": 0, "no_match": 0, "distractor_match": 0}
        
        for attack_path in tqdm(synthetic[:500], desc="Querying"):  # Limit to 500
            result = evaluator.query(attack_path)
            
            if result["match"] is None:
                syn_results["no_match"] += 1
            elif is_correct_match(attack_path, result["match"]):
                syn_results["correct"] += 1
            elif "distractor" in result["match"].lower() or DISTRACTOR_DIR in result["match"]:
                syn_results["distractor_match"] += 1
            else:
                syn_results["wrong_match"] += 1
        
        syn_total = min(len(synthetic), 500)
        print(f"\n  Synthetic attacks: {syn_results['correct']}/{syn_total} correct ({100*syn_results['correct']/syn_total:.1f}%)")
    
    print("\n" + "=" * 60)
    print("✅ Evaluation complete!")
    print("=" * 60)
    
    # Cleanup
    print("\n🧹 Cleaning up evaluation index...")
    shutil.rmtree(Path(EVAL_INDEX_PATH).parent, ignore_errors=True)


if __name__ == "__main__":
    run_evaluation()
