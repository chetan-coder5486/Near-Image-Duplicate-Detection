import os
import sys
import glob
from tqdm import tqdm
from PIL import Image

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from main import detect_duplicate, build_hash_db, indexer, verifier, HASH_DB
from src.sieves import compute_dhash

# --- CONFIG ---
SYNTHETIC_DIR = "data/synthetic_attacks"
ORIGINAL_DIR = "data/raw/distractors"

def evaluate_recall():
    print("--- 📊 Starting Evaluation (Recall Test) ---")
    
    # 1. Clear existing data for fresh evaluation
    HASH_DB.clear()
    indexer.reset()  # Reset FAISS index
    
    # 2. Identify Test Pairs
    attack_files = glob.glob(os.path.join(SYNTHETIC_DIR, "*_attack.jpg"))
    if not attack_files:
        print("❌ No synthetic attacks found. Run 'data/generate_attacks.py' first.")
        return

    print(f"Found {len(attack_files)} test cases.")
    
    # 3. First pass: Add ONLY originals to database
    print("Populating Database with Originals...")
    valid_test_pairs = []  # Store valid (original, attack) pairs
    
    for attack_path in tqdm(attack_files[:100]):
        filename = os.path.basename(attack_path)
        original_name = filename.replace("_attack.jpg", ".jpg")
        original_path = os.path.join(ORIGINAL_DIR, original_name)
        
        # Try alternate extensions
        if not os.path.exists(original_path):
            original_path = os.path.join(ORIGINAL_DIR, original_name.replace(".jpg", ".JPEG"))
        
        if os.path.exists(original_path):
            try:
                # Add to hash DB
                img = Image.open(original_path).convert("RGB")
                dhash = compute_dhash(img)
                HASH_DB[dhash] = original_path
                
                # Add to FAISS index
                embedding = verifier.get_embedding(original_path)
                indexer.add(os.path.basename(original_path), embedding)
                
                valid_test_pairs.append((original_path, attack_path))
            except Exception as e:
                print(f"⚠️ Failed to add {original_path}: {e}")

    print(f"Added {len(valid_test_pairs)} originals to database.")

    # 4. Second pass: Query with attacks
    true_positives = 0
    false_negatives = 0
    
    print("\nRunning Queries...")
    for original_path, attack_path in tqdm(valid_test_pairs):
        expected_original = os.path.basename(original_path)
        
        try:
            result = detect_duplicate(attack_path)
            
            if result['status'] == 'Duplicate (Stage 1 - Sieve)' or result['status'] == 'Duplicate (Stage 2 - Verifier)':
                match_name = os.path.basename(result['match']) if result['match'] else None
                if match_name == expected_original:
                    true_positives += 1
                # else: found wrong duplicate (rare edge case)
            else:
                false_negatives += 1
        except Exception as e:
            print(f"⚠️ Query failed for {attack_path}: {e}")
            false_negatives += 1

    # 5. Calculate Metrics
    total_tests = len(valid_test_pairs)
    recall = true_positives / total_tests if total_tests > 0 else 0
    
    print(f"\n--- 📈 Results ---")
    print(f"Total Queries: {total_tests}")
    print(f"Correct Matches (True Positives): {true_positives}")
    print(f"Missed Matches (False Negatives): {false_negatives}")
    print(f"Recall Score: {recall:.2%}")
    
    if recall < 0.80:
        print("⚠️ Recall is low. Try LOWERING the Vector Threshold in main.py (e.g. 0.75 -> 0.70)")
    else:
        print("✅ System is performing well!")

if __name__ == "__main__":
    evaluate_recall()