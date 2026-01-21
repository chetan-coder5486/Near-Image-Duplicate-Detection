import sys
import os
import numpy as np
import shutil

# Add src to python path to import our code
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.indexer import Indexer

def test_faiss_logic():
    print("--- 🧪 Testing Step 5: FAISS Search Engine ---")
    
    # 1. Setup clean environment
    test_dir = "data/test_processed"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    
    # 2. Initialize
    indexer = Indexer(index_path=f"{test_dir}/index.bin", metadata_path=f"{test_dir}/meta.pkl")
    
    # 3. Create Fake Data (3 vectors)
    # We create random vectors and normalize them (required for Cosine Similarity)
    vecs = np.random.random((3, 512)).astype('float32')
    # L2 Normalization logic: v / ||v||
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    
    filenames = ["photo_A.jpg", "photo_B.jpg", "photo_C.jpg"]
    
    # 4. Add to Index and Save
    indexer.add_vectors(vecs, filenames)
    indexer.save()
    
    # 5. Simulate a Server Restart (Reload from disk)
    print("\n🔄 Simulating Server Restart...")
    new_indexer = Indexer(index_path=f"{test_dir}/index.bin", metadata_path=f"{test_dir}/meta.pkl")
    
    # 6. Search Test: Query with the EXACT same vector as photo_A
    print("\n🔎 Searching for 'photo_A' vector...")
    query = vecs[0].reshape(1, 512) # Reshape to (1, 512)
    results = new_indexer.search(query, k=1)
    
    print(f"Result: {results}")
    
    # 7. Verification
    # Score should be 1.0 (or 0.999999) because it's an exact match
    match_name = results[0]['filename']
    match_score = results[0]['score']
    
    if match_name == "photo_A.jpg" and match_score > 0.99:
        print("\n✅ SUCCESS: Search Engine correctly identified the image.")
    else:
        print(f"\n❌ FAILURE: Expected photo_A.jpg with score 1.0, got {match_name} with {match_score}")

    # Cleanup
    shutil.rmtree(test_dir)

if __name__ == "__main__":
    test_faiss_logic()