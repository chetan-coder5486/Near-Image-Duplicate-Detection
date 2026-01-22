"""
Script to build FAISS vector index from images in the data directory
Run this once before starting the Streamlit app
"""

import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
from src.verifier import SSCDVerifier
from src.search import VectorStore
from src.config import SSCD_MODEL_PATH

def get_image_paths(root_dir="data/data/raw", extensions=('.jpg', '.jpeg', '.png', '.webp', '.bmp')):
    """Recursively find all image files"""
    image_paths = []
    for ext in extensions:
        image_paths.extend(Path(root_dir).rglob(f'*{ext}'))
        image_paths.extend(Path(root_dir).rglob(f'*{ext.upper()}'))
    return sorted(set(image_paths))

def build_index():
    """Build FAISS index from all images in data directory"""
    
    print("🔍 Near-Image Duplicate Detection - Index Builder")
    print("=" * 50)
    
    # Initialize verifier
    print("\n📦 Loading SSCD model...")
    verifier = SSCDVerifier(SSCD_MODEL_PATH)
    
    # Find all images
    print("\n🖼️  Scanning for images...")
    image_paths = get_image_paths()
    
    if len(image_paths) == 0:
        print("❌ No images found in data/data/raw directory")
        print("   Please ensure images are in the correct location")
        return
    
    print(f"✅ Found {len(image_paths)} images")
    
    # Compute embeddings
    print("\n⚡ Computing embeddings...")
    embeddings = []
    metadata = []
    failed_images = []
    
    for img_path in tqdm(image_paths, desc="Processing images"):
        try:
            # Get embedding
            embedding = verifier.get_embedding(str(img_path))
            embeddings.append(embedding)
            
            # Store metadata
            metadata.append({
                'path': str(img_path),
                'filename': img_path.name
            })
        except Exception as e:
            failed_images.append((str(img_path), str(e)))
            print(f"\n⚠️  Failed to process {img_path}: {e}")
    
    if len(embeddings) == 0:
        print("❌ No embeddings computed. Please check your image files.")
        return
    
    print(f"\n✅ Successfully computed {len(embeddings)} embeddings")
    if failed_images:
        print(f"⚠️  Failed to process {len(failed_images)} images")
    
    # Build FAISS index
    print("\n🔨 Building FAISS index...")
    embeddings_array = np.array(embeddings)
    
    vector_store = VectorStore()
    vector_store.build_index(embeddings_array, metadata)
    
    print("✅ Index built and saved!")
    print(f"   - Index: data/faiss_index.idx")
    print(f"   - Metadata: data/metadata.pkl")
    print("\n🚀 Ready to launch the app with: streamlit run app.py")

if __name__ == "__main__":
    build_index()
