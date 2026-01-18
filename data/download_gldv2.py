import os
import requests
import tarfile
from tqdm import tqdm

# Official S3 Bucket URL for the first chunk of the Training set (approx 500MB - 1GB)
# This contains roughly 10,000 images, perfect for your "Distractor" set.
URL = "https://s3.amazonaws.com/google-landmark/train/images_000.tar"
OUTPUT_DIR = "data/raw/distractors"

def download_chunk():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    tar_path = os.path.join(OUTPUT_DIR, "images_000.tar")
    
    # 1. Download the Tar file
    print(f"Downloading GLDv2 Chunk 0 from {URL}...")
    try:
        response = requests.get(URL, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(tar_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True) as t:
                for chunk in response.iter_content(chunk_size=4096):
                    t.update(len(chunk))
                    f.write(chunk)
                    
        print("Download complete. Extracting...")
        
        # 2. Extract images
        with tarfile.open(tar_path) as tar:
            # The tar contains folders like 0/1/2/image.jpg
            # We want to flatten this into data/raw/distractors/image.jpg
            for member in tqdm(tar.getmembers(), desc="Extracting"):
                if member.isfile() and member.name.endswith('.jpg'):
                    # Flatten the path
                    member.name = os.path.basename(member.name) 
                    tar.extract(member, OUTPUT_DIR)
        
        # 3. Cleanup
        os.remove(tar_path)
        print(f"Success! Images ready in {OUTPUT_DIR}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    download_chunk()