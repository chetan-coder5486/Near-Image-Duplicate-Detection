import os
import requests
import tarfile
from io import BytesIO
from tqdm import tqdm

# Stable mirrors for the INRIA Copydays dataset
URLS = {
    "original": "https://dl.fbaipublicfiles.com/vissl/datasets/copydays_original.tar.gz",
    "strong": "https://dl.fbaipublicfiles.com/vissl/datasets/copydays_strong.tar.gz"
}

BASE_DIR = "data/raw/copydays"

def download_and_extract(url, name):
    output_dir = os.path.join(BASE_DIR, name)
    
    # Check if already exists and not empty
    if os.path.exists(output_dir) and len(os.listdir(output_dir)) > 0:
        print(f"[{name}] already exists at {output_dir}. Skipping.")
        return

    os.makedirs(output_dir, exist_ok=True)
    print(f"Downloading {name} from {url}...")
    
    try:
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        # Download into memory buffer
        buffer = BytesIO()
        with tqdm(total=total_size, unit='B', unit_scale=True) as t:
            for chunk in response.iter_content(chunk_size=8192):
                t.update(len(chunk))
                buffer.write(chunk)
        
        print(f"Extracting {name}...")
        buffer.seek(0)
        with tarfile.open(fileobj=buffer, mode="r:gz") as tar:
            # Flatten directory structure if needed
            tar.extractall(path=output_dir)
            
        print(f"Success! Saved to {output_dir}")

    except Exception as e:
        print(f"Error processing {name}: {e}")

def main():
    print("Setting up INRIA Copydays dataset...")
    for key, url in URLS.items():
        download_and_extract(url, key)
    print("Done.")

if __name__ == "__main__":
    main()