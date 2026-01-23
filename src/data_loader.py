from PIL import Image
import os

def load_image(image_path: str) -> Image.Image:
    """
    Standardized image loader for the entire project.
    - Handles file not found errors.
    - Converts all images to RGB (fixes PNG transparency/Grayscale crashes).
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at: {image_path}")

    try:
        img = Image.open(image_path)
        
        # Crucial: Convert to RGB. 
        # Deep Learning models crash on Grayscale (1 channel) or RGBA (4 channels).
        if img.mode!= 'RGB':
            img = img.convert('RGB')
            
        return img
    except Exception as e:
        raise ValueError(f"Failed to process image {image_path}: {e}")