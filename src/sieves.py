from PIL import Image
import imagehash
from typing import Dict, List, Tuple

def compute_dhash(image: Image.Image, hash_size: int = 8) -> str:
    """
    Compute 64-bit Difference Hash (dHash) for an image.

    Args:
        image (PIL.Image): Input image
        hash_size (int): Hash size (default 8 -> 64-bit)

    Returns:
        str: Hexadecimal hash string
    """
    # Convert to grayscale for robustness
    image = image.convert("L")

    # imagehash internally resizes to (hash_size+1) x hash_size (9x8)
    dhash = imagehash.dhash(image, hash_size=hash_size)

    return str(dhash)


def hamming_distance(hash1: str, hash2: str) -> int:
    """
    Compute Hamming distance between two hexadecimal hashes.

    Args:
        hash1 (str): First hash
        hash2 (str): Second hash

    Returns:
        int: Number of differing bits
    """
    h1 = imagehash.hex_to_hash(hash1)
    h2 = imagehash.hex_to_hash(hash2)

    return h1 - h2


def find_near_duplicates(
    query_hash: str,
    hash_db: Dict[str, str],
    max_distance: int = 3
) -> List[Tuple[str, int]]:
    """
    Find near-duplicate images using Hamming distance.

    Args:
        query_hash (str): dHash of query image
        hash_db (dict): {image_id: dHash}
        max_distance (int): Allowed Hamming distance

    Returns:
        list: [(image_id, distance), ...]
    """
    matches = []

    for img_id, db_hash in hash_db.items():
        dist = hamming_distance(query_hash, db_hash)
        if dist <= max_distance:
            matches.append((img_id, dist))

    # Sort by similarity (lowest distance = best match)
    return sorted(matches, key=lambda x: x[1])


def is_duplicate(
    image: Image.Image,
    hash_db: Dict[str, str],
    threshold: int = 3
) -> Tuple[bool, List[Tuple[str, int]]]:
    """
    Full Sieve check: image -> hash -> compare -> decision.

    Args:
        image (PIL.Image): Query image
        hash_db (dict): Stored hashes
        threshold (int): Hamming distance threshold

    Returns:
        (bool, list): (is_duplicate, matches)
    """
    query_hash = compute_dhash(image)
    matches = find_near_duplicates(query_hash, hash_db, threshold)

    return (len(matches) > 0), matches
