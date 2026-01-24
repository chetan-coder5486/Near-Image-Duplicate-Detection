from pathlib import Path
from typing import Dict, List, Optional
from PIL import Image

from src.sieves import compute_dhash, hamming_distance
from src.verifier import SSCDVerifier
from src.indexer import Indexer
from src.config import HASH_HAMMING_THRESHOLD, SSCD_SIM_THRESHOLD, SSCD_MODEL_PATH


def build_hash_db(image_folder: str) -> Dict[str, str]:
    """
    Build an in-memory hash database for quick sieve filtering.
    Returns a mapping of hash -> image path.
    """
    hash_db: Dict[str, str] = {}
    folder = Path(image_folder)

    if not folder.exists():
        raise FileNotFoundError(f"Image folder not found: {folder}")

    for img_path in folder.iterdir():
        if not img_path.is_file():
            continue

        try:
            img = Image.open(img_path).convert("RGB")
            dhash = compute_dhash(img)
            hash_db[dhash] = str(img_path)
        except Exception:
            # Skip unreadable files to keep startup resilient.
            continue

    return hash_db


class DuplicateDetector:
    """
    Two-stage duplicate detector that first performs a dHash sieve and then
    confirms with SSCD + FAISS.
    """

    def __init__(
        self,
        image_dir: str,
        hash_db: Optional[Dict[str, str]] = None,
        verifier: Optional[SSCDVerifier] = None,
        indexer: Optional[Indexer] = None,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.verifier = verifier or SSCDVerifier(SSCD_MODEL_PATH)
        self.indexer = indexer or Indexer()
        self.hash_db = hash_db if hash_db is not None else build_hash_db(image_dir)

    def sieve(self, query_image: Image.Image, query_path: str = None, max_matches: int = 3) -> List[Dict]:
        """
        Run the dHash sieve and return near-duplicate hits sorted by distance.
        Filters out self-matches if query_path is provided.
        """
        matches: List[Dict] = []
        q_hash = compute_dhash(query_image)
        query_resolved = str(Path(query_path).resolve()) if query_path else None

        for db_hash, img_path in self.hash_db.items():
            # Skip self-matches
            if query_resolved and Path(img_path).resolve() == Path(query_resolved).resolve():
                continue
                
            dist = hamming_distance(q_hash, db_hash)
            if dist <= HASH_HAMMING_THRESHOLD:
                matches.append({"filename": img_path, "distance": dist})

        matches.sort(key=lambda x: x["distance"])
        return matches[:max_matches]

    def verify(self, image_path: str, top_k: int = 3) -> List[Dict]:
        """
        Run SSCD + FAISS search for the given image and return top-k results.
        Filters out self-matches.
        """
        query_vec = self.verifier.get_embedding(image_path)
        query_path = str(Path(image_path).resolve())
        
        # Get more results to account for filtering
        results = self.indexer.search(query_vec, k=top_k + 5)
        
        # Filter out self-matches (same file path)
        filtered = [r for r in results if Path(r["filename"]).resolve() != Path(query_path).resolve()]
        
        return filtered[:top_k]

    def detect(self, image_path: str, top_k: int = 3) -> Dict:
        """
        Full funnel duplicate check. Returns structured decision data.
        """
        query_image = Image.open(image_path).convert("RGB")

        sieve_matches = self.sieve(query_image, query_path=image_path, max_matches=top_k)
        if sieve_matches:
            best = sieve_matches[0]
            return {
                "is_duplicate": True,
                "stage": "sieve",
                "match": best["filename"],
                "score": best["distance"],
                "sieve_matches": sieve_matches,
                "verifier_matches": [],
            }

        verifier_matches = self.verify(image_path, top_k=top_k)
        
        # Filter matches that meet the similarity threshold
        valid_matches = [m for m in verifier_matches if m.get("score", 0.0) >= SSCD_SIM_THRESHOLD]
        best = valid_matches[0] if valid_matches else None
        is_duplicate = bool(best)

        return {
            "is_duplicate": is_duplicate,
            "stage": "verifier" if valid_matches else "unique",
            "match": best.get("filename") if best else None,
            "score": best.get("score") if best else None,
            "sieve_matches": sieve_matches,
            "verifier_matches": valid_matches,  # Only return matches above threshold
        }
