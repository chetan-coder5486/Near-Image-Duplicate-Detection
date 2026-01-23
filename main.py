from src.config import IMAGE_DIR, TOP_K
from src.pipeline import DuplicateDetector, build_hash_db


def run_example(test_image: str) -> None:
    hash_db = build_hash_db(IMAGE_DIR)
    detector = DuplicateDetector(image_dir=IMAGE_DIR, hash_db=hash_db)
    result = detector.detect(test_image, top_k=TOP_K)
    print(result)


if __name__ == "__main__":
    demo_image = "data/raw/copydays/strong/214402.jpg"
    run_example(demo_image)
