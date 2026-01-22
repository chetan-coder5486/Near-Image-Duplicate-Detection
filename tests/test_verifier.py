from src.verifier import SSCDVerifier
from src.config import SSCD_MODEL_PATH

verifier = SSCDVerifier(SSCD_MODEL_PATH)

img1_path = "data/raw/copydays/original/201200.jpg"
img2_path = "data/raw/copydays/strong/201201.jpg"

v1 = verifier.get_embedding(img1_path)
v2 = verifier.get_embedding(img2_path)

similarity = v1 @ v2
print("Similarity:", similarity)
