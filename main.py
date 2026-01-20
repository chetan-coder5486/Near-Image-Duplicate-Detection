from src.verifier import SSCDVerifier
from src.config import SSCD_SIM_THRESHOLD
verifier = SSCDVerifier()
query_vec = verifier.get_embedding(query_image)

# later passed to FAISS (Step 5)
