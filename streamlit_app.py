import sys
from pathlib import Path
import streamlit as st
from PIL import Image

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import IMAGE_DIR, TOP_K, UPLOAD_DIR
from src.pipeline import DuplicateDetector, build_hash_db


@st.cache_resource
def load_detector():
    hash_db = build_hash_db(IMAGE_DIR)
    return DuplicateDetector(image_dir=IMAGE_DIR, hash_db=hash_db)


st.set_page_config(page_title="Duplicate Finder", layout="wide", initial_sidebar_state="collapsed")

# Inject custom CSS
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] { padding-top: 0; }
    .main { padding: 1rem; }
    h1 { margin: 0 0 1.5rem 0; font-size: 2.5rem; }
    [data-testid="stMetric"] { 
        background: none;
        border: none;
    }
    .stImage { border-radius: 8px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

st.title("🔍 Duplicate Finder")

col1, col2 = st.columns([1, 1.2], gap="large")

with col1:
    st.markdown("### Upload")
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed")
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, width="stretch")
        
        upload_dir = Path(UPLOAD_DIR)
        upload_dir.mkdir(parents=True, exist_ok=True)
        temp_path = upload_dir / uploaded_file.name
        image.save(temp_path)
        
        if st.button("Search", use_container_width=True, type="primary"):
            detector = load_detector()
            result = detector.detect(str(temp_path), top_k=TOP_K)
            st.session_state.result = result

with col2:
    if "result" in st.session_state:
        result = st.session_state.result
        
        if result["is_duplicate"]:
            st.markdown("### 🚨 Duplicates Found")
        else:
            st.markdown("### ✅ Similar Images")
        
        if result.get("verifier_matches"):
    # Filter valid images first
         valid_matches = [m for m in result["verifier_matches"] if Path(m["filename"]).exists()]
    
         if valid_matches:
            cols = st.columns(len(valid_matches))
            for col, match in zip(cols, valid_matches):
                with col:
                    img = Image.open(match["filename"])
                    st.image(img, width="stretch")
                    st.caption(f"{match['score']:.3f}")
