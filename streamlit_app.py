import sys
from pathlib import Path
import streamlit as st
from PIL import Image
import io
import base64

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
    /* Disable page scrolling completely */
    html, body {
        overflow: hidden !important;
        height: 100vh;
        margin: 0;
        padding: 0;
    }
    
    [data-testid="stAppViewContainer"] { 
        overflow: hidden !important;
        height: 100vh;
        max-height: 100vh;
    }
    
    .main { 
        padding: 0.5rem 1rem !important;
        overflow: hidden !important;
        height: 100vh;
        max-height: 100vh;
    }
    
    section[data-testid="stVerticalBlock"] {
        overflow: hidden !important;
        gap: 0.5rem !important;
    }
    
    [data-testid="column"] {
        overflow: hidden !important;
    }
    
    h1 { 
        margin: 0 0 0.5rem 0 !important; 
        font-size: 1.8rem !important;
        padding: 0 !important;
    }
    
    h3 {
        margin: 0 0 0.5rem 0 !important;
        font-size: 1.1rem !important;
    }
    
    .stButton > button {
        margin-top: 0.5rem !important;
    }
    
    /* Fixed height results container - CRITICAL */
    .results-container {
        height: calc(100vh - 140px) !important;
        overflow-y: auto !important;
        overflow-x: hidden !important;
        padding: 10px;
        background: #f8f9fa;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
    }
    
    /* Custom scrollbar */
    .results-container::-webkit-scrollbar {
        width: 8px;
    }
    
    .results-container::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    
    .results-container::-webkit-scrollbar-thumb {
        background: #888;
        border-radius: 10px;
    }
    
    .results-container::-webkit-scrollbar-thumb:hover {
        background: #555;
    }
    
    /* Medium sized result images */
    .result-image {
        max-width: 100%;
        max-height: 280px;
        object-fit: contain;
        margin: 0 auto 8px auto;
        display: block;
        border-radius: 8px;
        background: white;
        padding: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .result-item {
        margin-bottom: 20px;
        text-align: center;
        background: white;
        padding: 10px;
        border-radius: 8px;
    }
    
    .score-badge {
        display: inline-block;
        padding: 5px 12px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 20px;
        font-weight: 600;
        font-size: 13px;
        margin: 5px 0;
    }
    
    .filename-text {
        color: #666;
        font-size: 11px;
        margin-top: 5px;
        word-break: break-all;
    }
    
    /* Hide streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Remove extra spacing */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 0 !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("🔍 Duplicate Finder")

col1, col2 = st.columns([1, 1.2], gap="medium")

with col1:
    st.markdown("### Upload")
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed")
    
    if uploaded_file is not None:
        original_image = Image.open(uploaded_file)
        
        # Create thumbnail for display (max 250px height)
        display_image = original_image.copy()
        display_image.thumbnail((400, 250), Image.Resampling.LANCZOS)
        
        # Display the thumbnail
        st.image(display_image)
        
        # Save original image for processing
        upload_dir = Path(UPLOAD_DIR)
        upload_dir.mkdir(parents=True, exist_ok=True)
        temp_path = upload_dir / uploaded_file.name
        original_image.save(temp_path)
        
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
                # Build complete HTML in one block to avoid Streamlit spacing
                html_parts = ['<div class="results-container">']
                
                for match in valid_matches:
                    filename = Path(match["filename"]).name
                    score = match["score"]
                    img = Image.open(match["filename"])
                    
                    # Convert image to base64
                    buffered = io.BytesIO()
                    img.save(buffered, format="JPEG")
                    img_str = base64.b64encode(buffered.getvalue()).decode()
                    
                    # Add item to HTML
                    html_parts.append(
                        f'<div class="result-item">'
                        f'<img src="data:image/jpeg;base64,{img_str}" class="result-image">'
                        f'<div><span class="score-badge">Score: {score:.3f}</span></div>'
                        f'<div class="filename-text">{filename}</div>'
                        f'</div>'
                    )
                
                html_parts.append('</div>')
                
                # Render as single HTML block
                st.markdown(''.join(html_parts), unsafe_allow_html=True)
