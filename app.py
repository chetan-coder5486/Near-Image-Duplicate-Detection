import base64
import os
from mimetypes import guess_type

import numpy as np
import streamlit as st
from PIL import Image

from src.config import SSCD_MODEL_PATH, SSCD_SIM_THRESHOLD
from src.search import VectorStore
from src.verifier import SSCDVerifier

# Page config
st.set_page_config(
    page_title="Image Duplicate Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔍 Near-Image Duplicate Detection")
st.markdown("Upload an image to find similar duplicates in the database")

# Lightweight styling to make the grid/cards consistent
st.markdown(
    """
    <style>
        /* Reduce default padding and create a neutral background */
        .block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
        /* Card grid */
        .result-grid {display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 16px;}
        .result-card {background: #f9fafb; border: 1px solid #e5e7eb; border-radius: 12px; padding: 12px; box-shadow: 0 3px 12px rgba(0,0,0,0.05);}
        .result-card h4 {margin: 0 0 8px 0; font-size: 15px;}
        .result-image {width: 100%; height: 220px; object-fit: cover; border-radius: 10px; border: 1px solid #e5e7eb; background: #fff;}
        .metric-row {display: flex; gap: 8px; margin: 10px 0 6px 0;}
        .metric-pill {padding: 6px 10px; border-radius: 999px; font-size: 12px; font-weight: 600; border: 1px solid #e5e7eb;}
        .metric-green {background: #ecfdf3; color: #166534; border-color: #bbf7d0;}
        .metric-blue {background: #eff6ff; color: #1d4ed8; border-color: #bfdbfe;}
        .meta {font-size: 12px; color: #4b5563; overflow-wrap: anywhere; margin-top: 4px;}
        .missing {height: 220px; display: grid; place-items: center; border: 1px dashed #cbd5e1; border-radius: 10px; color: #6b7280; background: #fff;}
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize session state
if 'verifier' not in st.session_state:
    with st.spinner("Loading SSCD model..."):
        st.session_state.verifier = SSCDVerifier(SSCD_MODEL_PATH)

if 'vector_store' not in st.session_state:
    with st.spinner("Loading vector index..."):
        vector_store = VectorStore()
        try:
            vector_store.load_index()
            st.session_state.vector_store = vector_store
        except FileNotFoundError:
            st.warning("Vector index not found. Please run `build_index.py` first to create it.")
            st.session_state.vector_store = None

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Configuration")
    k = st.slider("Number of duplicates to show", 1, 20, 5)
    similarity_threshold = st.slider(
        "Similarity threshold",
        0.0, 1.0, SSCD_SIM_THRESHOLD,
        help="Only show results with similarity above this threshold"
    )

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Upload Image")
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['jpg', 'jpeg', 'png', 'webp', 'bmp']
    )

with col2:
    st.subheader("🖼️ Preview")
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, use_column_width=True)

# Process uploaded image
if uploaded_file and st.session_state.vector_store:
    st.markdown("---")
    
    # Get embedding for uploaded image
    with st.spinner("Computing embedding..."):
        # Save temp file
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        try:
            query_embedding = st.session_state.verifier.get_embedding(temp_path)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    # Search for similar images
    with st.spinner(f"Searching for {k} nearest duplicates..."):
        results = st.session_state.vector_store.search(query_embedding, k=k*2)
        
        # Filter by similarity threshold
        filtered_results = [
            r for r in results 
            if r['similarity'] >= similarity_threshold
        ][:k]
    
    # Display results
    st.subheader(f"🎯 Found {len(filtered_results)} Similar Images")
    
    def _img_to_base64(img_path: str) -> tuple[str, str]:
        """Return base64 string and mime type for a given image path"""
        try:
            with open(img_path, "rb") as f:
                data = f.read()
            b64 = base64.b64encode(data).decode("utf-8")
            mime, _ = guess_type(img_path)
            return b64, mime or "image/jpeg"
        except Exception:
            return "", "image/jpeg"

    if filtered_results:
        cards_html = ["<div class='result-grid'>"]
        for result in filtered_results:
            img_path = result['metadata'].get('path', '')
            filename = result['metadata'].get('filename', 'N/A')
            b64, mime = _img_to_base64(img_path) if os.path.exists(img_path) else ("", "image/jpeg")
            image_tag = (
                f"<img class='result-image' src='data:{mime};base64,{b64}' alt='{filename}'>"
                if b64 else "<div class='missing'>Image not found</div>"
            )
            cards_html.append(
                f"""
                <div class='result-card'>
                    {image_tag}
                    <div class='metric-row'>
                        <div class='metric-pill metric-green'>Similarity: {result['similarity']:.3f}</div>
                        <div class='metric-pill metric-blue'>Distance: {result['distance']:.3f}</div>
                    </div>
                    <div class='meta'><b>File:</b> {filename}</div>
                    <div class='meta'><b>Path:</b> {img_path}</div>
                </div>
                """
            )
        cards_html.append("</div>")
        st.markdown("\n".join(cards_html), unsafe_allow_html=True)

        # Summary table
        st.subheader("📊 Results Summary")
        summary_data = []
        for i, result in enumerate(filtered_results, 1):
            summary_data.append({
                "Rank": i,
                "Similarity": f"{result['similarity']:.4f}",
                "Distance": f"{result['distance']:.4f}",
                "File": result['metadata'].get('filename', 'N/A')
            })
        st.dataframe(summary_data, use_container_width=True)
    else:
        st.info(f"No images found with similarity >= {similarity_threshold}")

elif uploaded_file and not st.session_state.vector_store:
    st.error("Vector store not initialized. Please run `build_index.py` first.")

else:
    st.info("👆 Upload an image to get started!")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <small>Powered by SSCD (Self-Supervised Copy Detection)</small>
    </div>
    """,
    unsafe_allow_html=True
)
