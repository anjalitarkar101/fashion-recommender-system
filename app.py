# ==========================================
# app.py - Fashion Recommender System
# ==========================================

import streamlit as st
import os
from PIL import Image
from feature_extractor import load_model, extract_features
from recommend import load_data, get_recommendations

# ==========================================
# Page Configuration
# ==========================================

st.set_page_config(
    page_title="Fashion Recommender System",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# Sidebar
# ==========================================

with st.sidebar:
    st.markdown("### 👗 About This App")
    st.markdown("""
        This app recommends similar fashion items using Deep Learning.

        **How it works:**
        1. 📤 Upload a fashion image
        2. 🔍 AI extracts features 
        3. 🎯 Finds similar items 
        4. 📊 Shows top 5 recommendations
    """)

    st.markdown("---")

    st.markdown("### 🧠 Model Info")
    st.markdown("""
        - **Model:** ResNet50 (ImageNet weights)
        - **Algorithm:** K-Nearest Neighbors (KNN)
        - **Similarity Metric:** Euclidean Distance
        - **Feature Vector:** 2048 dimensions
    """)

# ==========================================
# Main Content
# ==========================================

st.title("👗 Fashion Recommender System")
st.markdown("Upload an image to find similar fashion items!")

# Load model
with st.spinner("🔄 Loading ResNet50 model..."):
    model = load_model()

# Load data
with st.spinner("🔄 Loading fashion database..."):
    feature_vectors, filenames = load_data()

# Check if data loaded
if feature_vectors is None or filenames is None:
    st.error("❌ Feature files not found! Please run: python feature_extractor.py")
    st.stop()

st.success(f"✅ Loaded {len(filenames)} fashion items successfully!")

# ==========================================
# Upload Section
# ==========================================

st.markdown("---")

uploaded_file = st.file_uploader(
    "📤 Choose a fashion image...",
    type=['jpg', 'jpeg', 'png'],
    help="Upload a clear image of a fashion item"
)

if uploaded_file is not None:
    # Create uploads folder if it doesn't exist
    os.makedirs('uploads', exist_ok=True)
    
    # Save uploaded file
    file_path = os.path.join('uploads', uploaded_file.name)
    with open(file_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())

    # Display uploaded image
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("📸 Your Image")
        display_image = Image.open(file_path)
        st.image(display_image, width=300)
        st.caption(f"📄 {uploaded_file.name}")
        st.caption(f"📦 {uploaded_file.size / 1024:.1f} KB")

    # Find recommendations
    with st.spinner("🔍 Finding similar items..."):
        feature_vector = extract_features(file_path, model)
        indices = get_recommendations(feature_vector, feature_vectors, n=5)

    # Display recommendations
    with col2:
        st.subheader("🛍️ Similar Items")
        cols = st.columns(5)

        for idx, col in enumerate(cols):
            if idx < len(indices):
                try:
                    img_path = filenames[indices[idx]]
                    col.image(img_path, use_column_width=True)
                    col.caption(f"Item {idx + 1}")
                except:
                    col.error("Not found")
            else:
                col.caption("No more")

else:
    st.info("👆 Upload an image to get recommendations!")

# ==========================================
# Footer
# ==========================================

st.markdown("---")
st.caption("🔍 Powered by ResNet50")