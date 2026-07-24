# ==========================================
# recommend.py - Fashion Recommender Functions
# ==========================================

import pickle
import numpy as np
from sklearn.neighbors import NearestNeighbors
import streamlit as st


# ==========================================
# Load Data Function
# ==========================================

@st.cache_data
def load_data():
    """
    Load feature embeddings and filenames from models folder.

    Returns:
        tuple: (feature_list, filenames)
    """
    try:
        with open('models/embeddings.pkl', 'rb') as f:
            feature_vectors = np.array(pickle.load(f))
        with open('models/filenames.pkl', 'rb') as f:
            filenames = pickle.load(f)
        return feature_vectors, filenames
    except FileNotFoundError:
        return None, None


# ==========================================
# Recommendation Function
# ==========================================

def get_recommendations(feature_vector, feature_vectors, n=5):
    """
    Args:
        features: Feature vector of uploaded image
        feature_list: List of all feature vectors
        n: Number of recommendations

    Returns:
        list: Indices of recommended items
    """
    neighbors = NearestNeighbors(
        n_neighbors=n + 1,  # +1 because first is the query image itself
        algorithm='brute',
        metric='euclidean'
    )
    neighbors.fit(feature_vectors)

    distances, indices = neighbors.kneighbors([feature_vector])

    # Skip the first one (query image itself)
    return indices[0][1:]