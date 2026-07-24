#!/bin/bash
echo "=========================================="
echo "👗 Fashion Recommender System Setup"
echo "=========================================="

# Create directories
mkdir -p data models uploads

# Install dependencies
pip install -r requirements.txt

echo ""
echo "✅ Setup complete!"
echo "1. Add images to 'images' folder"
echo "2. Run: python feature_extractor.py"
echo "3. Run: streamlit run app.py"
echo "=========================================="