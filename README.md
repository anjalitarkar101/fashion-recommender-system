# 👗 Fashion Recommender System

## 📖 Overview
A **Fashion Recommender System** built with Streamlit that suggests similar fashion items based on uploaded images. The system uses **ResNet50** (pre-trained on ImageNet) for feature extraction and **K-Nearest Neighbors (KNN)** for finding similar items.

---

## ✨ Features
- 👗 Image Upload - Upload any fashion item image
- 🔍 Feature Extraction - Uses ResNet50 to extract 2048-dimensional feature vectors
- 🎯 Smart Recommendations - Finds top 5 similar items using KNN
- 🖼️ Visual Results - Displays recommended items with thumbnails
- 🎨 Clean UI - User-friendly interface with sidebar information

---

## 🛠️ Technologies Used
- **Python 3.10+**  - Core programming language
- **TensorFlow/Keras**  - Deep learning framework
- **ResNet50**  - Pre-trained model for feature extraction
- **Streamlit**  - Web application framework
- **NumPy**  - Numerical operations
- **Scikit-learn**  - KNN for similarity search
- **Pillow**  - Image processing
- **OpenCV**  - Image handling

---

## 📁 Project Structure
```
fashion-recommender-system/
├── app.py                    # Main Streamlit application (UI)
├── recommend.py              # Recommendation functions
├── feature_extractor.py      # Feature extraction script
├── requirements.txt          # Python dependencies
├── setup.sh                  # Setup script
├── .gitignore               # Git ignore file
├── data/                     # Fashion images (gitignored)
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── models/                   # Generated files (gitignored)
│   ├── embeddings.pkl
│   └── filenames.pkl
├── uploads/                  # Temporary uploads (gitignored)
└── README.md                 # Project documentation
```

## 🚀 Installation & Setup

### Prerequisites
- Python 3.10 or higher
- pip package manager

### Step 1: Clone the Repository
```bash
git clone https://github.com/anjalitarkar101/fashion-recommender-system.git
cd fashion-recommender
```

### Step 2: Run Setup Script
```bash
chmod +x setup.sh
./setup.sh
```

This will:
- Create required directories (data/, models/, uploads/)
- Install all dependencies

### Step 3: Download Dataset
- **Source:** Kaggle
- **Link:** https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset
- **Name:** Fashion Product Images Dataset
- **Files:** 44,441 fashion product images
- **Categories:** Dresses, T-shirts, Jeans, Bags, Shoes, and more

After downloading, extract and place the image files in the `data/` folder:
```
data/
├── image1.jpg
├── image2.jpg
└── ... (all fashion images)
```

### Step 4: Extract Features
```bash
python feature_extractor.py
```

This will:
- Load ResNet50 model
- Extract features from all images
- Save embeddings and filenames to models/ folder

### Step 5: Run the Application
```bash
streamlit run app.py
Open your browser and navigate to http://localhost:8501
```

---

## 📊 How It Works

**Feature Extraction**
- Loads image and resizes to 224x224
- Passes through ResNet50 (without top layers)
- Extracts 2048-dimensional feature vector
- Normalizes the vector using L2 normalization

**Similarity Search**
- Uses KNN (K-Nearest Neighbors) algorithm
- Measures similarity using Euclidean Distance
- Returns top 5 most similar items

**Recommendation**
- User uploads an image
- System extracts its feature vector
- Finds nearest neighbors in the feature space
- Displays similar items with thumbnails

---

## 🔧 Dependencies
```txt
tensorflow==2.13.0
numpy==1.24.3
opencv-python==4.8.1.78
Pillow==10.0.0
scikit-learn==1.3.0
tqdm==4.66.0
streamlit==1.28.0
```

---

## 📝 Usage Guide
1. Click "Choose a fashion image" to upload your image
2. Wait for the system to process the image
3. View the top 5 similar fashion items
4. Explore different items to discover new styles!

---

## 📄 License
This project is licensed under the MIT License.

© 2026 Anjali Tarkar. All rights reserved.

---

## 👩‍💻 Author
**Anjali Tarkar**
- GitHub: https://github.com/anjalitarkar101
- Email: anjalitarkar101@gmail.com

---

## ⭐ Show Your Support
If you find this project useful, please give it a star on GitHub!

---

## 🙏 Acknowledgments
- Param Aggarwal - For the Fashion Product Images Dataset on Kaggle
- ResNet50 - Pre-trained model from Keras Applications
- Streamlit - For the awesome web framework
- Scikit-learn - For KNN implementation

