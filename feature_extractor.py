# ==================================================
# feature_extractor.py - Fashion Recommender System
# ==================================================

import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import numpy as np
from numpy.linalg import norm
import os
import pickle
from tqdm import tqdm


def load_model():

    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False
    model = tf.keras.Sequential([base_model, GlobalMaxPooling2D()])
    return model

def extract_features(img_path, model):

    # Step 1: Load image and resize to 224x224 pixels
    img = image.load_img(img_path, target_size=(224, 224))

    # Step 2: Convert PIL image to numpy array
    img_array = image.img_to_array(img)

    # Step 3: Add batch dimension
    expanded_img_array = np.expand_dims(img_array, axis=0)

    # Step 4: Preprocess for ResNet50
    preprocessed_img = preprocess_input(expanded_img_array)

    # Step 5: Extract features using the model
    result = model.predict(preprocessed_img, verbose=0)

    # Step 6: Flatten the output to 1D vector
    flattened_result = result.flatten()

    # Step 7: Normalize the feature vector
    normalized_result = flattened_result / norm(flattened_result)
    return normalized_result


def main():
    print("=" * 60)
    print("👗 Fashion Recommender - Feature Extractor")
    print("=" * 60)

    # Load model
    print("🔄 Loading ResNet50 model...")
    model = load_model()
    print(f"✅ Model loaded! Feature vector size: {model.output_shape[-1]}")

    # Get filenames
    print("🔄 Getting all image filenames...")
    image_folder = 'data'
    filenames = []
    for file in os.listdir(image_folder):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            filenames.append(os.path.join(image_folder, file))

    # Save filenames
    with open('models/filenames.pkl', 'wb') as f:
        pickle.dump(filenames, f)
    print("💾 Saved filenames.pkl")

    # Extract feature vectors
    print("🔄 Extracting feature vectors...")
    feature_vectors = []

    for file in tqdm(filenames, desc="Processing data"):
       feature_vector = extract_features(file, model)
       feature_vectors.append(feature_vector)

    # Save feature vectors
    with open('models/embeddings.pkl', 'wb') as f:
        pickle.dump(np.array(feature_vectors), f)
    print("💾 Saved embeddings.pkl")


    print("\n" + "=" * 60)
    print("📌 Now run: streamlit run app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()