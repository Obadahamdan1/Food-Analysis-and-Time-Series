import os
import pickle
import json
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, ResNet50


IMAGE_PATH = "C:\\New folder\\food_Classification\\Steak.jpg" # Change this to any image path
MODEL_PATH = "svm_model.pkl"
CLASS_INDEX_PATH = "class_indices.json"
#Load Class Indices
with open(CLASS_INDEX_PATH, 'r') as f:
    class_indices = json.load(f)

print(f"Class Indices: {class_indices}")
index_to_class = {v: k for k, v in class_indices.items()}
#LOAD MODEL AND CLASS INDICES
with open(MODEL_PATH, 'rb') as f:
    svm = pickle.load(f)

with open(CLASS_INDEX_PATH, 'r') as f:
    class_indices = json.load(f)

# Reverse the class index dictionary for mapping class indices to class names
index_to_class = {v: k for k, v in class_indices.items()}
#Calorie estimates per serving
calorie_dict = {
    "bread": 265,            # calories per 100g
    "Dairy product": 150,    # average for items like cheese/yogurt
    "Dessert": 350,          # average slice of cake/pie
    "Egg": 155,              # per 100g (approx 2 large eggs)
    "Fried Food": 312,       # general fried snack
    "Meat": 250,             # average for grilled meat
    "Rice": 130,             # per 100g of cooked rice
    "Seafood": 200           # average for cooked seafood
}

#LOAD RESNET50 FOR FEATURE EXTRACTION
resnet_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

#IMAGE PREPROCESSING FUNCTION
def load_and_preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

#PROCESS AND PREDICT
img_tensor = load_and_preprocess_image(IMAGE_PATH)

#Debugging: check if the image is loaded and preprocessed correctly
print(f"Image Tensor Shape: {img_tensor.shape}")

#PROCESS AND PREDICT WITH PROBABILITIES
from sklearn.preprocessing import LabelBinarizer

#Extract features
features = resnet_model.predict(img_tensor, verbose=0)

#Get decision scores for all classes
#Get probability scores
probas = svm.predict_proba(features)

#Get top 3 predictions
top_indices = np.argsort(probas[0])[::-1][:3]
print("\nTop 3 Predictions with Probabilities:")
for idx in top_indices:
    label = index_to_class.get(idx, "Unknown")
    prob = probas[0][idx]
    print(f"{label}: {prob:.2%}")

#Calorie estimate for top-1 prediction
top_label = index_to_class[top_indices[0]]
estimated_calories = calorie_dict.get(top_label, "N/A")
print(f"\nEstimated Calories for '{top_label}': {estimated_calories} kcal per serving(100 gram)")

if probas[0][top_indices[0]] < 0.8:
    print("Low confidence prediction — double-check the image.")
