#importing libraries
import os
import shutil
import numpy as np
from tqdm import tqdm #shows progress bars
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50 #pretrained CNN 
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import kagglehub
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import label_binarize

#SETUP
#selecting classes uploading dataset
SELECTED_CLASSES = ['bread', 'Dairy product', 'Dessert', 'Egg', 'Fried Food','Meat','Rice','Seafood']
NUM_IMAGES_PER_CLASS = 300
DATASET_PATH = kagglehub.dataset_download("trolukovich/food11-image-dataset")
print("Dataset downloaded to:", DATASET_PATH)

#Raw folders
#folders paths(Raw)
ORIGINAL_TRAIN_DIR = os.path.join(DATASET_PATH, "training")
ORIGINAL_VAL_DIR = os.path.join(DATASET_PATH, "validation")
ORIGINAL_EVAL_DIR = os.path.join(DATASET_PATH, "evaluation")

# Filtered folders (for 5-class setup)
#folders paths(filtered)
BASE_OUTPUT = "filtered_food11"
FILTERED_TRAIN = os.path.join(BASE_OUTPUT, "train")
FILTERED_VAL = os.path.join(BASE_OUTPUT, "val")
FILTERED_TEST = os.path.join(BASE_OUTPUT, "test")

#function to filter dataset into 8 classes, 300 images
def create_filtered_dataset(src, dst, classes, max_per_class):
    if os.path.exists(dst):
        shutil.rmtree(dst)
    os.makedirs(dst)

    for cls in classes:
        src_class_dir = os.path.join(src, cls)
        dst_class_dir = os.path.join(dst, cls)
        os.makedirs(dst_class_dir)
        images = os.listdir(src_class_dir)[:max_per_class]
        for img in images:
            shutil.copy(os.path.join(src_class_dir, img), os.path.join(dst_class_dir, img))


#Create filtered datasets
create_filtered_dataset(ORIGINAL_TRAIN_DIR, FILTERED_TRAIN, SELECTED_CLASSES, NUM_IMAGES_PER_CLASS)
create_filtered_dataset(ORIGINAL_VAL_DIR, FILTERED_VAL, SELECTED_CLASSES, NUM_IMAGES_PER_CLASS)
create_filtered_dataset(ORIGINAL_EVAL_DIR, FILTERED_TEST, SELECTED_CLASSES, NUM_IMAGES_PER_CLASS)

#LOAD DATA with preprocessing
datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

#resizes them to 224x224, ResNet-specific preprocessing 
train_data = datagen.flow_from_directory(FILTERED_TRAIN, target_size=(224, 224), batch_size=32, shuffle=False)
val_data = datagen.flow_from_directory(FILTERED_VAL, target_size=(224, 224), batch_size=32, shuffle=False)
test_data = datagen.flow_from_directory(FILTERED_TEST, target_size=(224, 224), batch_size=32, shuffle=False)

#FEATURE EXTRACTION
resnet_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

#function to pass batches through ResNet to extract deep features.
def extract_features(generator):
    features = []
    for _ in tqdm(range(len(generator))):
        batch, _ = next(generator)
        batch_features = resnet_model.predict(batch, verbose=0)
        features.append(batch_features)
    return np.vstack(features)

X_train = extract_features(train_data)
X_val = extract_features(val_data)
X_test = extract_features(test_data)

y_train = train_data.classes
y_val = val_data.classes
y_test = test_data.classes

#TRAIN + EVALUATE SVM
svm = SVC(kernel='linear', probability=True)
svm.fit(X_train, y_train)

y_val_pred = svm.predict(X_val)
y_test_pred = svm.predict(X_test)

# Metrics for SVM
print(f"SVM Validation Accuracy: {accuracy_score(y_val, y_val_pred):.4f}")
print(f"SVM Test Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
print(f"SVM Precision: {precision_score(y_test, y_test_pred, average='weighted'):.4f}")
print(f"SVM Recall: {recall_score(y_test, y_test_pred, average='weighted'):.4f}")
print(f"SVM F1-Score: {f1_score(y_test, y_test_pred, average='weighted'):.4f}")
print(f"SVM AUC-ROC: {roc_auc_score(label_binarize(y_test, classes=np.unique(y_train)), svm.predict_proba(X_test), average='macro', multi_class='ovr'):.4f}")

#Confusion Matrix for SVM
cm = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=train_data.class_indices.keys(), yticklabels=train_data.class_indices.keys())
plt.title("SVM Confusion Matrix")
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

#TRAIN + EVALUATE MLP
mlp = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=300, random_state=42)
mlp.fit(X_train, y_train)

y_val_pred_mlp = mlp.predict(X_val)
y_test_pred_mlp = mlp.predict(X_test)

#Metrics for MLP
print(f"MLP Validation Accuracy: {accuracy_score(y_val, y_val_pred_mlp):.4f}")
print(f"MLP Test Accuracy: {accuracy_score(y_test, y_test_pred_mlp):.4f}")
print(f"MLP Precision: {precision_score(y_test, y_test_pred_mlp, average='weighted'):.4f}")
print(f"MLP Recall: {recall_score(y_test, y_test_pred_mlp, average='weighted'):.4f}")
print(f"MLP F1-Score: {f1_score(y_test, y_test_pred_mlp, average='weighted'):.4f}")
print(f"MLP AUC-ROC: {roc_auc_score(label_binarize(y_test, classes=np.unique(y_train)), mlp.predict_proba(X_test), average='macro', multi_class='ovr'):.4f}")

#Confusion Matrix for MLP
cm_mlp = confusion_matrix(y_test, y_test_pred_mlp)
plt.figure(figsize=(10, 7))
sns.heatmap(cm_mlp, annot=True, fmt='d', cmap='Blues', xticklabels=train_data.class_indices.keys(), yticklabels=train_data.class_indices.keys())
plt.title("MLP Confusion Matrix")
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

#TRAIN + EVALUATE Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

y_val_pred_rf = rf.predict(X_val)
y_test_pred_rf = rf.predict(X_test)

#Metrics for Random Forest
print(f"Random Forest Validation Accuracy: {accuracy_score(y_val, y_val_pred_rf):.4f}")
print(f"Random Forest Test Accuracy: {accuracy_score(y_test, y_test_pred_rf):.4f}")
print(f"Random Forest Precision: {precision_score(y_test, y_test_pred_rf, average='weighted'):.4f}")
print(f"Random Forest Recall: {recall_score(y_test, y_test_pred_rf, average='weighted'):.4f}")
print(f"Random Forest F1-Score: {f1_score(y_test, y_test_pred_rf, average='weighted'):.4f}")
print(f"Random Forest AUC-ROC: {roc_auc_score(label_binarize(y_test, classes=np.unique(y_train)), rf.predict_proba(X_test), average='macro', multi_class='ovr'):.4f}")

#Confusion Matrix for Random Forest
cm_rf = confusion_matrix(y_test, y_test_pred_rf)
plt.figure(figsize=(10, 7))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', xticklabels=train_data.class_indices.keys(), yticklabels=train_data.class_indices.keys())
plt.title("Random Forest Confusion Matrix")
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

#Save the trained model
import pickle
with open('svm_model.pkl', 'wb') as model_file:
    pickle.dump(svm, model_file)

import json
#   Save class indices
with open('class_indices.json', 'w') as f:
    json.dump(train_data.class_indices, f)
print("Class indices mapping:", train_data.class_indices)