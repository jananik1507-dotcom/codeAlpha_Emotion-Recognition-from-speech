import librosa
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

# ----------------------------
# STEP 1: FEATURE EXTRACTION
# ----------------------------
def extract_features(file_path):
    audio, sample_rate = librosa.load(file_path, duration=3, offset=0.5)
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    return np.mean(mfcc.T, axis=0)

# ----------------------------
# STEP 2: LOAD DATASET
# ----------------------------
dataset_path = "dataset"

X = []
y = []

for file in os.listdir(dataset_path):
    if file.endswith(".wav"):
        path = os.path.join(dataset_path, file)

        features = extract_features(path)
        X.append(features)

        # ----------------------------
        # STEP 3: LABELS
        # ----------------------------
        if "happy" in file:
            y.append(0)
        elif "sad" in file:
            y.append(1)
        elif "angry" in file:
            y.append(2)
        else:
            y.append(3)

print("Features extracted:", len(X))

# ----------------------------
# STEP 4: TRAIN MODEL
# ----------------------------
X = np.array(X)
y = np.array(y)

# Check if enough data
if len(X) < 2:
    print("Not enough data to train model!")
    exit()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500)
model.fit(X_train, y_train)

print("Training completed")
print("Accuracy:", model.score(X_test, y_test))

# ----------------------------
# STEP 5: PREDICTION FUNCTION
# ----------------------------
def predict(file_path):
    feature = extract_features(file_path)
    feature = np.array(feature).reshape(1, -1)

    result = model.predict(feature)

    emotions = ["Happy", "Sad", "Angry", "Neutral"]
    print("Predicted Emotion:", emotions[result[0]])

# ----------------------------
# STEP 6: TEST PREDICTION (OPTIONAL)
# ----------------------------
# predict("dataset/happy1.wav")