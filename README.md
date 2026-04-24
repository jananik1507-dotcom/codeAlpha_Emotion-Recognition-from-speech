Task 9: Emotional Speech Recognition

Objective

To build a system that detects human emotions from speech/audio input using Python.




Description

This project analyzes audio signals and predicts the emotional state of a speaker such as:

Happy

Sad

Angry

Neutral

Fearful


It uses audio processing and machine learning techniques to classify emotions.




Features

Accepts audio input (.wav files)

Extracts audio features (MFCC, pitch, energy)

Predicts emotion from speech

Displays predicted emotion in output





Requirements

Install required libraries:

pip install numpy
pip install librosa
pip install scikit-learn
pip install soundfile




How to Run

python emotion_recognition.py




Working Process

1. Load audio file


2. Extract features using Librosa


3. Train or load ML model


4. Predict emotion class


5. Display result






Sample Output

Input Audio: sample.wav
Predicted Emotion: Happy




Example Code Flow

import librosa
import numpy as np

file = "sample.wav"
audio, sr = librosa.load(file)
features = librosa.feature.mfcc(y=audio, sr=sr)
print("Features extracted")




Error Handling

Handles invalid audio file formats

Handles missing or corrupted files

Handles empty audio input





Skills Learned

Audio signal processing

Feature extraction (MFCC)

Machine learning basics

Data classification




Output

The system prints detected emotion in the terminal after processing audio input.