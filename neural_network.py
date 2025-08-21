import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, SimpleRNN, Embedding
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Fraud Detection with Banknotes

print("\n--- Fraud Detection ---")

# Load dataset
df = pd.read_csv("banknotes.csv")
X = df.drop("class", axis=1).values
y = df["class"].values

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build Neural Network
fraud_model = Sequential([
    Dense(16, activation="relu", input_shape=(X_train.shape[1],)),
    Dense(8, activation="relu"),
    Dense(1, activation="sigmoid")
])

fraud_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
fraud_model.fit(X_train, y_train, epochs=10, validation_split=0.2, verbose=1)

loss, acc = fraud_model.evaluate(X_test, y_test, verbose=0)
print(f"Fraud Detection Accuracy: {acc:.4f}")

# Edge Detection (Computer Vision)

print("\n--- Edge Detection ---")

image = cv2.imread("edgedetection_input.input", cv2.IMREAD_GRAYSCALE)
edges = cv2.Canny(image, 100, 200)

cv2.imwrite("edgedetection_output.png", edges)
print("Edge detection result saved as edgedetection_output.png")

# Convolutional Neural Network (MNIST)

print("\n--- CNN on MNIST ---")

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalize and reshape
X_train = X_train.reshape(-1, 28, 28, 1) / 255.0
X_test = X_test.reshape(-1, 28, 28, 1) / 255.0

# Build CNN
cnn_model = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(28,28,1)),
    MaxPooling2D((2,2)),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation="relu"),
    Dense(10, activation="softmax")
])

cnn_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
cnn_model.fit(X_train, y_train, epochs=3, validation_split=0.1, verbose=1)

loss, acc = cnn_model.evaluate(X_test, y_test, verbose=0)
print(f"MNIST CNN Accuracy: {acc:.4f}")

# Save model (directory only, no .keras extension)
cnn_model.save("MNIST_cnn_model")
print("CNN model saved as MNIST_cnn_model/")


