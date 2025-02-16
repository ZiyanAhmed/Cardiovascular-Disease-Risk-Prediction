import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load Dataset
data = pd.read_csv('cardio_train.csv', sep=';')

# Calculate BMI before dropping 'weight' and 'height'
data['BMI'] = (data['weight'] / ((data['height'] / 100) ** 2)).astype(int)  # Calculate BMI

# Drop unnecessary columns
data.drop(columns=['id', 'height', 'weight'], inplace=True)

# Convert age from days to years
data['age'] = (data['age'] / 365).astype(int)

# Rename target column
data.rename(columns={'cardio': 'Cardio_disease'}, inplace=True)

# Check for missing values before dropping
print("Missing Values Before Dropping:", data.isnull().sum())

# Drop missing values if any
data.dropna(inplace=True)

# Standardize Features
feature_columns = ['age', 'BMI', 'ap_hi', 'ap_lo']
scaler = StandardScaler()
data[feature_columns] = scaler.fit_transform(data[feature_columns])

# Split Data
X = data.drop(columns=['Cardio_disease'])
y = data['Cardio_disease']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Build Neural Network Model
model = Sequential([
    Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary classification
])

# Compile Model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train Model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# Evaluate Model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy:.4f}')

# Plot Training History
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Model Training Performance')
plt.show()
