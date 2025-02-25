## Cardiovascular Risk Prediction

This project aims to predict an individual's risk of developing cardiovascular disease using machine learning techniques. It utilizes an Artificial Neural Network (ANN) trained on a dataset with 10 health parameters.

1. Project Overview
Cardiovascular disease is one of the leading causes of death globally. This project employs machine learning techniques to predict whether an individual is at risk of developing cardiovascular disease based on key health indicators.

2. Dataset
The dataset (cardio_train.csv) contains health-related features for individuals, with a target variable (Cardio_disease) indicating the presence (1) or absence (0) of cardiovascular disease.

Features Used
Age (converted from days to years)
Gender (1 = Female, 2 = Male)
BMI (calculated from height and weight)
Systolic Blood Pressure (ap_hi)
Diastolic Blood Pressure (ap_lo)
Smoking Status (binary)
Alcohol Consumption Status (binary)
Physical Activity Level (binary)
Blood Glucose Level (categorical: normal, above normal, well above normal)
Blood Cholesterol Level (categorical: normal, above normal, well above normal)
3. Model Implementation
The prediction model is built using TensorFlow and Keras, following these steps:

Data Preprocessing
The BMI feature is computed before dropping height and weight.
id is dropped as it is not relevant.
Age is converted from days to years.
Standardization is applied to continuous numerical features (age, BMI, ap_hi, ap_lo).
The dataset is split into 80% training and 20% testing while ensuring class balance.
Model Architecture
A fully connected Artificial Neural Network (ANN) is used:

Input Layer: Accepts all preprocessed features.
Hidden Layers:
Dense (16 neurons, ReLU activation)
Dropout (0.2) (to prevent overfitting)
Dense (32 neurons, ReLU activation)
Dropout (0.2)
Dense (16 neurons, ReLU activation)
Output Layer: A single neuron with sigmoid activation for binary classification.
Training Details
Optimizer: Adam
Loss Function: Binary Crossentropy
Metrics: Accuracy
Epochs: 50
Batch Size: 32
Model Performance
The training history is plotted to visualize accuracy trends. The final accuracy is printed after evaluation.

4. Installation and Setup
To run this project, install the required dependencies using:

sh
Copy
Edit
pip install -r requirements.txt
Requirements
Python 3.x
NumPy
Pandas
Matplotlib
TensorFlow
Scikit-learn
Joblib
5. Running the Project
Run the following command to execute the script:

sh
Copy
Edit
python risk_pre.py
6. Results and Visualizations
The trained model's accuracy is displayed in the terminal.
A plot of training accuracy vs. validation accuracy is generated.
7. Future Improvements
Hyperparameter tuning for better accuracy.
Consider feature engineering for more precise predictions.
Implement a web interface for user-friendly predictions.
