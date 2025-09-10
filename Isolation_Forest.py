# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import numpy as np

print("Step 1: Loading the dataset...")
try:
    df = pd.read_csv('C://Users//Shobit//PycharmProjects//Hackathons//pythonProject//dataset//ai4i2020.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: The file 'ai4i2020.csv' was not found.")
    print("Please make sure the dataset is in the same directory as the script.")
    exit()

print("\nStep 2: Preparing data for Isolation Forest (training on normal data only)...")
normal_data = df[df['Machine failure'] == 0].copy()
all_data_for_prediction = df.copy()

features = ['Type', 'Air temperature [K]', 'Process temperature [K]',
            'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']

X_train_normal = normal_data[features]
X_all = all_data_for_prediction[features]
y_true = all_data_for_prediction['Machine failure']
print("Data prepared.")

print("\nStep 3: Creating a preprocessing pipeline...")
numerical_features = ['Air temperature [K]', 'Process temperature [K]',
                      'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
categorical_features = ['Type']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(), categorical_features)
    ],
    remainder='passthrough'
)
print("Preprocessing pipeline created.")

print("\nStep 4: Building and training the Isolation Forest model...")
model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                 ('classifier', IsolationForest(
                                     contamination=0.0339,
                                     random_state=42))])

model_pipeline.fit(X_train_normal)
print("Model training complete.")

print("\nStep 5: Evaluating the model on the entire dataset...")
predictions_raw = model_pipeline.predict(X_all)

y_pred = np.where(predictions_raw == -1, 1, 0)

print("\nClassification Report:")
print(classification_report(y_true, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))

print("\nStep 6: Saving the trained model...")
model_filename = 'isolation_forest_model.joblib'
joblib.dump(model_pipeline, model_filename)
print(f"Model successfully saved as '{model_filename}'.")
