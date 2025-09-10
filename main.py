# Import necessary libraries
import pandas as pd
import joblib
import numpy as np
import sys

print("Step 1: Loading the saved machine learning models...")
try:
    models = {
        "Logistic Regression": joblib.load('logistic_regression_model.joblib'),
        "Isolation Forest": joblib.load('isolation_forest_model.joblib'),
        "One-Class SVM": joblib.load('one_class_svm_model.joblib'),
        "Random Forest": joblib.load('random_forest_model.joblib'),
        "Gradient Boosting": joblib.load('gradient_boosting_model.joblib')
    }
    print("All models loaded successfully!")
except FileNotFoundError as e:
    print(f"Error: One or more model files were not found. Please ensure the files exist.")
    print(f"Missing file: {e}")
    sys.exit()

best_model_scores = {
    "Logistic Regression": 0.52,
    "Isolation Forest": 0.45,
    "One-Class SVM": 0.35,
    "Random Forest": 0.78,
    "Gradient Boosting": 0.81
}
best_model_name = max(best_model_scores, key=best_model_scores.get)
print(f"\nBased on previous performance, the best performing model is: {best_model_name}")

print("\nStep 3: Please enter the machine's current parameters to make a prediction.")
print("You will need to provide the following details:")
print("- Product Type: L (Low), M (Medium), or H (High)")
print("- Air temperature (in Kelvin)")
print("- Process temperature (in Kelvin)")
print("- Rotational speed (in rpm)")
print("- Torque (in Nm)")
print("- Tool wear (in min)")

user_input_data = {}

valid_types = ['L', 'M', 'H']
while True:
    product_type = input(f"Enter Product Type (L=Low, M=Medium, or H=High): ").upper()
    if product_type in valid_types:
        user_input_data['Type'] = product_type
        break
    else:
        print(f"Invalid input. Please enter L, M, or H.")

numerical_features = {
    'Air temperature [K]': 'Air temperature (in K)',
    'Process temperature [K]': 'Process temperature (in K)',
    'Rotational speed [rpm]': 'Rotational speed (in rpm)',
    'Torque [Nm]': 'Torque (in Nm)',
    'Tool wear [min]': 'Tool wear (in min)'
}

for feature_key, feature_prompt in numerical_features.items():
    while True:
        try:
            value = float(input(f"Enter {feature_prompt}: "))
            user_input_data[feature_key] = value
            break
        except ValueError:
            print("Invalid input. Please enter a valid number.")

input_df = pd.DataFrame([user_input_data])
print("\nInput data successfully collected.")

print("\nStep 4: Making predictions with each model...")

predictions = {}
for name, model in models.items():
    if name in ["Isolation Forest", "One-Class SVM"]:
        prediction_raw = model.predict(input_df)[0]
        prediction = 1 if prediction_raw == -1 else 0
    else:
        prediction = model.predict(input_df)[0]

    predictions[name] = prediction
    condition = "FAILURE DETECTED" if prediction == 1 else "Good Condition"
    print(f"  - {name} predicts: {condition}")

print("\n========================================================")
print("Step 5: Final Prediction")

best_prediction = predictions[best_model_name]
final_condition = "FAILURE DETECTED" if best_prediction == 1 else "Good Condition"

print(f"Based on the best performing model ({best_model_name}), the machine is in a:")
print(f"--> {final_condition} <--")

if final_condition == "FAILURE DETECTED":
    print("\nRecommendation: The machine's current parameters suggest a high probability of failure.")
    print("Immediate maintenance or inspection is recommended.")
else:
    print("\nRecommendation: The machine is operating within normal parameters.")
    print("Continue with routine monitoring.")

print("========================================================")
