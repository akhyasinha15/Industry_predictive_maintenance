# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

print("Step 1: Loading the dataset...")
try:
    df = pd.read_csv('C://Users//Shobit//PycharmProjects//Hackathons//pythonProject//dataset//ai4i2020.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: The file 'ai4i2020.csv' was not found.")
    print("Please make sure the dataset is in the same directory as the script.")
    exit()

print("\nStep 2: Separating features and target...")
features = ['Type', 'Air temperature [K]', 'Process temperature [K]',
            'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
target = 'Machine failure'

X = df[features]
y = df[target]
print("Features (X) and Target (y) separated.")

print("\nStep 3: Creating a preprocessing pipeline...")
numerical_features = ['Air temperature [K]', 'Process temperature [K]',
                      'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
categorical_features = ['Type']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])
print("Preprocessing pipeline created.")

print("\nStep 4: Splitting the data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("Data split complete. Training set size:", len(X_train), ", Testing set size:", len(X_test))

print("\nStep 5: Building and training the Gradient Boosting Machine model...")
model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                 ('classifier', GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42))])

model_pipeline.fit(X_train, y_train)
print("Model training complete.")

print("\nStep 6: Evaluating the model on the test set...")
y_pred = model_pipeline.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nStep 7: Saving the trained model...")
model_filename = 'gradient_boosting_model.joblib'
joblib.dump(model_pipeline, model_filename)
print(f"Model successfully saved as '{model_filename}'.")

