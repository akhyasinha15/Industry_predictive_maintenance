import pandas as pd
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS
from vercel_python import from_wsgi_app

app = Flask(__name__)
CORS(app)

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
    print(f"Error loading models: {e}")
    exit()

best_model_scores = {
    "Logistic Regression": 0.52,
    "Isolation Forest": 0.45,
    "One-Class SVM": 0.35,
    "Random Forest": 0.78,
    "Gradient Boosting": 0.81
}
best_model_name = max(best_model_scores, key=best_model_scores.get)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No input data provided'}), 400

        input_df = pd.DataFrame([data])
        predictions = {}
        for name, model in models.items():
            if name in ["Isolation Forest", "One-Class SVM"]:
                prediction_raw = model.predict(input_df)[0]
                prediction = 1 if prediction_raw == -1 else 0
            else:
                prediction = model.predict(input_df)[0]
            
            predictions[name] = int(prediction)

        best_prediction = predictions[best_model_name]
        
        if best_prediction == 1:
            final_condition = "FAILURE DETECTED"
            recommendation = "The machine's current parameters suggest a high probability of failure. Immediate maintenance or inspection is recommended."
        else:
            final_condition = "GOOD CONDITION"
            recommendation = "The machine is operating within normal parameters. Continue with routine monitoring."
        
        return jsonify({
            'status': 'success',
            'prediction': final_condition,
            'recommendation': recommendation,
            'best_model': best_model_name
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

handler = from_wsgi_app(app)
