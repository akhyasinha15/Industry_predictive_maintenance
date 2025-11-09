import easyocr
import cv2
import numpy as np
import pandas as pd
import joblib
import re
import time
import sys

# =======================
# STEP 1: Load all ML models
# =======================
print("🧠 Loading trained models...")
try:
    models = {
        "Logistic Regression": joblib.load('logistic_regression_model.joblib'),
        "Isolation Forest": joblib.load('isolation_forest_model.joblib'),
        "One-Class SVM": joblib.load('one_class_svm_model.joblib'),
        "Random Forest": joblib.load('random_forest_model.joblib'),
        "Gradient Boosting": joblib.load('gradient_boosting_model.joblib')
    }
    print("✅ All models loaded successfully!\n")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    sys.exit()

best_model_scores = {
    "Logistic Regression": 0.52,
    "Isolation Forest": 0.45,
    "One-Class SVM": 0.35,
    "Random Forest": 0.78,
    "Gradient Boosting": 0.81
}
best_model_name = max(best_model_scores, key=best_model_scores.get)
print(f"⭐ Best Performing Model: {best_model_name}\n")

# =======================
# STEP 2: Initialize OCR and Webcam
# =======================
reader = easyocr.Reader(['en'], gpu=False)

def open_camera():
    print("🔍 Initializing webcam...")
    for i in range(5):
        cap = cv2.VideoCapture(i)
        time.sleep(0.5)
        if cap.isOpened():
            print(f"✅ Webcam found at index {i}")
            return cap
        cap.release()
    raise ValueError("❌ No webcam found. Check your permissions or connection.")

# =======================
# STEP 3: OCR Reading Function (Digital Meter)
# =======================
def read_meter_live(label):
    print(f"\n📸 Place the '{label}' meter in front of the camera...")
    print("⏳ Hold steady until the value is captured automatically.\n")

    cap = open_camera()
    start_time = time.time()
    MAX_TIME = 45
    stable_count = 0
    readings = []
    last_value = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Frame not captured, retrying...")
            continue

        frame = cv2.resize(frame, (640, 480))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.convertScaleAbs(gray, alpha=2.0, beta=0)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)

        # Crop center region (where display likely is)
        h, w = gray.shape
        x1, y1, x2, y2 = int(w * 0.25), int(h * 0.3), int(w * 0.75), int(h * 0.7)
        roi = blur[y1:y2, x1:x2]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # OCR every 1 second for stability
        if int(time.time() - start_time) % 1 == 0:
            result = reader.readtext(roi, detail=0, paragraph=True)
            text = " ".join(result)
            numbers = re.findall(r"\d+\.\d+|\d+", text)
            if numbers:
                try:
                    val = float(numbers[0])
                    readings.append(val)
                    cv2.putText(frame, f"Detected: {val:.3f}",
                                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                    # Stability logic
                    if last_value and abs(val - last_value) < 0.02:
                        stable_count += 1
                    else:
                        stable_count = 0
                    last_value = val
                except:
                    pass

        # Show live feed
        elapsed = int(time.time() - start_time)
        cv2.putText(frame, f"⏱ {elapsed}s / {MAX_TIME}s", (450, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.imshow("Digital Meter Reader", frame)

        # Stop if stable or timeout
        if stable_count >= 8 or (elapsed > MAX_TIME and readings):
            final_value = np.mean(readings[-5:]) if len(readings) >= 5 else np.mean(readings)
            print(f"✅ Captured '{label}' Value: {final_value:.3f}\n")
            cap.release()
            cv2.destroyAllWindows()
            return float(final_value)

        if cv2.getWindowProperty("Digital Meter Reader", cv2.WND_PROP_VISIBLE) < 1:
            break
        if cv2.waitKey(30) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"⚠️ No stable reading captured for '{label}'. Defaulting to 0.\n")
    return 0.0


# =======================
# STEP 4: Capture 5 Parameters Automatically
# =======================
print("\n========== AUTOMATIC METER READING STARTED ==========\n")

features_to_read = [
    ("Air temperature [K]", "Air Temperature Meter"),
    ("Process temperature [K]", "Process Temperature Meter"),
    ("Rotational speed [rpm]", "Rotational Speed Meter"),
    ("Torque [Nm]", "Torque Meter"),
    ("Tool wear [min]", "Tool Wear Meter")
]

input_data = {}

for feature, label in features_to_read:
    val = read_meter_live(label)
    input_data[feature] = val

# Fixed product type (you can automate with OCR label later)
input_data['Type'] = 'M'

input_df = pd.DataFrame([input_data])
print("✅ All readings captured successfully!\n")
print(input_df)

# =======================
# STEP 5: Run Predictions
# =======================
print("\n🔮 Running predictions on captured data...\n")

predictions = {}
for name, model in models.items():
    try:
        if name in ["Isolation Forest", "One-Class SVM"]:
            pred_raw = model.predict(input_df)[0]
            prediction = 1 if pred_raw == -1 else 0
        else:
            prediction = model.predict(input_df)[0]

        predictions[name] = prediction
        condition = "FAILURE DETECTED ⚠️" if prediction == 1 else "Good Condition ✅"
        print(f"  - {name}: {condition}")
    except Exception as e:
        print(f"❌ {name} prediction failed: {e}")
        predictions[name] = None

# =======================
# STEP 6: Final Decision
# =======================
print("\n====================================================")
print("FINAL MACHINE CONDITION")
print("====================================================")

final_prediction = predictions.get(best_model_name, 0)
final_condition = "FAILURE DETECTED ⚠️" if final_prediction == 1 else "Good Condition ✅"

print(f"🏆 Best Model: {best_model_name}")
print(f"🧭 Final Machine Condition: {final_condition}")
print("====================================================")

if final_condition.startswith("FAILURE"):
    print("\n🛠️ Recommendation: Immediate maintenance or inspection required.")
else:
    print("\n✅ Recommendation: Machine operating within normal range.")
