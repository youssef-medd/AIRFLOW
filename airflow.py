from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import numpy as np
import joblib
import os

df = pd.read_csv("sensor_data.csv")
FEATURES = [
    "front_corr", "sideL_corr", "sideR_corr",
    "d_front", "d_sideL", "d_sideR",
    "asym", "turbulence", "internal_pressure",
    "vibration", "wind_speed"
]
X = df[FEATURES].values
y = df["label"].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
model = LogisticRegression(
    max_iter=300,
    multi_class='multinomial'
)
model.fit(X_train, y_train)
pred = model.predict(X_test)
print("Sensor model accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred))
print("W shape:", model.coef_.shape)
print("b shape:", model.intercept_.shape)
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("Sensor model saved -> model.pkl / scaler.pkl")

CV_MODEL_PATH = "cv_model.pt"
if os.path.exists(CV_MODEL_PATH):
    from ai_integration import AirflowAI
    ai = AirflowAI(
        sensor_model_path="model.pkl",
        cnn_model_path=CV_MODEL_PATH,
        scaler_path="scaler.pkl",
    )
    sample = X_test[0]
    sensor_raw = scaler.inverse_transform(sample.reshape(1, -1))[0]
    print("\nRunning fused AI inference on first test sample:")
    import torch
    dummy_img = torch.randn(3, 224, 224)
    result = ai.predict_fused(sensor_raw, dummy_img)
    print(f"  Label      : {result['label']}")
    print(f"  Confidence : {result['confidence']:.3f}")
else:
    print(f"\nSkipping fused inference — {CV_MODEL_PATH} not found.")
    print("Train the vision model first: python cv_train.py --data <image_root>")
