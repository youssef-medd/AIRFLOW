# ============================================
#  TRAINING MODEL (NO WINDOWS)
# ============================================

import argparse
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import SENSOR_FEATURES as FEATURES
from data_validator import validate


def main() -> None:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--model_type", default="logistic", choices=["logistic", "random_forest"])
    args, _ = p.parse_known_args()

    df = pd.read_csv("sensor_data.csv")

    report = validate(df)
    print(f"[validator] rows={report['rows']}  duplicates={report['duplicates']}")
    if report["missing"]:
        print("[validator] missing values:", report["missing"])
    if report["out_of_range"]:
        print("[validator] out-of-range columns:", list(report["out_of_range"].keys()))
    if report["high_corr_pairs"]:
        print("[validator] highly correlated pairs:", report["high_corr_pairs"])

    X = df[FEATURES].values
    y = df["label"].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    if args.model_type == "random_forest":
        model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    else:
        model = LogisticRegression(max_iter=300, multi_class="multinomial")

    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print(f"Sensor model ({args.model_type}) accuracy:", accuracy_score(y_test, pred))
    print(classification_report(y_test, pred))
    joblib.dump(model, "model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    print("Sensor model saved -> model.pkl / scaler.pkl")

    CV_MODEL_PATH = "cv_model.pt"
    if os.path.exists(CV_MODEL_PATH):
        import torch
        from ai_integration import AirflowAI
        ai = AirflowAI(
            sensor_model_path="model.pkl",
            cnn_model_path=CV_MODEL_PATH,
            scaler_path="scaler.pkl",
        )
        sample = X_test[0]
        sensor_raw = scaler.inverse_transform(sample.reshape(1, -1))[0]
        print("\nRunning fused AI inference on first test sample:")
        dummy_img = torch.randn(3, 224, 224)
        result = ai.predict_fused(sensor_raw, dummy_img)
        print(f"  Label      : {result['label']}")
        print(f"  Confidence : {result['confidence']:.3f}")
    else:
        print(f"\nSkipping fused inference — {CV_MODEL_PATH} not found.")
        print("Train the vision model first: python cv_train.py --data <image_root>")


if __name__ == "__main__":
    main()
