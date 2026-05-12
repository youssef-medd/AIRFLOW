import numpy as np
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, f1_score
)
from config import LABEL_NAMES


def evaluate_sensor_model(model, scaler, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    X_scaled = scaler.transform(X_test)
    y_pred   = model.predict(X_scaled)
    acc      = accuracy_score(y_test, y_pred)
    f1       = f1_score(y_test, y_pred, average="weighted")
    cm       = confusion_matrix(y_test, y_pred)
    report   = classification_report(y_test, y_pred, target_names=LABEL_NAMES)
    return {"accuracy": acc, "f1_weighted": f1, "confusion_matrix": cm, "report": report}


def print_metrics(results: dict) -> None:
    print(f"Accuracy   : {results['accuracy']:.4f}")
    print(f"F1 (w-avg) : {results['f1_weighted']:.4f}")
    print("\nClassification Report:")
    print(results["report"])
    print("Confusion Matrix:")
    header = "       " + "  ".join(f"{n[:4]:>6}" for n in LABEL_NAMES)
    print(header)
    for i, row in enumerate(results["confusion_matrix"]):
        print(f"{LABEL_NAMES[i][:6]:>6} " + "  ".join(f"{v:>6}" for v in row))
