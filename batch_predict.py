import argparse
import numpy as np
import pandas as pd
import joblib

from config import SENSOR_FEATURES, LABEL_NAMES


def batch_predict(model_path: str, scaler_path: str, csv_path: str,
                  out_path: str = None) -> pd.DataFrame:
    model  = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    df      = pd.read_csv(csv_path)
    missing = [f for f in SENSOR_FEATURES if f not in df.columns]
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    X       = df[SENSOR_FEATURES].values
    X_sc    = scaler.transform(X)
    preds   = model.predict(X_sc)
    probas  = model.predict_proba(X_sc)

    df["predicted_label"] = preds
    df["confidence"]      = probas.max(axis=1).round(4)
    for i, name in enumerate(LABEL_NAMES):
        df[f"proba_{name}"] = probas[:, i].round(4)

    if out_path:
        df.to_csv(out_path, index=False)
        print(f"Results saved -> {out_path}")
    return df


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model",  required=True)
    p.add_argument("--scaler", required=True)
    p.add_argument("--csv",    required=True)
    p.add_argument("--out",    default=None)
    args = p.parse_args()

    results = batch_predict(args.model, args.scaler, args.csv, args.out)
    print(results[["predicted_label", "confidence"]].to_string(index=False))
