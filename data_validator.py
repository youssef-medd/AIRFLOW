import pandas as pd
import numpy as np
from config import SENSOR_FEATURES, LABEL_NAMES, SENSOR_RANGES


def check_missing_values(df: pd.DataFrame) -> dict:
    missing = df[SENSOR_FEATURES + ["label"]].isnull().sum()
    total = len(df)
    result = {col: {"count": int(n), "pct": round(n / total * 100, 2)}
              for col, n in missing.items() if n > 0}
    return result


def check_value_ranges(df: pd.DataFrame) -> dict:
    out_of_range = {}
    for col, (lo, hi) in SENSOR_RANGES.items():
        if col not in df.columns:
            continue
        mask = (df[col] < lo) | (df[col] > hi)
        n = int(mask.sum())
        if n:
            out_of_range[col] = {"count": n, "min": float(df[col].min()), "max": float(df[col].max()), "range": (lo, hi)}
    return out_of_range


def check_class_balance(df: pd.DataFrame) -> dict:
    counts = df["label"].value_counts().to_dict()
    total = len(df)
    balance = {LABEL_NAMES[k] if isinstance(k, int) else k: {"count": int(v), "pct": round(v / total * 100, 2)}
               for k, v in counts.items()}
    return balance


def check_duplicates(df: pd.DataFrame) -> int:
    return int(df.duplicated(subset=SENSOR_FEATURES).sum())


def check_feature_correlation(df: pd.DataFrame, threshold: float = 0.95) -> list[tuple]:
    corr = df[SENSOR_FEATURES].corr().abs()
    pairs = []
    cols = corr.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            if corr.iloc[i, j] >= threshold:
                pairs.append((cols[i], cols[j], round(float(corr.iloc[i, j]), 4)))
    return pairs


def validate(df: pd.DataFrame, corr_threshold: float = 0.95) -> dict:
    return {
        "rows": len(df),
        "missing": check_missing_values(df),
        "out_of_range": check_value_ranges(df),
        "class_balance": check_class_balance(df),
        "duplicates": check_duplicates(df),
        "high_corr_pairs": check_feature_correlation(df, corr_threshold),
    }
