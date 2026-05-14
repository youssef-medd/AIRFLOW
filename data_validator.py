import pandas as pd
import numpy as np
from config import SENSOR_FEATURES, LABEL_NAMES


def check_missing_values(df: pd.DataFrame) -> dict:
    missing = df[SENSOR_FEATURES + ["label"]].isnull().sum()
    total = len(df)
    result = {col: {"count": int(n), "pct": round(n / total * 100, 2)}
              for col, n in missing.items() if n > 0}
    return result
