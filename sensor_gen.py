import numpy as np
import pandas as pd
from config import SENSOR_FEATURES, LABEL_NAMES

# Per-class mean offsets that make each condition distinguishable
_CLASS_OFFSETS = {
    "normal":     {"turbulence": 0.0, "asym": 0.0, "d_front": 0.5},
    "turbulent":  {"turbulence": 2.5, "asym": 0.3, "vibration": 1.8},
    "asymmetric": {"asym": 2.0, "sideL_corr": -1.0, "sideR_corr": 1.0},
    "blocked":    {"d_front": -1.5, "internal_pressure": 1.5, "front_corr": -0.8},
}


def _generate_class(label: str, n: int, rng: np.random.Generator) -> pd.DataFrame:
    base = rng.standard_normal((n, len(SENSOR_FEATURES)))
    df   = pd.DataFrame(base, columns=SENSOR_FEATURES)
    for col, offset in _CLASS_OFFSETS[label].items():
        df[col] += offset
    df["label"] = label
    return df


def generate_sensor_data(n_per_class: int = 500, seed: int = 42) -> pd.DataFrame:
    rng    = np.random.default_rng(seed)
    frames = [_generate_class(lbl, n_per_class, rng) for lbl in LABEL_NAMES]
    df  = pd.concat(frames, ignore_index=True)
    idx = rng.permutation(len(df))
    return df.iloc[idx].reset_index(drop=True)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n",    type=int, default=500)
    p.add_argument("--out",  default="sensor_data.csv")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    df = generate_sensor_data(args.n, args.seed)
    df.to_csv(args.out, index=False)
    print(f"Generated {len(df)} rows -> {args.out}")
    print(df["label"].value_counts().to_string())
