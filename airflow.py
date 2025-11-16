# ============================================
#  TRAINING MODEL (NO WINDOWS)
# ============================================

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import joblib
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
print("Accuracy:", accuracy_score(y_test, pred))
print("W shape:", model.coef_.shape)
print("b shape:", model.intercept_.shape)
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

