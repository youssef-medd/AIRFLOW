import numpy as np
import joblib
import torch

from cv_model import build_model
from fusion import fuse_numpy, attention_fusion

SENSOR_FEATURES = [
    "front_corr", "sideL_corr", "sideR_corr",
    "d_front", "d_sideL", "d_sideR",
    "asym", "turbulence", "internal_pressure",
    "vibration", "wind_speed"
]

LABEL_NAMES = ["normal", "turbulent", "asymmetric", "blocked"]


class AirflowAI:
    def __init__(self, sensor_model_path: str, cnn_model_path: str,
                 scaler_path: str, arch: str = "resnet18"):
        self.scaler       = joblib.load(scaler_path)
        self.sensor_model = joblib.load(sensor_model_path)
        self.device       = "cuda" if torch.cuda.is_available() else "cpu"
        self.cnn          = build_model(arch, pretrained=False).to(self.device)
        self.cnn.load_state_dict(torch.load(cnn_model_path, map_location=self.device))
        self.cnn.eval()

    def predict_sensor(self, sensor_row: np.ndarray) -> int:
        scaled = self.scaler.transform(sensor_row.reshape(1, -1))
        return int(self.sensor_model.predict(scaled)[0])

    def predict_image(self, img_tensor: torch.Tensor) -> int:
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.cnn(img_tensor)
        return int(logits.argmax(1).item())

    def predict_fused(self, sensor_row: np.ndarray,
                      img_tensor: torch.Tensor) -> dict:
        scaled    = self.scaler.transform(sensor_row.reshape(1, -1))
        s_proba   = self.sensor_model.predict_proba(scaled)[0]

        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.cnn(img_tensor)
        v_proba = torch.softmax(logits, dim=1).cpu().numpy()[0]

        fused    = fuse_numpy(s_proba, v_proba)
        pred_idx = int(np.argmax(fused))
        return {
            "label":          LABEL_NAMES[pred_idx],
            "confidence":     float(fused[pred_idx]),
            "sensor_proba":   s_proba.tolist(),
            "vision_proba":   v_proba.tolist(),
            "fused_proba":    fused.tolist(),
        }
