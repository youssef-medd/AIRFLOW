import numpy as np
import torch
import torch.nn as nn


class SensorImageFusion(nn.Module):
    """Late fusion of sensor feature vector and CNN image embedding."""

    def __init__(self, sensor_dim: int, image_dim: int, num_classes: int = 4):
        super().__init__()
        self.sensor_encoder = nn.Sequential(
            nn.Linear(sensor_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.image_encoder = nn.Sequential(
            nn.Linear(image_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, sensor: torch.Tensor, image_emb: torch.Tensor) -> torch.Tensor:
        s = self.sensor_encoder(sensor)
        v = self.image_encoder(image_emb)
        return self.classifier(torch.cat([s, v], dim=-1))


def fuse_numpy(sensor_vec: np.ndarray, image_vec: np.ndarray,
               weights: tuple = (0.4, 0.6)) -> np.ndarray:
    # L2 normalization distorts probability vectors; blend then re-normalize by sum
    fused = weights[0] * sensor_vec + weights[1] * image_vec
    return fused / (fused.sum() + 1e-8)


def attention_fusion(sensor_vec: np.ndarray, image_vec: np.ndarray) -> np.ndarray:
    s_norm = sensor_vec / (np.linalg.norm(sensor_vec) + 1e-8)
    v_norm = image_vec  / (np.linalg.norm(image_vec)  + 1e-8)
    # Use entropy-based confidence: lower entropy = higher certainty = higher weight
    def _confidence(v: np.ndarray) -> float:
        p = np.abs(v) / (np.abs(v).sum() + 1e-8)
        return float(np.exp(-np.sum(p * np.log(p + 1e-8))))
    score_s = _confidence(sensor_vec)
    score_v = _confidence(image_vec)
    total   = score_s + score_v
    return (score_s / total) * s_norm + (score_v / total) * v_norm
