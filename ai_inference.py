import argparse
import time
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

from ai_integration import AirflowAI, SENSOR_FEATURES, LABEL_NAMES


EVAL_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def load_img_tensor(path: str) -> torch.Tensor:
    return EVAL_TRANSFORMS(Image.open(path).convert("RGB"))


def run_single(ai: AirflowAI, sensor_csv_row: dict, img_path: str):
    sensor_vec = np.array([sensor_csv_row[f] for f in SENSOR_FEATURES], dtype=np.float32)
    img_tensor = load_img_tensor(img_path)
    result     = ai.predict_fused(sensor_vec, img_tensor)
    print(f"Prediction : {result['label'].upper()}")
    print(f"Confidence : {result['confidence']:.3f}")
    print(f"Sensor     : {[f'{p:.3f}' for p in result['sensor_proba']]}")
    print(f"Vision     : {[f'{p:.3f}' for p in result['vision_proba']]}")
    print(f"Fused      : {[f'{p:.3f}' for p in result['fused_proba']]}")
    return result


def benchmark(ai: AirflowAI, n: int = 100):
    dummy_sensor = np.random.randn(len(SENSOR_FEATURES)).astype(np.float32)
    dummy_img    = torch.randn(3, 224, 224)
    start = time.perf_counter()
    for _ in range(n):
        ai.predict_fused(dummy_sensor, dummy_img)
    elapsed = time.perf_counter() - start
    print(f"{n} inferences in {elapsed:.3f}s — {elapsed/n*1000:.2f} ms/sample")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sensor_model", required=True)
    p.add_argument("--cnn_model",    required=True)
    p.add_argument("--scaler",       required=True)
    p.add_argument("--img",          required=True)
    p.add_argument("--benchmark",    action="store_true")
    args = p.parse_args()

    ai = AirflowAI(args.sensor_model, args.cnn_model, args.scaler)
    if args.benchmark:
        benchmark(ai)
    else:
        dummy_row = {f: 0.0 for f in SENSOR_FEATURES}
        run_single(ai, dummy_row, args.img)
