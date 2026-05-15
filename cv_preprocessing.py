import cv2
import numpy as np
from pathlib import Path


IMG_SIZE = (224, 224)


def load_image(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return img


def preprocess(img: np.ndarray) -> np.ndarray:
    img = cv2.resize(img, IMG_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img  = (img - mean) / std
    return img


def apply_edge_detection(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny((gray * 255).astype(np.uint8), 50, 150)
    return edges


def apply_optical_flow(prev: np.ndarray, curr: np.ndarray) -> np.ndarray:
    prev_gray = cv2.cvtColor((prev * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    curr_gray = cv2.cvtColor((curr * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )
    return flow


def batch_preprocess(paths: list) -> np.ndarray:
    imgs = []
    for p in paths:
        try:
            imgs.append(preprocess(load_image(p)))
        except Exception as exc:
            raise RuntimeError(f"Failed to preprocess image: {p}") from exc
    return np.stack(imgs)
