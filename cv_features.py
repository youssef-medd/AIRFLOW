import cv2
import numpy as np
from cv_preprocessing import preprocess, load_image


def extract_hog_features(img: np.ndarray, cell_size=(8, 8), block_size=(2, 2)) -> np.ndarray:
    gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    hog = cv2.HOGDescriptor(
        _winSize=(img.shape[1] // cell_size[1] * cell_size[1],
                  img.shape[0] // cell_size[0] * cell_size[0]),
        _blockSize=(block_size[1] * cell_size[1], block_size[0] * cell_size[0]),
        _blockStride=(cell_size[1], cell_size[0]),
        _cellSize=(cell_size[1], cell_size[0]),
        _nbins=9
    )
    return hog.compute(gray).flatten()


def extract_color_histogram(img: np.ndarray, bins=32) -> np.ndarray:
    hists = []
    for ch in range(3):
        h, _ = np.histogram(img[:, :, ch], bins=bins, range=(0.0, 1.0))
        hists.append(h.astype(np.float32))
    return np.concatenate(hists)


def extract_flow_magnitude(flow: np.ndarray) -> np.ndarray:
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return np.array([mag.mean(), mag.std(), mag.max(), ang.mean()])


def extract_turbulence_proxy(flow: np.ndarray) -> float:
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return float(mag.std())


def extract_all(img_path: str, prev_img: np.ndarray = None) -> np.ndarray:
    img = preprocess(load_image(img_path))
    hog   = extract_hog_features(img)
    chist = extract_color_histogram(img)
    parts = [hog, chist]
    if prev_img is not None:
        import cv2 as _cv2
        prev_gray = _cv2.cvtColor((prev_img * 255).astype(np.uint8), _cv2.COLOR_RGB2GRAY)
        curr_gray = _cv2.cvtColor((img    * 255).astype(np.uint8), _cv2.COLOR_RGB2GRAY)
        flow = _cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None,
            0.5, 3, 15, 3, 5, 1.2, 0
        )
        parts.append(extract_flow_magnitude(flow))
    return np.concatenate(parts)
