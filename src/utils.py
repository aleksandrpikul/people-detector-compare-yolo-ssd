from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import cv2
import numpy as np

@dataclass
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    score: float
    class_id: int

def draw_detections(frame: np.ndarray, dets: List[Detection], label: str = "person") -> np.ndarray:
    """Draws bounding boxes with confidence."""
    out = frame.copy()
    for d in dets:
        cv2.rectangle(out, (d.x1, d.y1), (d.x2, d.y2), (0, 255, 0), 2)
        txt = f"{label} {d.score:.2f}"
        cv2.putText(out, txt, (d.x1, max(0, d.y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
    return out

def letterbox(image: np.ndarray, new_shape=(640, 640), color=(114, 114, 114)) -> Tuple[np.ndarray, float, Tuple[int,int]]:
    """Resize with padding (YOLO-style). Returns (img, ratio, (pad_w, pad_h))."""
    h, w = image.shape[:2]
    new_w, new_h = new_shape
    r = min(new_w / w, new_h / h)
    resized = cv2.resize(image, (int(round(w * r)), int(round(h * r))), interpolation=cv2.INTER_LINEAR)
    pad_w = new_w - resized.shape[1]
    pad_h = new_h - resized.shape[0]
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return padded, r, (left, top)

def clip_box(x1, y1, x2, y2, w, h):
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(0, min(int(x2), w - 1))
    y2 = max(0, min(int(y2), h - 1))
    if x2 <= x1: x2 = min(w - 1, x1 + 1)
    if y2 <= y1: y2 = min(h - 1, y1 + 1)
    return x1, y1, x2, y2
