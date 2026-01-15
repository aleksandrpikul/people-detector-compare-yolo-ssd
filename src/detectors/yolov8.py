from __future__ import annotations
from typing import List, Tuple
import cv2
import numpy as np
from .base import BaseDetector
from ..utils import Detection, letterbox, clip_box

class YOLOv8ONNXDetector(BaseDetector):
    """YOLOv8n ONNX via OpenCV DNN. Outputs raw predictions -> NMS in Python."""
    name = "yolov8"

    def __init__(self, model_path: str, conf_thres: float = 0.25, iou_thres: float = 0.45, input_size: int = 640):
        self.model_path = model_path
        self.conf_thres = float(conf_thres)
        self.iou_thres = float(iou_thres)
        self.input_size = int(input_size)

        self.net = cv2.dnn.readNetFromONNX(model_path)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        # COCO person id for YOLOv8 exports is 0
        self.person_class_id = 0

    def detect_people(self, frame_bgr: np.ndarray) -> List[Detection]:
        h0, w0 = frame_bgr.shape[:2]
        img, r, (pad_w, pad_h) = letterbox(frame_bgr, (self.input_size, self.input_size))
        blob = cv2.dnn.blobFromImage(img, scalefactor=1/255.0, size=(self.input_size, self.input_size),
                                     swapRB=True, crop=False)
        self.net.setInput(blob)
        preds = self.net.forward()

        # Handle output layouts:
        # (1, 84, 8400) or (1, 8400, 84)
        if preds.ndim == 3:
            p = preds[0]
            if p.shape[0] in (84, 85):
                p = p.transpose(1, 0)
        else:
            p = preds

        n, d = p.shape
        boxes = []
        scores = []

        for i in range(n):
            row = p[i]
            if d == 84:
                cx, cy, w, h = row[:4]
                cls_scores = row[4:]
                conf = float(cls_scores[self.person_class_id])
            elif d == 85:
                cx, cy, w, h = row[:4]
                obj = float(row[4])
                cls_scores = row[5:]
                conf = obj * float(cls_scores[self.person_class_id])
            else:
                # Unknown format
                continue
            if conf < self.conf_thres:
                continue

            # Convert from input-space (letterboxed) to original
            x1 = (cx - w / 2 - pad_w) / r
            y1 = (cy - h / 2 - pad_h) / r
            x2 = (cx + w / 2 - pad_w) / r
            y2 = (cy + h / 2 - pad_h) / r
            x1, y1, x2, y2 = clip_box(x1, y1, x2, y2, w0, h0)

            boxes.append([x1, y1, x2 - x1, y2 - y1])  # xywh for NMSBoxes
            scores.append(conf)

        if not boxes:
            return []

        idxs = cv2.dnn.NMSBoxes(boxes, scores, self.conf_thres, self.iou_thres)
        dets: List[Detection] = []
        if len(idxs) == 0:
            return dets

        for j in idxs.flatten().tolist():
            x, y, bw, bh = boxes[j]
            dets.append(Detection(
                x1=int(x),
                y1=int(y),
                x2=int(x + bw),
                y2=int(y + bh),
                score=float(scores[j]),
                class_id=self.person_class_id
            ))
        return dets
