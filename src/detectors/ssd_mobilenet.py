from __future__ import annotations
from typing import List
import cv2
import numpy as np
from .base import BaseDetector
from ..utils import Detection, clip_box

class SSDMobileNetV1ONNXDetector(BaseDetector):
    """SSD MobileNetV1 ONNX (TF-style graph) via OpenCV DNN.

    Model is expected to output:
      - detection_boxes: [1, N, 4] in normalized ymin,xmin,ymax,xmax
      - detection_scores: [1, N]
      - detection_classes: [1, N] (float, COCO ids)
      - num_detections: [1]
    """
    name = "ssd"

    def __init__(self, model_path: str, conf_thres: float = 0.30, input_size: int = 300):
        self.model_path = model_path
        self.conf_thres = float(conf_thres)
        self.input_size = int(input_size)

        self.net = cv2.dnn.readNetFromONNX(model_path)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        # COCO person class id is 1 for SSD models from TF zoo
        self.person_class_id = 1

    def detect_people(self, frame_bgr: np.ndarray) -> List[Detection]:
        h0, w0 = frame_bgr.shape[:2]
        # TF models often expect RGB uint8 NHWC
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
        inp = resized.astype(np.uint8)[None, :, :, :]  # NHWC uint8
        self.net.setInput(inp)
        outs = self.net.forward(self.net.getUnconnectedOutLayersNames())

        # Map outputs by shape
        boxes = None
        scores = None
        classes = None
        num = None
        for o in outs:
            if o.ndim == 3 and o.shape[-1] == 4:
                boxes = o
            elif o.ndim == 2 and o.shape[-1] >= 1:
                # could be scores or num_detections
                pass

        # OpenCV may return outputs in fixed order:
        # ['detection_boxes:0', 'detection_classes:0', 'detection_scores:0', 'num_detections:0']
        names = self.net.getUnconnectedOutLayersNames()
        named = {n: outs[i] for i, n in enumerate(names)}

        # fallback keys
        for k in list(named.keys()):
            if 'boxes' in k: boxes = named[k]
            if 'scores' in k: scores = named[k]
            if 'classes' in k: classes = named[k]
            if 'num' in k: num = named[k]

        if boxes is None or scores is None or classes is None:
            # Unknown model signature
            return []

        boxes = boxes[0]
        scores = scores[0]
        classes = classes[0]

        dets: List[Detection] = []
        for i in range(min(len(scores), len(boxes), len(classes))):
            sc = float(scores[i])
            if sc < self.conf_thres:
                continue
            cls = int(classes[i])
            if cls != self.person_class_id:
                continue
            y1, x1, y2, x2 = boxes[i]
            x1 = x1 * w0
            x2 = x2 * w0
            y1 = y1 * h0
            y2 = y2 * h0
            x1, y1, x2, y2 = clip_box(x1, y1, x2, y2, w0, h0)
            dets.append(Detection(x1=x1, y1=y1, x2=x2, y2=y2, score=sc, class_id=cls))
        return dets
