from __future__ import annotations

from typing import List, Tuple, Optional
import numpy as np
import onnxruntime as ort
import cv2

from .base import Detection, BaseDetector


class SSDMobileNetV1ONNXDetector(BaseDetector):
    def __init__(self, model_path: str, conf_thres: float = 0.3, input_size: int = 320):
        self.model_path = model_path
        self.conf_thres = float(conf_thres)
        self.input_size = int(input_size)

        providers = ["CPUExecutionProvider"]
        try:
            if "CUDAExecutionProvider" in ort.get_available_providers():
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        except Exception:
            pass

        self.sess = ort.InferenceSession(self.model_path, providers=providers)
        self.inp = self.sess.get_inputs()[0]
        self.inp_name = self.inp.name
        self.inp_type = self.inp.type
        self.inp_shape = self.inp.shape

        self.nchw = False
        if isinstance(self.inp_shape, (list, tuple)) and len(self.inp_shape) == 4:
            if self.inp_shape[1] == 3:
                self.nchw = True

        self.out_names = [o.name for o in self.sess.get_outputs()]

    def _preprocess(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
        h0, w0 = frame_bgr.shape[:2]
        img = cv2.resize(frame_bgr, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if "uint8" in self.inp_type:
            blob = img_rgb.astype(np.uint8)
        else:
            blob = img_rgb.astype(np.float32) / 255.0

        if self.nchw:
            blob = np.transpose(blob, (2, 0, 1))
        blob = np.expand_dims(blob, axis=0)

        return blob, (w0, h0)

    def _pick_outputs(self, outs: List[np.ndarray]):
        name_to_arr = {n: a for n, a in zip(self.out_names, outs)}

        def find_by_substr(substrs):
            for n in self.out_names:
                ln = n.lower()
                if any(s in ln for s in substrs):
                    return name_to_arr[n]
            return None

        boxes = find_by_substr(["detection_boxes", "boxes"])
        scores = find_by_substr(["detection_scores", "scores", "score"])
        classes = find_by_substr(["detection_classes", "classes", "class"])
        num = find_by_substr(["num_detections", "num"])

        num_det = None
        if num is not None:
            try:
                num_det = int(np.squeeze(num).item())
            except Exception:
                num_det = None

        return boxes, scores, classes, num_det

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        blob, (w0, h0) = self._preprocess(frame_bgr)
        outs = self.sess.run(None, {self.inp_name: blob})

        dets: List[Detection] = []

        boxes, scores, classes, num_det = self._pick_outputs(outs)

        # TF2-style outputs
        if boxes is not None and scores is not None and classes is not None:
            boxes = np.squeeze(boxes, axis=0) if boxes.ndim == 3 else boxes
            scores = np.squeeze(scores, axis=0) if scores.ndim >= 2 else scores
            classes = np.squeeze(classes, axis=0) if classes.ndim >= 2 else classes

            N = boxes.shape[0]
            if num_det is not None:
                N = min(N, num_det)

            for i in range(N):
                sc = float(scores[i])
                if sc < self.conf_thres:
                    continue
                cls = int(classes[i])
                if cls != 1:  # person
                    continue

                y1, x1, y2, x2 = boxes[i].tolist()

                if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 2.0:
                    x1 *= w0; x2 *= w0
                    y1 *= h0; y2 *= h0
                else:
                    sx = w0 / self.input_size
                    sy = h0 / self.input_size
                    x1 *= sx; x2 *= sx
                    y1 *= sy; y2 *= sy

                x1 = max(0, min(w0 - 1, x1))
                x2 = max(0, min(w0 - 1, x2))
                y1 = max(0, min(h0 - 1, y1))
                y2 = max(0, min(h0 - 1, y2))

                if x2 <= x1 or y2 <= y1:
                    continue

                dets.append(Detection(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2), score=sc, cls=cls))
            return dets

        # SSD-12 style outputs
        cand = None
        for a in outs:
            if isinstance(a, np.ndarray) and a.ndim >= 2 and a.shape[-1] == 7:
                cand = a
                break

        if cand is not None:
            arr = cand.reshape(-1, 7)
            for row in arr:
                _, label, sc, x1, y1, x2, y2 = row.tolist()
                sc = float(sc)
                if sc < self.conf_thres:
                    continue
                if int(label) != 1:
                    continue

                if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 2.0:
                    x1 *= w0; x2 *= w0
                    y1 *= h0; y2 *= h0
                else:
                    sx = w0 / self.input_size
                    sy = h0 / self.input_size
                    x1 *= sx; x2 *= sx
                    y1 *= sy; y2 *= sy

                x1 = max(0, min(w0 - 1, x1))
                x2 = max(0, min(w0 - 1, x2))
                y1 = max(0, min(h0 - 1, y1))
                y2 = max(0, min(h0 - 1, y2))

                if x2 <= x1 or y2 <= y1:
                    continue

                dets.append(Detection(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2), score=sc, cls=int(label)))

        return dets
