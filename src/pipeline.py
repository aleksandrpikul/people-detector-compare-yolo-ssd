from __future__ import annotations
from dataclasses import asdict
from typing import Dict, Optional, Tuple, List
import time
import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from .utils import draw_detections, Detection
from .detectors.base import BaseDetector

def process_video(
    detector: BaseDetector,
    input_path: str,
    output_path: str,
    metrics_path: Optional[str] = None,
    max_frames: Optional[int] = None,
    draw: bool = True,
) -> Dict:
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # mp4v is widely supported cross-platform
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    times = []
    det_counts = []
    mean_scores = []

    total = nframes if (nframes > 0 and max_frames is None) else (max_frames or 0)
    pbar = tqdm(total=total if total else None, desc=f"{detector.name} inference", unit="frame")

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        if max_frames is not None and frame_idx > max_frames:
            break

        t0 = time.perf_counter()
        dets = detector.detect_people(frame)
        t1 = time.perf_counter()
        times.append(t1 - t0)

        det_counts.append(len(dets))
        mean_scores.append(float(np.mean([d.score for d in dets])) if dets else 0.0)

        if draw:
            frame_out = draw_detections(frame, dets, label="person")
        else:
            frame_out = frame
        writer.write(frame_out)
        pbar.update(1)

    pbar.close()
    cap.release()
    writer.release()

    import numpy as np
    arr = np.array(times, dtype=np.float64) if times else np.array([0.0], dtype=np.float64)

    metrics = {
        "model": detector.name,
        "input": str(input_path),
        "output": str(output_path),
        "frames_processed": int(frame_idx),
        "video_fps": float(fps),
        "avg_infer_time_s": float(arr.mean()),
        "p50_infer_time_s": float(np.percentile(arr, 50)),
        "p95_infer_time_s": float(np.percentile(arr, 95)),
        "approx_fps_infer_only": float(1.0 / arr.mean()) if arr.mean() > 0 else 0.0,
        "avg_people_per_frame": float(np.mean(det_counts)) if det_counts else 0.0,
        "avg_confidence_per_frame": float(np.mean(mean_scores)) if mean_scores else 0.0,
    }

    if metrics_path is None:
        metrics_path = str(out_path.with_suffix(out_path.suffix + ".metrics.json"))
    Path(metrics_path).write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    return metrics
