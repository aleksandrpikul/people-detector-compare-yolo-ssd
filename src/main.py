from __future__ import annotations
import argparse
from pathlib import Path

from .pipeline import process_video
from .detectors.yolov8 import YOLOv8ONNXDetector
from .detectors.ssd_mobilenet import SSDMobileNetV1ONNXDetector

def build_detector(model: str, models_dir: Path, conf: float, iou: float):
    if model == "yolov8":
        mp = models_dir / "yolov8n.onnx"
        if not mp.exists():
            raise FileNotFoundError(f"Missing model file: {mp}. Run: python -m src.download_models")
        return YOLOv8ONNXDetector(str(mp), conf_thres=conf, iou_thres=iou)
    if model == "ssd":
        mp = models_dir / "ssd_mobilenet_v1_12.onnx"
        if not mp.exists():
            raise FileNotFoundError(f"Missing model file: {mp}. Run: python -m src.download_models")
        return SSDMobileNetV1ONNXDetector(str(mp), conf_thres=conf)
    raise ValueError(f"Unknown model: {model}. Use yolov8|ssd")

def parse_args():
    p = argparse.ArgumentParser(description="People detection on video with 2 selectable detectors")
    p.add_argument("--input", required=True, help="Input video path")
    p.add_argument("--output", required=True, help="Output video path")
    p.add_argument("--model", required=True, choices=["yolov8", "ssd"], help="Detector to use")
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    p.add_argument("--iou", type=float, default=0.45, help="IoU threshold for NMS (YOLO only)")
    p.add_argument("--max-frames", type=int, default=None, help="Process only first N frames (debug)")
    return p.parse_args()

def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    models_dir = repo_root / "models"

    det = build_detector(args.model, models_dir, args.conf, args.iou)
    metrics = process_video(
        detector=det,
        input_path=args.input,
        output_path=args.output,
        max_frames=args.max_frames,
    )
    print("\nDone. Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    main()
