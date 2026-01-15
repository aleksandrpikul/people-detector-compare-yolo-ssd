from __future__ import annotations
import json
from pathlib import Path

TEMPLATE_PATH = Path(__file__).resolve().parents[1] / "reports" / "report.md"
OUT_PATH = Path(__file__).resolve().parents[1] / "reports" / "report_filled.md"

def main():
    root = Path(__file__).resolve().parents[1]
    yolom = root / "outputs" / "crowd_yolov8.mp4.metrics.json"
    ssdm = root / "outputs" / "crowd_ssd.mp4.metrics.json"

    if not yolom.exists() or not ssdm.exists():
        raise FileNotFoundError("Run inference first to produce metrics json files in outputs/")

    y = json.loads(yolom.read_text(encoding="utf-8"))
    s = json.loads(ssdm.read_text(encoding="utf-8"))

    md = TEMPLATE_PATH.read_text(encoding="utf-8")
    table = (
        "| Модель | avg_infer_time_s | p50 | p95 | approx_fps_infer_only | avg_people_per_frame | avg_confidence_per_frame |\n"
        "|---|---:|---:|---:|---:|---:|---:|\n"
        f"| YOLOv8n | {y['avg_infer_time_s']:.4f} | {y['p50_infer_time_s']:.4f} | {y['p95_infer_time_s']:.4f} | {y['approx_fps_infer_only']:.2f} | {y['avg_people_per_frame']:.2f} | {y['avg_confidence_per_frame']:.2f} |\n"
        f"| SSD-MobileNetV1 | {s['avg_infer_time_s']:.4f} | {s['p50_infer_time_s']:.4f} | {s['p95_infer_time_s']:.4f} | {s['approx_fps_infer_only']:.2f} | {s['avg_people_per_frame']:.2f} | {s['avg_confidence_per_frame']:.2f} |\n"
    )

    # replace first occurrence of the empty table header block
    import re
    md2 = re.sub(r"\| Модель \| avg_infer_time_s[\s\S]*?\| SSD-MobileNetV1 \|\s*\|\s*\|\s*\|\s*\|\s*\|\s*\|\s*\|",
                 table.strip(), md, count=1)
    OUT_PATH.write_text(md2, encoding="utf-8")
    print(f"Saved: {OUT_PATH}")

if __name__ == "__main__":
    main()
