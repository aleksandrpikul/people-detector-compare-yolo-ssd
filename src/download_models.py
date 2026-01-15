from __future__ import annotations
import hashlib
from pathlib import Path
from urllib.request import urlopen

MODELS = {
    # YOLOv8n ONNX (13MB). Source repo contains ONNX directly.
    "yolov8n.onnx": {
        "url": "https://github.com/Hyuto/yolov8-onnxruntime-web/raw/master/public/model/yolov8n.onnx",
        "sha256": None,
    },
    # SSD MobileNet V1 ONNX (27MB). Official ONNX Model Zoo (Git LFS); this URL may require alternative mirror.
    "ssd_mobilenet_v1_12.onnx": {
        "url": "https://huggingface.co/onnxmodelzoo/ssd_mobilenet_v1_12/resolve/main/ssd_mobilenet_v1_12.onnx?download=true",
        "sha256": None,
    },
}

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def download(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url) as r, dst.open("wb") as f:
        while True:
            chunk = r.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)

def main():
    models_dir = Path(__file__).resolve().parents[1] / "models"
    for name, meta in MODELS.items():
        dst = models_dir / name
        if dst.exists():
            print(f"[OK] exists: {dst}")
            continue
        print(f"[DL] {name} <- {meta['url']}")
        download(meta["url"], dst)
        print(f"[OK] saved: {dst} ({dst.stat().st_size/1024/1024:.1f} MB)")
        if meta.get("sha256"):
            got = sha256_file(dst)
            assert got.lower() == meta["sha256"].lower(), f"SHA256 mismatch for {name}"

if __name__ == "__main__":
    main()
