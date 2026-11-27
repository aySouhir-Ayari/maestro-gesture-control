"""Entraîne YOLOv8 (approche B) sur le dataset préparé par prepare_hagrid.py.

Utilisation (GPU conseillé : voir notebooks/train_yolo_colab.ipynb pour Google Colab) :
    pip install -r requirements-yolo.txt
    python scripts/train_yolo.py --epochs 50
Le meilleur modèle est copié dans models/maestro_yolo.pt -> python main.py --classifier yolo
"""
import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from maestro import config  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(config.YOLO_DATASET_DIR / "data.yaml"))
    parser.add_argument("--model", default="yolov8n.pt", help="yolov8n (rapide) ou yolov8s (plus précis)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=config.YOLO_IMGSZ)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", default=None, help="0 (GPU), cpu...")
    args = parser.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(
        data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, device=args.device,
        project=str(ROOT / "runs"), name="maestro", exist_ok=True,
        fliplr=0.0,           # PAS de miroir horizontal : « index à gauche » deviendrait « à droite »
        hsv_v=0.5,            # variations de luminosité fortes : robustesse à l'éclairage
        patience=15,
    )
    best = ROOT / "runs" / "maestro" / "weights" / "best.pt"
    config.MODELS_DIR.mkdir(exist_ok=True)
    shutil.copy2(best, config.YOLO_MODEL_PATH)
    metrics = YOLO(str(config.YOLO_MODEL_PATH)).val(data=args.data, imgsz=args.imgsz, plots=True,
                                                    project=str(ROOT / "runs"), name="maestro_val",
                                                    exist_ok=True)
    print(f"\nmAP50 = {metrics.box.map50:.3f} | mAP50-95 = {metrics.box.map:.3f}")
    print(f"Modèle : {config.YOLO_MODEL_PATH}   (matrice de confusion : runs/maestro_val/)")


if __name__ == "__main__":
    main()
