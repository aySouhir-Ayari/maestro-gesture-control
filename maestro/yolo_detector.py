"""Approche B : YOLOv8 détecte les mains et classe le geste directement dans l'image.

Fonctionnement hybride dans Maestro :
- YOLO regarde l'image entière (set_frame) : il trouve chaque main, même dans une scène chargée,
  et donne son geste ;
- MediaPipe garde les 21 points clés, utiles pour le pincement (volume), le balayage et le seek ;
- predict(lm) associe la main MediaPipe à la boîte YOLO qui la recouvre le mieux.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import config
from .features import finger_states
from .gestures import Prediction


@dataclass
class Detection:
    label: str
    confidence: float
    box: tuple[float, float, float, float]   # x0, y0, x1, y1 normalisés


def iou(a, b) -> float:
    ix0, iy0, ix1, iy1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(ix1 - ix0, 0) * max(iy1 - iy0, 0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def landmark_box(lm: np.ndarray, margin: float = 0.25):
    x0, y0, x1, y1 = lm[:, 0].min(), lm[:, 1].min(), lm[:, 0].max(), lm[:, 1].max()
    mx, my = (x1 - x0) * margin / 2, (y1 - y0) * margin / 2
    return float(x0 - mx), float(y0 - my), float(x1 + mx), float(y1 + my)


def match(detections: list[Detection], lm: np.ndarray, min_iou: float = 0.1) -> Detection | None:
    """Boîte YOLO correspondant à la main MediaPipe (meilleur recouvrement)."""
    box = landmark_box(lm)
    best = max(detections, key=lambda d: iou(d.box, box), default=None)
    return best if best is not None and iou(best.box, box) >= min_iou else None


class YoloClassifier:
    name = "YOLOv8 (approche B)"

    def __init__(self, path=config.YOLO_MODEL_PATH, conf=config.YOLO_CONF, imgsz=config.YOLO_IMGSZ,
                 every=config.YOLO_EVERY):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise SystemExit("Approche B : installez d'abord  pip install -r requirements-yolo.txt") from exc
        if not path.exists():
            raise SystemExit(f"Modèle YOLO absent : {path}\n"
                             "Entraînez-le (notebooks/train_yolo_colab.ipynb) puis copiez maestro_yolo.pt dans models/.")
        self.model = YOLO(str(path))
        self.names = self.model.names
        self.conf, self.imgsz, self.every = conf, imgsz, max(1, every)
        self.detections: list[Detection] = []
        self._n = 0

    def detect(self, rgb: np.ndarray) -> list[Detection]:
        """Toutes les mains de l'image avec leur geste."""
        res = self.model.predict(np.ascontiguousarray(rgb[..., ::-1]), imgsz=self.imgsz, conf=self.conf, verbose=False)[0]
        h, w = rgb.shape[:2]
        out = []
        for (x0, y0, x1, y1), c, k in zip(res.boxes.xyxy.tolist(), res.boxes.conf.tolist(),
                                          res.boxes.cls.tolist()):
            out.append(Detection(str(self.names[int(k)]), float(c), (x0 / w, y0 / h, x1 / w, y1 / h)))
        return out

    def set_frame(self, rgb: np.ndarray):
        if self._n % self.every == 0:
            self.detections = self.detect(rgb)
        self._n += 1

    def predict(self, lm: np.ndarray) -> Prediction:
        det = match(self.detections, lm)
        if det is None:
            return Prediction(config.NONE, 0.0, finger_states(lm))
        return Prediction(det.label, det.confidence, finger_states(lm))
