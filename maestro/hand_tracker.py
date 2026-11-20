"""Détection et suivi de la main avec MediaPipe Hand Landmarker (API Tasks)."""
from __future__ import annotations

import time
import urllib.request
from dataclasses import dataclass

import numpy as np

from . import config


@dataclass
class Hand:
    landmarks: np.ndarray   # (21, 3) coordonnées normalisées x, y ∈ [0,1], z relatif
    handedness: str         # "Left" / "Right"
    score: float

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        x, y = self.landmarks[:, 0], self.landmarks[:, 1]
        return float(x.min()), float(y.min()), float(x.max()), float(y.max())

    @property
    def area(self) -> float:
        x0, y0, x1, y1 = self.bbox
        return (x1 - x0) * (y1 - y0)


def ensure_model() -> str:
    """Télécharge le modèle MediaPipe s'il est absent."""
    path = config.HAND_MODEL_PATH
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[Maestro] Téléchargement du modèle MediaPipe -> {path}")
        tmp = path.with_suffix(".part")
        try:
            urllib.request.urlretrieve(config.HAND_MODEL_URL, tmp)
        except Exception as exc:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(
                f"Téléchargement impossible ({exc}). Téléchargez manuellement\n"
                f"  {config.HAND_MODEL_URL}\net placez le fichier dans {path}") from exc
        tmp.replace(path)
    return str(path)


class HandTracker:
    """Encapsule MediaPipe en mode VIDEO : détection + suivi image par image."""

    def __init__(self, max_hands: int = config.MAX_HANDS):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self._mp = mp
        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=ensure_model()),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=config.MIN_DETECTION_CONF,
            min_hand_presence_confidence=config.MIN_PRESENCE_CONF,
            min_tracking_confidence=config.MIN_TRACKING_CONF,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._t0 = time.monotonic()
        self._last_ts = -1

    def process(self, frame_rgb: np.ndarray) -> list[Hand]:
        """Renvoie toutes les mains détectées dans l'image RGB."""
        ts = int((time.monotonic() - self._t0) * 1000)
        ts = max(ts, self._last_ts + 1)  # horodatage strictement croissant exigé
        self._last_ts = ts
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=frame_rgb)
        result = self._landmarker.detect_for_video(image, ts)

        hands = []
        for lms, handed in zip(result.hand_landmarks, result.handedness):
            arr = np.array([[p.x, p.y, p.z] for p in lms], dtype=np.float32)
            hands.append(Hand(arr, handed[0].category_name, handed[0].score))
        return hands

    def close(self):
        self._landmarker.close()


def select_main_hand(hands: list[Hand]) -> Hand | None:
    """Plusieurs mains dans l'image : on garde la plus grande (la plus proche de la caméra)."""
    return max(hands, key=lambda h: h.area) if hands else None
