"""Ouverture de la webcam (plus rapide avec DirectShow sous Windows)."""
import os

import cv2

from . import config


def open_camera(index: int = config.CAMERA_INDEX) -> cv2.VideoCapture:
    backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
    cap = cv2.VideoCapture(index, backend)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir la webcam n°{index}. Essayez --camera 1.")
    return cap
