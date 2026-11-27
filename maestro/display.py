"""Fenêtre d'affichage : image agrandie, fenêtre adaptée à l'écran, plein écran (touche F)."""
from __future__ import annotations

import cv2

from . import config

WINDOW = "Maestro"


def screen_size() -> tuple[int, int]:
    """Taille de l'écran principal (Windows), sinon 1280x720 par défaut."""
    try:
        import ctypes

        user32 = ctypes.windll.user32
        try:
            user32.SetProcessDPIAware()
        except Exception:
            pass
        return int(user32.GetSystemMetrics(0)), int(user32.GetSystemMetrics(1))
    except Exception:
        return 1280, 720


def to_canvas(frame, height: int = config.DISPLAY_HEIGHT):
    """Agrandit l'image de la caméra (en gardant ses proportions) avant d'y dessiner l'interface."""
    h, w = frame.shape[:2]
    if h == height:
        return frame
    width = round(w * height / h)
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)


class Window:
    def __init__(self, name: str = WINDOW):
        self.name = name
        self.fullscreen = False
        self._sized = False
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)

    def show(self, canvas):
        if not self._sized:
            # première image : fenêtre la plus grande possible dans l'écran, proportions gardées
            sw, sh = screen_size()
            h, w = canvas.shape[:2]
            k = min(sw * config.WINDOW_SCREEN_FRACTION / w, sh * config.WINDOW_SCREEN_FRACTION / h)
            cv2.resizeWindow(self.name, int(w * k), int(h * k))
            cv2.moveWindow(self.name, max(int((sw - w * k) / 2), 0), max(int((sh - h * k) / 2) - 20, 0))
            self._sized = True
            if config.START_FULLSCREEN:
                self.toggle_fullscreen()
        cv2.imshow(self.name, canvas)

    def toggle_fullscreen(self):
        self.fullscreen = not self.fullscreen
        cv2.setWindowProperty(self.name, cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN if self.fullscreen else cv2.WINDOW_NORMAL)
