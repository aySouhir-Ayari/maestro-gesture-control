"""Commandes continues : volume au pincement et barre de progression « attrapée »."""
from __future__ import annotations

import numpy as np

from . import config
from .features import finger_extended, palm_center, pinch_ratio


class PinchVolume:
    """Main gauche : majeur, annulaire et auriculaire repliés, l'écart pouce-index donne le volume.

    Ouvrir la main (quitter la pose) fige le volume à sa dernière valeur.
    """

    def __init__(self, vmin=config.PINCH_MIN, vmax=config.PINCH_MAX,
                 engage_frames=config.PINCH_ENGAGE_FRAMES, smoothing=config.PINCH_SMOOTHING):
        self.vmin, self.vmax = vmin, vmax
        self.engage_frames = engage_frames
        self.smoothing = smoothing
        self._count = 0
        self.value: float | None = None   # volume lissé pendant le pincement
        self.active = False

    @staticmethod
    def in_pose(lm: np.ndarray) -> bool:
        """Majeur, annulaire, auriculaire repliés ET index pas recroquevillé dans la paume
        (sinon un simple poing fermé couperait le son)."""
        p = lm[:, :2]
        index_free = np.linalg.norm(p[8] - p[0]) > 0.9 * np.linalg.norm(p[6] - p[0])
        return index_free and not any(finger_extended(lm, f) for f in ("middle", "ring", "pinky"))

    def ratio_to_volume(self, ratio: float) -> float:
        return float(np.clip((ratio - self.vmin) / (self.vmax - self.vmin), 0.0, 1.0))

    def update(self, lm: np.ndarray | None) -> float | None:
        """Renvoie le volume à appliquer (0-1) quand la pose est tenue, sinon None."""
        if lm is None or not self.in_pose(lm):
            self._count, self.active, self.value = 0, False, None
            return None
        self._count += 1
        if self._count < self.engage_frames:
            return None
        target = self.ratio_to_volume(pinch_ratio(lm))
        self.value = target if self.value is None else (
            self.value + self.smoothing * (target - self.value))
        self.active = True
        return round(self.value, 3)


class SeekScrubber:
    """Main droite : pincer (pouce contre index) attrape la barre de progression,
    glisser horizontalement avance ou recule, relâcher applique le saut."""

    def __init__(self, on=config.SCRUB_PINCH_ON, off=config.SCRUB_PINCH_OFF,
                 engage_frames=config.SCRUB_ENGAGE_FRAMES,
                 seconds_per_width=config.SCRUB_SECONDS_PER_WIDTH):
        self.on, self.off = on, off
        self.engage_frames = engage_frames
        self.seconds_per_width = seconds_per_width
        self._count = 0
        self.active = False
        self._x0 = 0.0
        self._start_pos = 0.0
        self.target: float | None = None

    def cancel(self):
        self._count, self.active, self.target = 0, False, None

    def update(self, lm: np.ndarray | None, position: float, duration: float | None):
        """Renvoie la position (s) à appliquer au moment du relâchement, sinon None."""
        if lm is None:
            self.cancel()
            return None
        ratio = pinch_ratio(lm)
        x = float(palm_center(lm)[0])
        if not self.active:
            self._count = self._count + 1 if ratio < self.on else 0
            if self._count >= self.engage_frames:
                self.active, self._x0, self._start_pos = True, x, position
                self.target = position
            return None
        if ratio > self.off:               # relâché : on applique le saut
            result = self.target
            self.cancel()
            return result
        t = self._start_pos + (x - self._x0) * self.seconds_per_width
        hi = duration if duration else max(t, 0.0)
        self.target = float(np.clip(t, 0.0, hi))
        return None
