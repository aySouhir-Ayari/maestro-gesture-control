"""Suivi temporel de la main : trajectoire, vitesse et détection des swipes.

On garde l'historique récent de la position du centre de la paume
(coordonnées normalisées de l'image). Un swipe est un déplacement rapide,
ample et majoritairement dans une direction.
"""
from __future__ import annotations

import time
from collections import deque

import numpy as np

from . import config

LEFT, RIGHT, UP, DOWN = "left", "right", "up", "down"


class MotionTracker:
    def __init__(self, window_s=config.SWIPE_WINDOW_S, min_dist=config.SWIPE_MIN_DIST,
                 directionality=config.SWIPE_DIRECTIONALITY, cooldown_s=config.SWIPE_COOLDOWN_S,
                 clock=time.monotonic):
        self.window_s = window_s
        self.min_dist = min_dist
        self.directionality = directionality
        self.cooldown_s = cooldown_s
        self.clock = clock
        self.history: deque[tuple[float, np.ndarray]] = deque()
        self._last_swipe = -1e9

    def reset(self):
        self.history.clear()

    def update(self, point) -> str | None:
        """Ajoute la position de la main ; renvoie la direction d'un swipe détecté, sinon None."""
        now = self.clock()
        self.history.append((now, np.asarray(point, dtype=float)))
        while self.history and now - self.history[0][0] > self.window_s:
            self.history.popleft()
        if now - self._last_swipe < self.cooldown_s or len(self.history) < 3:
            return None

        start, end = self.history[0][1], self.history[-1][1]
        dx, dy = end - start
        direction = None
        if abs(dx) >= self.min_dist and abs(dx) >= self.directionality * abs(dy):
            direction = RIGHT if dx > 0 else LEFT
        elif abs(dy) >= self.min_dist and abs(dy) >= self.directionality * abs(dx):
            direction = DOWN if dy > 0 else UP  # y vers le bas dans l'image
        if direction:
            self._last_swipe = now
            self.history.clear()
        return direction

    def speed(self, span_s: float = 0.15) -> float:
        """Vitesse récente de la main (fraction d'image par seconde)."""
        if len(self.history) < 2:
            return 0.0
        t_end, p_end = self.history[-1]
        for t, p in self.history:
            if t_end - t <= span_s:
                dt = t_end - t
                return float(np.linalg.norm(p_end - p) / dt) if dt > 1e-6 else 0.0
        return 0.0

    def trail(self) -> list[np.ndarray]:
        """Points récents, pour dessiner la traînée du mouvement."""
        return [p for _, p in self.history]
