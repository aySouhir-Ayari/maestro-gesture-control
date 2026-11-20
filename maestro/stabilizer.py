"""Stabilisation temporelle : évite les commandes déclenchées par erreur.

Un geste n'est validé que s'il reste identique sur plusieurs images consécutives.
Ensuite :
- les gestes ponctuels (lecture/pause, pistes) ne se déclenchent qu'une fois :
  il faut relâcher le geste pour le refaire ;
- les gestes de volume se répètent tant qu'ils sont maintenus.
"""
from __future__ import annotations

import time

from . import config


class GestureStabilizer:
    def __init__(self, stable_frames=config.STABLE_FRAMES, cooldown=config.COOLDOWN_S,
                 repeat_interval=config.REPEAT_INTERVAL_S, clock=time.monotonic):
        self.stable_frames = stable_frames
        self.cooldown = cooldown
        self.repeat_interval = repeat_interval
        self.clock = clock
        self._candidate = config.NONE
        self._count = 0
        self._fired_label = None
        self._last_fire = -1e9

    @property
    def progress(self) -> float:
        """Avancement de la validation (0 -> 1), utile pour l'affichage."""
        if self._candidate == config.NONE:
            return 0.0
        return min(1.0, self._count / self.stable_frames)

    def update(self, label: str) -> str | None:
        """Reçoit le geste prédit pour l'image courante.
        Renvoie le geste validé à exécuter maintenant, sinon None."""
        now = self.clock()
        if label != self._candidate:
            self._candidate, self._count = label, 1
            if label != self._fired_label:
                self._fired_label = None  # geste relâché : on peut le refaire
            return None

        self._count += 1
        if label == config.NONE or self._count < self.stable_frames:
            return None

        if self._fired_label == label:
            # Geste déjà exécuté et toujours maintenu
            if label in config.REPEATABLE and now - self._last_fire >= self.repeat_interval:
                self._last_fire = now
                return label
            return None

        if now - self._last_fire < self.cooldown:
            return None
        self._fired_label = label
        self._last_fire = now
        return label
