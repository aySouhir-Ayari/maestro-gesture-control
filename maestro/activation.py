"""État veille / écoute de Maestro.

- En veille : la caméra tourne mais aucune détection n'est faite (pas de fausse commande).
- Actif : détection, tracking et commandes.
- Retour automatique en veille si aucune main n'est vue pendant `auto_sleep_s` secondes.
"""
from __future__ import annotations

import time

from . import config
from .voice import ACTIVATE, DEACTIVATE


class ActivationController:
    def __init__(self, start_active: bool, auto_sleep_s: float = config.AUTO_SLEEP_S,
                 clock=time.monotonic):
        self.clock = clock
        self.active = start_active
        self.auto_sleep_s = auto_sleep_s
        self._last_hand = clock()
        self.message, self._message_t = None, -1e9

    def _set(self, active: bool, message: str) -> bool:
        changed = active != self.active
        self.active = active
        self._last_hand = self.clock()
        self.message, self._message_t = message, self.clock()
        return changed

    def handle(self, event: str) -> bool:
        """Applique un événement vocal. Renvoie True si l'état a changé."""
        if event == ACTIVATE:
            return self._set(True, "Maestro vous ecoute")
        if event == DEACTIVATE:
            return self._set(False, "Maestro en veille")
        return False

    def toggle(self) -> bool:
        return self.handle(DEACTIVATE if self.active else ACTIVATE)

    def update(self, hand_present: bool) -> bool:
        """À appeler à chaque image quand Maestro est actif. Renvoie True s'il s'endort."""
        now = self.clock()
        if not self.active:
            return False
        if hand_present:
            self._last_hand = now
            return False
        if self.auto_sleep_s and now - self._last_hand > self.auto_sleep_s:
            return self._set(False, "Aucune main : retour en veille")
        return False

    @property
    def banner(self) -> str | None:
        """Message à afficher pendant 2 s après un changement d'état."""
        return self.message if self.clock() - self._message_t < 2.0 else None
