"""Cerveau de Maestro : rôles des deux mains, gestes, mouvements et modes d'affichage.

- Main « volume » (gauche par défaut) : volume continu au pincement.
- Main « morceaux » (droite par défaut) :
    gestes statiques (lecture/pause, volume ±, suivant, précédent),
    swipe gauche/droite -> playlist en carrousel (poing fermé = jouer le morceau choisi),
    swipe vers le haut   -> fiche du morceau en cours,
    pincer + glisser     -> avancer / reculer dans le morceau.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from . import config
from .controls import PinchVolume, SeekScrubber
from .features import palm_center
from .gestures import Prediction
from .hand_tracker import Hand
from .motion import DOWN, LEFT, RIGHT, UP, MotionTracker
from .stabilizer import GestureStabilizer

NORMAL, CAROUSEL = "normal", "carousel"
VOLUME_ROLE, TRACKS_ROLE = "volume", "tracks"


@dataclass
class FrameState:
    """Ce que l'interface doit afficher pour l'image courante."""
    track_hand: Hand | None = None
    volume_hand: Hand | None = None
    pred: Prediction | None = None
    progress: float = 0.0
    mode: str = NORMAL
    carousel_index: int = 0
    info_visible: bool = False
    volume_active: bool = False
    scrub_active: bool = False
    scrub_target: float | None = None
    trail: list = field(default_factory=list)
    events: list[str] = field(default_factory=list)
    # actions validées pour les animations : (type, position (x, y) normalisée de la main)
    actions: list[tuple[str, tuple[float, float] | None]] = field(default_factory=list)


def _side(hand: Hand) -> str:
    side = hand.handedness
    if config.SWAP_HANDEDNESS:
        side = "Left" if side == "Right" else "Right"
    return side


def assign_roles(hands: list[Hand], left_handed: bool = False) -> tuple[Hand | None, Hand | None]:
    """Renvoie (main_volume, main_morceaux)."""
    volume_side = "Right" if left_handed else "Left"
    if len(hands) >= 2:
        # deux mains : on garde les deux plus grandes
        a, b = sorted(hands, key=lambda h: h.area, reverse=True)[:2]
        if _side(a) == _side(b):
            # MediaPipe hésite : en vue miroir, la main gauche est à gauche de l'image
            a, b = sorted((a, b), key=lambda h: palm_center(h.landmarks)[0])
            left, right = a, b
            return (right, left) if left_handed else (left, right)
        vol = a if _side(a) == volume_side else b
        return vol, (b if vol is a else a)
    if len(hands) == 1:
        h = hands[0]
        return (h, None) if _side(h) == volume_side else (None, h)
    return None, None


class MaestroController:
    def __init__(self, classifier, player, left_handed: bool = config.LEFT_HANDED,
                 clock=time.monotonic):
        self.classifier = classifier
        self.player = player
        self.left_handed = left_handed
        self.clock = clock
        self.stabilizer = GestureStabilizer(clock=clock)
        self.motion = MotionTracker(clock=clock)
        self.pinch = PinchVolume()
        self.scrub = SeekScrubber()
        self.mode = NORMAL
        self.carousel_index = 0
        self._carousel_t = 0.0
        self._info_until = -1e9
        self._blocked_label = None
        self._pos = None
        self.last_command: str | None = None

    # --- utilitaires ----------------------------------------------------------
    def reset(self):
        """Passage en veille : on oublie tout geste en cours."""
        self.stabilizer.reset()
        self.motion.reset()
        self.pinch.update(None)
        self.scrub.cancel()
        self.mode = NORMAL
        self._blocked_label = None

    def show_info(self, seconds: float = config.INFO_DURATION_S):
        """Affiche la fiche du morceau (ex. quand un nouveau morceau commence)."""
        self._info_until = max(self._info_until, self.clock() + seconds)

    def _act(self, st: FrameState, kind: str):
        st.actions.append((kind, self._pos))

    def _open_carousel(self, st: FrameState):
        self.mode = CAROUSEL
        self.carousel_index = self.player.index
        self._carousel_t = self.clock()
        st.events.append("Playlist")
        self._act(st, "playlist")

    def _close_carousel(self):
        self.mode = NORMAL

    # --- une image ------------------------------------------------------------
    def step(self, hands: list[Hand]) -> FrameState:
        now = self.clock()
        st = FrameState()
        vol_hand, track_hand = assign_roles(hands, self.left_handed)
        st.volume_hand, st.track_hand = vol_hand, track_hand

        # 1. Main volume : pincement continu
        v = self.pinch.update(vol_hand.landmarks if vol_hand else None)
        if v is not None and abs(v - self.player.volume) >= 0.01:
            self.player.set_volume(v)
        st.volume_active = self.pinch.active

        # 2. Main morceaux
        if track_hand is None:
            self.motion.reset()
            self.scrub.cancel()
            self.stabilizer.update(config.NONE)
        else:
            self._track_hand_step(track_hand.landmarks, st, now)

        # 3. Délais d'affichage
        if self.mode == CAROUSEL and now - self._carousel_t > config.CAROUSEL_TIMEOUT_S:
            self._close_carousel()
        st.mode, st.carousel_index = self.mode, self.carousel_index
        st.info_visible = now < self._info_until
        st.progress = self.stabilizer.progress
        return st

    def _track_hand_step(self, lm, st: FrameState, now: float):
        self._pos = tuple(float(v) for v in palm_center(lm))
        # Avancer / reculer : pincer + glisser (mode normal uniquement)
        if self.mode == NORMAL and getattr(self.player, "can_seek", True):
            seek_to = self.scrub.update(lm, self.player.position, self.player.duration)
            if seek_to is not None:
                delta = seek_to - self.player.position
                self.player.seek(seek_to)
                st.events.append(f"{'Avance' if delta >= 0 else 'Recul'} {abs(delta):.0f} s")
                self._act(st, "seek_forward" if delta >= 0 else "seek_back")
        st.scrub_active, st.scrub_target = self.scrub.active, self.scrub.target
        if self.scrub.active:
            self.motion.reset()
            self.stabilizer.update(config.NONE)
            return

        # Swipes : trajectoire de la paume
        swipe = self.motion.update(palm_center(lm))
        st.trail = self.motion.trail()
        if swipe:
            self._on_swipe(swipe, st, now)
            pred = self.classifier.predict(lm)
            st.pred = pred
            self._blocked_label = pred.label   # le geste tenu après un swipe ne compte pas
            self.stabilizer.reset()
            return

        # Gestes statiques : seulement si la main est immobile
        pred = self.classifier.predict(lm)
        st.pred = pred
        label = pred.label
        if self.motion.speed() > config.STILL_SPEED:
            label = config.NONE
        if self._blocked_label is not None:
            if label == self._blocked_label:
                label = config.NONE
            else:
                self._blocked_label = None
        fired = self.stabilizer.update(label)
        if not fired:
            return
        if self.mode == CAROUSEL:
            if fired == config.FIST:
                self.player.play_index(self.carousel_index)
                self._close_carousel()
                self.last_command = "play_index"
                st.events.append(f"Lecture : {self.player.track_name}")
                self._act(st, "select")
            return
        if fired in config.COMMANDS:
            self.last_command = self.player.execute(config.COMMANDS[fired])
            st.events.append(config.GESTURE_NAMES_FR[fired])
            kind = self.last_command
            if kind == "play_pause":
                kind = "play" if getattr(self.player, "playing", True) else "pause"
            self._act(st, kind)

    def _on_swipe(self, swipe: str, st: FrameState, now: float):
        n = max(len(self.player.tracks), 1)
        if self.mode == NORMAL:
            if swipe in (LEFT, RIGHT) and not getattr(self.player, "has_playlist", True):
                # lecteur sans liste (touches média) : swipe = morceau suivant / précédent
                cmd = "next_track" if swipe == LEFT else "previous_track"
                self.last_command = self.player.execute(cmd)
                st.events.append("Morceau suivant" if swipe == LEFT else "Morceau precedent")
                self._act(st, cmd)
            elif swipe in (LEFT, RIGHT):
                self._open_carousel(st)
            elif swipe == UP:
                if now < self._info_until:
                    self._info_until = -1e9
                else:
                    self._info_until = now + config.INFO_DURATION_S
                    st.events.append("Infos du morceau")
                    self._act(st, "info")
            elif swipe == DOWN:
                self._info_until = -1e9
            return
        # mode carrousel
        self._carousel_t = now
        if swipe == LEFT:      # comme sur un téléphone : glisser à gauche = suivant
            self.carousel_index = (self.carousel_index + 1) % n
            self._act(st, "browse_next")
        elif swipe == RIGHT:
            self.carousel_index = (self.carousel_index - 1) % n
            self._act(st, "browse_prev")
        else:                  # haut ou bas : fermer la playlist
            self._close_carousel()
            st.events.append("Playlist fermee")
