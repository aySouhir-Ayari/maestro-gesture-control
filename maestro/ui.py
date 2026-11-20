"""Affichage OpenCV : squelette de la main, geste reconnu, état du lecteur."""
from __future__ import annotations

import unicodedata

import cv2
import numpy as np

from . import config

PINK = (189, 56, 255)   # BGR
CYAN = (238, 211, 34)
GREEN = (128, 222, 74)
GREY = (150, 150, 150)
WHITE = (250, 250, 250)

CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10),
               (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17),
               (17, 18), (18, 19), (19, 20)]
FINGER_LABELS = ["Pouce", "Index", "Majeur", "Annul.", "Auric."]


def ascii_text(s: str) -> str:
    """OpenCV ne sait pas afficher les accents : « levé » -> « leve »."""
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()


def draw_hand(frame, lm, main=True):
    h, w = frame.shape[:2]
    pts = [(int(x * w), int(y * h)) for x, y, _ in lm]
    color = CYAN if main else GREY
    for a, b in CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], color, 2, cv2.LINE_AA)
    for k, p in enumerate(pts):
        r = 6 if k in (4, 8, 12, 16, 20) else 4
        cv2.circle(frame, p, r, PINK if main else GREY, -1, cv2.LINE_AA)
    if main:
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        cv2.rectangle(frame, (min(xs) - 15, min(ys) - 15), (max(xs) + 15, max(ys) + 15), PINK, 1)


def _panel(frame, x0, y0, x1, y1, alpha=0.55):
    sub = frame[y0:y1, x0:x1]
    frame[y0:y1, x0:x1] = cv2.addWeighted(sub, 1 - alpha, np.zeros_like(sub), alpha, 0)


def draw_hud(frame, pred, progress, player, last_command, fps, classifier_name):
    h, w = frame.shape[:2]
    _panel(frame, 0, 0, w, 110)
    label = pred.label if pred else config.NONE
    cv2.putText(frame, "MAESTRO", (20, 40), cv2.FONT_HERSHEY_DUPLEX, 1.1, PINK, 2, cv2.LINE_AA)
    cv2.putText(frame, ascii_text(f"Geste : {config.GESTURE_NAMES_FR[label]}"), (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2, cv2.LINE_AA)
    if pred and pred.confidence < 1.0:
        cv2.putText(frame, f"{pred.confidence:.0%}", (380, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREY, 1)

    # barre de stabilité
    cv2.rectangle(frame, (20, 92), (320, 100), (70, 60, 90), -1)
    cv2.rectangle(frame, (20, 92), (20 + int(300 * progress), 100),
                  GREEN if progress >= 1 else CYAN, -1)

    # états des doigts
    if pred and pred.states:
        for k, key in enumerate(["thumb", "index", "middle", "ring", "pinky"]):
            cx = w - 330 + k * 62
            on = pred.states[key]
            cv2.circle(frame, (cx, 45), 16, GREEN if on else (90, 90, 110), -1, cv2.LINE_AA)
            cv2.putText(frame, FINGER_LABELS[k], (cx - 24, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1)

    # lecteur
    _panel(frame, 0, h - 70, w, h)
    state = "Lecture" if player.playing else "Pause"
    cv2.putText(frame, ascii_text(f"{state} | {player.track_name}"), (20, h - 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2, cv2.LINE_AA)
    cv2.putText(frame, "Vol", (20, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREY, 1)
    cv2.rectangle(frame, (55, h - 22), (255, h - 14), (70, 60, 90), -1)
    cv2.rectangle(frame, (55, h - 22), (55 + int(200 * player.volume), h - 14), CYAN, -1)
    if last_command:
        cv2.putText(frame, f"> {last_command}", (300, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, GREEN, 2)
    cv2.putText(frame, ascii_text(f"{fps:4.1f} FPS | {classifier_name} | Q : quitter"), (w - 420, h - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREY, 1)
