"""Extraction de caractéristiques géométriques à partir des 21 repères.

Numérotation MediaPipe :
0 poignet | pouce 1-4 | index 5-8 | majeur 9-12 | annulaire 13-16 | auriculaire 17-20
(pour chaque doigt : MCP, PIP, DIP, TIP ; pour le pouce : CMC, MCP, IP, TIP)
"""
from __future__ import annotations

import numpy as np

WRIST = 0
THUMB = (1, 2, 3, 4)
FINGERS = {
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}
FINGER_ORDER = ["thumb", "index", "middle", "ring", "pinky"]


def _xy(lm: np.ndarray) -> np.ndarray:
    return lm[:, :2].astype(np.float64)


def hand_size(lm: np.ndarray) -> float:
    """Échelle de la main : distance poignet -> base du majeur."""
    p = _xy(lm)
    return float(np.linalg.norm(p[9] - p[WRIST])) + 1e-6


def angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle ABC en degrés."""
    v1, v2 = a - b, c - b
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def finger_extended(lm: np.ndarray, name: str) -> bool:
    """Doigt tendu si le bout est plus loin du poignet que l'articulation PIP
    et si le doigt est globalement droit."""
    p = _xy(lm)
    mcp, pip, dip, tip = FINGERS[name]
    farther = np.linalg.norm(p[tip] - p[WRIST]) > 1.1 * np.linalg.norm(p[pip] - p[WRIST])
    straight = angle(p[mcp], p[pip], p[tip]) > 150
    return bool(farther and straight)


def thumb_extended(lm: np.ndarray) -> bool:
    """Pouce tendu s'il est droit et que son bout s'éloigne de la base de
    l'auriculaire (un pouce replié vient au contraire vers l'intérieur de la paume)."""
    p = _xy(lm)
    cmc, mcp, ip, tip = THUMB
    straight = angle(p[mcp], p[ip], p[tip]) > 150
    away = np.linalg.norm(p[tip] - p[17]) > 1.15 * np.linalg.norm(p[mcp] - p[17])
    return bool(straight and away)


def finger_states(lm: np.ndarray) -> dict[str, bool]:
    states = {"thumb": thumb_extended(lm)}
    for name in FINGERS:
        states[name] = finger_extended(lm, name)
    return states


def direction(lm: np.ndarray, base: int, tip: int) -> np.ndarray:
    """Vecteur unitaire base -> bout (repère image : x vers la droite, y vers le bas)."""
    p = _xy(lm)
    v = p[tip] - p[base]
    return v / (np.linalg.norm(v) + 1e-9)


def normalized_vector(lm: np.ndarray) -> np.ndarray:
    """Vecteur de 63 valeurs pour le classifieur ML :
    repères recentrés sur le poignet et divisés par la taille de la main
    (invariant à la position et à la distance de la caméra)."""
    pts = lm.astype(np.float64).copy()
    pts -= pts[WRIST]
    pts /= hand_size(lm)
    return pts.flatten()


def pinch_ratio(lm: np.ndarray) -> float:
    """Écart bout du pouce - bout de l'index, divisé par la taille de la main."""
    p = _xy(lm)
    return float(np.linalg.norm(p[4] - p[8]) / hand_size(lm))


def palm_center(lm: np.ndarray) -> np.ndarray:
    """Centre de la paume (poignet + bases des doigts), stable quand les doigts bougent."""
    p = _xy(lm)
    return p[[0, 5, 9, 13, 17]].mean(axis=0)
