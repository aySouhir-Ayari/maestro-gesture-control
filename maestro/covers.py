"""Pochettes générées pour la playlist, style « affiche électro » violet nuit :
anneaux concentriques, ondes, grille en perspective et étoile rose."""
from __future__ import annotations

import math
import unicodedata
from functools import lru_cache

import cv2
import numpy as np

# couleurs BGR
LAVENDER = (255, 170, 190)
PINK = (210, 120, 255)
HOT_PINK = (200, 80, 255)
WHITE = (255, 255, 255)


def _ascii(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()


def _background(w: int, h: int, seed: int) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    t = (yy / h) * 0.8 + (xx / w) * 0.2
    top = np.array([110, 30, 50], np.float32)      # violet nuit (BGR)
    bottom = np.array([60, 15, 25], np.float32)
    img = top * (1 - t[..., None]) + bottom * t[..., None]
    cx, cy = w * (0.3 + 0.4 * ((seed * 37) % 10) / 10), h * 0.35
    glow = np.exp(-(((xx - cx) / (w * 0.6)) ** 2 + ((yy - cy) / (h * 0.5)) ** 2))
    img += glow[..., None] * np.array([90, 40, 70], np.float32)
    return np.clip(img, 0, 255).astype(np.uint8)


def _star(img, cx, cy, r, color=HOT_PINK):
    """Étoile à 4 branches avec halo."""
    pts = []
    for k in range(8):
        a = math.pi / 4 * k - math.pi / 2
        rr = r if k % 2 == 0 else r * 0.18
        pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
    glow = np.zeros_like(img)
    cv2.circle(glow, (int(cx), int(cy)), int(r * 0.8), color, -1, cv2.LINE_AA)
    img[:] = cv2.add(img, (cv2.GaussianBlur(glow, (0, 0), r * 0.5) * 0.5).astype(img.dtype))
    cv2.fillPoly(img, [np.array(pts, np.int32)], color, cv2.LINE_AA)
    cv2.fillPoly(img, [np.array([(cx + (p[0] - cx) * 0.45, cy + (p[1] - cy) * 0.45) for p in pts],
                                np.int32)], (240, 200, 255), cv2.LINE_AA)


def _rings(img, w, h):
    cx, cy = w // 2, int(h * 0.42)
    for k, r in enumerate(range(8, int(w * 0.45), 7)):
        col = PINK if k % 3 == 0 else LAVENDER
        cv2.circle(img, (cx, cy), r, col, 1, cv2.LINE_AA)
    # quart de disque décalé, comme un vinyle découpé
    mask = np.zeros(img.shape[:2], np.uint8)
    cv2.ellipse(mask, (cx, cy), (int(w * 0.45), int(w * 0.45)), 0, 270, 360, 255, -1)
    img[mask > 0] = (img[mask > 0] * 0.55).astype(np.uint8)
    _star(img, cx, cy, w * 0.07)


def _waves(img, w, h, seed):
    for k in range(22):
        pts = []
        for x in range(-10, w + 10, 4):
            y = (h * 0.30 + k * 5
                 + 28 * math.sin(x / w * 2 * math.pi * 1.2 + k * 0.18 + seed)
                 + 16 * math.sin(x / w * 2 * math.pi * 2.7 - k * 0.1))
            pts.append((x, int(y)))
        col = PINK if k % 4 == 0 else LAVENDER
        cv2.polylines(img, [np.array(pts, np.int32)], False, col, 1, cv2.LINE_AA)
    _star(img, w * 0.28, h * 0.45, w * 0.12)
    _star(img, w * 0.78, h * 0.18, w * 0.06)


def _grid(img, w, h):
    vx, vy = w // 2, int(h * 0.45)
    for k in range(-8, 9):
        cv2.line(img, (vx, vy), (vx + k * w // 6, h), LAVENDER, 1, cv2.LINE_AA)
    for j in range(1, 10):
        y = vy + int((h - vy) * (j / 10) ** 1.8)
        cv2.line(img, (0, y), (w, y), LAVENDER, 1, cv2.LINE_AA)
    for r in range(10, int(w * 0.5), 12):
        cv2.ellipse(img, (vx, vy + 20), (r, r // 3), 0, 0, 360, PINK, 1, cv2.LINE_AA)
    _star(img, vx, int(h * 0.30), w * 0.11)
    _star(img, vx, int(h * 0.52), w * 0.05)


def _text_center(img, text, y, scale, thick, color=WHITE, max_w=None):
    font = cv2.FONT_HERSHEY_DUPLEX
    max_w = max_w or img.shape[1] - 16
    while True:
        (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
        if tw <= max_w or scale < 0.35:
            break
        scale *= 0.9
    cv2.putText(img, text, ((img.shape[1] - tw) // 2, y), font, scale, color, thick, cv2.LINE_AA)


@lru_cache(maxsize=64)
def make_cover(index: int, title: str, w: int = 220, h: int = 300) -> np.ndarray:
    """Pochette du morceau n° index (image BGR w×h), toujours la même pour un morceau."""
    img = _background(w, h, index)
    style = index % 3
    if style == 0:
        _rings(img, w, h)
    elif style == 1:
        _waves(img, w, h, index)
    else:
        _grid(img, w, h)

    artist = ""
    name = _ascii(title)
    if " - " in name:  # « Artiste - Titre » ou « 01 - Titre »
        left, name = [x.strip() for x in name.split(" - ", 1)]
        artist = "" if left.isdigit() else left
    words = name.upper().split() or ["MAESTRO"]
    line1 = " ".join(words[:2])
    line2 = " ".join(words[2:4])
    cv2.putText(img, "MAESTRO", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.38, WHITE, 1, cv2.LINE_AA)
    cv2.putText(img, f"N.{index + 1:02d}", (w - 50, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.38, WHITE, 1,
                cv2.LINE_AA)
    base = (h - 72 if line2 else h - 46) if style != 2 else int(h * 0.16)
    _text_center(img, line1, base, 0.85, 2)
    if line2:
        _text_center(img, line2, base + 30, 0.85, 2)
    if artist:
        _text_center(img, artist.upper(), base - 30 if style != 2 else base + (60 if line2 else 28),
                     0.45, 1, LAVENDER)
    _text_center(img, "PLAY THE NIGHT AWAY", h - 12, 0.33, 1, LAVENDER)
    return img


def make_art_cover(index: int, title: str, artist: str, art: np.ndarray, w: int = 220, h: int = 300) -> np.ndarray:
    """Pochette au même format que make_cover, avec la vraie image de l'album (Spotify)."""
    img = _background(w, h, index)
    side = w - 20
    a = art
    if a.shape[0] != a.shape[1]:   # recadrage carré au centre
        k = min(a.shape[:2])
        y, x = (a.shape[0] - k) // 2, (a.shape[1] - k) // 2
        a = a[y:y + k, x:x + k]
    img[10:10 + side, 10:10 + side] = cv2.resize(a, (side, side), interpolation=cv2.INTER_AREA)
    cv2.rectangle(img, (10, 10), (10 + side, 10 + side), (255, 255, 255), 1, cv2.LINE_AA)
    _text_center(img, _ascii(title).upper()[:22], 10 + side + 30, 0.6, 2)
    _text_center(img, _ascii(artist).upper()[:30], 10 + side + 55, 0.4, 1, LAVENDER)
    return img
