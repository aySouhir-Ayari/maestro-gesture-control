"""Interface OpenCV de Maestro, style « électro violet nuit ».

Toutes les tailles sont proportionnelles à la hauteur de l'image (échelle S = hauteur / 540) :
l'interface reste lisible et sans chevauchement quelle que soit la taille de la fenêtre.

- barre du haut : logo, geste reconnu, validation, doigts, état ÉCOUTE / VEILLE ;
- barre de lecture : pochette, titre, progression, volume, baisse de volume active ;
- animations à chaque geste validé (onde de choc sur la main + icône au centre) ;
- vues : playlist en carrousel, fiche du morceau, barre « attrapée », jauge de volume.
"""
from __future__ import annotations

import math
import time
import unicodedata

import cv2
import numpy as np

from . import config

# Couleurs (BGR)
PINK = (200, 80, 255)
LAVENDER = (255, 170, 190)
VIOLET = (190, 60, 120)
CYAN = (238, 211, 34)
GREEN = (128, 222, 74)
GREY = (160, 150, 160)
WHITE = (250, 250, 250)
NIGHT = (45, 12, 28)

FONT = cv2.FONT_HERSHEY_DUPLEX
SMALL = cv2.FONT_HERSHEY_SIMPLEX

CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10),
               (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17),
               (17, 18), (18, 19), (19, 20)]
FINGER_LETTERS = ["P", "I", "M", "A", "A"]


# --- échelle et utilitaires ---------------------------------------------------
def scale_of(frame) -> float:
    """Facteur d'échelle de l'interface : 1 pour une image de 540 px de haut."""
    return frame.shape[0] / 540


def top_h(frame) -> int:
    return int(64 * scale_of(frame))


def bar_h(frame) -> int:
    return int(88 * scale_of(frame))


def ascii_text(s: str) -> str:
    """OpenCV ne sait pas afficher les accents : « levé » -> « leve »."""
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()


def fmt_time(sec) -> str:
    if sec is None:
        return "--:--"
    sec = max(0, int(sec))
    return f"{sec // 60}:{sec % 60:02d}"


def _glass(frame, x0, y0, x1, y1, alpha=0.72, tint=NIGHT):
    """Panneau semi-transparent teinté violet nuit."""
    x0, y0 = max(int(x0), 0), max(int(y0), 0)
    x1, y1 = min(int(x1), frame.shape[1]), min(int(y1), frame.shape[0])
    if x1 <= x0 or y1 <= y0:
        return
    sub = frame[y0:y1, x0:x1]
    frame[y0:y1, x0:x1] = cv2.addWeighted(sub, 1 - alpha, np.full_like(sub, tint), alpha, 0)


_panel = _glass  # compatibilité


def _text(frame, txt, org, scale, color, thick=1, font=SMALL):
    cv2.putText(frame, ascii_text(txt), (int(org[0]), int(org[1])), font, scale, color,
                max(1, int(round(thick))), cv2.LINE_AA)


def _text_w(txt, scale, thick=1, font=SMALL) -> int:
    return cv2.getTextSize(ascii_text(txt), font, scale, max(1, int(round(thick))))[0][0]


def _fit(txt, scale, max_w, thick=1, font=SMALL) -> str:
    """Raccourcit le texte (avec « ... ») pour qu'il tienne dans max_w pixels."""
    txt = ascii_text(txt)
    if _text_w(txt, scale, thick, font) <= max_w:
        return txt
    while txt and _text_w(txt + "...", scale, thick, font) > max_w:
        txt = txt[:-1]
    return txt + "..."


def _paste(frame, img, cx, cy, alpha=1.0):
    h, w = img.shape[:2]
    x0, y0 = int(cx - w // 2), int(cy - h // 2)
    fx0, fy0 = max(x0, 0), max(y0, 0)
    fx1, fy1 = min(x0 + w, frame.shape[1]), min(y0 + h, frame.shape[0])
    if fx1 <= fx0 or fy1 <= fy0:
        return
    sub = img[fy0 - y0:fy1 - y0, fx0 - x0:fx1 - x0]
    roi = frame[fy0:fy1, fx0:fx1]
    frame[fy0:fy1, fx0:fx1] = cv2.addWeighted(sub, alpha, roi, 1 - alpha, 0)


def _mix(c1, c2, t):
    return tuple(int(a + (b - a) * t) for a, b in zip(c1, c2))


# --- icônes dessinées (pas d'emoji dans OpenCV) ------------------------------
def draw_icon(img, kind: str, c, s: int, color=WHITE, thick=3):
    """Icône vectorielle centrée en c, de demi-taille s."""
    x, y, s = int(c[0]), int(c[1]), max(int(s), 3)
    thick = max(int(thick), 1)

    def tri(cx, cy, r, direction=1):
        pts = np.array([(cx - r * 0.6 * direction, cy - r), (cx - r * 0.6 * direction, cy + r),
                        (cx + r * 0.9 * direction, cy)], np.int32)
        cv2.fillPoly(img, [pts], color, cv2.LINE_AA)

    if kind in ("play", "select"):
        tri(x, y, s)
        if kind == "select":
            cv2.circle(img, (x, y), int(s * 1.6), color, 2, cv2.LINE_AA)
    elif kind == "pause":
        for dx in (-s * 0.45, s * 0.45):
            cv2.rectangle(img, (int(x + dx - s * 0.22), y - s), (int(x + dx + s * 0.22), y + s), color, -1)
    elif kind in ("next_track", "previous_track", "seek_forward", "seek_back"):
        d = 1 if kind in ("next_track", "seek_forward") else -1
        tri(x - d * s * 0.5, y, s * 0.75, d)
        tri(x + d * s * 0.45, y, s * 0.75, d)
        if "track" in kind:
            bx = int(x + d * s * 1.25)
            cv2.rectangle(img, (bx - max(2, s // 8), y - int(s * 0.75)), (bx + max(2, s // 8), y + int(s * 0.75)),
                          color, -1)
    elif kind in ("volume_up", "volume_down", "volume"):
        body = np.array([(x - s, y - s * 0.35), (x - s * 0.5, y - s * 0.35), (x, y - s * 0.85),
                         (x, y + s * 0.85), (x - s * 0.5, y + s * 0.35), (x - s, y + s * 0.35)], np.int32)
        cv2.fillPoly(img, [body], color, cv2.LINE_AA)
        if kind == "volume":
            for r in (0.45, 0.85):
                cv2.ellipse(img, (x, y), (int(s * r), int(s * r)), 0, -45, 45, color, 2, cv2.LINE_AA)
        else:
            cx = int(x + s * 0.75)
            cv2.line(img, (cx - int(s * 0.35), y), (cx + int(s * 0.35), y), color, thick, cv2.LINE_AA)
            if kind == "volume_up":
                cv2.line(img, (cx, y - int(s * 0.35)), (cx, y + int(s * 0.35)), color, thick, cv2.LINE_AA)
    elif kind == "playlist":
        for dy in (-0.6, 0, 0.6):
            yy = int(y + dy * s)
            cv2.circle(img, (int(x - s * 0.8), yy), max(2, s // 8), color, -1, cv2.LINE_AA)
            cv2.line(img, (int(x - s * 0.5), yy), (int(x + s * 0.9), yy), color, thick, cv2.LINE_AA)
    elif kind in ("browse_next", "browse_prev"):
        d = 1 if kind == "browse_next" else -1
        pts = np.array([(x - d * s * 0.4, y - s * 0.8), (x + d * s * 0.4, y), (x - d * s * 0.4, y + s * 0.8)],
                       np.int32)
        cv2.polylines(img, [pts], False, color, thick + 1, cv2.LINE_AA)
    elif kind == "info":
        cv2.circle(img, (x, y), s, color, 2, cv2.LINE_AA)
        cv2.circle(img, (x, int(y - s * 0.45)), max(2, s // 8), color, -1, cv2.LINE_AA)
        cv2.line(img, (x, int(y - s * 0.1)), (x, int(y + s * 0.55)), color, thick, cv2.LINE_AA)
    elif kind == "mic":
        cv2.ellipse(img, (x, int(y - s * 0.2)), (max(int(s * 0.35), 2), max(int(s * 0.6), 2)), 0, 0, 360, color,
                    -1, cv2.LINE_AA)
        cv2.ellipse(img, (x, int(y - s * 0.1)), (int(s * 0.65), int(s * 0.65)), 0, 0, 180, color, 2, cv2.LINE_AA)
        cv2.line(img, (x, int(y + s * 0.55)), (x, int(y + s * 0.9)), color, 2, cv2.LINE_AA)


def _star(img, c, r, color=PINK):
    x, y = c
    pts = []
    for k in range(8):
        a = math.pi / 4 * k - math.pi / 2
        rr = r if k % 2 == 0 else r * 0.2
        pts.append((x + rr * math.cos(a), y + rr * math.sin(a)))
    cv2.fillPoly(img, [np.array(pts, np.int32)], color, cv2.LINE_AA)


# --- mains ------------------------------------------------------------------
def draw_hand(frame, lm, main=True, tag: str | None = None, color=None):
    h, w = frame.shape[:2]
    S = scale_of(frame)
    pts = [(int(x * w), int(y * h)) for x, y, _ in lm]
    color = color or (CYAN if main else GREY)
    lw = max(2, int(2 * S))
    for a, b in CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], color, lw, cv2.LINE_AA)
    for k, p in enumerate(pts):
        r = int((6 if k in (4, 8, 12, 16, 20) else 4) * S)
        cv2.circle(frame, p, r, PINK if main else GREY, -1, cv2.LINE_AA)
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    m = int(15 * S)
    if main:
        cv2.rectangle(frame, (min(xs) - m, min(ys) - m), (max(xs) + m, max(ys) + m), color, 1)
    if tag:
        tx = int(np.clip(min(xs) - m, 5, w - 160 * S))
        ty = int(np.clip(min(ys) - 25 * S, top_h(frame) + 25 * S, h - bar_h(frame) - 20 * S))
        _text(frame, tag, (tx, ty), 0.6 * S, color, 2 * S)


def draw_trail(frame, pts):
    """Traînée de la paume : visualise la trajectoire analysée pour les swipes."""
    h, w = frame.shape[:2]
    S = scale_of(frame)
    pix = [(int(p[0] * w), int(p[1] * h)) for p in pts]
    for k in range(1, len(pix)):
        cv2.line(frame, pix[k - 1], pix[k], PINK, max(1, int((1 + 5 * k / max(len(pix), 1)) * S)), cv2.LINE_AA)


def draw_volume_gauge(frame, lm, volume: float, active: bool):
    """Ligne pouce-index et jauge de volume à côté de la main « volume »."""
    h, w = frame.shape[:2]
    S = scale_of(frame)
    t = (int(lm[4][0] * w), int(lm[4][1] * h))
    i = (int(lm[8][0] * w), int(lm[8][1] * h))
    color = GREEN if active else GREY
    cv2.line(frame, t, i, color, max(2, int(3 * S)), cv2.LINE_AA)
    for p in (t, i):
        cv2.circle(frame, p, int(9 * S), color, 2, cv2.LINE_AA)
    x = max(int(min(lm[:, 0]) * w - 45 * S), 10)
    y0, y1 = int(min(lm[:, 1]) * h), int(max(lm[:, 1]) * h)
    bw = int(14 * S)
    cv2.rectangle(frame, (x, y0), (x + bw, y1), (70, 40, 60), -1)
    cv2.rectangle(frame, (x, y1 - int((y1 - y0) * volume)), (x + bw, y1), color, -1)
    _text(frame, f"{int(volume * 100)}%", (x - 6 * S, y0 - 10 * S), 0.65 * S, color, 2 * S)


# --- onde sonore (désactivée par défaut : config.SHOW_VISUALIZER) -------------
class Visualizer:
    """Barres qui suivent le spectre réel du morceau (option, désactivée par défaut)."""

    def __init__(self, bands: int = config.VISUALIZER_BANDS):
        self.bands = bands
        self.display = np.zeros(bands, np.float32)

    def draw(self, frame, levels, playing: bool, gain: float = 1.0):
        h, w = frame.shape[:2]
        S = scale_of(frame)
        k = np.arange(self.bands)
        t = time.monotonic()
        if not playing:
            target = 0.05 + 0.03 * np.sin(t * 1.5 + k * 0.4)
        elif levels is None:
            target = np.clip(0.25 + 0.22 * np.sin(t * 6 + k * 0.7) * np.sin(t * 2.3 + k * 0.23), 0.05, 1)
        else:
            target = np.clip(np.interp(np.linspace(0, len(levels) - 1, self.bands), np.arange(len(levels)),
                                       levels), 0.03, 1)
        self.display = np.maximum(target * (0.4 + 0.6 * gain), self.display * 0.82)
        yc, half = int(h - bar_h(frame) - 42 * S), int(34 * S)
        bw = (w - 40) / self.bands
        for i, v in enumerate(self.display):
            x0, x1 = int(20 + i * bw + 1.5), int(20 + (i + 1) * bw - 1.5)
            hh = max(int(v * half), 2)
            cv2.rectangle(frame, (x0, yc - hh), (x1, yc + hh), _mix(VIOLET, PINK, float(v)), -1)


# --- animations des gestes validés -------------------------------------------
ACTION_LABELS = {
    "play": "Lecture", "pause": "Pause", "volume_up": "Volume +", "volume_down": "Volume -",
    "next_track": "Morceau suivant", "previous_track": "Morceau precedent", "playlist": "Playlist",
    "browse_next": "Suivant", "browse_prev": "Precedent", "select": "Lecture du morceau",
    "info": "Infos du morceau", "seek_forward": "Avance", "seek_back": "Recul",
}
NO_TOAST = {"playlist", "browse_next", "browse_prev"}   # le carrousel suffit comme retour visuel


def _ease_out_back(x: float) -> float:
    c1, c3 = 1.70158, 2.70158
    return 1 + c3 * (x - 1) ** 3 + c1 * (x - 1) ** 2


class Animator:
    """Onde de choc sur la main + icône qui « pop » au centre à chaque geste validé."""

    def __init__(self, duration: float = config.ANIMATION_S):
        self.duration = duration
        self.items: list[tuple[str, tuple[float, float] | None, float]] = []

    def add(self, kind: str, pos=None):
        self.items.append((kind, pos, time.monotonic()))

    def draw(self, frame):
        h, w = frame.shape[:2]
        S = scale_of(frame)
        now = time.monotonic()
        self.items = [it for it in self.items if now - it[2] < self.duration]
        for kind, pos, t0 in self.items:
            p = (now - t0) / self.duration
            overlay = frame.copy()
            if pos is not None:  # onde de choc sur la main
                c = (int(pos[0] * w), int(pos[1] * h))
                for k in range(3):
                    r = int((25 + (p * 1.4 - k * 0.15) * 130) * S)
                    if r > 0:
                        cv2.circle(overlay, c, r, PINK if k % 2 == 0 else LAVENDER, max(1, int((4 - k) * S)),
                                   cv2.LINE_AA)
            if kind not in NO_TOAST:  # icône au centre
                pop = _ease_out_back(min(p * 3.5, 1.0))
                r = int(58 * S * pop)
                cx, cy = w // 2, h // 2 - int(30 * S)
                if r > 4:
                    cv2.circle(overlay, (cx, cy), r + int(10 * S), (80, 25, 60), -1, cv2.LINE_AA)
                    cv2.circle(overlay, (cx, cy), r + int(10 * S), PINK, max(2, int(3 * S)), cv2.LINE_AA)
                    draw_icon(overlay, kind, (cx, cy), int(r * 0.45), WHITE, 3 * S)
                label = ACTION_LABELS.get(kind, kind)
                lw = _text_w(label, 1.0 * S, 2 * S, FONT)
                _text(overlay, label, (cx - lw // 2, cy + 110 * S), 1.0 * S, WHITE, 2 * S, FONT)
            alpha = float(np.clip((1 - p) * 1.6, 0, 1))
            frame[:] = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)


# --- barres principales -------------------------------------------------------
def draw_hud(frame, pred, progress, player, banner, fps, classifier_name, visualizer: Visualizer | None = None):
    """Barre du haut, barre de lecture (et onde sonore si activée)."""
    h, w = frame.shape[:2]
    S = scale_of(frame)
    if visualizer is not None and config.SHOW_VISUALIZER:
        visualizer.draw(frame, player.levels() if hasattr(player, "levels") else None, player.playing,
                        getattr(player, "_gain", 1.0))
    _draw_top_bar(frame, pred, progress, fps, classifier_name)
    _draw_player_bar(frame, player)
    if banner:
        txt = _fit(banner, 0.65 * S, w * 0.6)
        bw = _text_w(txt, 0.65 * S) + int(40 * S)
        x0, y0 = (w - bw) // 2, top_h(frame) + int(12 * S)
        _glass(frame, x0, y0, x0 + bw, y0 + 34 * S, 0.8)
        cv2.rectangle(frame, (x0, y0), (x0 + bw, int(y0 + 34 * S)), PINK, 1, cv2.LINE_AA)
        _text(frame, txt, (x0 + 20 * S, y0 + 24 * S), 0.65 * S, WHITE)


def _draw_top_bar(frame, pred, progress, fps, classifier_name):
    h, w = frame.shape[:2]
    S = scale_of(frame)
    th = top_h(frame)
    _glass(frame, 0, 0, w, th, 0.75)
    cv2.line(frame, (0, th), (w, th), (120, 40, 110), 1)
    cy = int(32 * S)
    _star(frame, (int(26 * S), cy), 13 * S, PINK)
    _text(frame, "MAESTRO", (46 * S, cy + 11 * S), 0.95 * S, PINK, 2 * S, FONT)

    label = pred.label if pred else config.NONE
    name = config.GESTURE_NAMES_FR[label]
    # anneau de validation, puis nom du geste et, en dessous, les doigts tendus / pliés
    cx = int(max(0.27 * w, 225 * S))
    r = int(16 * S)
    cv2.circle(frame, (cx, cy), r, (90, 50, 80), max(2, int(3 * S)), cv2.LINE_AA)
    if progress > 0:
        cv2.ellipse(frame, (cx, cy), (r, r), -90, 0, int(360 * progress),
                    GREEN if progress >= 1 else PINK, max(2, int(3 * S)), cv2.LINE_AA)
    nx = int(cx + 28 * S)
    room = int(w - 175 * S) - nx
    _text(frame, _fit(name, 0.68 * S, room, 2 * S), (nx, 28 * S), 0.68 * S,
          WHITE if label != config.NONE else GREY, 2 * S)
    if pred and pred.states:
        for k, key in enumerate(["thumb", "index", "middle", "ring", "pinky"]):
            px, py = int(nx + 9 * S + k * 24 * S), int(47 * S)
            on = pred.states[key]
            cv2.circle(frame, (px, py), int(9 * S), GREEN if on else (80, 50, 75), -1, cv2.LINE_AA)
            _text(frame, FINGER_LETTERS[k], (px - 4 * S, py + 4 * S), 0.35 * S, NIGHT if on else GREY, S)
    info = f"{fps:4.1f} FPS  {classifier_name}"
    _text(frame, info, (w - _text_w(info, 0.42 * S) - 12 * S, th - 6 * S), 0.42 * S, GREY)


def cover_for(player, i: int, title: str):
    """Vraie pochette si le lecteur en fournit une (Spotify), sinon pochette générée."""
    from .covers import make_cover

    art = player.cover_of(i) if hasattr(player, "cover_of") else None
    return art if art is not None else make_cover(i, title)


def _draw_player_bar(frame, player):
    h, w = frame.shape[:2]
    S = scale_of(frame)
    bh = bar_h(frame)
    y0 = h - bh
    _glass(frame, 0, y0, w, h, 0.82)
    cv2.line(frame, (0, y0), (w, y0), (120, 40, 110), 1)

    # 1) pochette + titre : 0 -> 33 % de la largeur
    x = int(14 * S)
    cw, ch = int(54 * S), int(74 * S)
    if player.tracks:
        cover = cv2.resize(cover_for(player, player.index, player.track_name), (cw, ch), interpolation=cv2.INTER_AREA)
        _paste(frame, cover, x + cw // 2, y0 + bh // 2)
    info = player.info() if hasattr(player, "info") else {"title": player.track_name, "artist": ""}
    tx = x + cw + int(14 * S)
    max_w = int(w * 0.33) - tx
    _text(frame, _fit(info["title"], 0.62 * S, max_w, 1, FONT), (tx, y0 + 36 * S), 0.62 * S, WHITE, 1, FONT)
    _text(frame, _fit(info.get("artist", ""), 0.5 * S, max_w), (tx, y0 + 62 * S), 0.5 * S, LAVENDER)

    # 2) lecture/pause + progression : 36 % -> 70 %
    bx0, bx1 = int(w * 0.36), int(w * 0.70)
    draw_icon(frame, "pause" if player.playing else "play", ((bx0 + bx1) // 2, y0 + 28 * S), 11 * S, WHITE)
    pos, dur = player.position, player.duration
    by = int(y0 + 62 * S)
    cv2.line(frame, (bx0, by), (bx1, by), (90, 50, 80), max(3, int(4 * S)), cv2.LINE_AA)
    if dur:
        xp = bx0 + int((bx1 - bx0) * min(pos / dur, 1))
        cv2.line(frame, (bx0, by), (xp, by), PINK, max(3, int(4 * S)), cv2.LINE_AA)
        cv2.circle(frame, (xp, by), int(6 * S), WHITE, -1, cv2.LINE_AA)
    _text(frame, fmt_time(pos), (bx0, by - 12 * S), 0.45 * S, GREY)
    end = fmt_time(dur)
    _text(frame, end, (bx1 - _text_w(end, 0.45 * S), by - 12 * S), 0.45 * S, GREY)

    # 3) volume (+ indicateur de baisse quand on parle) : 74 % -> fin
    vx = int(w * 0.74)
    vy = int(y0 + 40 * S)
    ducked = getattr(player, "ducked", False)
    draw_icon(frame, "volume", (vx + 12 * S, vy), 11 * S, LAVENDER if not ducked else GREY)
    pct = f"{int(player.volume * 100)}%"
    pct_x = w - _text_w(pct, 0.55 * S) - int(14 * S)
    lx0, lx1 = int(vx + 34 * S), pct_x - int(12 * S)
    cv2.line(frame, (lx0, vy), (lx1, vy), (90, 50, 80), max(3, int(4 * S)), cv2.LINE_AA)
    eff = player.volume * getattr(player, "_gain", 1.0)
    cv2.line(frame, (lx0, vy), (lx0 + int((lx1 - lx0) * eff), vy), CYAN, max(3, int(4 * S)), cv2.LINE_AA)
    _text(frame, pct, (pct_x, vy + 6 * S), 0.55 * S, WHITE)
    if ducked:
        draw_icon(frame, "mic", (vx + 12 * S, y0 + 70 * S), 9 * S, PINK)
        _text(frame, "on vous ecoute...", (vx + 30 * S, y0 + 76 * S), 0.45 * S, PINK)


def draw_status(frame, active: bool, voice_on: bool, banner: str | None = None):
    """Badge ECOUTE / VEILLE, écran de veille et message de changement d'état."""
    h, w = frame.shape[:2]
    S = scale_of(frame)
    if not active:
        frame[:] = (frame * 0.3).astype(frame.dtype)
        _star(frame, (w // 2, int(h // 2 - 90 * S)), 26 * S, PINK)
        lw = _text_w("MAESTRO", 1.8 * S, 3 * S, FONT)
        _text(frame, "MAESTRO", ((w - lw) // 2, h // 2 - 20 * S), 1.8 * S, PINK, 3 * S, FONT)
        hint = "Dites << Maestro >> pour commencer" if voice_on else "Appuyez sur M pour commencer"
        hw = _text_w(hint, 0.8 * S, S, FONT)
        _text(frame, hint, ((w - hw) // 2, h // 2 + 30 * S), 0.8 * S, WHITE, S, FONT)
    label, color = ("ECOUTE", GREEN) if active else ("VEILLE", GREY)
    x, y = int(w - 150 * S), int(26 * S)
    cv2.circle(frame, (x, y), int(7 * S), color, -1, cv2.LINE_AA)
    if active:
        r = int((7 + 6 * (0.5 + 0.5 * math.sin(time.monotonic() * 4))) * S)
        cv2.circle(frame, (x, y), r, color, 1, cv2.LINE_AA)
    _text(frame, label, (x + 14 * S, y + 7 * S), 0.6 * S, color, 2 * S)
    if voice_on:
        draw_icon(frame, "mic", (w - 32 * S, y), 8 * S, color)
    _text(frame, "H : aide", (x + 14 * S, y + 30 * S), 0.45 * S, GREY)
    if banner:
        bw = _text_w(banner, 0.9 * S, 2 * S, FONT) + int(50 * S)
        x0, y0 = (w - bw) // 2, int(top_h(frame) + 50 * S)
        _glass(frame, x0, y0, x0 + bw, y0 + 48 * S, 0.85)
        cv2.rectangle(frame, (x0, y0), (x0 + bw, int(y0 + 48 * S)), PINK, 2, cv2.LINE_AA)
        _text(frame, banner, (x0 + 25 * S, y0 + 33 * S), 0.9 * S, WHITE, 2 * S, FONT)


def help_lines(player, left_handed: bool) -> list[tuple[str, str]]:
    """Liste (geste, effet) adaptée au lecteur et aux rôles des mains."""
    vol, trk = ("DROITE", "GAUCHE") if left_handed else ("GAUCHE", "DROITE")
    has_list = getattr(player, "has_playlist", True)
    can_seek = getattr(player, "can_seek", True)
    lines = [
        ("", f"MAIN {vol} = VOLUME"),
        ("Pincer pouce-index", "ecarter = plus fort, serrer = moins fort"),
        ("", f"MAIN {trk} = MORCEAUX"),
        ("Paume ouverte", "lecture / pause"),
        ("Pouce leve / baisse", "volume + / -"),
        ("Deux doigts (V)", "morceau suivant"),
        ("Index vers la gauche", "morceau precedent"),
    ]
    if has_list:
        lines += [("Swipe gauche / droite", "ouvre la playlist et la parcourt"),
                  ("Poing ferme", "joue le morceau choisi dans la playlist"),
                  ("Swipe haut / bas (playlist)", "ferme la playlist")]
    else:
        lines += [("Swipe gauche / droite", "morceau suivant / precedent")]
    lines += [("Swipe vers le haut", "fiche du morceau")]
    if can_seek:
        lines += [("Pincer + glisser", "avancer / reculer dans le morceau")]
    lines += [("", "VOIX ET CLAVIER"),
              ("<< Maestro >>", "active les gestes (la musique baisse)"),
              ("<< Maestro stop >>", "veille + pause"),
              ("M  /  F  /  H  /  Q", "veille  /  plein ecran  /  aide  /  quitter")]
    return lines


def draw_help(frame, player, left_handed: bool):
    """Panneau d'aide (touche H) : tous les gestes disponibles."""
    h, w = frame.shape[:2]
    S = scale_of(frame)
    lines = help_lines(player, left_handed)
    lh = min(26 * S, (h - top_h(frame) - 82 * S) / len(lines))   # tient toujours dans l'écran
    pw, ph = int(min(w * 0.86, 900 * S)), int(len(lines) * lh + 62 * S)
    x0, y0 = (w - pw) // 2, max(int((h - ph) // 2), int(top_h(frame) + 8 * S))
    _glass(frame, x0, y0, x0 + pw, y0 + ph, 0.92)
    cv2.rectangle(frame, (x0, y0), (x0 + pw, y0 + ph), PINK, 2, cv2.LINE_AA)
    _text(frame, "GESTES MAESTRO", (x0 + 25 * S, y0 + 38 * S), 0.9 * S, PINK, 2 * S, FONT)
    _text(frame, "H pour fermer", (x0 + pw - 150 * S, y0 + 36 * S), 0.5 * S, GREY)
    y = y0 + 38 * S + lh
    col2 = x0 + int(pw * 0.42)
    for gesture, effect in lines:
        if not gesture:
            _text(frame, effect, (x0 + 25 * S, y), 0.6 * S, CYAN, 2 * S)
        else:
            _text(frame, _fit(gesture, 0.58 * S, col2 - x0 - 50 * S), (x0 + 45 * S, y), 0.58 * S, WHITE, max(1, int(S)))
            _text(frame, _fit(effect, 0.55 * S, x0 + pw - col2 - 20 * S), (col2, y), 0.55 * S, LAVENDER)
        y += lh


# --- vues ---------------------------------------------------------------------
def draw_scrub(frame, position: float, target: float, duration):
    """Barre de progression « attrapée » : aperçu de la nouvelle position."""
    h, w = frame.shape[:2]
    S = scale_of(frame)
    x0, x1, y = int(w * 0.12), int(w * 0.88), int(h // 2 + 100 * S)
    _glass(frame, x0 - 30 * S, y - 70 * S, x1 + 30 * S, y + 40 * S, 0.8)
    cv2.rectangle(frame, (int(x0 - 30 * S), int(y - 70 * S)), (int(x1 + 30 * S), int(y + 40 * S)), PINK, 1,
                  cv2.LINE_AA)
    cv2.line(frame, (x0, y), (x1, y), (90, 60, 100), int(6 * S), cv2.LINE_AA)
    if duration:
        xp = x0 + int((x1 - x0) * min(position / duration, 1))
        xt = x0 + int((x1 - x0) * min(target / duration, 1))
        cv2.line(frame, (x0, y), (xt, y), PINK, int(6 * S), cv2.LINE_AA)
        cv2.circle(frame, (xp, y), int(7 * S), GREY, -1, cv2.LINE_AA)
        cv2.circle(frame, (xt, y), int(12 * S), WHITE, -1, cv2.LINE_AA)
    delta = target - position
    draw_icon(frame, "seek_forward" if delta >= 0 else "seek_back", (x0 + 12 * S, y - 33 * S), 10 * S, PINK)
    _text(frame, f"{fmt_time(target)}  ({'+' if delta >= 0 else '-'}{abs(delta):.0f} s)", (x0 + 40 * S, y - 25 * S),
          0.9 * S, WHITE, 2 * S, FONT)
    hint = "Glissez puis relachez"
    _text(frame, hint, (x1 - _text_w(hint, 0.6 * S), y - 25 * S), 0.6 * S, GREY)


def draw_carousel(frame, player, selected: int):
    """Playlist en ligne horizontale : le morceau sélectionné au centre."""
    h, w = frame.shape[:2]
    S = scale_of(frame)
    n = len(player.tracks)
    _glass(frame, 0, top_h(frame), w, h - bar_h(frame), 0.78)
    if n == 0:
        _text(frame, "Playlist vide : ajoutez des fichiers dans music/", (60 * S, h // 2), 0.8 * S, WHITE, 2 * S)
        return
    cy = int(h // 2 - 35 * S)
    left, right = min(2, (n - 1) // 2), min(2, n // 2)
    sizes = {0: (180, 245), 1: (130, 178), 2: (92, 126)}
    for off in sorted(range(-left, right + 1), key=abs, reverse=True):
        i = (selected + off) % n
        cw, ch = int(sizes[abs(off)][0] * S), int(sizes[abs(off)][1] * S)
        cover = cv2.resize(cover_for(player, i, player.title_of(i)), (cw, ch), interpolation=cv2.INTER_AREA)
        cx = int(w // 2 + off * (165 if abs(off) == 1 else 140 * 2 - 5) * S)
        _paste(frame, cover, cx, cy, 1.0 if off == 0 else 0.5)
        if off == 0:
            cv2.rectangle(frame, (cx - cw // 2 - 4, cy - ch // 2 - 4), (cx + cw // 2 + 4, cy + ch // 2 + 4),
                          PINK, max(2, int(3 * S)), cv2.LINE_AA)
        if i == player.index:
            _text(frame, "EN COURS", (cx - 38 * S, cy - ch // 2 - 12 * S), 0.5 * S, GREEN, 2 * S)
    title = _fit(f"{selected % n + 1}/{n}  {player.title_of(selected % n)}", 0.8 * S, w * 0.9, 2 * S, FONT)
    tw = _text_w(title, 0.8 * S, 2 * S, FONT)
    _text(frame, title, ((w - tw) // 2, cy + 155 * S), 0.8 * S, WHITE, 2 * S, FONT)
    hint = _fit("Swipe < > : parcourir   |   Poing ferme : jouer   |   Swipe haut : fermer", 0.5 * S, w * 0.95)
    tw = _text_w(hint, 0.5 * S)
    _text(frame, hint, ((w - tw) // 2, cy + 183 * S), 0.5 * S, LAVENDER)
    for x, kind in ((40 * S, "browse_prev"), (w - 40 * S, "browse_next")):
        draw_icon(frame, kind, (x, cy), 16 * S, PINK, 3 * S)


def draw_info_card(frame, player):
    """Fiche du morceau en cours (swipe vers le haut)."""
    info = player.info()
    h, w = frame.shape[:2]
    S = scale_of(frame)
    cw = int(min(470 * S, w * 0.6))
    x0, y0, x1, y1 = w - cw - int(20 * S), int(top_h(frame) + 60 * S), w - int(20 * S), int(top_h(frame) + 270 * S)
    _glass(frame, x0, y0, x1, y1, 0.85)
    cv2.rectangle(frame, (x0, y0), (x1, y1), PINK, 2, cv2.LINE_AA)
    if player.tracks:
        cover = cv2.resize(cover_for(player, player.index, player.track_name), (int(135 * S), int(185 * S)))
        _paste(frame, cover, x0 + int(85 * S), (y0 + y1) // 2)
    tx = int(x0 + 170 * S)
    max_w = x1 - tx - int(15 * S)
    _text(frame, _fit(info["title"], 0.8 * S, max_w, 2 * S, FONT), (tx, y0 + 40 * S), 0.8 * S, WHITE, 2 * S, FONT)
    lines = [
        info["artist"],
        f"Piste {info['index']}/{info['count']}  |  {info['format']}",
        f"{fmt_time(info['position'])} / {fmt_time(info['duration'])}",
        f"Volume {int(info['volume'] * 100)}%  |  {'Lecture' if info['playing'] else 'Pause'}",
    ]
    for k, line in enumerate(lines):
        _text(frame, _fit(line, 0.6 * S, max_w), (tx, y0 + (75 + 30 * k) * S), 0.6 * S,
              LAVENDER if k == 0 else WHITE)
    if info["duration"]:
        bx0, bx1, by = tx, x1 - int(20 * S), int(y1 - 22 * S)
        cv2.line(frame, (bx0, by), (bx1, by), (90, 60, 100), max(3, int(4 * S)))
        cv2.line(frame, (bx0, by), (bx0 + int((bx1 - bx0) * min(info["position"] / info["duration"], 1)), by),
                 PINK, max(3, int(4 * S)))
