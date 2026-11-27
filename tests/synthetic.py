"""Mains synthétiques (21 repères) pour tester les règles sans webcam."""
import math

import numpy as np

# base des doigts (MCP) relative au poignet, main verticale, unités arbitraires
_MCP = {"index": (-0.09, -0.30), "middle": (-0.02, -0.32), "ring": (0.05, -0.30), "pinky": (0.11, -0.26)}
_LEN = {"index": (0.12, 0.08, 0.07), "middle": (0.13, 0.09, 0.07),
        "ring": (0.12, 0.08, 0.06), "pinky": (0.09, 0.06, 0.05)}


def _finger(mcp, lens, extended):
    mcp = np.array(mcp)
    up = np.array([0.0, -1.0])
    if extended:
        pip = mcp + up * lens[0]
        dip = pip + up * lens[1]
        tip = dip + up * lens[2]
    else:  # replié : le doigt revient vers la paume
        pip = mcp + up * lens[0] * 0.6
        dip = pip + np.array([0.0, 1.0]) * lens[1] * 0.6 + np.array([0.01, 0])
        tip = dip + np.array([0.0, 1.0]) * lens[2] * 0.6
    return [mcp, pip, dip, tip]


def _thumb(mode):
    cmc = np.array([-0.10, -0.06])
    if mode == "side":      # écarté sur le côté (paume ouverte)
        d = np.array([-0.85, -0.53])
    elif mode == "out":     # pouce sorti du poing, perpendiculaire aux doigts
        d = np.array([-1.0, -0.05])
    else:                   # replié contre la paume
        mcp = cmc + np.array([0.04, -0.06])
        return [cmc, mcp, mcp + np.array([0.05, -0.02]), mcp + np.array([0.08, 0.02])]
    d = d / np.linalg.norm(d)
    return [cmc, cmc + d * 0.08, cmc + d * 0.15, cmc + d * 0.21]


def make_hand(thumb="side", fingers=(1, 1, 1, 1), rotation_deg=0.0, center=(0.5, 0.75), scale=1.0):
    pts = [np.array([0.0, 0.0])] + _thumb(thumb)
    for name, ext in zip(_MCP, fingers):
        pts += _finger(_MCP[name], _LEN[name], bool(ext))
    pts = np.array(pts)
    r = math.radians(rotation_deg)
    rot = np.array([[math.cos(r), -math.sin(r)], [math.sin(r), math.cos(r)]])
    pivot = np.array([0.0, -0.15])
    pts = (pts - pivot) @ rot.T + pivot
    pts = pts * scale + np.array(center)
    return np.hstack([pts, np.zeros((21, 1))]).astype(np.float32)


POSES = {
    "palm": dict(thumb="side", fingers=(1, 1, 1, 1)),
    # poing tourné de côté (jointures verticales), pouce vers le haut / le bas
    "thumbs_up": dict(thumb="out", fingers=(0, 0, 0, 0), rotation_deg=90),
    "thumbs_down": dict(thumb="out", fingers=(0, 0, 0, 0), rotation_deg=270),
    "two_fingers": dict(thumb="fold", fingers=(1, 1, 0, 0)),
    "index_left": dict(thumb="fold", fingers=(1, 0, 0, 0), rotation_deg=-90),
    "fist": dict(thumb="fold", fingers=(0, 0, 0, 0)),          # poing fermé
    "none": dict(thumb="fold", fingers=(1, 1, 1, 0)),          # trois doigts : aucun geste
}


def pinch_hand(gap: float, center=(0.5, 0.75), scale=1.0):
    """Main en pose « pincement » : index tendu, autres doigts repliés,
    bout du pouce placé à `gap` (fraction de taille de main) du bout de l'index."""
    lm = make_hand(thumb="side", fingers=(1, 0, 0, 0), center=center, scale=scale)
    size = np.linalg.norm(lm[9, :2] - lm[0, :2])
    lm[4, :2] = lm[8, :2] + np.array([-gap * size, 0.0])
    lm[3, :2] = lm[4, :2] + (lm[2, :2] - lm[4, :2]) * 0.4   # pouce resté droit
    return lm
