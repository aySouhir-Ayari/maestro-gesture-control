"""Spectre audio du morceau, pour l'onde sonore animée synchronisée avec la musique.

Le fichier audio est analysé une fois, en arrière-plan (FFT par fenêtres de 50 ms,
bandes de fréquences logarithmiques). Pendant la lecture, on lit simplement la ligne
qui correspond à la position actuelle : aucun calcul lourd à chaque image.
"""
from __future__ import annotations

import threading
import wave
from pathlib import Path

import numpy as np

from . import config


def _load_samples(path: Path) -> tuple[np.ndarray, int] | None:
    """Échantillons mono (float32, -1..1) et fréquence d'échantillonnage."""
    try:
        if path.suffix.lower() == ".wav":
            with wave.open(str(path)) as w:
                sr, ch, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
                raw = w.readframes(w.getnframes())
            dtype = {1: np.uint8, 2: np.int16, 4: np.int32}[width]
            x = np.frombuffer(raw, dtype).astype(np.float32)
            if width == 1:
                x = x - 128
            x = x.reshape(-1, ch).mean(axis=1) / float(np.iinfo(dtype).max if width > 1 else 128)
            return x, sr
        import pygame  # mp3 / ogg : décodés par pygame (mixer déjà initialisé par le lecteur)

        snd = pygame.mixer.Sound(str(path))
        arr = pygame.sndarray.array(snd).astype(np.float32)
        if arr.ndim == 2:
            arr = arr.mean(axis=1)
        sr = pygame.mixer.get_init()[0]
        return arr / 32768.0, sr
    except Exception as exc:
        print(f"[Maestro] Analyse audio impossible pour {path.name} : {exc}")
        return None


def compute_spectrum(samples: np.ndarray, sr: int, bands: int = config.VISUALIZER_BANDS,
                     fps: int = config.VISUALIZER_FPS) -> np.ndarray:
    """Matrice (nb_fenêtres, bandes) de niveaux entre 0 et 1."""
    hop = max(int(sr / fps), 1)
    n_fft = 1 << (2 * hop - 1).bit_length()
    window = np.hanning(n_fft).astype(np.float32)
    n_frames = max(len(samples) // hop, 1)
    padded = np.concatenate([samples, np.zeros(n_fft, np.float32)])
    freqs = np.fft.rfftfreq(n_fft, 1 / sr)
    edges = np.geomspace(40, min(12000, sr / 2 - 1), bands + 1)
    idx = np.searchsorted(freqs, edges)
    out = np.zeros((n_frames, bands), np.float32)
    for f in range(n_frames):
        seg = padded[f * hop: f * hop + n_fft] * window
        mag = np.abs(np.fft.rfft(seg))
        for b in range(bands):
            lo, hi = idx[b], max(idx[b + 1], idx[b] + 1)
            out[f, b] = mag[lo:hi].mean()
    db = 20 * np.log10(out + 1e-6)
    ref = np.percentile(db, 98)
    return np.clip((db - (ref - 50)) / 50, 0, 1).astype(np.float32)


class SpectrumCache:
    """Analyse les morceaux en arrière-plan et garde les résultats en mémoire."""

    def __init__(self):
        self._data: dict[Path, np.ndarray | None] = {}
        self._pending: set[Path] = set()
        self._lock = threading.Lock()

    def request(self, path: Path):
        with self._lock:
            if path in self._data or path in self._pending:
                return
            self._pending.add(path)
        threading.Thread(target=self._work, args=(path,), daemon=True).start()

    def _work(self, path: Path):
        loaded = _load_samples(path)
        spec = compute_spectrum(*loaded) if loaded else None
        with self._lock:
            self._data[path] = spec
            self._pending.discard(path)

    def levels(self, path: Path | None, position: float) -> np.ndarray | None:
        """Niveaux des bandes à la position donnée (s), ou None si pas encore analysé."""
        if path is None:
            return None
        self.request(path)
        spec = self._data.get(path)
        if spec is None:
            return None
        i = int(position * config.VISUALIZER_FPS)
        return spec[min(max(i, 0), len(spec) - 1)]
