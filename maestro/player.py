"""Lecteur musical (pygame.mixer) piloté par les gestes."""
from __future__ import annotations

import time
import wave
from pathlib import Path

import numpy as np  # noqa: F401  (annotation de levels)

from . import config


class MusicPlayer:
    def __init__(self, music_dir: Path = config.MUSIC_DIR, clock=time.monotonic):
        self.tracks = sorted(p for p in Path(music_dir).glob("*")
                             if p.suffix.lower() in config.SUPPORTED_AUDIO)
        self.index = 0
        self.volume = 0.5
        self.playing = False
        self._started = False
        self.enabled = False
        self.clock = clock
        self._offset = 0.0          # position (s) au dernier play / saut
        self._sim_t0 = None         # mode simulation : début de la lecture
        self._durations: dict[int, float | None] = {}
        self._gain = 1.0            # 1 = normal, < 1 = baissé pendant qu'on parle
        self._duck_until = -1e9
        self._last_tick = None
        try:
            import pygame

            self._pg = pygame
            pygame.mixer.init()
            pygame.mixer.music.set_volume(self.volume)
            self.enabled = bool(self.tracks)
            if config.SHOW_VISUALIZER:   # analyse audio seulement si l'onde est affichée
                from .visualizer import SpectrumCache

                self.spectrum = SpectrumCache()
        except Exception as exc:  # pas de carte son, etc.
            print(f"[Maestro] Audio indisponible : {exc}")
        if not self.tracks:
            print(f"[Maestro] Aucun fichier audio dans {music_dir} : mode simulation.")

    spectrum = None  # analyse audio pour l'onde sonore (créée si l'audio fonctionne)

    # --- informations -------------------------------------------------------
    @property
    def current_path(self) -> Path | None:
        return self.tracks[self.index] if self.tracks else None

    def levels(self) -> "np.ndarray | None":
        """Niveaux du spectre à la position actuelle (pour l'onde sonore animée)."""
        if self.spectrum is None or not self._started:
            return None
        return self.spectrum.levels(self.current_path, self.position)

    @property
    def ducked(self) -> bool:
        return self._gain < 0.99
    @property
    def track_name(self) -> str:
        return self.tracks[self.index].stem if self.tracks else "(aucune piste)"

    def cover_of(self, i: int):
        """Vraie pochette : image intégrée au fichier (MP3/FLAC/M4A), sinon catalogue iTunes
        si le nom de fichier est « Artiste - Titre ». None -> pochette générée."""
        if not self.tracks:
            return None
        if not hasattr(self, "_artwork"):
            from .artwork import ArtworkCache
            self._artwork, self._art_covers = ArtworkCache(), {}
        path = self.tracks[i % len(self.tracks)]
        key = str(path)
        if key in self._art_covers:
            return self._art_covers[key]
        artist, title = "", path.stem
        if " - " in path.stem:
            left, right = (x.strip() for x in path.stem.split(" - ", 1))
            if not left.isdigit():
                artist, title = left, right

        def load():
            from .artwork import embedded_artwork, itunes_artwork
            data = embedded_artwork(path)
            if data is None and artist and config.ONLINE_ARTWORK:
                data = itunes_artwork(artist, title)
            return data

        art = self._artwork.get(key, load)
        if art is None:
            if self._artwork.ready(key):
                self._art_covers[key] = None       # aucune vraie pochette : générée
            return None
        from .covers import make_art_cover
        self._art_covers[key] = make_art_cover(i, title, artist or "Artiste inconnu", art)
        return self._art_covers[key]

    def title_of(self, i: int) -> str:
        return self.tracks[i].stem if self.tracks else "(aucune piste)"

    def duration_of(self, i: int) -> float | None:
        """Durée d'un morceau en secondes (calculée une fois puis mémorisée)."""
        if not self.tracks:
            return None
        if i not in self._durations:
            path, d = self.tracks[i], None
            try:
                if path.suffix.lower() == ".wav":
                    with wave.open(str(path)) as w:
                        d = w.getnframes() / w.getframerate()
                elif self.enabled:
                    d = self._pg.mixer.Sound(str(path)).get_length()
            except Exception:
                d = None
            self._durations[i] = d
        return self._durations[i]

    @property
    def duration(self) -> float | None:
        return self.duration_of(self.index)

    @property
    def position(self) -> float:
        """Position actuelle dans le morceau (s)."""
        if not self._started:
            return 0.0
        if self.enabled:
            elapsed = max(self._pg.mixer.music.get_pos(), 0) / 1000
        else:
            elapsed = (self.clock() - self._sim_t0) if (self.playing and self._sim_t0 is not None) else 0.0
        pos = self._offset + elapsed
        d = self.duration
        return min(pos, d) if d else pos

    def info(self) -> dict:
        """Fiche du morceau en cours (swipe vers le haut)."""
        title = self.track_name
        artist = None
        if " - " in title:  # nom de fichier « Artiste - Titre » ou « 01 - Titre »
            left, right = title.split(" - ", 1)
            if not left.strip().isdigit():
                artist = left.strip()
            title = right.strip()
        path = self.tracks[self.index] if self.tracks else None
        return {
            "title": title,
            "artist": artist or "Artiste inconnu",
            "index": self.index + 1,
            "count": len(self.tracks),
            "position": self.position,
            "duration": self.duration,
            "format": path.suffix[1:].upper() if path else "-",
            "volume": self.volume,
            "playing": self.playing,
        }

    # --- commandes ----------------------------------------------------------
    def execute(self, command: str) -> str:
        getattr(self, command)()
        return command

    def _load(self, play=True, start=0.0):
        self._offset = start
        self._started = True
        if not self.enabled:  # mode simulation (pas de fichier audio)
            self.playing = play
            self._sim_t0 = self.clock() if play else None
            return
        self._pg.mixer.music.load(str(self.tracks[self.index]))
        self._pg.mixer.music.play(start=start)
        self.playing = True
        if not play:
            self._pg.mixer.music.pause()
            self.playing = False

    def play_pause(self):
        if not self._started:
            self._load()
        elif self.playing:
            if self.enabled:
                self._pg.mixer.music.pause()
            else:
                self._offset = self.position
            self.playing = False
        else:
            if self.enabled:
                self._pg.mixer.music.unpause()
            else:
                self._sim_t0 = self.clock()
            self.playing = True

    def pause(self):
        """Met en pause si la musique joue (sans effet sinon)."""
        if self.playing:
            self.play_pause()

    def volume_up(self):
        self.set_volume(self.volume + config.VOLUME_STEP)

    def volume_down(self):
        self.set_volume(self.volume - config.VOLUME_STEP)

    def set_volume(self, v: float):
        self.volume = round(min(1.0, max(0.0, v)), 2)
        self._apply_volume()

    def _apply_volume(self):
        if self.enabled:
            self._pg.mixer.music.set_volume(self.volume * self._gain)

    def duck(self, seconds: float = config.DUCK_HOLD_S):
        """Baisse temporairement la musique (on parle à Maestro). Prolongé à chaque appel."""
        if config.DUCK_ENABLED:
            self._duck_until = max(self._duck_until, self.clock() + seconds)

    def unduck(self):
        self._duck_until = -1e9

    def next_track(self):
        if self.tracks:
            self.play_index((self.index + 1) % len(self.tracks))

    def previous_track(self):
        if self.tracks:
            self.play_index((self.index - 1) % len(self.tracks))

    def play_index(self, i: int):
        """Joue le morceau n° i depuis le début (choix dans la playlist)."""
        if self.tracks:
            self.index = i % len(self.tracks)
            self._load()

    def seek(self, seconds: float):
        """Avance / recule à la position donnée (s), en gardant l'état lecture/pause."""
        if not self.tracks:
            return
        d = self.duration
        seconds = max(0.0, min(seconds, (d - 0.5) if d else seconds))
        was_playing = self.playing or not self._started
        try:
            self._load(play=was_playing, start=seconds)
        except Exception as exc:  # format qui ne permet pas le saut
            print(f"[Maestro] Avance/recul impossible pour ce fichier : {exc}")

    def tick(self):
        """À appeler à chaque image : fondu de la baisse de volume
        et enchaînement de la piste suivante en fin de morceau."""
        now = self.clock()
        dt = 0.0 if self._last_tick is None else now - self._last_tick
        self._last_tick = now
        target = config.DUCK_LEVEL if now < self._duck_until else 1.0
        if self._gain != target:
            step = dt / max(config.DUCK_FADE_S, 1e-3)
            self._gain = (max(target, self._gain - step) if target < self._gain
                          else min(target, self._gain + step))
            self._apply_volume()
        if self.enabled and self.playing and not self._pg.mixer.music.get_busy():
            self.next_track()

    def close(self):
        if self.enabled:
            self._pg.mixer.quit()
