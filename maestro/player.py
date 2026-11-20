"""Lecteur musical (pygame.mixer) piloté par les gestes."""
from __future__ import annotations

from pathlib import Path

from . import config


class MusicPlayer:
    def __init__(self, music_dir: Path = config.MUSIC_DIR):
        self.tracks = sorted(p for p in Path(music_dir).glob("*")
                             if p.suffix.lower() in config.SUPPORTED_AUDIO)
        self.index = 0
        self.volume = 0.5
        self.playing = False
        self._started = False
        self.enabled = False
        try:
            import pygame

            self._pg = pygame
            pygame.mixer.init()
            pygame.mixer.music.set_volume(self.volume)
            self.enabled = bool(self.tracks)
        except Exception as exc:  # pas de carte son, etc.
            print(f"[Maestro] Audio indisponible : {exc}")
        if not self.tracks:
            print(f"[Maestro] Aucun fichier audio dans {music_dir} : mode simulation.")

    # --- informations -------------------------------------------------------
    @property
    def track_name(self) -> str:
        return self.tracks[self.index].stem if self.tracks else "(aucune piste)"

    # --- commandes ----------------------------------------------------------
    def execute(self, command: str) -> str:
        getattr(self, command)()
        return command

    def _load(self, play=True):
        if not self.enabled:  # mode simulation (pas de fichier audio)
            self._started = True
            self.playing = play
            return
        self._pg.mixer.music.load(str(self.tracks[self.index]))
        self._pg.mixer.music.play()
        self._started = True
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
            self.playing = False
        else:
            if self.enabled:
                self._pg.mixer.music.unpause()
            self.playing = True

    def volume_up(self):
        self._set_volume(self.volume + config.VOLUME_STEP)

    def volume_down(self):
        self._set_volume(self.volume - config.VOLUME_STEP)

    def _set_volume(self, v):
        self.volume = round(min(1.0, max(0.0, v)), 2)
        if self.enabled:
            self._pg.mixer.music.set_volume(self.volume)

    def next_track(self):
        if self.tracks:
            self.index = (self.index + 1) % len(self.tracks)
            self._load()

    def previous_track(self):
        if self.tracks:
            self.index = (self.index - 1) % len(self.tracks)
            self._load()

    def tick(self):
        """À appeler à chaque image : enchaîne la piste suivante en fin de morceau."""
        if self.enabled and self.playing and not self._pg.mixer.music.get_busy():
            self.next_track()

    def close(self):
        if self.enabled:
            self._pg.mixer.quit()
