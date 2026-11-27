"""Lecteur Spotify : les gestes pilotent l'application Spotify (vraie musique, vos playlists).

Même interface que MusicPlayer : le contrôleur et l'interface ne voient pas la différence.
La musique est jouée PAR l'application Spotify (PC ou téléphone) ; Maestro envoie les commandes
via l'API Web Spotify (bibliothèque spotipy).

Conditions (règles Spotify depuis février 2026) :
- compte Spotify **Premium** (obligatoire pour le mode développement et le contrôle de la lecture) ;
- une application créée sur https://developer.spotify.com/dashboard (Client ID + Client Secret) ;
- l'application Spotify ouverte sur un appareil (PC, téléphone...) ;
- la playlist doit vous appartenir (ou être collaborative) pour que la liste des morceaux soit lisible.

Les appels réseau se font dans un fil d'exécution séparé : la vidéo ne se fige jamais.
"""
from __future__ import annotations

import queue
import re
import threading
import time
import urllib.request

import numpy as np

from . import config

SCOPES = ("user-read-playback-state user-modify-playback-state user-read-currently-playing "
          "playlist-read-private playlist-read-collaborative")


def playlist_id_from(text: str) -> str | None:
    """Accepte un lien, une URI spotify:playlist:... ou un identifiant ; sinon None (= un nom)."""
    text = (text or "").strip()
    m = re.search(r"playlist[/:]([A-Za-z0-9]{22})", text)
    if m:
        return m.group(1)
    return text if re.fullmatch(r"[A-Za-z0-9]{22}", text) else None


def parse_track(entry: dict) -> dict | None:
    """Une entrée de playlist -> morceau (format 2026 « item » ou ancien « track »)."""
    t = entry.get("item") or entry.get("track")
    if not t or t.get("type", "track") != "track" or not t.get("uri") or t.get("is_local"):
        return None
    images = (t.get("album") or {}).get("images") or []
    return {
        "uri": t["uri"],
        "title": t.get("name", "?"),
        "artist": ", ".join(a.get("name", "") for a in t.get("artists", [])) or "Artiste inconnu",
        "album": (t.get("album") or {}).get("name", ""),
        "duration": (t.get("duration_ms") or 0) / 1000 or None,
        "image": min(images, key=lambda im: abs((im.get("width") or 300) - 300))["url"] if images else None,
    }


class SpotifyError(RuntimeError):
    pass


def friendly_error(exc: Exception) -> str:
    status = getattr(exc, "http_status", None)
    msg = str(getattr(exc, "msg", exc))
    if status == 403:
        return "Spotify refuse la commande (compte Premium requis ?)"
    if status == 404:
        return "Aucun appareil Spotify actif : ouvrez Spotify et lancez un morceau une fois"
    if status == 429:
        return "Trop de requêtes Spotify : patientez quelques secondes"
    if status == 401:
        return "Connexion Spotify expirée : relancez scripts/spotify_setup.py"
    return f"Spotify : {msg[:80]}"


class SpotifyPlayer:
    POLL_S = 1.0            # fréquence de lecture de l'état (morceau, position, pause)
    VOLUME_SEND_S = 0.3     # le volume au pincement change à chaque image : envoyé au plus 3 fois/s

    def __init__(self, sp=None, clock=time.monotonic, start_thread=True):
        self.clock = clock
        self.sp = sp or self._connect()
        self.playlist_name, self.playlist_uri, self.tracks = self._load_playlist(config.SPOTIFY_PLAYLIST)
        self.index = 0
        self.volume = 0.5
        self.playing = False
        self._started = False
        self._position = 0.0
        self._pos_t = clock()
        self._gain = 1.0
        self._duck_until = -1e9
        self._sent_volume = None
        self._volume_t = -1e9
        self._local_until = -1e9      # après une commande, on fait confiance à l'état local
        self._covers: dict[int, np.ndarray | None] = {}
        self._loading_covers: set[int] = set()
        self.device_id = None
        self.last_error = None
        self._jobs: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self.enabled = True
        self._refresh_device()
        self._poll()
        if start_thread:
            self._thread = threading.Thread(target=self._worker, daemon=True)
            self._thread.start()
        print(f"[Maestro] Spotify : playlist « {self.playlist_name} » ({len(self.tracks)} morceaux)")

    # --- connexion et playlist ---------------------------------------------------------------
    @staticmethod
    def _connect():
        try:
            import spotipy
            from spotipy.oauth2 import SpotifyOAuth
        except ImportError as exc:
            raise SystemExit("Spotify : installez d'abord  pip install spotipy") from exc
        if not (config.SPOTIFY_CLIENT_ID and config.SPOTIFY_CLIENT_SECRET):
            raise SystemExit("Spotify : renseignez SPOTIFY_CLIENT_ID et SPOTIFY_CLIENT_SECRET dans le fichier .env\n"
                             "(voir README, section Spotify).")
        auth = SpotifyOAuth(client_id=config.SPOTIFY_CLIENT_ID, client_secret=config.SPOTIFY_CLIENT_SECRET,
                            redirect_uri=config.SPOTIFY_REDIRECT_URI, scope=SCOPES,
                            cache_path=str(config.SPOTIFY_CACHE), open_browser=True)
        return spotipy.Spotify(auth_manager=auth, requests_timeout=6, retries=1)

    def my_playlists(self) -> list[dict]:
        out, offset = [], 0
        while True:
            page = self.sp.current_user_playlists(limit=50, offset=offset)
            out += [p for p in page.get("items", []) if p]
            if not page.get("next"):
                return out
            offset += 50

    def _playlist_items(self, pid: str) -> list[dict]:
        tracks, offset = [], 0
        while True:
            try:   # nouvel endpoint (2026) puis ancien en secours
                page = self.sp._get(f"playlists/{pid}/items", limit=50, offset=offset)
            except Exception:
                page = self.sp._get(f"playlists/{pid}/tracks", limit=50, offset=offset)
            tracks += [t for t in map(parse_track, page.get("items", [])) if t]
            if not page.get("next"):
                return tracks
            offset += 50

    def _load_playlist(self, wanted: str):
        pid = playlist_id_from(wanted)
        if pid is None:
            lists = self.my_playlists()
            if not lists:
                raise SystemExit("Spotify : aucune playlist dans votre compte.")
            if wanted:
                match = [p for p in lists if p["name"].strip().lower() == wanted.strip().lower()]
                if not match:
                    names = ", ".join(p["name"] for p in lists[:10])
                    raise SystemExit(f"Spotify : playlist « {wanted} » introuvable. Vos playlists : {names}")
                pl = match[0]
            else:
                pl = lists[0]
            pid = pl["id"]
        pl = self.sp._get(f"playlists/{pid}", fields="name,uri")
        tracks = self._playlist_items(pid)
        if not tracks:
            raise SystemExit(f"Spotify : impossible de lire les morceaux de « {pl.get('name')} ».\n"
                             "Depuis 2026, seules VOS playlists (ou collaboratives) sont lisibles : "
                             "copiez la playlist dans une nouvelle playlist à vous.")
        return pl.get("name", "Spotify"), pl.get("uri"), tracks

    def _refresh_device(self):
        try:
            devices = self.sp.devices().get("devices", [])
        except Exception as exc:
            self.last_error = friendly_error(exc)
            return
        wanted = config.SPOTIFY_DEVICE.strip().lower()
        pick = ([d for d in devices if wanted and wanted in d["name"].lower()]
                or [d for d in devices if d.get("is_active")] or devices)
        if pick:
            self.device_id = pick[0]["id"]
            if pick[0].get("volume_percent") is not None and self._sent_volume is None:
                self.volume = pick[0]["volume_percent"] / 100
                self._sent_volume = self._target_volume()
        else:
            self.device_id = None
            self.last_error = "Aucun appareil Spotify : ouvrez l'application Spotify"
            print(f"[Maestro] {self.last_error}")

    # --- fil réseau ----------------------------------------------------------------------------
    def _submit(self, fn, *args, **kwargs):
        self._local_until = self.clock() + 2.0
        self._jobs.put((fn, args, kwargs))

    def _call(self, fn, *args, **kwargs):
        try:
            fn(*args, **kwargs)
            self.last_error = None
        except Exception as exc:
            if getattr(exc, "http_status", None) == 404:      # appareil inactif : on le réveille
                self._refresh_device()
                if self.device_id and "device_id" in kwargs:
                    kwargs["device_id"] = self.device_id
                    try:
                        fn(*args, **kwargs)
                        self.last_error = None
                        return
                    except Exception as exc2:
                        exc = exc2
            self.last_error = friendly_error(exc)
            print(f"[Maestro] {self.last_error}")

    def _worker(self):
        last_poll = 0.0
        while not self._stop.is_set():
            try:
                fn, args, kwargs = self._jobs.get(timeout=0.1)
                self._call(fn, *args, **kwargs)
            except queue.Empty:
                pass
            self._flush_volume()
            if time.monotonic() - last_poll >= self.POLL_S:
                last_poll = time.monotonic()
                self._poll()

    def _target_volume(self) -> int:
        return int(round(100 * self.volume * self._gain))

    def _flush_volume(self, force=False):
        v = self._target_volume()
        if v != self._sent_volume and (force or self.clock() - self._volume_t >= self.VOLUME_SEND_S):
            self._volume_t = self.clock()
            self._sent_volume = v
            self._call(self.sp.volume, v, device_id=self.device_id)

    def _poll(self):
        """Met à jour l'état depuis Spotify (morceau changé sur le téléphone, fin de morceau...)."""
        try:
            pb = self.sp.current_playback()
        except Exception as exc:
            self.last_error = friendly_error(exc)
            return
        if not pb or self.clock() < self._local_until:
            return
        item = pb.get("item") or {}
        uri = item.get("uri")
        for i, t in enumerate(self.tracks):
            if t["uri"] == uri:
                self.index = i
                break
        self.playing = bool(pb.get("is_playing"))
        self._started = self._started or uri is not None
        self._position = (pb.get("progress_ms") or 0) / 1000
        self._pos_t = self.clock()
        dev = pb.get("device") or {}
        if dev.get("id"):
            self.device_id = dev["id"]

    # --- informations (même interface que MusicPlayer) ---------------------------------------
    spectrum = None

    def levels(self):
        return None            # pas d'accès au son de Spotify

    @property
    def ducked(self) -> bool:
        return self._gain < 0.99

    @property
    def current_path(self):
        return None

    @property
    def track_name(self) -> str:
        return self.title_of(self.index)

    def title_of(self, i: int) -> str:
        if not self.tracks:
            return "(aucune piste)"
        t = self.tracks[i % len(self.tracks)]
        return f"{t['artist']} - {t['title']}"

    def duration_of(self, i: int) -> float | None:
        return self.tracks[i % len(self.tracks)]["duration"] if self.tracks else None

    @property
    def duration(self) -> float | None:
        return self.duration_of(self.index)

    @property
    def position(self) -> float:
        pos = self._position + ((self.clock() - self._pos_t) if self.playing else 0.0)
        d = self.duration
        return min(pos, d) if d else pos

    def info(self) -> dict:
        t = self.tracks[self.index] if self.tracks else {}
        return {"title": t.get("title", "-"), "artist": t.get("artist", "-"), "index": self.index + 1,
                "count": len(self.tracks), "position": self.position, "duration": self.duration,
                "format": "Spotify", "volume": self.volume, "playing": self.playing,
                "album": t.get("album", ""), "playlist": self.playlist_name}

    def cover_of(self, i: int):
        """Vraie pochette de l'album (téléchargée en arrière-plan) ; None tant qu'elle n'est pas prête."""
        if not self.tracks:
            return None
        i %= len(self.tracks)
        if i not in self._covers and i not in self._loading_covers and self.tracks[i]["image"]:
            self._loading_covers.add(i)
            threading.Thread(target=self._download_cover, args=(i,), daemon=True).start()
        return self._covers.get(i)

    def _download_cover(self, i: int):
        import cv2

        from .covers import make_art_cover
        t = self.tracks[i]
        try:
            with urllib.request.urlopen(t["image"], timeout=6) as r:
                art = cv2.imdecode(np.frombuffer(r.read(), np.uint8), cv2.IMREAD_COLOR)
            self._covers[i] = make_art_cover(i, t["title"], t["artist"], art) if art is not None else None
        except Exception:
            self._covers[i] = None

    # --- commandes ----------------------------------------------------------------------------
    def execute(self, command: str) -> str:
        getattr(self, command)()
        return command

    def play_index(self, i: int):
        if not self.tracks:
            return
        self.index = i % len(self.tracks)
        self.playing, self._started = True, True
        self._position, self._pos_t = 0.0, self.clock()
        if self.playlist_uri:
            self._submit(self.sp.start_playback, device_id=self.device_id, context_uri=self.playlist_uri,
                         offset={"position": self.index})
        else:
            self._submit(self.sp.start_playback, device_id=self.device_id,
                         uris=[t["uri"] for t in self.tracks], offset={"position": self.index})

    def play_pause(self):
        if not self._started:
            self.play_index(self.index)
        elif self.playing:
            self._position, self._pos_t = self.position, self.clock()
            self.playing = False
            self._submit(self.sp.pause_playback, device_id=self.device_id)
        else:
            self._pos_t = self.clock()
            self.playing = True
            self._submit(self.sp.start_playback, device_id=self.device_id)

    def pause(self):
        if self.playing:
            self.play_pause()

    def next_track(self):
        self.play_index(self.index + 1)

    def previous_track(self):
        self.play_index(self.index - 1)

    def seek(self, seconds: float):
        d = self.duration
        seconds = max(0.0, min(seconds, (d - 1) if d else seconds))
        self._position, self._pos_t = seconds, self.clock()
        self._submit(self.sp.seek_track, int(seconds * 1000), device_id=self.device_id)

    def volume_up(self):
        self.set_volume(self.volume + config.VOLUME_STEP)

    def volume_down(self):
        self.set_volume(self.volume - config.VOLUME_STEP)

    def set_volume(self, v: float):
        self.volume = round(min(1.0, max(0.0, v)), 2)   # envoyé par le fil réseau (regroupé)

    def duck(self, seconds: float = config.DUCK_HOLD_S):
        if config.DUCK_ENABLED:
            self._duck_until = max(self._duck_until, self.clock() + seconds)
            self._gain = config.DUCK_LEVEL

    def unduck(self):
        self._duck_until = -1e9
        self._gain = 1.0

    def tick(self):
        if self._gain < 1.0 and self.clock() >= self._duck_until:
            self._gain = 1.0

    def close(self):
        self._stop.set()
