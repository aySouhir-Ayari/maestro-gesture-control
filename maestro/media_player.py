"""Lecteur « media » : pilote l'application Spotify GRATUITE (ou tout lecteur) par les touches média de Windows.

Pas d'API ni de compte Premium : Maestro appuie virtuellement sur les touches
lecture/pause, suivant, précédent du clavier, exactement comme un casque Bluetooth.
- titre du morceau : lu dans le titre de la fenêtre Spotify (« Artiste - Titre ») ;
- volume au pincement : volume de Windows (précis avec pycaw, sinon par touches volume +/-).

Avec les paquets winrt (pip install -r requirements-media.txt), Maestro lit aussi ce que Windows
sait du morceau en cours : titre, artiste, album, position, durée et la VRAIE pochette envoyée
par Spotify. Sinon la pochette est cherchée dans le catalogue iTunes (artiste + titre).

Limites : pas de liste des morceaux de la playlist (le swipe gauche/droite passe directement
au morceau suivant/précédent), pas d'avance/recul dans le morceau.
Avec Spotify gratuit, Spotify peut limiter le nombre de morceaux passés par heure.
"""
from __future__ import annotations

import sys
import threading
import time

from . import config

VK_VOLUME_DOWN, VK_VOLUME_UP = 0xAE, 0xAF
VK_NEXT, VK_PREV, VK_PLAY_PAUSE = 0xB0, 0xB1, 0xB3
KEY_VOLUME_STEP = 0.02           # une touche volume = 2 % sur Windows
IDLE_TITLES = {"spotify", "spotify free", "spotify premium", "spotify gratuit"}


# --- accès Windows (ctypes, aucune dépendance) ------------------------------------------------
def press_key(vk: int):
    import ctypes
    user32 = ctypes.windll.user32
    user32.keybd_event(vk, 0, 0, 0)
    user32.keybd_event(vk, 0, 2, 0)          # KEYEVENTF_KEYUP


def window_titles_of(exe_name: str) -> list[str]:
    """Titres des fenêtres visibles appartenant au programme exe_name (ex. Spotify.exe)."""
    import ctypes
    from ctypes import wintypes
    user32, kernel32 = ctypes.windll.user32, ctypes.windll.kernel32
    titles = []

    def callback(hwnd, _):
        if not user32.IsWindowVisible(hwnd):
            return True
        n = user32.GetWindowTextLengthW(hwnd)
        if n == 0:
            return True
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        h = kernel32.OpenProcess(0x1000, False, pid.value)      # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return True
        try:
            buf, size = ctypes.create_unicode_buffer(512), wintypes.DWORD(512)
            if kernel32.QueryFullProcessImageNameW(h, 0, buf, ctypes.byref(size)):
                if buf.value.lower().endswith(exe_name.lower()):
                    title = ctypes.create_unicode_buffer(n + 1)
                    user32.GetWindowTextW(hwnd, title, n + 1)
                    titles.append(title.value)
        finally:
            kernel32.CloseHandle(h)
        return True

    proto = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    user32.EnumWindows(proto(callback), 0)
    return titles


class WindowsMediaSession:
    """Session média de Windows (celle affichée par le menu volume) : infos + pochette du morceau.

    Fonctionne dans un fil séparé (API asynchrone WinRT). available = False si winrt est absent.
    """
    POLL_S = 0.7

    def __init__(self, app_hint: str):
        self.app_hint = app_hint.lower().removesuffix(".exe")
        self.available = False
        self.title = self.artist = self.album = None
        self.playing = None
        self.position = 0.0
        self.duration = None
        self.thumbnail: bytes | None = None
        self.thumb_title = None            # titre auquel correspond self.thumbnail
        self.updated = 0.0
        try:
            import importlib
            importlib.import_module("winrt.windows.media.control")
        except Exception:
            return
        self.available = True
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        import asyncio
        asyncio.run(self._loop())

    async def _loop(self):
        import asyncio

        from winrt.windows.media.control import GlobalSystemMediaTransportControlsSessionManager as Manager
        try:
            manager = await Manager.request_async()
        except Exception:
            self.available = False
            return
        while True:
            try:
                await self._update(manager)
            except Exception:
                pass
            await asyncio.sleep(self.POLL_S)

    def _session(self, manager):
        sessions = list(manager.get_sessions())
        for s in sessions:
            if self.app_hint in (s.source_app_user_model_id or "").lower():
                return s
        return manager.get_current_session()

    @staticmethod
    def _seconds(value) -> float:
        if value is None:
            return 0.0
        if hasattr(value, "total_seconds"):
            return float(value.total_seconds())
        return float(getattr(value, "duration", value)) / 1e7      # ticks de 100 ns

    async def _update(self, manager):
        session = self._session(manager)
        if session is None:
            self.title = None
            return
        props = await session.try_get_media_properties_async()
        title, artist = props.title or None, props.artist or ""
        info = session.get_playback_info()
        self.playing = int(info.playback_status) == 4             # 4 = en lecture
        tl = session.get_timeline_properties()
        self.duration = self._seconds(tl.end_time) or None
        self.position = self._seconds(tl.position)
        if title and title != self.thumb_title and props.thumbnail is not None:
            self.thumbnail = await self._read_thumbnail(props.thumbnail)
            self.thumb_title = title if self.thumbnail else None
        self.title, self.artist, self.album = title, artist, props.album_title or ""
        self.updated = time.monotonic()

    @staticmethod
    async def _read_thumbnail(ref) -> bytes | None:
        from winrt.windows.storage.streams import Buffer, DataReader, InputStreamOptions
        stream = await ref.open_read_async()
        size = int(stream.size)
        if size <= 0:
            return None
        buf = Buffer(size)
        await stream.read_async(buf, size, InputStreamOptions.READ_AHEAD)
        try:
            return bytes(memoryview(buf))
        except TypeError:
            data = bytearray(buf.length)
            DataReader.from_buffer(buf).read_bytes(data)
            return bytes(data)


class WindowsVolume:
    """Volume général de Windows via pycaw (pip install pycaw). None si indisponible."""

    @staticmethod
    def create():
        try:
            from ctypes import POINTER, cast

            from comtypes import CLSCTX_ALL
            from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
            speakers = AudioUtilities.GetSpeakers()
            ep = getattr(speakers, "EndpointVolume", None)
            if ep is None:
                iface = speakers.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
                ep = cast(iface, POINTER(IAudioEndpointVolume))
            vol = WindowsVolume()
            vol._ep = ep
            return vol
        except Exception:
            return None

    def get(self) -> float:
        return float(self._ep.GetMasterVolumeLevelScalar())

    def set(self, v: float):
        self._ep.SetMasterVolumeLevelScalar(float(v), None)


def split_title(window_title: str | None):
    """« Artiste - Titre » -> (titre, artiste) ; None si rien ne joue (fenêtre « Spotify Free »)."""
    if not window_title or window_title.strip().lower() in IDLE_TITLES:
        return None
    if " - " in window_title:
        artist, title = window_title.split(" - ", 1)
        return title.strip(), artist.strip()
    return window_title.strip(), ""


class MediaKeysPlayer:
    has_playlist = False     # le contrôleur : swipe gauche/droite = suivant/précédent
    can_seek = False
    spectrum = None
    enabled = True
    index = 0
    current_path = None
    TITLE_POLL_S = 0.5
    VOLUME_SEND_S = 0.08

    def __init__(self, press=None, read_titles=None, volume_api="auto", clock=time.monotonic, session="auto",
                 artwork_loader=None):
        if press is None and sys.platform != "win32":
            raise SystemExit("Le lecteur « media » (touches média) fonctionne seulement sous Windows.")
        self.press = press or press_key
        self.read_titles = read_titles or (lambda: window_titles_of(config.MEDIA_APP_EXE))
        self.vol_api = WindowsVolume.create() if volume_api == "auto" else volume_api
        self.clock = clock
        self.volume = self.vol_api.get() if self.vol_api else 0.5
        self._sent_volume = self.volume
        self._volume_t = -1e9
        self._gain = 1.0
        self._duck_until = -1e9
        self._title_t = -1e9
        self.title, self.artist = None, ""
        self.album = ""
        self.playing = False
        from .artwork import ArtworkCache, itunes_artwork
        self.session = WindowsMediaSession(config.MEDIA_APP_EXE) if session == "auto" else session
        self._artwork = ArtworkCache()
        self._itunes = artwork_loader or itunes_artwork
        self._art_title = None
        self._art_cover = None
        self.app_found = False
        self.last_error = None
        self._read_title()
        if not self.vol_api:
            print("[Maestro] Volume par touches +/- (installez pycaw pour un volume précis : pip install pycaw)")
        if not self.app_found:
            self.last_error = f"Ouvrez l'application {config.MEDIA_APP_EXE.removesuffix('.exe')}"
            print(f"[Maestro] {self.last_error}")

    # --- état ---------------------------------------------------------------------------------
    def _read_title(self):
        sess = self.session
        if sess is not None and sess.available and sess.title and time.monotonic() - sess.updated < 3:
            # infos de Windows (plus complètes que le titre de la fenêtre)
            self.app_found = True
            self.title, self.artist, self.album = sess.title, sess.artist or "", sess.album or ""
            self.playing = bool(sess.playing)
            self.last_error = None
            return
        try:
            titles = self.read_titles()
        except Exception:
            titles = []
        self.app_found = bool(titles)
        parsed = next((p for p in map(split_title, titles) if p), None)
        if parsed:
            self.title, self.artist = parsed
            self.playing = True
            self.last_error = None
        elif self.app_found:
            self.playing = False            # Spotify ouvert mais en pause (titre « Spotify Free »)

    @property
    def position(self) -> float:
        sess = self.session
        if sess is not None and sess.available and sess.title == self.title:
            return sess.position
        return 0.0

    @property
    def duration(self):
        sess = self.session
        if sess is not None and sess.available and sess.title == self.title:
            return sess.duration
        return None

    def cover_of(self, i: int):
        """Vraie pochette du morceau en cours (Spotify via Windows, sinon catalogue iTunes)."""
        if not self.title:
            return None
        key = f"{self.artist} - {self.title}"
        sess = self.session
        if sess is not None and sess.available and sess.thumb_title != self.title and \
                not self._artwork.ready(key) and time.monotonic() - sess.updated < 2:
            return None                            # la pochette de Windows arrive : on attend
        title, artist, album = self.title, self.artist, self.album

        def load():
            if sess is not None and sess.thumb_title == title and sess.thumbnail:
                return sess.thumbnail
            return self._itunes(artist, title) if config.ONLINE_ARTWORK else None

        art = self._artwork.get(key, load)
        if art is None:
            return None
        if key != self._art_title:
            from .covers import make_art_cover
            self._art_cover, self._art_title = make_art_cover(0, title, artist or album, art), key
        return self._art_cover

    @property
    def tracks(self) -> list[str]:
        return [self.track_name]

    @property
    def track_name(self) -> str:
        if not self.title:
            return "Spotify : lancez un morceau"
        return f"{self.artist} - {self.title}" if self.artist else self.title

    def title_of(self, i: int) -> str:
        return self.track_name

    def duration_of(self, i: int):
        return None

    def levels(self):
        return None

    @property
    def ducked(self) -> bool:
        return self._gain < 0.99

    def info(self) -> dict:
        return {"title": self.title or "-", "artist": self.artist or "Spotify", "index": 1, "count": 1,
                "position": self.position, "duration": self.duration, "format": "Spotify",
                "volume": self.volume, "playing": self.playing}

    # --- commandes ----------------------------------------------------------------------------
    def execute(self, command: str) -> str:
        getattr(self, command)()
        return command

    def play_pause(self):
        self.press(VK_PLAY_PAUSE)
        self.playing = not self.playing
        self._title_t = self.clock() + 1.0      # laisser le temps à Spotify de changer son titre

    def pause(self):
        if self.playing:
            self.play_pause()

    def next_track(self):
        self.press(VK_NEXT)
        self._title_t = self.clock() - self.TITLE_POLL_S + 0.4

    def previous_track(self):
        self.press(VK_PREV)
        self._title_t = self.clock() - self.TITLE_POLL_S + 0.4

    def play_index(self, i: int):
        pass                                   # pas de liste : sans effet

    def seek(self, seconds: float):
        pass

    def volume_up(self):
        self.set_volume(self.volume + config.VOLUME_STEP)

    def volume_down(self):
        self.set_volume(self.volume - config.VOLUME_STEP)

    def set_volume(self, v: float):
        self.volume = round(min(1.0, max(0.0, v)), 2)
        self._apply_volume()

    def _apply_volume(self, force=False):
        target = round(self.volume * self._gain, 2)
        if abs(target - self._sent_volume) < 0.005:
            return
        if not force and self.clock() - self._volume_t < self.VOLUME_SEND_S:
            return
        self._volume_t = self.clock()
        if self.vol_api:
            try:
                self.vol_api.set(target)
                self._sent_volume = target
            except Exception as exc:
                self.last_error = f"Volume Windows : {exc}"
            return
        steps = round((target - self._sent_volume) / KEY_VOLUME_STEP)
        for _ in range(min(abs(steps), 25)):
            self.press(VK_VOLUME_UP if steps > 0 else VK_VOLUME_DOWN)
        self._sent_volume = round(self._sent_volume + KEY_VOLUME_STEP * steps, 2)

    def duck(self, seconds: float = config.DUCK_HOLD_S):
        if config.DUCK_ENABLED:
            self._duck_until = max(self._duck_until, self.clock() + seconds)
            if self._gain != config.DUCK_LEVEL:
                self._gain = config.DUCK_LEVEL
                self._apply_volume(force=True)

    def unduck(self):
        self._duck_until = -1e9
        if self._gain != 1.0:
            self._gain = 1.0
            self._apply_volume(force=True)

    def tick(self):
        now = self.clock()
        if self._gain < 1.0 and now >= self._duck_until:
            self.unduck()
        self._apply_volume()                   # termine un volume regroupé
        if now - self._title_t >= self.TITLE_POLL_S:
            self._title_t = now
            self._read_title()

    def close(self):
        if self._gain != 1.0:
            self.unduck()
