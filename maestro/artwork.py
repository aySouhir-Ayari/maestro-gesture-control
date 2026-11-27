"""Vraies pochettes des morceaux (pas de pochette générée quand on peut l'éviter).

Sources, dans l'ordre :
1. l'image fournie par le lecteur lui-même (Spotify via Windows, ou pochette intégrée au MP3) ;
2. recherche « artiste + titre » dans le catalogue iTunes (gratuit, sans clé) ;
3. sinon seulement : pochette générée (covers.make_cover).
Les téléchargements se font en arrière-plan : la vidéo ne se fige pas.
"""
from __future__ import annotations

import json
import threading
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

ITUNES_URL = "https://itunes.apple.com/search?{}"


def decode_image(data: bytes | None):
    if not data:
        return None
    import cv2
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    return img if img is not None and img.size else None


def _clean(s: str) -> str:
    """Enlève « (feat. ...) », « - Remastered 2011 »... qui gênent la recherche."""
    import re
    s = re.sub(r"\s*[\(\[][^)\]]*(feat|ft\.|remaster|version|edit|live)[^)\]]*[\)\]]", "", s, flags=re.I)
    s = re.sub(r"\s+-\s+.*(remaster|version|edit|live).*$", "", s, flags=re.I)
    return s.strip()


def itunes_artwork(artist: str, title: str, timeout: float = 6) -> bytes | None:
    """Pochette 600x600 du morceau trouvée dans le catalogue iTunes, ou None."""
    term = f"{_clean(artist)} {_clean(title)}".strip()
    if not term:
        return None
    url = ITUNES_URL.format(urllib.parse.urlencode({"term": term, "entity": "song", "limit": 5}))
    with urllib.request.urlopen(url, timeout=timeout) as r:
        results = json.loads(r.read().decode("utf-8")).get("results", [])
    if not results:
        return None
    want = _clean(title).lower()
    best = next((x for x in results if want and want in x.get("trackName", "").lower()), results[0])
    art = best.get("artworkUrl100")
    if not art:
        return None
    with urllib.request.urlopen(art.replace("100x100bb", "600x600bb"), timeout=timeout) as r:
        return r.read()


def embedded_artwork(path: Path) -> bytes | None:
    """Pochette intégrée au fichier audio (MP3, FLAC, M4A, OGG) ; nécessite mutagen."""
    try:
        import mutagen
    except ImportError:
        return None
    try:
        f = mutagen.File(str(path))
    except Exception:
        return None
    if f is None:
        return None
    for pic in getattr(f, "pictures", []) or []:            # FLAC
        return pic.data
    tags = f.tags or {}
    try:
        for key in tags.keys():
            if str(key).startswith("APIC"):                   # MP3 (ID3)
                return tags[key].data
        if "covr" in tags:                                     # M4A
            return bytes(tags["covr"][0])
        if "metadata_block_picture" in tags:                   # OGG
            import base64
            from mutagen.flac import Picture
            return Picture(base64.b64decode(tags["metadata_block_picture"][0])).data
    except Exception:
        return None
    return None


class ArtworkCache:
    """Pochettes prêtes à afficher, chargées une fois par morceau dans un fil séparé."""

    def __init__(self):
        self._images: dict[str, np.ndarray | None] = {}
        self._loading: set[str] = set()
        self._lock = threading.Lock()

    def get(self, key: str, loader):
        """Image BGR si prête ; sinon lance loader() (-> bytes ou None) et renvoie None."""
        with self._lock:
            if key in self._images:
                return self._images[key]
            if key in self._loading:
                return None
            self._loading.add(key)
        threading.Thread(target=self._load, args=(key, loader), daemon=True).start()
        return None

    def _load(self, key, loader):
        try:
            img = decode_image(loader())
        except Exception:
            img = None
        with self._lock:
            self._images[key] = img
            self._loading.discard(key)

    def ready(self, key: str) -> bool:
        return key in self._images
