"""Activation vocale : dites « Maestro » pour lancer la détection, « Maestro stop » pour l'arrêter.

Reconnaissance vocale hors ligne avec Vosk (petit modèle français, téléchargé au
premier lancement). La reconnaissance est limitée à quelques mots (grammaire) :
elle est plus rapide et déclenche moins souvent par erreur.
"""
from __future__ import annotations

import difflib
import json
import queue
import threading
import time
import urllib.request
import zipfile
from pathlib import Path

from . import config

ACTIVATE = "activate"
DEACTIVATE = "deactivate"
SPEAKING = "speaking"   # « Maestro » entendu en début de phrase : on baisse la musique
ERROR = "error"


class VoiceCommandParser:
    """Transforme une phrase reconnue en événement d'activation / désactivation.

    Vosk coupe souvent « Maestro… stop » en deux phrases (« maestro » puis « stop »)
    quand on marque une pause : « stop » est donc accepté s'il arrive peu après « Maestro ».
    """

    def __init__(self, wake_words=config.WAKE_WORDS, stop_words=config.STOP_WORDS,
                 require_wake_for_stop=config.REQUIRE_WAKE_FOR_STOP,
                 stop_window_s=config.STOP_WINDOW_S, threshold=0.75, clock=time.monotonic):
        self.wake_words = [w.lower() for w in wake_words]
        self.stop_words = [w.lower() for w in stop_words]
        self.require_wake_for_stop = require_wake_for_stop
        self.stop_window_s = stop_window_s
        self.threshold = threshold
        self.clock = clock
        self._last_wake = -1e9

    def _match(self, words, targets) -> bool:
        # comparaison approximative : « mestro » ou « maestros » comptent aussi
        return any(difflib.SequenceMatcher(None, w, t).ratio() >= self.threshold
                   for w in words for t in targets)

    def mentions_wake(self, text: str) -> bool:
        """Le mot d'activation apparaît-il (même dans une phrase pas encore terminée) ?"""
        return self._match(text.lower().replace(",", " ").split(), self.wake_words)

    def parse(self, text: str) -> str | None:
        words = text.lower().replace(",", " ").split()
        if not words:
            return None
        now = self.clock()
        wake = self._match(words, self.wake_words)
        stop = self._match(words, self.stop_words)
        recent_wake = now - self._last_wake <= self.stop_window_s
        if stop and (wake or recent_wake or not self.require_wake_for_stop):
            self._last_wake = -1e9
            return DEACTIVATE
        if wake:
            self._last_wake = now
            return ACTIVATE
        return None


def ensure_vosk_model() -> Path:
    """Télécharge et décompresse le modèle Vosk français s'il est absent."""
    target = config.VOSK_MODEL_DIR
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    zip_path = target.with_suffix(".zip")
    print(f"[Maestro] Téléchargement du modèle vocal (~40 Mo) -> {target}")
    try:
        urllib.request.urlretrieve(config.VOSK_MODEL_URL, zip_path)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(target.parent)
    except Exception as exc:
        raise RuntimeError(
            f"Téléchargement du modèle vocal impossible ({exc}).\n"
            f"Téléchargez {config.VOSK_MODEL_URL}\net décompressez-le dans {target.parent}") from exc
    finally:
        zip_path.unlink(missing_ok=True)
    return target


class VoiceListener:
    """Écoute le micro dans un thread et place les événements dans une file."""

    SAMPLE_RATE = 16000

    def __init__(self, parser: VoiceCommandParser | None = None, debug: bool = False):
        self.parser = parser or VoiceCommandParser()
        self.events: queue.Queue = queue.Queue()
        self.debug = debug
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()
        return self

    def poll(self) -> list[tuple[str, str]]:
        """Événements reçus depuis le dernier appel : [(type, phrase), ...]."""
        out = []
        while not self.events.empty():
            out.append(self.events.get_nowait())
        return out

    def close(self):
        self._stop.set()

    def _run(self):
        try:
            import sounddevice as sd
            import vosk

            vosk.SetLogLevel(-1)
            model = vosk.Model(str(ensure_vosk_model()))
            words = self.parser.wake_words + self.parser.stop_words + ["[unk]"]
            rec = (vosk.KaldiRecognizer(model, self.SAMPLE_RATE, json.dumps(words))
                   if config.VOICE_USE_GRAMMAR else vosk.KaldiRecognizer(model, self.SAMPLE_RATE))
            block = 2000  # 1/8 s : réaction rapide
            last_speaking = 0.0
            with sd.RawInputStream(samplerate=self.SAMPLE_RATE, blocksize=block,
                                   dtype="int16", channels=1) as stream:
                print("[Maestro] Micro actif : dites « Maestro » pour commencer.")
                while not self._stop.is_set():
                    data, _ = stream.read(block)
                    if rec.AcceptWaveform(bytes(data)):
                        text = json.loads(rec.Result()).get("text", "")
                        if self.debug and text:
                            print(f"[voix] « {text} »")
                        event = self.parser.parse(text)
                        if event:
                            self.events.put((event, text))
                    else:
                        # résultat partiel : « Maestro » est en train d'être prononcé
                        partial = json.loads(rec.PartialResult()).get("partial", "")
                        now = time.monotonic()
                        if partial and now - last_speaking > 0.5 and self.parser.mentions_wake(partial):
                            last_speaking = now
                            self.events.put((SPEAKING, partial))
        except Exception as exc:  # pas de micro, Vosk absent, etc.
            self.events.put((ERROR, str(exc)))
