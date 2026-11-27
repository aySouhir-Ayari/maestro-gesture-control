"""Vérifie le micro et la reconnaissance du mot « Maestro ».

Ce script teste SEULEMENT le micro : il n'ouvre pas la caméra.
Pour utiliser Maestro avec les gestes : python main.py

Utilisation : python scripts/test_voice.py
Parlez : chaque phrase reconnue s'affiche, avec l'événement déclenché.
Ctrl+C pour quitter.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from maestro.voice import ERROR, VoiceListener  # noqa: E402


def main():
    listener = VoiceListener(debug=True).start()
    print("TEST DU MICRO UNIQUEMENT (pas de caméra). Pour l'application : python main.py")
    print("Dites « Maestro » puis « Maestro stop »... (Ctrl+C pour quitter)")
    try:
        while True:
            for event, text in listener.poll():
                if event == ERROR:
                    print("Erreur :", text)
                    return
                print(f"  -> {event.upper()}  (« {text} »)")
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        listener.close()


if __name__ == "__main__":
    main()
