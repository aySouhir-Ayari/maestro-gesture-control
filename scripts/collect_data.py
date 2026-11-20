"""Collecte d'un dataset personnel de gestes (repères MediaPipe -> CSV).

Utilisation : python scripts/collect_data.py
  Maintenez un geste devant la caméra et gardez une touche appuyée :
    0 = aucun geste   1 = paume ouverte   2 = pouce levé
    3 = pouce baissé  4 = deux doigts     5 = index à gauche
  Chaque image enregistrée ajoute une ligne : label + 63 coordonnées normalisées.
  Q pour quitter. Variez la distance, l'angle, l'éclairage et la main utilisée.
"""
import csv
import sys
from collections import Counter
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from maestro import config  # noqa: E402
from maestro.camera import open_camera  # noqa: E402
from maestro.features import normalized_vector  # noqa: E402
from maestro.hand_tracker import HandTracker, select_main_hand  # noqa: E402
from maestro.ui import ascii_text, draw_hand  # noqa: E402

KEYS = {ord(str(i)): label for i, label in enumerate(config.LABELS)}


def main():
    config.DATA_DIR.mkdir(exist_ok=True)
    new_file = not config.DATASET_CSV.exists()
    counts = Counter()
    if not new_file:
        with open(config.DATASET_CSV, newline="") as f:
            counts.update(row[0] for row in csv.reader(f) if row and row[0] != "label")

    tracker, cap = HandTracker(), open_camera()
    with open(config.DATASET_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["label"] + [f"{a}{i}" for i in range(21) for a in "xyz"])
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            hand = select_main_hand(tracker.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if hand is not None:
                draw_hand(frame, hand.landmarks)
                if key in KEYS:
                    label = KEYS[key]
                    writer.writerow([label] + [f"{v:.5f}" for v in normalized_vector(hand.landmarks)])
                    counts[label] += 1
            for i, label in enumerate(config.LABELS):
                txt = ascii_text(f"{i} {config.GESTURE_NAMES_FR[label]} : {counts[label]}")
                cv2.putText(frame, txt, (15, 30 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (250, 250, 250), 2)
            cv2.imshow("Maestro - collecte", frame)
    cap.release()
    tracker.close()
    cv2.destroyAllWindows()
    print("Échantillons :", dict(counts), "->", config.DATASET_CSV)


if __name__ == "__main__":
    main()
