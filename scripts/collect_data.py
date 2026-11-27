"""Collecte de votre propre dataset de gestes.

Utilisation :
    python scripts/collect_data.py --lighting normal
    python scripts/collect_data.py --lighting faible        (refaites une session par éclairage)

Maintenez un geste devant la caméra et gardez la touche correspondante appuyée :
    0 = aucun geste   1 = paume ouverte   2 = pouce levé
    3 = pouce baissé  4 = deux doigts     5 = index à gauche
    6 = poing fermé (sert à choisir un morceau dans la playlist)
Q pour quitter. Variez la distance, l'angle, la main utilisée et l'arrière-plan.

Chaque échantillon ajoute une ligne à data/landmarks.csv :
    label, éclairage, image, main, boîte de la main (x0 y0 x1 y1), 63 coordonnées normalisées.
Une image sur N (--image-every) est aussi enregistrée dans data/images/<geste>/ :
elles servent à évaluer l'approche B (YOLO) sur les mêmes gestes, et à entraîner YOLO
sur les gestes absents de HaGRID (index à gauche).
"""
import argparse
import csv
import sys
import time
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
LIGHTINGS = ["normal", "faible", "fort", "contre-jour"]
HEADER = (["label", "lighting", "image", "hand", "x_min", "y_min", "x_max", "y_max"]
          + [f"{a}{i}" for i in range(21) for a in "xyz"])


def hand_box(lm, margin=0.25):
    """Boîte englobante de la main (normalisée), élargie de 25 % comme les annotations HaGRID."""
    x0, y0 = lm[:, 0].min(), lm[:, 1].min()
    x1, y1 = lm[:, 0].max(), lm[:, 1].max()
    mx, my = (x1 - x0) * margin / 2, (y1 - y0) * margin / 2
    return max(x0 - mx, 0), max(y0 - my, 0), min(x1 + mx, 1), min(y1 + my, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lighting", choices=LIGHTINGS, default="normal",
                        help="condition d'éclairage de cette session")
    parser.add_argument("--image-every", type=int, default=4, help="garder 1 image sur N échantillons")
    parser.add_argument("--camera", type=int, default=config.CAMERA_INDEX)
    args = parser.parse_args()

    config.DATA_DIR.mkdir(exist_ok=True)
    images_dir = config.DATA_DIR / "images"
    new_file = not config.DATASET_CSV.exists()
    counts = Counter()
    if not new_file:
        with open(config.DATASET_CSV, newline="") as f:
            reader = csv.reader(f)
            header = next(reader, [])
            if header and header[:2] != ["label", "lighting"]:
                sys.exit(f"{config.DATASET_CSV} a l'ancien format : renommez-le puis relancez.")
            counts.update(row[0] for row in reader if row)

    tracker, cap = HandTracker(), open_camera(args.camera)
    session = time.strftime("%Y%m%d-%H%M%S")
    n_saved = 0
    with open(config.DATASET_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(HEADER)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            hand = select_main_hand(tracker.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            view = frame.copy()
            if hand is not None:
                draw_hand(view, hand.landmarks)
                if key in KEYS:
                    label = KEYS[key]
                    image_rel = ""
                    if n_saved % max(args.image_every, 1) == 0:
                        out = images_dir / label / f"{session}_{args.lighting}_{n_saved:05d}.jpg"
                        out.parent.mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(str(out), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                        image_rel = out.relative_to(config.DATA_DIR).as_posix()
                    box = hand_box(hand.landmarks)
                    writer.writerow([label, args.lighting, image_rel, hand.handedness]
                                    + [f"{v:.4f}" for v in box]
                                    + [f"{v:.5f}" for v in normalized_vector(hand.landmarks)])
                    counts[label] += 1
                    n_saved += 1
            cv2.putText(view, f"Eclairage : {args.lighting}", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (200, 80, 255), 2)
            for i, label in enumerate(config.LABELS):
                txt = ascii_text(f"{i} {config.GESTURE_NAMES_FR[label]} : {counts[label]}")
                cv2.putText(view, txt, (15, 55 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (250, 250, 250), 2)
            cv2.imshow("Maestro - collecte", view)
    cap.release()
    tracker.close()
    cv2.destroyAllWindows()
    print("Échantillons :", dict(counts), "->", config.DATASET_CSV)


if __name__ == "__main__":
    main()
