"""Maestro : lancez `python main.py` puis faites un geste devant la webcam.

Options :
  --classifier rules|ml   règles géométriques (défaut) ou classifieur entraîné
  --camera N              index de la webcam (défaut 0)
"""
import argparse
import time

import cv2

from maestro import config
from maestro.camera import open_camera
from maestro.gestures import MLClassifier, RuleClassifier
from maestro.hand_tracker import HandTracker, select_main_hand
from maestro.player import MusicPlayer
from maestro.stabilizer import GestureStabilizer
from maestro.ui import draw_hand, draw_hud


def main():
    parser = argparse.ArgumentParser(description="Maestro - contrôle musical par gestes")
    parser.add_argument("--classifier", choices=["rules", "ml"], default="rules")
    parser.add_argument("--camera", type=int, default=config.CAMERA_INDEX)
    args = parser.parse_args()

    classifier = MLClassifier() if args.classifier == "ml" else RuleClassifier()
    tracker = HandTracker()
    stabilizer = GestureStabilizer()
    player = MusicPlayer()
    cap = open_camera(args.camera)

    last_command, last_command_t = None, 0.0
    fps, prev_t = 0.0, time.monotonic()
    print(f"[Maestro] Prêt ({classifier.name}). Appuyez sur Q pour quitter.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if config.MIRROR:
                frame = cv2.flip(frame, 1)

            # 1. Détection + suivi de toutes les mains
            hands = tracker.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            main_hand = select_main_hand(hands)

            # 2. Classification du geste de la main principale
            pred = classifier.predict(main_hand.landmarks) if main_hand else None
            label = pred.label if pred else config.NONE

            # 3. Validation temporelle puis commande
            fired = stabilizer.update(label)
            if fired:
                last_command = player.execute(config.COMMANDS[fired])
                last_command_t = time.monotonic()
                print(f"[Maestro] {config.GESTURE_NAMES_FR[fired]} -> {last_command}")
            player.tick()

            # 4. Affichage
            for hand in hands:
                draw_hand(frame, hand.landmarks, main=hand is main_hand)
            now = time.monotonic()
            fps = 0.9 * fps + 0.1 / max(now - prev_t, 1e-6)
            prev_t = now
            shown = last_command if now - last_command_t < 1.5 else None
            draw_hud(frame, pred, stabilizer.progress, player, shown, fps, classifier.name)

            cv2.imshow("Maestro", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        cap.release()
        tracker.close()
        player.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
