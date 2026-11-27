"""Maestro : lancez `python main.py`, dites « Maestro », puis faites un geste devant la webcam.

Main droite (morceaux) : gestes, swipe gauche/droite = playlist, swipe haut = infos,
                         pincer + glisser = avancer / reculer.
Main gauche (volume)   : écart pouce-index = volume.

Options :
  --classifier rules|ml|yolo   règles géométriques (défaut), classifieur entraîné ou YOLOv8 (approche B)
  --player local|media|spotify  fichiers de music/ (défaut), Spotify gratuit (touches média)
                          ou API Spotify (Premium)
  --camera N              index de la webcam (défaut 0)
  --no-voice              sans activation vocale : la détection démarre tout de suite
  --voice-debug           affiche dans le terminal les phrases reconnues
  --left-handed           inverse les rôles des mains

Touches : M = activer / mettre en veille, F = plein écran, Q = quitter.
"""
import argparse
import time

import cv2

from maestro import config
from maestro.activation import ActivationController
from maestro.camera import open_camera
from maestro.controller import CAROUSEL, MaestroController
from maestro.display import Window, to_canvas
from maestro.gestures import MLClassifier, RuleClassifier
from maestro.hand_tracker import HandTracker
from maestro.player import MusicPlayer
from maestro.ui import (GREEN, Animator, Visualizer, draw_carousel, draw_hand, draw_hud,
                        draw_help, draw_info_card, draw_scrub, draw_status, draw_trail, draw_volume_gauge)
from maestro.voice import ACTIVATE, DEACTIVATE, ERROR, SPEAKING, VoiceListener


def main():
    parser = argparse.ArgumentParser(description="Maestro - contrôle musical par gestes")
    parser.add_argument("--classifier", choices=["rules", "ml", "yolo"],
                        default=config.CLASSIFIER if config.CLASSIFIER in ("rules", "ml", "yolo") else "rules")
    parser.add_argument("--player", choices=["local", "media", "spotify"],
                        default=config.PLAYER if config.PLAYER in ("local", "media", "spotify") else "local",
                        help="local = fichiers de music/, media = Spotify gratuit (touches média), "
                             "spotify = API Spotify (Premium)")
    parser.add_argument("--camera", type=int, default=config.CAMERA_INDEX)
    parser.add_argument("--no-voice", action="store_true", help="désactive l'activation vocale")
    parser.add_argument("--voice-debug", action="store_true")
    parser.add_argument("--left-handed", action="store_true", help="main droite = volume")
    args = parser.parse_args()

    if args.classifier == "yolo":
        from maestro.yolo_detector import YoloClassifier
        classifier = YoloClassifier()
    else:
        classifier = MLClassifier() if args.classifier == "ml" else RuleClassifier()
    tracker = HandTracker()
    if args.player == "media":
        from maestro.media_player import MediaKeysPlayer
        player = MediaKeysPlayer()
    elif args.player == "spotify":
        from maestro.spotify_player import SpotifyPlayer
        player = SpotifyPlayer()
    else:
        player = MusicPlayer()
    brain = MaestroController(classifier, player, left_handed=args.left_handed or config.LEFT_HANDED)
    cap = open_camera(args.camera)

    voice = None
    if config.VOICE_ENABLED and not args.no_voice:
        voice = VoiceListener(debug=args.voice_debug).start()
    # Avec la voix : démarrage en veille. Sans : détection immédiate.
    activation = ActivationController(start_active=voice is None,
                                      auto_sleep_s=config.AUTO_SLEEP_S if voice else 0)

    visualizer, animator = Visualizer(), Animator()
    window = Window()
    banner, banner_t = None, 0.0
    shown_error = None
    show_help = False
    last_track = None
    fps, prev_t = 0.0, time.monotonic()
    print(f"[Maestro] Prêt ({classifier.name}). M : activer/veille, H : aide, F : plein écran, Q : quitter.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if config.MIRROR:
                frame = cv2.flip(frame, 1)

            # 0. Commandes vocales « Maestro » / « Maestro stop »
            if voice:
                for event, text in voice.poll():
                    if event == ERROR:
                        print(f"[Maestro] Voix indisponible ({text}). Utilisez la touche M.")
                        voice.close()
                        voice = None
                        activation.auto_sleep_s = 0
                        break
                    # baisse automatique de la musique dès qu'on commence à dire « Maestro »
                    if event in (SPEAKING, ACTIVATE):
                        player.duck()
                    elif event == DEACTIVATE:
                        player.unduck()
                    if activation.handle(event):
                        print(f"[Maestro] « {text} » -> {activation.message}")
                        if not activation.active and config.STOP_PAUSES_MUSIC:
                            player.pause()

            state = None
            if activation.active:
                # 1. Détection + suivi des mains (sur l'image d'origine, plus rapide)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if hasattr(classifier, "set_frame"):   # approche B : YOLO regarde l'image
                    classifier.set_frame(rgb)
                hands = tracker.process(rgb)
                state = brain.step(hands)
                for msg in state.events:
                    print(f"[Maestro] {msg}")
                    banner, banner_t = msg, time.monotonic()
                for kind, pos in state.actions:
                    animator.add(kind, pos)
                present = state.track_hand is not None or state.volume_hand is not None
                if activation.update(present):
                    print(f"[Maestro] {activation.message}")
                    brain.reset()
            else:
                brain.reset()
            player.tick()
            if player.track_name != last_track:       # nouveau morceau : fiche + pochette
                if last_track is not None and config.SHOW_NOW_PLAYING_S > 0:
                    brain.show_info(config.SHOW_NOW_PLAYING_S)
                last_track = player.track_name
            err = getattr(player, "last_error", None)   # Spotify : appareil absent, Premium...
            if err and err != shown_error:
                banner, banner_t = err, time.monotonic()
            shown_error = err

            # 2. Affichage sur une image agrandie (coordonnées des mains normalisées : inchangées)
            frame = to_canvas(frame)
            if state:
                if state.volume_hand:
                    draw_hand(frame, state.volume_hand.landmarks, main=False, tag="VOLUME",
                              color=GREEN)
                    draw_volume_gauge(frame, state.volume_hand.landmarks, player.volume,
                                      state.volume_active)
                if state.track_hand:
                    draw_trail(frame, state.trail)
                    draw_hand(frame, state.track_hand.landmarks, main=True, tag="MORCEAUX")
            now = time.monotonic()
            fps = 0.9 * fps + 0.1 / max(now - prev_t, 1e-6)
            prev_t = now
            shown = banner if now - banner_t < 1.5 else None
            draw_hud(frame, state.pred if state else None, state.progress if state else 0.0,
                     player, shown, fps, classifier.name, visualizer)
            if state and state.mode == CAROUSEL:
                draw_carousel(frame, player, state.carousel_index)
            elif state and state.info_visible:
                draw_info_card(frame, player)
            if state and state.scrub_active and state.scrub_target is not None:
                draw_scrub(frame, player.position, state.scrub_target, player.duration)
            animator.draw(frame)
            draw_status(frame, activation.active, voice is not None, activation.banner)
            if show_help:
                draw_help(frame, player, args.left_handed or config.LEFT_HANDED)

            window.show(frame)
            key = cv2.waitKey(1) & 0xFF
            if ord("A") <= key <= ord("Z"):     # majuscules (Verr. Maj) acceptées
                key += 32
            if key in (ord("q"), 27):
                break
            if key == ord("h"):
                show_help = not show_help
            if key == ord("f"):
                window.toggle_fullscreen()
            if key == ord("m"):
                activation.toggle()
                print(f"[Maestro] {activation.message}")
                if not activation.active and config.STOP_PAUSES_MUSIC:
                    player.pause()
    finally:
        if voice:
            voice.close()
        cap.release()
        tracker.close()
        player.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
