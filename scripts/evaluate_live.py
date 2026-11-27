"""Mode évaluation en direct (webcam) : protocole guidé, conditions réelles d'utilisation.

Utilisation :
    python scripts/evaluate_live.py --classifier rules --lighting normal
    python scripts/evaluate_live.py --classifier yolo --lighting faible --rounds 2

Déroulement (suivez les consignes à l'écran) :
  1. pour chaque geste : 3 s de préparation, puis 6 s pour le faire.
     -> réussite, temps jusqu'à la validation, geste validé par erreur ;
  2. 30 s de repos : bougez naturellement, parlez, SANS faire de commande.
     -> fausses activations par minute.
  FPS et latence (détection + classification) sont mesurés en continu.
Résultats : reports/live_results.csv (une ligne par session, pour comparer) + reports/live_<...>.md
Touche Q : arrêter.
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from maestro import config  # noqa: E402
from maestro.camera import open_camera  # noqa: E402
from maestro.display import Window, to_canvas  # noqa: E402
from maestro.hand_tracker import HandTracker, select_main_hand  # noqa: E402
from maestro.stabilizer import GestureStabilizer  # noqa: E402
from maestro.ui import ascii_text, draw_hand  # noqa: E402

PREP_S, TRY_S, REST_S = 3.0, 6.0, 30.0
TARGETS = config.GESTURES + [config.FIST]


def make_classifier(kind):
    if kind == "yolo":
        from maestro.yolo_detector import YoloClassifier
        return YoloClassifier()
    if kind == "ml":
        from maestro.gestures import MLClassifier
        return MLClassifier()
    from maestro.gestures import RuleClassifier
    return RuleClassifier()


def build_protocol(rounds):
    steps = []
    for r in range(rounds):
        for g in TARGETS:
            steps += [("prep", g, PREP_S), ("try", g, TRY_S)]
    steps.append(("rest", config.NONE, REST_S))
    return steps


def text(img, s, y, scale=1.0, color=(255, 255, 255)):
    S = img.shape[0] / 540
    cv2.putText(img, ascii_text(s), (int(20 * S), int(y * S)), cv2.FONT_HERSHEY_SIMPLEX, 0.8 * scale * S,
                (0, 0, 0), int(5 * S))
    cv2.putText(img, ascii_text(s), (int(20 * S), int(y * S)), cv2.FONT_HERSHEY_SIMPLEX, 0.8 * scale * S,
                color, max(1, int(2 * S)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--classifier", choices=["rules", "ml", "yolo"], default="rules")
    parser.add_argument("--lighting", default="normal", help="étiquette de la condition (normal, faible...)")
    parser.add_argument("--rounds", type=int, default=1, help="nombre de passages par geste")
    parser.add_argument("--camera", type=int, default=config.CAMERA_INDEX)
    args = parser.parse_args()

    classifier = make_classifier(args.classifier)
    tracker, cap, window = HandTracker(), open_camera(args.camera), Window("Maestro - evaluation")
    stab = GestureStabilizer()
    steps = build_protocol(args.rounds)
    trials, false_acts, frame_ms, latency_ms = [], [], [], []
    step_i, step_t0, result = 0, time.monotonic(), None
    prev = time.monotonic()

    while step_i < len(steps):
        ok, frame = cap.read()
        if not ok:
            break
        if config.MIRROR:
            frame = cv2.flip(frame, 1)
        now = time.monotonic()
        frame_ms.append((now - prev) * 1000)
        prev = now
        phase, target, dur = steps[step_i]
        elapsed = now - step_t0

        t0 = time.perf_counter()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if hasattr(classifier, "set_frame"):
            classifier.set_frame(rgb)
        hand = select_main_hand(tracker.process(rgb))
        label = classifier.predict(hand.landmarks).label if hand is not None else config.NONE
        latency_ms.append((time.perf_counter() - t0) * 1000)
        fired = stab.update(label)

        if phase == "try" and result is None:
            result = {"target": target, "ok": False, "time_s": None, "wrong": ""}
        if fired and phase == "try" and not result["ok"]:
            if fired == target:
                result.update(ok=True, time_s=elapsed)
            elif not result["wrong"]:
                result["wrong"] = fired
        elif fired and phase == "rest":
            false_acts.append(fired)

        view = to_canvas(frame)
        if hand is not None:
            draw_hand(view, hand.landmarks)
        name = config.GESTURE_NAMES_FR.get(target, target)
        if phase == "prep":
            text(view, f"Preparez : {name}", 60, 1.3, (255, 120, 220))
            text(view, f"debut dans {dur - elapsed:.0f} s", 100)
        elif phase == "try":
            text(view, f"FAITES : {name}", 60, 1.3, (120, 255, 160))
            text(view, f"{dur - elapsed:.1f} s   " + ("VALIDE !" if result["ok"] else ""), 100)
        else:
            text(view, "Repos : bougez naturellement, sans commande", 60, 1.0, (255, 200, 120))
            text(view, f"{dur - elapsed:.0f} s | fausses activations : {len(false_acts)}", 100)
        text(view, f"vu : {config.GESTURE_NAMES_FR.get(label, label)}  |  {args.classifier}  |  "
                   f"{1000 / max(np.mean(frame_ms[-30:]), 1e-6):.0f} FPS", 520, 0.8)
        window.show(view)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break

        if elapsed >= dur:
            if phase == "try":
                trials.append(result)
                result = None
            stab.reset()
            step_i += 1
            step_t0 = time.monotonic()

    cap.release()
    tracker.close()
    cv2.destroyAllWindows()
    if not trials:
        sys.exit("Évaluation interrompue avant la fin du premier geste.")

    rest_done = step_i >= len(steps)
    success = np.mean([t["ok"] for t in trials])
    times = [t["time_s"] for t in trials if t["ok"]]
    wrong = sum(bool(t["wrong"]) for t in trials)
    fa_min = len(false_acts) / (REST_S / 60) if rest_done else None
    fps = 1000 / np.mean(frame_ms[5:] or frame_ms)
    summary = {
        "date": time.strftime("%Y-%m-%d %H:%M"), "classifier": classifier.name, "lighting": args.lighting,
        "trials": len(trials), "success_rate": f"{success:.3f}",
        "time_to_validate_s": f"{np.median(times):.2f}" if times else "",
        "wrong_validations": wrong, "false_activations_per_min": "" if fa_min is None else f"{fa_min:.1f}",
        "fps": f"{fps:.1f}", "latency_median_ms": f"{np.median(latency_ms):.1f}",
        "latency_p95_ms": f"{np.percentile(latency_ms, 95):.1f}",
    }
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    out_csv = reports / "live_results.csv"
    new = not out_csv.exists()
    with open(out_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary))
        if new:
            w.writeheader()
        w.writerow(summary)

    lines = [f"# Évaluation en direct : {classifier.name}, éclairage {args.lighting}", "",
             f"- Taux de réussite : **{success:.0%}** ({len(trials)} essais)",
             f"- Temps médian jusqu'à la validation : {summary['time_to_validate_s'] or '-'} s",
             f"- Mauvais gestes validés : {wrong}",
             f"- Fausses activations : {summary['false_activations_per_min'] or '-'} / min",
             f"- FPS : {fps:.1f} | latence médiane {summary['latency_median_ms']} ms "
             f"(p95 {summary['latency_p95_ms']} ms)", "", "| Geste | Validé | Temps (s) | Erreur |", "|---|---|---|---|"]
    for t in trials:
        name = config.GESTURE_NAMES_FR.get(t["target"], t["target"])
        took = "" if t["time_s"] is None else f"{t['time_s']:.2f}"
        err = config.GESTURE_NAMES_FR.get(t["wrong"], t["wrong"])
        lines.append(f"| {name} | {'oui' if t['ok'] else 'non'} | {took} | {err} |")
    out_md = reports / f"live_{args.classifier}_{args.lighting}.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nRésultats ajoutés à {out_csv}")


if __name__ == "__main__":
    main()
