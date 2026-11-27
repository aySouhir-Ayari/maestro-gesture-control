"""Mode évaluation hors ligne : compare les approches sur les MÊMES images.

    A1  MediaPipe + règles géométriques
    A2  MediaPipe + classifieur ML (si models/gesture_classifier.joblib existe)
    B   YOLOv8 entraîné sur HaGRID (si models/maestro_yolo.pt existe)

Utilisation :
    python scripts/evaluate.py                  vos images (data/images, avec éclairage)
    python scripts/evaluate.py --source yolo    images de validation YOLO (HaGRID : scènes réelles)
    python scripts/evaluate.py --methods rules ml

Mesures : accuracy, précision, rappel, F1 macro, matrice de confusion, latence (ms, médiane et p95),
FPS équivalent, taux de main détectée, fausses activations (commande prédite alors qu'aucun geste),
accuracy par éclairage. Résultats : reports/evaluation_<source>.md + reports/eval_*.png
"""
import argparse
import csv
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from maestro import config  # noqa: E402
from maestro.evaluation import latency_stats, markdown_table, metrics, save_confusion  # noqa: E402


def own_images():
    """(chemin, geste, éclairage) depuis data/landmarks.csv (images de collect_data.py)."""
    out = []
    with open(config.DATASET_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("image") and (config.DATA_DIR / row["image"]).exists():
                out.append((config.DATA_DIR / row["image"], row["label"], row.get("lighting", "normal")))
    return out


def yolo_val_images():
    """(chemin, geste de la plus grande main, source) depuis data/yolo/images/val."""
    out = []
    root = config.YOLO_DATASET_DIR
    for img in sorted((root / "images" / "val").glob("*.jpg")):
        lbl = root / "labels" / "val" / f"{img.stem}.txt"
        if not lbl.exists():
            continue
        boxes = [ln.split() for ln in lbl.read_text().splitlines() if ln.strip()]
        if boxes:
            biggest = max(boxes, key=lambda b: float(b[3]) * float(b[4]))
            out.append((img, config.LABELS[int(biggest[0])], img.stem.split("_")[0]))
    return out


class MediaPipeImage:
    """MediaPipe en mode IMAGE (chaque image est indépendante)."""

    def __init__(self):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        from maestro.hand_tracker import ensure_model

        self.mp = mp
        self.lm = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=ensure_model()),
            running_mode=vision.RunningMode.IMAGE, num_hands=config.MAX_HANDS,
            min_hand_detection_confidence=config.MIN_DETECTION_CONF))

    def main_hand(self, rgb):
        res = self.lm.detect(self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb))
        hands = [np.array([[p.x, p.y, p.z] for p in h], np.float32) for h in res.hand_landmarks]
        if not hands:
            return None
        return max(hands, key=lambda a: np.ptp(a[:, 0]) * np.ptp(a[:, 1]))


def build_methods(names):
    methods = {}
    if "rules" in names or "ml" in names:
        mp_img = MediaPipeImage()

        def landmarks_method(clf):
            def run(rgb):
                lm = mp_img.main_hand(rgb)
                return (config.NONE, False) if lm is None else (clf.predict(lm).label, True)
            return run

        if "rules" in names:
            from maestro.gestures import RuleClassifier
            methods["A1 MediaPipe + règles"] = landmarks_method(RuleClassifier())
        if "ml" in names:
            if config.ML_MODEL_PATH.exists():
                from maestro.gestures import MLClassifier
                clf = MLClassifier()
                methods[f"A2 MediaPipe + {clf.name}"] = landmarks_method(clf)
            else:
                print("(classifieur ML absent : lancez scripts/train_classifier.py)")
    if "yolo" in names:
        if config.YOLO_MODEL_PATH.exists():
            from maestro.yolo_detector import YoloClassifier
            yolo = YoloClassifier()

            def run_yolo(rgb):
                dets = yolo.detect(rgb)
                if not dets:
                    return config.NONE, False
                d = max(dets, key=lambda d: (d.box[2] - d.box[0]) * (d.box[3] - d.box[1]))
                return d.label, True
            methods["B YOLOv8 (HaGRID)"] = run_yolo
        else:
            print("(modèle YOLO absent : models/maestro_yolo.pt)")
    return methods


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["own", "yolo"], default="own")
    parser.add_argument("--methods", nargs="+", default=["rules", "ml", "yolo"])
    parser.add_argument("--limit", type=int, default=0, help="nombre max d'images (0 = toutes)")
    args = parser.parse_args()

    samples = own_images() if args.source == "own" else yolo_val_images()
    if args.limit:
        samples = samples[:: max(1, len(samples) // args.limit)][:args.limit]
    if not samples:
        sys.exit("Aucune image : lancez collect_data.py (ou prepare_hagrid.py pour --source yolo).")
    print(f"{len(samples)} images ({args.source})")
    images = [(cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB), lab, grp) for p, lab, grp in samples]
    y_true = [lab for _, lab, _ in images]
    groups = sorted({g for _, _, g in images})
    labels = [lab for lab in config.LABELS if lab in set(y_true)]

    reports = ROOT / "reports"
    rows = []
    for name, run in build_methods(args.methods).items():
        run(images[0][0])                     # préchauffage (chargement du modèle)
        preds, detected, times = [], [], []
        for rgb, _, _ in images:
            t0 = time.perf_counter()
            label, found = run(rgb)
            times.append((time.perf_counter() - t0) * 1000)
            preds.append(label)
            detected.append(found)
        m = metrics(y_true, preds, labels)
        from sklearn.metrics import precision_score, recall_score
        none_idx = [i for i, t in enumerate(y_true) if t == config.NONE]
        false_act = (sum(preds[i] != config.NONE for i in none_idx) / len(none_idx)) if none_idx else None
        row = {"method": name, "accuracy": m["accuracy"],
               "precision": precision_score(y_true, preds, labels=labels, average="macro", zero_division=0),
               "recall": recall_score(y_true, preds, labels=labels, average="macro", zero_division=0),
               "f1": m["f1"], "detect": float(np.mean(detected)), "false_act": false_act,
               **latency_stats(times)}
        per_group = defaultdict(list)
        for (_, t, g), p in zip(images, preds):
            per_group[g].append(t == p)
        for g in groups:
            row[f"acc_{g}"] = float(np.mean(per_group[g]))
        rows.append(row)
        print(f"\n=== {name} ===  F1 {m['f1']:.3f} | {row['median_ms']:.1f} ms | ~{row['fps_equiv']:.0f} FPS")
        print(m["report"])
        tag = name.split()[0].lower()
        save_confusion(reports / f"eval_{args.source}_{tag}.png", y_true, preds, labels, f"{name} - F1 {m['f1']:.2f}")

    cols = [("method", "Méthode", ""), ("accuracy", "Accuracy", "{:.3f}"), ("precision", "Précision", "{:.3f}"),
            ("recall", "Rappel", "{:.3f}"), ("f1", "F1 macro", "{:.3f}"), ("detect", "Main détectée", "{:.1%}"),
            ("false_act", "Fausses activations", "{:.1%}"), ("median_ms", "Latence méd. (ms)", "{:.1f}"),
            ("p95_ms", "Latence p95 (ms)", "{:.1f}"), ("fps_equiv", "FPS équiv.", "{:.0f}")]
    group_cols = [("method", "Méthode", "")] + [(f"acc_{g}", g, "{:.3f}") for g in groups]
    title = "vos images (par éclairage)" if args.source == "own" else "validation YOLO (par source)"
    md = (f"# Évaluation hors ligne : {title}\n\n- Images : {len(images)}\n"
          f"- Gestes : {', '.join(labels)}\n\n## Résultats globaux\n\n{markdown_table(rows, cols)}\n\n"
          f"## Accuracy par {'éclairage' if args.source == 'own' else 'source'}\n\n"
          f"{markdown_table(rows, group_cols)}\n\n"
          "- *Latence* : détection + classification d'une image, sur cette machine.\n"
          "- *Fausses activations* : part des images « aucun geste » classées comme une commande.\n"
          "- *Main détectée* : sinon le geste est compté « none ».\n"
          f"- Matrices de confusion : `reports/eval_{args.source}_*.png`\n")
    out = reports / f"evaluation_{args.source}.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text(md, encoding="utf-8")
    print(f"\nRapport : {out}")


if __name__ == "__main__":
    main()
