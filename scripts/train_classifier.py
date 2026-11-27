"""Entraîne des classifieurs sur VOS gestes et les compare aux règles géométriques.

Utilisation :
    python scripts/train_classifier.py
    python scripts/train_classifier.py --test-lighting faible   (entraîné sans « faible », testé dessus)

Méthodes comparées sur le même jeu de test :
    Règles géométriques (baseline, sans entraînement), Random Forest, SVM, MLP.
Produit :
    reports/ml_comparison.md        tableau accuracy / F1 / latence + accuracy par éclairage
    reports/confusion_<methode>.png matrices de confusion
    models/gesture_classifier.joblib  meilleur modèle (python main.py --classifier ml)
"""
import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from maestro import config  # noqa: E402
from maestro.evaluation import (FEATURE_COLS, landmarks_from_vector, latency_stats,  # noqa: E402
                                markdown_table, metrics, predict_timed, save_confusion)
from maestro.gestures import RuleClassifier  # noqa: E402

MODELS = {
    "Random Forest": lambda: RandomForestClassifier(n_estimators=200, random_state=42),
    "SVM (RBF)": lambda: make_pipeline(StandardScaler(), SVC(C=10, probability=True, random_state=42)),
    "MLP": lambda: make_pipeline(StandardScaler(), MLPClassifier((128, 64), max_iter=800, random_state=42)),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=str(config.DATASET_CSV))
    parser.add_argument("--test-lighting", default=None,
                        help="éclairage réservé au test (mesure la robustesse à un éclairage jamais vu)")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    if "lighting" not in df:
        df["lighting"] = "normal"
    print(f"{len(df)} échantillons | gestes : {df['label'].value_counts().to_dict()}")
    print(f"éclairages : {df['lighting'].value_counts().to_dict()}")

    if args.test_lighting:
        test_mask = df["lighting"] == args.test_lighting
        if not test_mask.any():
            sys.exit(f"Aucun échantillon avec l'éclairage « {args.test_lighting} ».")
        tr, te = df[~test_mask], df[test_mask]
    else:
        tr, te = train_test_split(df, test_size=0.25, stratify=df["label"], random_state=42)
    X_tr, y_tr = tr[FEATURE_COLS].to_numpy(np.float64), tr["label"].to_numpy()
    X_te, y_te = te[FEATURE_COLS].to_numpy(np.float64), te["label"].to_numpy()
    labels = [lab for lab in config.LABELS if lab in set(y_tr) | set(y_te)]

    reports = ROOT / "reports"
    rows, best = [], (None, -1.0, None)

    def evaluate(name, preds, times):
        m = metrics(list(y_te), preds, labels)
        row = {"method": name, "accuracy": m["accuracy"], "f1": m["f1"], **latency_stats(times)}
        for light in sorted(te["lighting"].unique()):
            mask = (te["lighting"] == light).to_numpy()
            row[f"acc_{light}"] = float(np.mean(np.asarray(preds)[mask] == y_te[mask]))
        rows.append(row)
        print(f"\n=== {name} === accuracy {m['accuracy']:.3f} | F1 macro {m['f1']:.3f} | "
              f"{row['median_ms']:.3f} ms/échantillon")
        print(m["report"])
        save_confusion(reports / f"confusion_{name.split()[0].lower()}.png", list(y_te), preds, labels,
                       f"{name} - F1 {m['f1']:.2f}")
        return m

    # Baseline : règles géométriques (aucun entraînement)
    rules = RuleClassifier()
    preds, times = predict_timed(lambda v: rules.predict(landmarks_from_vector(v)).label, X_te)
    evaluate("Regles geometriques", preds, times)

    for name, make in MODELS.items():
        model = make().fit(X_tr, y_tr)
        preds, times = predict_timed(lambda v: str(model.predict(v.reshape(1, -1))[0]), X_te)
        m = evaluate(name, preds, times)
        if m["f1"] > best[1]:
            best = (name, m["f1"], model)

    config.MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump({"name": best[0], "model": best[2]}, config.ML_MODEL_PATH)

    lights = sorted(te["lighting"].unique())
    cols = [("method", "Méthode", ""), ("accuracy", "Accuracy", "{:.3f}"), ("f1", "F1 macro", "{:.3f}"),
            ("median_ms", "Latence (ms)", "{:.3f}")] + [(f"acc_{lt}", f"Acc. {lt}", "{:.3f}") for lt in lights]
    split = (f"entraînement sans « {args.test_lighting} », test sur « {args.test_lighting} »"
             if args.test_lighting else "75 % entraînement / 25 % test (stratifié)")
    md = (f"# Classifieur sur les points clés : comparaison\n\n"
          f"- Données : {len(df)} échantillons ({len(tr)} entraînement, {len(te)} test)\n"
          f"- Découpage : {split}\n"
          f"- Meilleur modèle sauvegardé : **{best[0]}** (F1 {best[1]:.3f})\n\n"
          + markdown_table(rows, cols)
          + "\n\nLatence = temps de classification seul (hors détection MediaPipe).\n")
    (reports / "ml_comparison.md").write_text(md, encoding="utf-8")
    print(f"\nMeilleur modèle : {best[0]} (F1 {best[1]:.3f}) -> {config.ML_MODEL_PATH}")
    print(f"Rapport : {reports / 'ml_comparison.md'}")


if __name__ == "__main__":
    main()
