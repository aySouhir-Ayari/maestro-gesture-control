"""Entraîne et compare des classifieurs sur les repères collectés.

Utilisation : python scripts/train_classifier.py [--csv data/landmarks.csv]
Produit :
  - un rapport par modèle (accuracy, F1, précision/rappel par classe)
  - les matrices de confusion dans reports/
  - le meilleur modèle dans models/gesture_classifier.joblib (utilisé par main.py --classifier ml)
"""
import argparse
import sys
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,  # noqa: E402
                             classification_report, f1_score)
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.neural_network import MLPClassifier  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import SVC  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from maestro import config  # noqa: E402

MODELS = {
    "Random Forest": lambda: RandomForestClassifier(n_estimators=200, random_state=42),
    "SVM (RBF)": lambda: make_pipeline(StandardScaler(), SVC(C=10, probability=True, random_state=42)),
    "MLP": lambda: make_pipeline(StandardScaler(), MLPClassifier((128, 64), max_iter=600, random_state=42)),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=str(config.DATASET_CSV))
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    X, y = df.drop(columns="label").to_numpy(np.float64), df["label"].to_numpy()
    print(f"{len(df)} échantillons | répartition : {df['label'].value_counts().to_dict()}")
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    best = (None, -1.0, None)
    for name, make in MODELS.items():
        model = make().fit(X_tr, y_tr)
        t0 = time.perf_counter()
        y_pred = model.predict(X_te)
        latency_ms = (time.perf_counter() - t0) / len(X_te) * 1000
        acc, f1 = accuracy_score(y_te, y_pred), f1_score(y_te, y_pred, average="macro")
        print(f"\n=== {name} === accuracy {acc:.3f} | F1 macro {f1:.3f} | {latency_ms:.3f} ms/échantillon")
        print(classification_report(y_te, y_pred, zero_division=0))
        ConfusionMatrixDisplay.from_predictions(y_te, y_pred, xticks_rotation=45, cmap="RdPu")
        plt.title(f"{name} - F1 {f1:.2f}")
        plt.tight_layout()
        plt.savefig(reports / f"confusion_{name.split()[0].lower()}.png", dpi=150)
        plt.close()
        if f1 > best[1]:
            best = (name, f1, model)

    config.MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump({"name": best[0], "model": best[2]}, config.ML_MODEL_PATH)
    print(f"\nMeilleur modèle : {best[0]} (F1 {best[1]:.3f}) -> {config.ML_MODEL_PATH}")


if __name__ == "__main__":
    main()
