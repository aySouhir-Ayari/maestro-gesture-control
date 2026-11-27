"""Outils communs d'évaluation : métriques, matrices de confusion, rapports Markdown."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

FEATURE_COLS = [f"{a}{i}" for i in range(21) for a in "xyz"]


def metrics(y_true, y_pred, labels=None) -> dict:
    """Accuracy, F1 macro et rapport par classe."""
    from sklearn.metrics import accuracy_score, classification_report, f1_score

    labels = labels or sorted(set(y_true) | set(y_pred))
    return {
        "n": len(y_true),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "report": classification_report(y_true, y_pred, labels=labels, zero_division=0),
    }


def save_confusion(path: Path, y_true, y_pred, labels, title: str):
    """Matrice de confusion normalisée par ligne (rappel par classe), en PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, labels=labels, normalize="true",
                                            values_format=".2f", cmap="RdPu", ax=ax, colorbar=False,
                                            xticks_rotation=45)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def latency_stats(ms: list[float]) -> dict:
    a = np.asarray(ms, dtype=float) if ms else np.zeros(1)
    return {"median_ms": float(np.median(a)), "p95_ms": float(np.percentile(a, 95)),
            "fps_equiv": float(1000 / max(np.mean(a), 1e-6))}


def markdown_table(rows: list[dict], columns: list[tuple[str, str, str]]) -> str:
    """columns : (clé, titre, format) ; format ex. '{:.3f}'."""
    head = "| " + " | ".join(t for _, t, _ in columns) + " |"
    sep = "|" + "|".join("---" for _ in columns) + "|"
    body = []
    for r in rows:
        cells = []
        for key, _, fmt in columns:
            v = r.get(key)
            cells.append("-" if v is None else (fmt.format(v) if fmt else str(v)))
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([head, sep, *body])


def predict_timed(predict_fn, items) -> tuple[list[str], list[float]]:
    """Applique predict_fn à chaque élément en mesurant le temps (ms)."""
    preds, times = [], []
    for it in items:
        t0 = time.perf_counter()
        preds.append(predict_fn(it))
        times.append((time.perf_counter() - t0) * 1000)
    return preds, times


def landmarks_from_vector(vec) -> np.ndarray:
    """Vecteur normalisé (63) -> repères (21, 3). Les règles géométriques sont invariantes
    par translation et changement d'échelle : on peut les appliquer directement."""
    return np.asarray(vec, dtype=np.float32).reshape(21, 3)

