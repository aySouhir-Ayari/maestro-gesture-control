"""Classification des gestes : règles géométriques (baseline) ou classifieur ML."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import config
from .features import direction, finger_states, normalized_vector


@dataclass
class Prediction:
    label: str
    confidence: float
    states: dict[str, bool] | None = None


class RuleClassifier:
    """Baseline : combinaison des doigts tendus + direction du doigt principal."""

    name = "règles géométriques"

    def predict(self, lm: np.ndarray) -> Prediction:
        s = finger_states(lm)
        th, ix, md, rg, pk = (s[k] for k in ["thumb", "index", "middle", "ring", "pinky"])
        others_folded = not (md or rg or pk)

        # Paume ouverte : les 5 doigts tendus
        if th and ix and md and rg and pk:
            return Prediction("palm", 1.0, s)

        # Pouce levé / baissé : seul le pouce est tendu, orienté verticalement
        if th and not ix and others_folded:
            d = direction(lm, 2, 4)  # MCP -> bout du pouce
            if d[1] < -0.6:
                return Prediction("thumbs_up", 1.0, s)
            if d[1] > 0.6:
                return Prediction("thumbs_down", 1.0, s)

        # Deux doigts (V) : index + majeur tendus, annulaire et auriculaire pliés
        if ix and md and not rg and not pk:
            return Prediction("two_fingers", 1.0, s)

        # Index à gauche : seul l'index tendu, pointé vers la gauche de l'image
        if ix and others_folded:
            d = direction(lm, 5, 8)
            if d[0] < -0.6 and abs(d[0]) > abs(d[1]):
                return Prediction("index_left", 1.0, s)

        # Poing fermé : les quatre doigts repliés (sert à valider un choix dans la playlist)
        if not ix and others_folded:
            return Prediction(config.FIST, 1.0, s)

        return Prediction(config.NONE, 1.0, s)


class MLClassifier:
    """Classifieur entraîné (Random Forest / SVM) sur les repères normalisés."""

    def __init__(self, path=config.ML_MODEL_PATH):
        import joblib

        bundle = joblib.load(path)
        self.model = bundle["model"]
        self.name = bundle.get("name", "classifieur ML")

    def predict(self, lm: np.ndarray) -> Prediction:
        x = normalized_vector(lm).reshape(1, -1)
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(x)[0]
            k = int(np.argmax(proba))
            label, conf = str(self.model.classes_[k]), float(proba[k])
        else:
            label, conf = str(self.model.predict(x)[0]), 1.0
        if conf < config.MIN_CONFIDENCE:
            label = config.NONE
        return Prediction(label, conf, finger_states(lm))
