# Résultats expérimentaux

Les scripts écrivent leurs sorties dans `reports/`, qui n'est pas publié. Les résultats définitifs (tableaux et figures) sont recopiés ici, avec leur analyse.

> État : protocole et outils prêts. Les tableaux seront remplis après la collecte complète et l'entraînement de YOLO.

## Protocole

| Expérience | Commande | Données |
|---|---|---|
| E1 : règles vs RF / SVM / MLP | `python scripts/train_classifier.py` | dataset Maestro, 25 % test stratifié |
| E2 : robustesse à un éclairage jamais vu | `python scripts/train_classifier.py --test-lighting faible` (et `contre-jour`) | dataset Maestro |
| E3 : A1 vs A2 vs B, mêmes images | `python scripts/evaluate.py` | images Maestro, par éclairage |
| E4 : A1 vs A2 vs B en scènes réelles | `python scripts/evaluate.py --source yolo` | validation HaGRID + Maestro |
| E5 : détection YOLOv8 | `python scripts/train_yolo.py` (Colab) | HaGRID + Maestro |
| E6 : utilisation réelle | `python scripts/evaluate_live.py --classifier <rules/ml/yolo> --lighting <...>` | webcam, protocole guidé |

## E1, E3 : comparaison des approches

| Méthode | Accuracy | Précision | Rappel | F1 macro | Main détectée | Fausses activations | Latence méd. (ms) | FPS |
|---|---|---|---|---|---|---|---|---|
| A1 MediaPipe + règles | | | | | | | | |
| A2 MediaPipe + (meilleur ML) | | | | | | | | |
| B YOLOv8n (HaGRID) | | | | | | | | |

## E2, E3 : robustesse à l'éclairage (accuracy)

| Méthode | normal | faible | fort | contre-jour |
|---|---|---|---|---|
| A1 | | | | |
| A2 | | | | |
| B | | | | |

## E4 : scènes réelles (validation HaGRID)

| Méthode | F1 macro | Main détectée | Fausses activations |
|---|---|---|---|
| A1 | | | |
| A2 | | | |
| B | | | |

## E5 : détection YOLOv8n

| mAP50 | mAP50-95 | Précision | Rappel |
|---|---|---|---|
| | | | |

## E6 : évaluation en direct

| Méthode | Éclairage | Taux de réussite | Temps de validation (s) | Mauvais gestes validés | Fausses activations / min | FPS |
|---|---|---|---|---|---|---|
| A1 règles | normal | | | | | |
| A1 règles | faible | | | | | |

Premières sessions de test réalisées : `reports/live_rules_normal.md` et `reports/live_rules_faible.md`, à reporter ici.

## Figures

Pour chaque méthode, ajouter ici les figures suivantes, recopiées depuis `reports/` et `runs/` :
- matrices de confusion (`eval_*.png`, `confusion_*.png`) ;
- courbes d'entraînement YOLO (`results.png`).

## Analyse critique (à rédiger)

- **Quelle approche est la meilleure, et à quel coût ?** Comparer précision, latence et besoin en données.
- **Quels gestes sont confondus, et pourquoi ?** Par exemple pouce levé et poing (pouce peu visible), ou deux doigts et paume selon l'angle.
- **Effet de l'éclairage :** quelle approche résiste le mieux au contre-jour ? Le MediaPipe des approches A1 et A2 perd-il la main avant que YOLO échoue ?
- **Scènes réelles :** apport de YOLO quand il y a plusieurs mains et un fond chargé.
- **Fausses activations :** effet de la stabilisation temporelle et de la condition « main immobile ».
- **Limites :** peu de personnes dans notre dataset, une seule webcam, classe `index_left` absente de HaGRID.
- **Pistes d'amélioration :** davantage de participants, augmentation de données, modèle temporel pour les gestes dynamiques.
