# Modèles

Les fichiers de modèles ne sont pas publiés sur GitHub (taille). Ils sont téléchargés automatiquement ou produits par nos scripts.

| Fichier | Approche | Description | Origine |
|---|---|---|---|
| `hand_landmarker.task` | A1, A2, B (hybride) | MediaPipe Hand Landmarker : détection de paume + 21 points clés 3D par main | téléchargé automatiquement au premier lancement (Google, Apache 2.0) |
| `gesture_classifier.joblib` | A2 | meilleur classifieur sur points clés (Random Forest, SVM RBF ou MLP), choisi au F1 macro | `python scripts/train_classifier.py` |
| `maestro_yolo.pt` | B | YOLOv8n fine-tuné sur HaGRID + nos images, 7 classes | `notebooks/train_yolo_colab.ipynb` (Google Colab), puis copié ici |
| `vosk-model-small-fr-0.22/` | activation vocale | reconnaissance vocale française hors ligne (environ 40 Mo) | téléchargé automatiquement (Alpha Cephei, Apache 2.0) |

## Détails d'entraînement

**A2 : classifieurs sur points clés** (`scripts/train_classifier.py`)
- **Entrée :** vecteur de 63 valeurs normalisées.
- **Découpage :** 75 % / 25 % stratifié, ou test sur un éclairage jamais vu (`--test-lighting`).
- **Modèles :**
  - Random Forest : 200 arbres ;
  - SVM : noyau RBF, C = 10, données standardisées ;
  - MLP : couches 128-64, données standardisées.
- **Sorties :** le meilleur modèle selon le F1 macro, et le rapport `reports/ml_comparison.md`.

**B : YOLOv8n** (`scripts/train_yolo.py`)
- **Point de départ :** poids préentraînés COCO (`yolov8n.pt`, Ultralytics).
- **Entraînement :** 50 époques, image de 416 px, batch de 64 sur GPU T4, arrêt anticipé (patience 15).
- **Augmentations :** variations de luminosité fortes (`hsv_v = 0.5`). Le miroir horizontal est **désactivé** (`fliplr = 0`), pour ne pas confondre « index à gauche » et « index à droite ».
- **Sorties :** mAP50 et mAP50-95, matrice de confusion et courbes dans `runs/maestro_val/`.

## Utilisation

```bash
python main.py                      # A1 : règles géométriques (aucun modèle entraîné)
python main.py --classifier ml      # A2 : models/gesture_classifier.joblib
python main.py --classifier yolo    # B  : models/maestro_yolo.pt
```
