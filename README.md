# Maestro 🎵✋

**Reconnaissance de gestes de la main en temps réel pour le contrôle musical sans contact.**

Maestro filme la main avec une webcam, la détecte dans la scène (même avec un arrière-plan chargé ou plusieurs mains), reconnaît le geste et pilote un lecteur musical, comme un chef d'orchestre.

| Geste | Commande |
|---|---|
| ✋ Paume ouverte | Lecture / Pause |
| 👍 Pouce levé | Volume + (répété si maintenu) |
| 👎 Pouce baissé | Volume − (répété si maintenu) |
| ✌️ Deux doigts | Piste suivante |
| 👈 Index à gauche | Piste précédente |

## Pipeline

```
Webcam (OpenCV) → Détection + 21 points clés + suivi (MediaPipe Hand Landmarker)
  → choix de la main principale (la plus grande)
  → classification du geste (règles géométriques OU classifieur ML)
  → stabilisation temporelle (geste stable ~0,3 s, anti-répétition)
  → commande musicale (pygame)
```

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/macOS : source .venv/bin/activate)
pip install -r requirements.txt
```

Placez quelques fichiers `.mp3`, `.ogg` ou `.wav` dans `music/`. Sans fichier, le lecteur fonctionne en mode simulation (l'état s'affiche à l'écran).
Le modèle MediaPipe (`models/hand_landmarker.task`) est téléchargé automatiquement au premier lancement.

## Utilisation

```bash
python main.py                    # approche A : règles géométriques (baseline)
python main.py --classifier ml    # classifieur entraîné sur vos données
python main.py --camera 1         # autre webcam
```

### Entraîner le classifieur ML

```bash
python scripts/collect_data.py      # touches 0-5 maintenues pour enregistrer chaque geste
python scripts/train_classifier.py  # compare Random Forest / SVM / MLP
```

Le script affiche accuracy, F1 et le rapport par classe, enregistre les matrices de confusion dans `reports/` et sauvegarde le meilleur modèle dans `models/`.

Conseil : collectez au moins 200 exemples par geste, avec les deux mains, à différentes distances et sous différents éclairages. Enregistrez aussi beaucoup de « 0 = aucun geste ».

### Tests

```bash
python -m pytest tests
```

Les tests vérifient les règles sur des mains synthétiques, l'invariance des caractéristiques, le choix de la main principale, la stabilisation et le lecteur.

## Structure

```
maestro/
├── main.py                  # application temps réel
├── maestro/
│   ├── config.py            # paramètres (seuils, gestes, commandes)
│   ├── camera.py            # ouverture de la webcam
│   ├── hand_tracker.py      # MediaPipe : détection, 21 repères, suivi, multi-mains
│   ├── features.py          # doigts tendus/pliés, directions, vecteur normalisé
│   ├── gestures.py          # RuleClassifier (baseline) + MLClassifier
│   ├── stabilizer.py        # validation temporelle et cooldown
│   ├── player.py            # lecteur pygame
│   └── ui.py                # affichage OpenCV
├── scripts/
│   ├── collect_data.py      # collecte du dataset personnel
│   └── train_classifier.py  # entraînement + évaluation
├── tests/                   # tests sans webcam
├── music/  models/  data/
```

## Feuille de route

- [x] Approche A : MediaPipe + règles géométriques + contrôle musical
- [x] Classifieur ML sur les points clés (RF / SVM / MLP) + évaluation
- [ ] Approche B : YOLOv8 fine-tuné sur [HaGRID](https://github.com/hukenovs/hagrid) (détection des mains dans des scènes réelles) + classification sur la main recadrée (MobileNetV2)
- [ ] Comparaison A vs B : accuracy, F1, matrice de confusion, latence, robustesse à l'éclairage
