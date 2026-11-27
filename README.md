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

## Deux mains, deux rôles

| Main | Geste | Effet |
|---|---|---|
| **Gauche** (volume) | majeur, annulaire, auriculaire repliés ; **écarter / rapprocher pouce et index** | volume continu (0 à 100 %) |
| **Droite** (morceaux) | les 5 gestes ci-dessus | lecture/pause, volume ±, suivant, précédent |
| Droite | **swipe gauche / droite** | ouvre la playlist en carrousel, puis fait défiler les morceaux |
| Droite | **poing fermé** (dans la playlist) | joue le morceau sélectionné |
| Droite | **swipe du bas vers le haut** | fiche du morceau en cours (titre, artiste, durée, format…) |
| Droite | **pincer (pouce contre index) puis glisser horizontalement** | avancer / reculer dans le morceau, appliqué au relâchement |

- Les swipes analysent la **trajectoire de la paume** dans le temps (≥ 20 % de l'image en moins de 0,45 s, mouvement majoritairement dans une direction).
- Un geste statique n'est validé que si la main est **immobile** : un swipe ne déclenche donc pas « paume ouverte ».
- Gaucher : `python main.py --left-handed`. Si la caméra inverse gauche et droite : `SWAP_HANDEDNESS = True` dans `maestro/config.py`.
- Les pochettes de la playlist sont générées automatiquement (style affiche électro).

## Activation vocale « Maestro »

Maestro démarre **en veille** : la caméra tourne mais aucune détection n'est faite.

| Vous dites | Effet |
|---|---|
| « **Maestro** » | lance la détection et le tracking des mains |
| « **Maestro, stop** » | remet Maestro en veille |

- Reconnaissance vocale **hors ligne** avec [Vosk](https://alphacephei.com/vosk/) (modèle français ~40 Mo, téléchargé au premier lancement), limitée à ces mots.
- Retour automatique en veille après 60 s sans main visible.
- Touche **M** : activer / mettre en veille au clavier (secours si le micro ne marche pas).
- `python main.py --no-voice` : sans micro, la détection démarre tout de suite.
- `python scripts/test_voice.py` : vérifie que le micro entend bien « Maestro ».

## Interface

- **Grand affichage** : l'image de la caméra est agrandie (720 px de haut) et la fenêtre occupe 90 % de l'écran ; toute l'interface s'adapte à la taille. Touche **F** = plein écran.
- **Animation à chaque geste validé** : onde de choc sur la main + icône au centre (lecture, pause, volume, pistes, avance/recul, infos).
- **Barre de lecture** : pochette, titre, artiste, progression, volume.
- **Baisse automatique du volume** : dès que le micro entend « Maestro… » (même avant la fin de la phrase), la musique descend à 25 % en fondu, le temps de donner la commande, puis remonte.

## Pipeline

```
« Maestro » (Vosk, micro) → activation
Webcam (OpenCV) → Détection + 21 points clés + suivi (MediaPipe Hand Landmarker)
  → choix de la main principale (la plus grande)
  → classification du geste : A1 règles géométriques | A2 classifieur ML | B YOLOv8 (HaGRID)
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

## Paramètres personnels : fichier `.env`

Le fichier `.env` (à la racine) change les réglages **sans toucher au code** : webcam, main dominante, classifieur par défaut, voix, affichage...

```ini
CAMERA_INDEX=1          # autre webcam
LEFT_HANDED=false       # main gauche = volume
CLASSIFIER=ml           # python main.py utilise le classifieur ML
START_FULLSCREEN=true
```

`.env.example` liste les principaux réglages avec leur explication ; tout paramètre de `maestro/config.py` peut y être écrit. `.env` reste sur votre PC (ignoré par git). Une variable d'environnement `MAESTRO_<NOM>` est prioritaire, et les options de la ligne de commande (`--camera`, `--classifier`...) aussi.

## Vraie musique avec Spotify GRATUIT (touches média)

Sans compte Premium, Maestro pilote l'application Spotify **comme un clavier multimédia** : il « appuie » sur les touches lecture/pause, suivant, précédent de Windows. Aucun compte développeur, aucune clé.

```powershell
pip install -r requirements-media.txt   # vraie pochette + titre + position (Windows), volume précis
python main.py --player media           # ou PLAYER=media dans .env
```

**Pochettes réelles** : Maestro affiche la vraie pochette du morceau en cours, celle que Spotify envoie à Windows (comme dans le menu volume de Windows). Sans les paquets `winrt`, elle est cherchée dans le catalogue iTunes (artiste + titre, connexion Internet). À chaque nouveau morceau, la fiche avec la pochette s'affiche 4 s (`SHOW_NOW_PLAYING_S`). Avec vos fichiers (`PLAYER=local`), la pochette intégrée au MP3 est utilisée (`mutagen`), ou celle du catalogue si le fichier s'appelle « Artiste - Titre.mp3 ».

Ouvrez Spotify (application PC) et lancez votre playlist. Puis :

| Geste | Effet |
|---|---|
| Paume ouverte | lecture / pause |
| Deux doigts · swipe ← | morceau suivant |
| Index à gauche · swipe → | morceau précédent |
| Pouce levé / baissé, pincement | volume (volume de Windows) |
| Swipe ↑ | fiche du morceau (titre et artiste lus dans la fenêtre Spotify) |

Limites : pas de carrousel de la playlist ni d'avance/recul (Spotify ne donne pas ces informations sans API) ; avec Spotify gratuit, Spotify peut limiter le nombre de morceaux passés par heure et ajouter des publicités. Fonctionne aussi avec d'autres lecteurs : `MEDIA_APP_EXE=` dans `.env` (titre lu dans leur fenêtre).

## Vraie musique : vos playlists Spotify (Premium)

Maestro peut piloter **l'application Spotify** (PC ou téléphone) : lecture/pause, volume au pincement, morceau suivant/précédent, playlist en carrousel avec les vraies pochettes, avance/recul, baisse du volume quand vous dites « Maestro ». Le son sort de Spotify ; Maestro envoie les commandes par l'API Web Spotify.

**Conditions (règles Spotify depuis février 2026)** : compte **Premium** obligatoire ; la liste des morceaux n'est lisible que pour **vos** playlists (ou collaboratives) : pour une playlist d'un autre, ajoutez ses titres dans une playlist à vous.

1. https://developer.spotify.com/dashboard → *Create app* : nom `Maestro`, *Redirect URI* `http://127.0.0.1:8888/callback`, API cochée : **Web API**.
2. Copiez le *Client ID* et le *Client Secret* dans `.env` :
   ```ini
   PLAYER=spotify
   SPOTIFY_CLIENT_ID=xxxxxxxx
   SPOTIFY_CLIENT_SECRET=xxxxxxxx
   SPOTIFY_PLAYLIST=Ma playlist
   ```
3. `pip install spotipy` puis `python scripts/spotify_setup.py` : autorisation dans le navigateur (une seule fois), vérification Premium, liste de vos playlists (★ = utilisables) et appareils.
4. Ouvrez Spotify, puis `python main.py --player spotify`.

## Utilisation

```bash
python main.py                    # approche A : règles géométriques (baseline)
python main.py --classifier ml    # classifieur entraîné sur vos données
python main.py --classifier yolo  # approche B : YOLOv8 + HaGRID (pip install -r requirements-yolo.txt)
python main.py --camera 1         # autre webcam
```

### Classifieur ML sur vos propres données (A2)

```bash
python scripts/collect_data.py --lighting normal   # touches 0-6 maintenues ; refaites avec faible, fort, contre-jour
python scripts/train_classifier.py                 # règles vs Random Forest / SVM / MLP
python scripts/train_classifier.py --test-lighting faible   # robustesse : test sur un éclairage jamais vu
```

`collect_data.py` enregistre les 21 points clés, l'éclairage et la boîte de la main dans `data/landmarks.csv`, et garde une image sur 4 dans `data/images/` (pour évaluer et entraîner YOLO sur les mêmes gestes).
`train_classifier.py` compare les **règles géométriques** (baseline) aux modèles entraînés sur le même jeu de test et écrit `reports/ml_comparison.md` (accuracy, F1, latence, accuracy par éclairage) + les matrices de confusion. Le meilleur modèle va dans `models/`.

Conseil : au moins 200 exemples par geste, avec les deux mains, à différentes distances, et beaucoup de « 0 = aucun geste ».

### Approche B : YOLOv8 + HaGRID

YOLO détecte **chaque main dans l'image entière** (scènes chargées, plusieurs personnes) et donne directement son geste. Dans Maestro, le mode est hybride : YOLO fournit le geste, MediaPipe garde les points clés pour le pincement, les swipes et le seek.

1. Ouvrez `notebooks/train_yolo_colab.ipynb` dans Google Colab (GPU gratuit, rien à installer sur le PC).
2. Le notebook télécharge HaGRID (échantillon 30k), ajoute vos images (`maestro_data.zip`), prépare le dataset (`scripts/prepare_hagrid.py`), entraîne YOLOv8n (`scripts/train_yolo.py`) et évalue.
3. Copiez `maestro_yolo.pt` dans `models/` puis `python main.py --classifier yolo`.

Correspondance HaGRID → Maestro : `palm`/`stop` → paume, `like` → pouce levé, `dislike` → pouce baissé, `peace` → deux doigts, `fist` → poing ; `call`, `ok`, `rock`... → aucun geste (YOLO apprend à ignorer les mains sans commande). « Index à gauche » n'existe pas dans HaGRID : il vient de vos images. Le miroir horizontal est désactivé à l'entraînement (sinon gauche ↔ droite).

### Mode évaluation

```bash
python scripts/evaluate.py                     # vos images, par éclairage : A1 vs A2 vs B
python scripts/evaluate.py --source yolo       # images de validation YOLO (scènes réelles HaGRID)
python scripts/evaluate_live.py --classifier rules --lighting normal   # webcam, protocole guidé
```

| Mesure | Hors ligne (`evaluate.py`) | En direct (`evaluate_live.py`) |
|---|---|---|
| Accuracy, précision, rappel, F1 macro | ✓ | taux de réussite par geste |
| Matrice de confusion | ✓ (`reports/eval_*.png`) | mauvais gestes validés |
| Latence (ms, médiane et p95) | ✓ | ✓ |
| FPS | équivalent | réel |
| Fausses activations | % d'images « aucun geste » prises pour une commande | par minute (30 s de repos) |
| Éclairage | accuracy par condition | une ligne par session dans `reports/live_results.csv` |

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
│   ├── controller.py        # rôles des mains, modes (normal, playlist, fiche)
│   ├── motion.py            # trajectoire de la main, vitesse, swipes
│   ├── controls.py          # volume au pincement, barre de progression attrapée
│   ├── covers.py            # pochettes générées de la playlist
│   ├── visualizer.py        # onde sonore (désactivée : SHOW_VISUALIZER)
│   ├── yolo_detector.py     # approche B : YOLOv8, association boîte ↔ main MediaPipe
│   ├── evaluation.py        # métriques, matrices de confusion, tableaux
│   ├── display.py           # fenêtre agrandie, plein écran
│   ├── stabilizer.py        # validation temporelle et cooldown
│   ├── voice.py             # mot d'activation « Maestro » (Vosk, hors ligne)
│   ├── activation.py        # état veille / écoute, veille automatique
│   ├── player.py            # lecteur pygame (fichiers locaux)
│   ├── spotify_player.py    # lecteur Spotify (API Web, Premium)
│   ├── media_player.py      # Spotify gratuit : touches média de Windows
│   ├── artwork.py           # vraies pochettes (Windows, fichier, catalogue iTunes)
│   └── ui.py                # affichage OpenCV
├── scripts/
│   ├── collect_data.py      # collecte du dataset personnel (points clés + images + éclairage)
│   ├── train_classifier.py  # A2 : RF / SVM / MLP comparés aux règles
│   ├── prepare_hagrid.py    # B : HaGRID + vos images -> format YOLO
│   ├── train_yolo.py        # B : entraînement YOLOv8
│   ├── evaluate.py          # évaluation hors ligne A1 / A2 / B
│   ├── evaluate_live.py     # évaluation webcam guidée
│   ├── spotify_setup.py     # connexion Spotify + diagnostic
│   └── test_voice.py        # test du micro et du mot « Maestro »
├── notebooks/train_yolo_colab.ipynb   # entraînement YOLO sur Google Colab
├── tests/                   # tests sans webcam
├── music/  models/  data/
```

## Feuille de route

- [x] Approche A : MediaPipe + règles géométriques + contrôle musical
- [x] Activation vocale « Maestro » / « Maestro stop »
- [x] Deux mains : volume au pincement (gauche), morceaux (droite)
- [x] Swipes (tracking temporel) : playlist en carrousel, fiche du morceau
- [x] Avancer / reculer en pinçant et glissant
- [x] Baisse automatique du volume quand on parle à Maestro
- [x] Interface agrandie et adaptative, animations des gestes
- [x] Classifieur ML sur les points clés (RF / SVM / MLP) + évaluation
- [x] Approche B : YOLOv8 entraîné sur [HaGRID](https://github.com/hukenovs/hagrid) + vos images (scènes réelles, plusieurs mains)
- [x] Mode évaluation : accuracy, F1, matrice de confusion, latence, FPS, fausses activations, éclairage
