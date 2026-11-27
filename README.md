# Maestro 🎵✋

**Reconnaissance de gestes de la main en temps réel pour contrôler la musique sans contact.**

Projet individuel du module de Computer Vision : Maestro filme l'utilisateur avec une simple webcam, détecte ses mains dans la scène, reconnaît le geste effectué et pilote un lecteur de musique, comme un chef d'orchestre.

## Sommaire

1. [Contexte et problématique](#1-contexte-et-problématique)
2. [Objectifs](#2-objectifs)
3. [Solution proposée](#3-solution-proposée)
4. [Approches et architectures](#4-approches-et-architectures)
5. [Choix techniques justifiés](#5-choix-techniques-justifiés)
6. [Datasets](#6-datasets)
7. [Évaluation expérimentale](#7-évaluation-expérimentale)
8. [Étapes de réalisation](#8-étapes-de-réalisation)
9. [Auteure](#9-auteure)
10. [Technologies et bibliothèques](#10-technologies-et-bibliothèques)
11. [Installation et exécution](#11-installation-et-exécution)
12. [Organisation du dépôt](#12-organisation-du-dépôt)
13. [Références](#13-références)

## 1. Contexte et problématique

On contrôle la musique en touchant un écran, un clavier ou une télécommande. Dans de nombreuses situations, ce contact est impossible ou peu pratique : mains sales ou mouillées (cuisine, atelier), activité sportive, conduite, présentation devant un public, personnes à mobilité réduite. Les assistants vocaux existent, mais ils sont gênés par la musique elle-même et par le bruit ambiant.

La main est une interface naturelle, et une webcam est présente sur presque tous les ordinateurs. Reconnaître des gestes de la main de façon **fiable** dans des **conditions réelles** reste pourtant un problème difficile de vision par ordinateur :

- **fond chargé** : objets, visages et autres personnes dans l'image ;
- **plusieurs mains** visibles : il faut choisir laquelle donne l'ordre ;
- **éclairage variable** : pièce sombre, contre-jour, forte lumière ;
- **distance et orientation** de la main qui changent sans cesse ;
- **temps réel** : la commande doit répondre en quelques dixièmes de seconde, sur un PC sans GPU ;
- **fausses activations** : un mouvement naturel (parler avec les mains, se gratter) ne doit pas changer la musique.

**Problématique :** comment reconnaître, en temps réel et avec une webcam ordinaire, un petit vocabulaire de gestes de la main de manière robuste aux scènes réelles et à l'éclairage, tout en limitant les commandes déclenchées par erreur ?

## 2. Objectifs

1. **Détecter et suivre** une ou deux mains dans le flux vidéo, y compris dans une scène chargée.
2. **Reconnaître** 6 gestes statiques (paume ouverte, pouce levé, pouce baissé, deux doigts, index vers la gauche, poing) et des **gestes dynamiques** (swipes, pincement).
3. **Comparer trois approches** de reconnaissance, du plus simple au plus avancé (règles géométriques, apprentissage automatique sur points clés, détecteur YOLOv8 entraîné sur HaGRID).
4. **Évaluer** chaque approche avec des métriques adaptées : accuracy, précision, rappel, F1, matrice de confusion, latence, FPS, fausses activations et robustesse à l'éclairage.
5. **Fonctionner en temps réel** (≥ 15 FPS) sur un ordinateur portable sans GPU.
6. Livrer une **application utilisable** : contrôle d'un lecteur local ou de Spotify, activation vocale, interface claire.

## 3. Solution proposée

```
« Maestro » (micro, Vosk hors ligne) → activation de la détection
Webcam (OpenCV)
  → détection des mains + 21 points clés + suivi (MediaPipe Hand Landmarker)
  → rôle de chaque main (une main = volume, l'autre = morceaux)
  → reconnaissance du geste :  A1 règles géométriques | A2 classifieur ML | B YOLOv8 (HaGRID)
  → gestes dynamiques : trajectoire de la paume (swipes), distance pouce-index (pincement)
  → stabilisation temporelle (geste stable ~0,3 s, main immobile, délai anti-répétition)
  → commande musicale (lecteur local, Spotify) + retour visuel animé
```

**Vocabulaire de gestes** (configuration par défaut : main droite = volume, main gauche = morceaux, inversable)

| Main | Geste | Commande |
|---|---|---|
| Volume | pincer pouce-index, écarter / rapprocher | volume continu 0-100 % |
| Morceaux | ✋ paume ouverte | lecture / pause |
| Morceaux | 👍 pouce levé · 👎 pouce baissé | volume + / − (répété si maintenu) |
| Morceaux | ✌️ deux doigts | morceau suivant |
| Morceaux | 👈 index vers la gauche | morceau précédent |
| Morceaux | swipe gauche / droite | ouvre la playlist en carrousel et la parcourt |
| Morceaux | ✊ poing (dans la playlist) | joue le morceau choisi |
| Morceaux | swipe vers le haut | fiche du morceau (pochette, titre, durée) |
| Morceaux | pincer puis glisser | avancer / reculer dans le morceau |
| Voix | « Maestro » / « Maestro stop » | active les gestes / veille + pause |

**Valeur ajoutée par rapport aux solutions existantes :**

- La plupart des démos de gestes se contentent d'un modèle préentraîné. Maestro **compare trois approches sur les mêmes images**, dont une entraînée sur nos propres données et une sur un dataset de scènes réelles.
- Les **fausses activations** sont traitées explicitement (main immobile, stabilisation temporelle, activation vocale) et **mesurées**.
- La **robustesse à l'éclairage** est évaluée sur des conditions jamais vues à l'entraînement.
- Le système combine **gestes statiques, gestes dynamiques et voix**, avec deux mains aux rôles différents.

## 4. Approches et architectures

| | Approche | Entrée | Modèle | Entraînement |
|---|---|---|---|---|
| **A1** | Règles géométriques (référence) | 21 points clés MediaPipe | doigts tendus/pliés, direction de l'index, position du pouce | aucun |
| **A2** | Classifieur sur points clés | vecteur de 63 valeurs normalisées (invariant à la position et à la taille) | Random Forest, SVM (noyau RBF), MLP (128-64) | nos données collectées |
| **B** | Détection + classification dans l'image entière | image RGB | **YOLOv8n** (détecteur à une étape, CNN) | HaGRID + nos images |

- **MediaPipe Hand Landmarker** (A1, A2) : un détecteur de paume suivi d'un réseau de régression qui donne 21 points 3D par main.
- **YOLOv8** (B) : fine-tuning du modèle préentraîné COCO pour détecter chaque main **et** donner son geste en une seule passe (7 classes : `none`, `palm`, `thumbs_up`, `thumbs_down`, `two_fingers`, `index_left`, `fist`). En mode hybride, YOLO donne le geste et MediaPipe garde les points clés pour le pincement et les swipes.
- **Gestes dynamiques** : suivi de la trajectoire du centre de la paume (déplacement ≥ 20 % de l'image en moins de 0,45 s, direction dominante).

## 5. Choix techniques justifiés

| Choix | Justification |
|---|---|
| MediaPipe pour les points clés | rapide sur CPU (temps réel sans GPU), 21 points 3D précis, gère deux mains ; donne une représentation compacte indépendante du fond |
| Règles géométriques comme référence | sans données ni entraînement, interprétables : elles mesurent ce qu'apporte réellement l'apprentissage |
| Random Forest / SVM / MLP sur points clés | peu de données suffisent (quelques centaines d'exemples par geste), entraînement en secondes, latence < 1 ms |
| Normalisation du vecteur (centré sur le poignet, mis à l'échelle) | rend la classification indépendante de la position et de la distance de la main |
| YOLOv8n | détecteur temps réel le plus utilisé, version *nano* adaptée au CPU, fine-tuning simple avec Ultralytics ; traite toute l'image (plusieurs mains, fond chargé) |
| HaGRID | grand dataset public de gestes en **scènes réelles**, très varié (personnes, pièces, éclairage, distance 0,5 à 4 m), annoté en boîtes englobantes au format adapté à YOLO |
| Données personnelles en plus | HaGRID ne contient pas le geste « index vers la gauche » ; nos images ajoutent aussi notre webcam et nos conditions d'éclairage |
| Pas de miroir horizontal à l'entraînement de YOLO | le miroir transformerait « index à gauche » en « index à droite » |
| Découpage train/test **par personne ou session** | évite qu'une même personne soit à la fois dans l'entraînement et le test (résultats trop optimistes sinon) |
| F1 macro en métrique principale | les classes ne sont pas équilibrées (beaucoup de « aucun geste ») ; le F1 macro donne le même poids à chaque geste |
| Fausses activations et latence mesurées | pour une interface, une commande déclenchée par erreur ou en retard est aussi grave qu'un geste mal reconnu |
| Vosk hors ligne pour la voix | fonctionne sans Internet, avec une grammaire réduite (« Maestro », « stop »), donc rapide et fiable |

## 6. Datasets

| Dataset | Rôle | Contenu | Lien | Licence |
|---|---|---|---|---|
| **HaGRID** (HAnd Gesture Recognition Image Dataset) | entraînement et évaluation de YOLOv8 (approche B) en scènes réelles | version initiale : 552 992 images RGB FullHD, 18 classes de gestes + `no_gesture`, plus de 34 000 personnes, boîtes englobantes + étiquettes ; découpage par personne | [github.com/hukenovs/hagrid](https://github.com/hukenovs/hagrid) | variante de CC BY-SA 4.0 |
| **HaGRID Sample 30k 384p** | version réduite utilisée en pratique (téléchargement léger, entraînement sur Colab) | 31 833 images HaGRID réduites en 384 p, 18 classes + `no_gesture`, mêmes annotations | [huggingface.co/datasets/cj-mills/hagrid-sample-30k-384p](https://huggingface.co/datasets/cj-mills/hagrid-sample-30k-384p) (aussi sur Kaggle) | CC BY-SA 4.0 |
| **Dataset Maestro** (collecté par l'auteure) | entraînement de l'approche A2, ajout du geste « index à gauche » à YOLO, évaluation par éclairage | 7 classes, 21 points clés + boîte de la main + 1 image sur 4, 4 conditions d'éclairage (normal, faible, fort, contre-jour) | créé avec `scripts/collect_data.py` | données personnelles : **non publiées** |

**Correspondance HaGRID → Maestro :** `palm`, `stop` → paume ; `like` → pouce levé ; `dislike` → pouce baissé ; `peace` → deux doigts ; `fist` → poing ; `no_gesture`, `call`, `ok`, `rock`, `one`… → aucun geste. Les classes ambiguës (`peace_inverted`, `three2`…) sont écartées.

**Prétraitement :**
- **HaGRID :** conversion des boîtes au format YOLO, échantillonnage équilibré par classe, images réduites à 640 px, découpage train/val par personne (`scripts/prepare_hagrid.py`).
- **Points clés :** normalisés (centrés sur le poignet, mis à l'échelle).

Les datasets ne sont **pas** déposés sur GitHub. Pour les télécharger et les organiser, voir [`data/README.md`](data/README.md).

## 7. Évaluation expérimentale

| Mesure | Pourquoi |
|---|---|
| Accuracy, précision, rappel, **F1 macro** | qualité de la reconnaissance, par geste et globale |
| Matrice de confusion | quels gestes sont confondus (ex. pouce levé / poing) |
| mAP50, mAP50-95 (YOLO) | qualité de la détection des mains |
| Taux de main détectée | une main manquée = commande impossible |
| Latence (ms, médiane et p95), FPS | contrainte temps réel |
| Fausses activations (% et par minute) | commandes déclenchées sans le vouloir |
| Accuracy par éclairage, test sur un éclairage jamais vu | robustesse |
| Taux de réussite et temps de validation en direct | expérience réelle de l'utilisateur |

Les trois approches sont évaluées **sur les mêmes images** :
- sur nos images, avec une ligne par éclairage ;
- sur les images de validation HaGRID (scènes réelles) ;
- en direct à la webcam, avec un protocole guidé.

**Premiers résultats (A1, règles géométriques, en direct)** : temps réel atteint (≈ 20 ms par image, 16,7 FPS), 67 % des gestes validés en éclairage normal comme faible, mais les gestes du pouce ne sont pas reconnus et il y a trop de fausses activations (12 à 26 par minute). Cela justifie les approches A2 et B.

Les résultats détaillés et leur analyse critique sont dans [`results/README.md`](results/README.md).

## 8. Étapes de réalisation

| # | Étape | Livrables | État |
|---|---|---|---|
| 1 | Étude du problème, choix des gestes et des datasets | sujet, vocabulaire de gestes | ✅ |
| 2 | Prototype A1 : MediaPipe + règles géométriques + lecteur | `main.py`, `maestro/` | ✅ |
| 3 | Interaction : deux mains, swipes, pincement, activation vocale, interface | application temps réel | ✅ |
| 4 | Collecte de notre dataset (gestes × éclairages) | `scripts/collect_data.py` | ✅ outil prêt, collecte en cours |
| 5 | Approche A2 : RF / SVM / MLP, comparaison avec les règles | `scripts/train_classifier.py` | ✅ code prêt, résultats à produire |
| 6 | Approche B : préparation HaGRID, fine-tuning YOLOv8 sur Colab | `scripts/prepare_hagrid.py`, `scripts/train_yolo.py`, notebook | ✅ code prêt, entraînement à lancer |
| 7 | Évaluation comparative hors ligne et en direct | `scripts/evaluate.py`, `scripts/evaluate_live.py` | ✅ code prêt, résultats à produire |
| 8 | Analyse critique des résultats, améliorations | `results/README.md` | 🔄 premiers résultats (A1 en direct) |
| 9 | Intégration de vraie musique (Spotify, pochettes) et démonstration | lecteurs `media` et `spotify` | ✅ |
| 10 | Rapport final et présentation | `docs/` | ⏳ |

## 9. Auteure

Projet **individuel** réalisé par :

| Nom | Prénom |
|---|---|
| Ayari | Souhir |

## 10. Technologies et bibliothèques

| Domaine | Outils |
|---|---|
| Langage | Python 3.10 |
| Vision | OpenCV, MediaPipe (Hand Landmarker), Ultralytics YOLOv8, PyTorch |
| Apprentissage automatique | scikit-learn (Random Forest, SVM, MLP), NumPy, Pandas, joblib |
| Évaluation et graphiques | scikit-learn.metrics, Matplotlib |
| Voix | Vosk (reconnaissance vocale hors ligne), sounddevice |
| Audio et lecteurs | pygame (fichiers locaux), touches média Windows + pycaw + WinRT (Spotify gratuit), spotipy (API Spotify, Premium) |
| Entraînement GPU | Google Colab (notebook fourni) |
| Qualité | pytest (tests sans webcam), Git/GitHub |

## 11. Installation et exécution

```bash
git clone https://github.com/aySouhir-Ayari/maestro-gesture-control.git
cd maestro-gesture-control
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/macOS : source .venv/bin/activate)
pip install -r requirements.txt
python main.py                  # puis dire « Maestro » (ou touche M) ; H = aide des gestes
```

- Les modèles MediaPipe et Vosk sont téléchargés automatiquement au premier lancement.
- Placez des fichiers `.mp3` / `.wav` dans `music/`, ou utilisez Spotify (voir plus bas).
- Les dépendances optionnelles sont dans `requirements-yolo.txt` (approche B) et `requirements-media.txt` (Spotify gratuit, vraies pochettes).

**Reproduire les expériences :**

```bash
python scripts/collect_data.py --lighting normal      # 1. collecte (refaire avec faible, fort, contre-jour)
python scripts/train_classifier.py                    # 2. approche A2 + comparaison avec les règles
#    3. approche B : notebooks/train_yolo_colab.ipynb sur Google Colab, puis copier maestro_yolo.pt dans models/
python scripts/evaluate.py                            # 4. comparaison A1 / A2 / B sur nos images
python scripts/evaluate.py --source yolo              #    ... et sur les scènes réelles HaGRID
python scripts/evaluate_live.py --classifier rules --lighting normal   # 5. évaluation en direct
python -m pytest tests                                # tests automatiques
```

### Guide d'utilisation détaillé

### Lancer l'application

```bash
python main.py                    # approche A : règles géométriques (baseline)
python main.py --classifier ml    # classifieur entraîné sur vos données
python main.py --classifier yolo  # approche B : YOLOv8 + HaGRID (pip install -r requirements-yolo.txt)
python main.py --camera 1         # autre webcam
```

#### Classifieur ML sur vos propres données (A2)

```bash
python scripts/collect_data.py --lighting normal   # touches 0-6 maintenues ; refaites avec faible, fort, contre-jour
python scripts/train_classifier.py                 # règles vs Random Forest / SVM / MLP
python scripts/train_classifier.py --test-lighting faible   # robustesse : test sur un éclairage jamais vu
```

`collect_data.py` enregistre les 21 points clés, l'éclairage et la boîte de la main dans `data/landmarks.csv`, et garde une image sur 4 dans `data/images/` (pour évaluer et entraîner YOLO sur les mêmes gestes).
`train_classifier.py` compare les **règles géométriques** (baseline) aux modèles entraînés sur le même jeu de test et écrit `reports/ml_comparison.md` (accuracy, F1, latence, accuracy par éclairage) + les matrices de confusion. Le meilleur modèle va dans `models/`.

Conseil : au moins 200 exemples par geste, avec les deux mains, à différentes distances, et beaucoup de « 0 = aucun geste ».

#### Approche B : YOLOv8 + HaGRID

YOLO détecte **chaque main dans l'image entière** (scènes chargées, plusieurs personnes) et donne directement son geste. Dans Maestro, le mode est hybride : YOLO fournit le geste, MediaPipe garde les points clés pour le pincement, les swipes et le seek.

1. Ouvrez `notebooks/train_yolo_colab.ipynb` dans Google Colab (GPU gratuit, rien à installer sur le PC).
2. Le notebook télécharge HaGRID (échantillon 30k), ajoute vos images (`maestro_data.zip`), prépare le dataset (`scripts/prepare_hagrid.py`), entraîne YOLOv8n (`scripts/train_yolo.py`) et évalue.
3. Copiez `maestro_yolo.pt` dans `models/` puis `python main.py --classifier yolo`.

Correspondance HaGRID → Maestro : `palm`/`stop` → paume, `like` → pouce levé, `dislike` → pouce baissé, `peace` → deux doigts, `fist` → poing ; `call`, `ok`, `rock`... → aucun geste (YOLO apprend à ignorer les mains sans commande). « Index à gauche » n'existe pas dans HaGRID : il vient de vos images. Le miroir horizontal est désactivé à l'entraînement (sinon gauche ↔ droite).

#### Mode évaluation

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

#### Tests

```bash
python -m pytest tests
```

Les tests vérifient les règles sur des mains synthétiques, l'invariance des caractéristiques, le choix de la main principale, la stabilisation et le lecteur.

### Paramètres personnels : fichier `.env`

Le fichier `.env` (à la racine) change les réglages **sans toucher au code** : webcam, main dominante, classifieur par défaut, voix, affichage...

```ini
CAMERA_INDEX=1          # autre webcam
LEFT_HANDED=false       # main gauche = volume
CLASSIFIER=ml           # python main.py utilise le classifieur ML
START_FULLSCREEN=true
```

`.env.example` liste les principaux réglages avec leur explication ; tout paramètre de `maestro/config.py` peut y être écrit. `.env` reste sur votre PC (ignoré par git). Une variable d'environnement `MAESTRO_<NOM>` est prioritaire, et les options de la ligne de commande (`--camera`, `--classifier`...) aussi.

### Vraie musique avec Spotify GRATUIT (touches média)

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

### Vraie musique : vos playlists Spotify (Premium)

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

## 12. Organisation du dépôt

```
maestro-gesture-control/
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
├── tests/                   # tests automatiques sans webcam
├── data/README.md           # téléchargement et organisation des datasets
├── models/README.md         # modèles utilisés / entraînés et comment les obtenir
├── results/README.md        # résultats expérimentaux et analyse critique
├── docs/references.md       # articles, méthodes, datasets et outils cités
├── music/                   # vos morceaux (non publiés)
├── requirements.txt         # dépendances principales
├── requirements-yolo.txt    # approche B (YOLOv8)
├── requirements-media.txt   # Spotify gratuit, vraies pochettes (Windows)
└── .env.example             # modèle de paramètres personnels
```

Le code de l'application est dans `maestro/` (bibliothèque) et `main.py` ; les expériences (collecte, entraînement, évaluation) sont dans `scripts/`. Cette organisation remplace le dossier `src/` de la structure indicative.

## 13. Références

Liste complète avec liens et licences : [`docs/references.md`](docs/references.md). Principales sources :

- Kapitanov et al., *HaGRID – HAnd Gesture Recognition Image Dataset*, WACV 2024 (arXiv:2206.08219).
- Zhang et al., *MediaPipe Hands: On-device Real-time Hand Tracking*, 2020 (arXiv:2006.10214).
- Jocher, Chaurasia, Qiu, *Ultralytics YOLOv8*, 2023 ; Redmon et al., *You Only Look Once*, CVPR 2016.
- Breiman, *Random Forests*, 2001 ; Cortes & Vapnik, *Support-Vector Networks*, 1995 ; Pedregosa et al., *Scikit-learn*, JMLR 2011.
- Vosk Speech Recognition Toolkit (Alpha Cephei).

Tout code, modèle ou dataset externe utilisé est cité dans `docs/references.md`.
