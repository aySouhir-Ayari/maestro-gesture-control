# Données

Aucun dataset n'est déposé sur GitHub : les fichiers sont trop volumineux et nos propres images sont des données personnelles. Ce dossier explique comment obtenir et organiser les données localement.

## 1. HaGRID (approche B : YOLOv8)

- **Source :** [github.com/hukenovs/hagrid](https://github.com/hukenovs/hagrid). Article : Kapitanov et al., *HaGRID – HAnd Gesture Recognition Image Dataset*, WACV 2024.
- **Version complète :** 552 992 images RGB FullHD (1920×1080), environ 716 Go.
  - 18 classes de gestes + `no_gesture`, plus de 34 000 personnes filmées entre 0,5 et 4 m de la caméra.
  - Découpage train/test par personne.
- **Version utilisée :** *HaGRID Sample 30k 384p*, soit 31 833 images réduites en 384 p avec les mêmes annotations.
  - Lien : [huggingface.co/datasets/cj-mills/hagrid-sample-30k-384p](https://huggingface.co/datasets/cj-mills/hagrid-sample-30k-384p).
  - Elle est assez légère pour Google Colab gratuit, et le notebook `notebooks/train_yolo_colab.ipynb` la télécharge automatiquement.
- **Annotations :** un fichier JSON par geste, de la forme `{image_id: {"bboxes": [[x, y, w, h], ...], "labels": [...], "user_id": ...}}`.
  - Les boîtes sont normalisées entre 0 et 1, avec (x, y) = coin en haut à gauche.
- **Licence :** variante de Creative Commons BY-SA 4.0. Citer l'article, utilisation non commerciale.

**Correspondance des classes HaGRID → Maestro**

| HaGRID | Maestro |
|---|---|
| `palm`, `stop` | `palm` (paume ouverte) |
| `like` | `thumbs_up` |
| `dislike` | `thumbs_down` |
| `peace` | `two_fingers` |
| `fist` | `fist` |
| `no_gesture`, `call`, `ok`, `rock`, `mute`, `four`, `three`, `one`, `two_up` | `none` (main sans commande) |
| `peace_inverted`, `stop_inverted`, `two_up_inverted`, `three2` | écartées (ambiguës) |

**Préparation** (`scripts/prepare_hagrid.py`) :

```bash
python scripts/prepare_hagrid.py --hagrid-ann <dossier des JSON> --hagrid-img <dossier des images> --per-class 800
```

1. Conversion des boîtes au format YOLO (`classe cx cy w h`).
2. Échantillonnage de N images par classe HaGRID.
3. Images réduites à 640 px de côté maximum.
4. Ajout de nos images.
5. Découpage train/val par personne (`user_id`), puis écriture de `data/yolo/data.yaml`.

## 2. Dataset Maestro (collecté par le groupe)

Il sert à trois choses : entraîner l'approche A2 (classifieur sur points clés), ajouter le geste `index_left`, absent de HaGRID, et évaluer la robustesse à l'éclairage.

```bash
python scripts/collect_data.py --lighting normal        # puis faible, fort, contre-jour
```

Pour enregistrer, il faut maintenir une touche du clavier pendant qu'on fait le geste :

| Touche | Classe |
|---|---|
| 0 | `none` (main relâchée, positions variées) |
| 1 | `palm` |
| 2 | `thumbs_up` |
| 3 | `thumbs_down` |
| 4 | `two_fingers` |
| 5 | `index_left` |
| 6 | `fist` |

- **Points clés :** `data/landmarks.csv`, avec une ligne par exemple.
  - Colonnes : `label`, `lighting`, `image`, `hand`, `x_min`, `y_min`, `x_max`, `y_max`, puis les 21 points (`x0`, `y0`, `z0`… `z20`).
  - Les points sont normalisés : centrés sur le poignet et mis à l'échelle.
- **Images :** `data/images/<classe>/<session>_<éclairage>_<n>.jpg`, une image sur 4. Elles servent à l'évaluation et à YOLO.
- **Objectif :** au moins 200 exemples par geste et par éclairage, avec les deux mains, à plusieurs distances, et plusieurs personnes si possible.

## 3. Organisation locale

```
data/
├── README.md            # ce fichier (seul fichier publié)
├── landmarks.csv        # points clés collectés          (non publié)
├── images/<classe>/     # images collectées              (non publié)
└── yolo/                # dataset YOLO préparé           (non publié)
    ├── images/{train,val}/
    ├── labels/{train,val}/
    └── data.yaml
```
