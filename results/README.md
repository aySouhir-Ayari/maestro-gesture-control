# Résultats

Cette page présente les résultats obtenus, leur analyse critique et les expériences qui restent à mener. Les scripts écrivent leurs sorties dans `reports/`, qui n'est pas publié ; les résultats retenus sont recopiés ici.

| Expérience | Objet | État |
|---|---|---|
| **E1** | Évaluation en direct de l'approche A1 (règles géométriques), éclairage normal et faible | ✅ réalisée (23/11/2026) |
| E2 | A1 règles vs A2 Random Forest / SVM / MLP sur le dataset Maestro | ⏳ en attente de la collecte des données |
| E3 | Robustesse à un éclairage jamais vu à l'entraînement | ⏳ en attente de la collecte des données |
| E4 | Fine-tuning de YOLOv8n sur HaGRID (approche B) | ⏳ entraînement à lancer sur Google Colab |
| E5 | Comparaison A1 / A2 / B sur les mêmes images (Maestro et scènes réelles HaGRID) | ⏳ après E2 et E4 |

---

## E1 : évaluation en direct de l'approche A1 (règles géométriques)

### Protocole

Commande : `python scripts/evaluate_live.py --classifier rules --lighting <normal|faible>`

- **Matériel :** webcam intégrée d'un PC portable (Windows), résolution demandée 960×540, sans GPU.
- **Gestes :** chacun des 6 gestes est demandé une fois, avec 3 s de préparation puis 6 s pour le faire valider.
- **Repos :** 30 s pendant lesquelles l'utilisatrice bouge naturellement **sans** faire de commande. Chaque geste validé pendant cette phase compte comme une fausse activation.
- **Validation d'un geste :** le même geste doit être reconnu sur 8 images consécutives (environ 0,3 à 0,5 s), main immobile, comme dans l'application.
- **Mesures :**
  - **taux de réussite :** part des gestes demandés qui ont été validés ;
  - **temps de validation :** délai entre la consigne et la validation ;
  - **mauvais gestes validés :** pendant un essai, un autre geste que celui demandé a été validé ;
  - **fausses activations par minute ;**
  - **FPS et latence :** latence = détection MediaPipe + classification, par image.

### Résultats globaux

| Éclairage | Taux de réussite | Temps médian de validation | Mauvais gestes validés | Fausses activations / min | FPS | Latence médiane | Latence p95 |
|---|---|---|---|---|---|---|---|
| Normal | **67 %** (4/6) | 3,09 s | 4 essais sur 6 | 12,0 | 16,7 | 19,7 ms | 25,8 ms |
| Faible | **67 %** (4/6) | 1,08 s | 0 | 26,0 | 16,7 | 20,4 ms | 26,7 ms |

### Résultats par geste

| Geste | Normal : validé (temps) | Normal : autre geste validé | Faible : validé (temps) |
|---|---|---|---|
| ✋ Paume ouverte | ✅ 5,19 s | poing fermé | ✅ 3,77 s |
| 👍 Pouce levé | ❌ | poing fermé | ❌ |
| 👎 Pouce baissé | ❌ | deux doigts | ❌ |
| ✌️ Deux doigts | ✅ 0,45 s | – | ✅ 0,86 s |
| 👈 Index à gauche | ✅ 5,69 s | poing fermé | ✅ 1,30 s |
| ✊ Poing fermé | ✅ 1,00 s | – | ✅ 0,80 s |

### Analyse critique

**1. Le temps réel est atteint.**
- La reconnaissance d'une image prend environ 20 ms (p95 < 27 ms), soit beaucoup moins que les 60 ms disponibles par image à 16,7 FPS.
- Le FPS est donc limité par la webcam et l'affichage, pas par le modèle : il reste de la marge pour une approche plus lourde (A2, ou YOLO une image sur deux).
- La latence ne change pas avec l'éclairage.

**2. Les gestes du pouce sont le point faible des règles.**
- Pouce levé et pouce baissé n'ont **jamais** été validés, dans les deux éclairages. En éclairage normal, ils ont été pris pour un **poing fermé** ou pour **deux doigts**.
- **Explication :** la règle décide si le pouce est tendu en comparant sa distance à la base de l'auriculaire. Face à la caméra, le pouce levé est vu de profil : ses points se projettent près du poing, et la profondeur (z) estimée par MediaPipe est peu fiable. Le seuil fixe ne s'adapte pas à l'orientation de la main.
- **Conséquence :** c'est la principale justification de l'approche A2. Un classifieur appris sur les 21 points peut apprendre cette frontière à partir d'exemples réels, au lieu d'un seuil choisi à la main.

**3. Le poing fermé est sur-détecté.**
- Dans 3 essais sur 6 (éclairage normal), un « poing fermé » a été validé alors qu'un autre geste était demandé.
- **Explication :** la main passe naturellement par une position fermée en changeant de geste. Si cette position dure plus que la fenêtre de validation, elle est validée.
- **Pistes :** exiger une durée plus longue pour le poing (le geste le plus « neutre »), ou ne l'accepter que dans le mode playlist où il a un sens.

**4. Trop de fausses activations.**
- 12 par minute en éclairage normal et 26 par minute en éclairage faible, alors que la cible pour une interface utilisable est inférieure à 1 par minute.
- **Explication :** les règles n'ont pas de classe « aucun geste » apprise. Toute main dont les doigts passent par une configuration reconnue déclenche une commande.
- La hausse en éclairage faible vient probablement de points clés plus bruités (MediaPipe moins sûr), qui font passer la main d'une configuration à l'autre.
- **Ce qui protège déjà l'application** (et n'était pas actif pendant ce test) :
  - le **mot d'activation « Maestro »** : hors activation, aucune commande n'est possible ;
  - la **veille automatique** après 60 s sans main.
- **Pistes :**
  - classe `none` apprise (A2, B) ;
  - augmenter `STABLE_FRAMES` (paramètre du `.env`) ;
  - mesurer le compromis : moins de fausses activations, mais une validation plus lente.

**5. L'éclairage faible ne dégrade pas la reconnaissance des gestes réussis.**
- Le taux de réussite est identique (67 %) et la validation est même plus rapide (1,08 s contre 3,09 s), car il n'y a pas eu de mauvais geste validé avant le bon.
- MediaPipe reste donc robuste à un éclairage faible pour la détection, mais il produit plus de fausses activations (point 4).

**6. Limites de cette expérience.**
- Une seule personne et un seul essai par geste et par éclairage, soit 12 essais au total : les pourcentages sont très incertains. Par exemple, 4/6 réussites donne un intervalle de confiance à 95 % d'environ 30 % à 90 %.
- Les éclairages « fort » et « contre-jour » n'ont pas encore été testés.
- Cette expérience est un **premier diagnostic**, pas une mesure définitive.
- **Prochaine étape :** répéter le protocole avec `--rounds 3` et sur les 4 éclairages, puis comparer avec A2 et B.

### Conclusion de E1

| Point fort | Point faible |
|---|---|
| Temps réel largement atteint (~20 ms par image) | Gestes du pouce non reconnus (0/4) |
| Gestes « deux doigts », « index à gauche », « poing » et « paume » reconnus dans les deux éclairages | Confusion fréquente avec le poing fermé |
| Aucun entraînement ni aucune donnée nécessaires | 12 à 26 fausses activations par minute |

Ces résultats **justifient expérimentalement** les deux autres approches du projet :
- **A2 (classifieur appris sur les points clés)** pour corriger les gestes du pouce et apprendre une vraie classe « aucun geste » ;
- **B (YOLOv8 sur HaGRID)** pour la robustesse en scènes réelles.

---

## Expériences à venir

### E2, E3 : A1 vs A2 (classifieur sur points clés)

```bash
python scripts/collect_data.py --lighting normal      # puis faible, fort, contre-jour (≈200 exemples / geste)
python scripts/train_classifier.py                    # E2 : règles vs RF / SVM / MLP
python scripts/train_classifier.py --test-lighting contre-jour   # E3 : éclairage jamais vu
```

| Méthode | Accuracy | F1 macro | Latence (ms) | Acc. normal | Acc. faible | Acc. fort | Acc. contre-jour |
|---|---|---|---|---|---|---|---|
| A1 Règles géométriques | | | | | | | |
| A2 Random Forest | | | | | | | |
| A2 SVM (RBF) | | | | | | | |
| A2 MLP | | | | | | | |

### E4 : YOLOv8n sur HaGRID

`notebooks/train_yolo_colab.ipynb` (Google Colab, GPU T4)

| mAP50 | mAP50-95 | Précision | Rappel |
|---|---|---|---|
| | | | |

### E5 : comparaison finale A1 / A2 / B sur les mêmes images

```bash
python scripts/evaluate.py                  # images Maestro, par éclairage
python scripts/evaluate.py --source yolo    # scènes réelles (validation HaGRID)
```

| Méthode | Accuracy | Précision | Rappel | F1 macro | Main détectée | Fausses activations | Latence (ms) |
|---|---|---|---|---|---|---|---|
| A1 | | | | | | | |
| A2 | | | | | | | |
| B | | | | | | | |

Figures à ajouter : matrices de confusion (`reports/eval_*.png`, `reports/confusion_*.png`) et courbes d'entraînement YOLO (`runs/maestro/results.png`).
