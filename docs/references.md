# Références

Toutes les sources externes (articles, datasets, modèles, bibliothèques et services) utilisées dans le projet.

## Datasets

1. **HaGRID** : A. Kapitanov, K. Kvanchiani, A. Nagaev, R. Kraynov, A. Makhliarchuk, *HaGRID – HAnd Gesture Recognition Image Dataset*, IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 2024.
   - arXiv : [2206.08219](https://arxiv.org/abs/2206.08219) · Dépôt : [github.com/hukenovs/hagrid](https://github.com/hukenovs/hagrid)
   - Licence : variante de CC BY-SA 4.0.
2. **HaGRIDv2** : A. Nuzhdin, A. Nagaev, A. Sautin, A. Kapitanov, K. Kvanchiani, *HaGRIDv2: 1M Images for Static and Dynamic Hand Gesture Recognition*, 2024.
   - arXiv : [2412.01508](https://arxiv.org/abs/2412.01508). Version étendue, citée pour les perspectives.
3. **HaGRID Sample 30k 384p** : C. Mills, sous-ensemble de 31 833 images de HaGRID réduites en 384 p.
   - Lien : [huggingface.co/datasets/cj-mills/hagrid-sample-30k-384p](https://huggingface.co/datasets/cj-mills/hagrid-sample-30k-384p)
   - Licence : CC BY-SA 4.0.

## Détection et suivi des mains

4. F. Zhang, V. Bazarevsky, A. Vakunov, A. Tkachenka, G. Sung, C.-L. Chang, M. Grundmann, *MediaPipe Hands: On-device Real-time Hand Tracking*, CVPR Workshop on Computer Vision for AR/VR, 2020.
   - arXiv : [2006.10214](https://arxiv.org/abs/2006.10214)
5. C. Lugaresi et al., *MediaPipe: A Framework for Building Perception Pipelines*, 2019.
   - arXiv : [1906.08172](https://arxiv.org/abs/1906.08172)
   - Modèle *Hand Landmarker* : [ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker) (Apache 2.0)

## Détection d'objets (approche B)

6. J. Redmon, S. Divvala, R. Girshick, A. Farhadi, *You Only Look Once: Unified, Real-Time Object Detection*, CVPR 2016.
   - arXiv : [1506.02640](https://arxiv.org/abs/1506.02640)
7. G. Jocher, A. Chaurasia, J. Qiu, *Ultralytics YOLOv8*, 2023.
   - Code et poids : [github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) (AGPL-3.0)
8. T.-Y. Lin et al., *Microsoft COCO: Common Objects in Context*, ECCV 2014. Poids de départ de YOLOv8.
   - arXiv : [1405.0312](https://arxiv.org/abs/1405.0312)

## Apprentissage automatique (approche A2) et évaluation

9. L. Breiman, *Random Forests*, Machine Learning, 45(1), 5-32, 2001.
10. C. Cortes, V. Vapnik, *Support-Vector Networks*, Machine Learning, 20(3), 273-297, 1995.
11. D. E. Rumelhart, G. E. Hinton, R. J. Williams, *Learning representations by back-propagating errors*, Nature, 323, 533-536, 1986. Base du MLP.
12. F. Pedregosa et al., *Scikit-learn: Machine Learning in Python*, Journal of Machine Learning Research, 12, 2825-2830, 2011.
    - [scikit-learn.org](https://scikit-learn.org)
    - Métriques utilisées : accuracy, précision, rappel, F1 macro, matrice de confusion.

## Voix, audio et lecteurs

13. **Vosk** : boîte à outils de reconnaissance vocale hors ligne, Alpha Cephei.
    - [alphacephei.com/vosk](https://alphacephei.com/vosk/) · modèle `vosk-model-small-fr-0.22` (Apache 2.0)
14. **pygame** : lecture audio des fichiers locaux. [pygame.org](https://www.pygame.org) (LGPL)
15. **Spotify Web API** et **spotipy** : lecteur Spotify (compte Premium).
    - [developer.spotify.com/documentation/web-api](https://developer.spotify.com/documentation/web-api) · [spotipy.readthedocs.io](https://spotipy.readthedocs.io)
16. **Windows Media Control** (WinRT, paquets `winrt-*`) et **pycaw** : titre, pochette et volume du lecteur Spotify gratuit.
17. **iTunes Search API** : recherche de pochettes d'album, sans clé.
    - [developer.apple.com, iTunes Search API](https://developer.apple.com/library/archive/documentation/AudioVideo/Conceptual/iTuneSearchAPI/)

## Bibliothèques générales

18. **OpenCV** : G. Bradski, *The OpenCV Library*, Dr. Dobb's Journal, 2000. [opencv.org](https://opencv.org) (Apache 2.0)
19. **NumPy** : C. R. Harris et al., *Array programming with NumPy*, Nature, 585, 357-362, 2020.
20. **Pandas**, **Matplotlib**, **PyTorch** (A. Paszke et al., NeurIPS 2019), **pytest**.
21. **Google Colab** : entraînement de YOLOv8 sur GPU.

## Contributions externes

- Le code de Maestro a été écrit par le groupe. Il utilise les bibliothèques et modèles ci-dessus sans modifier leur code.
- Les seuls poids préentraînés réutilisés sont MediaPipe Hand Landmarker, YOLOv8n COCO (point de départ du fine-tuning) et le modèle Vosk.
- Un assistant IA (Claude, Anthropic) a été utilisé comme aide à la programmation et à la rédaction de la documentation.
