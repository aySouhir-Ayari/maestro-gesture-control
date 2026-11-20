"""Paramètres globaux de Maestro."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
MUSIC_DIR = ROOT / "music"
DATA_DIR = ROOT / "data"

# Modèle MediaPipe Hand Landmarker (téléchargé automatiquement au premier lancement)
HAND_MODEL_PATH = MODELS_DIR / "hand_landmarker.task"
HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

# Classifieur ML (produit par scripts/train_classifier.py)
ML_MODEL_PATH = MODELS_DIR / "gesture_classifier.joblib"
DATASET_CSV = DATA_DIR / "landmarks.csv"

# Caméra
CAMERA_INDEX = 0
FRAME_WIDTH = 960
FRAME_HEIGHT = 540
MIRROR = True  # vue « miroir » : plus naturelle pour l'utilisateur

# Détection
MAX_HANDS = 2               # plusieurs mains possibles dans l'image
MIN_DETECTION_CONF = 0.6
MIN_PRESENCE_CONF = 0.6
MIN_TRACKING_CONF = 0.5

# Stabilisation temporelle
STABLE_FRAMES = 8           # nb d'images consécutives pour valider un geste (~0,3 s à 25 fps)
COOLDOWN_S = 1.0            # délai minimal entre deux commandes
REPEAT_INTERVAL_S = 0.6     # répétition pour les gestes de volume maintenus
MIN_CONFIDENCE = 0.6        # confiance minimale (classifieur ML)

# Lecteur
VOLUME_STEP = 0.1
SUPPORTED_AUDIO = (".mp3", ".ogg", ".wav")

# Les 5 gestes et leur commande
GESTURES = ["palm", "thumbs_up", "thumbs_down", "two_fingers", "index_left"]
NONE = "none"
LABELS = [NONE] + GESTURES

GESTURE_NAMES_FR = {
    "palm": "Paume ouverte",
    "thumbs_up": "Pouce levé",
    "thumbs_down": "Pouce baissé",
    "two_fingers": "Deux doigts",
    "index_left": "Index à gauche",
    NONE: "Aucun geste",
}

COMMANDS = {
    "palm": "play_pause",
    "thumbs_up": "volume_up",
    "thumbs_down": "volume_down",
    "two_fingers": "next_track",
    "index_left": "previous_track",
}
REPEATABLE = {"thumbs_up", "thumbs_down"}  # maintenir le geste répète la commande
