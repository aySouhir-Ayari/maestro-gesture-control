"""Paramètres globaux de Maestro.

Les valeurs ci-dessous sont les valeurs par défaut. Pour les changer sans toucher au code,
écrivez-les dans le fichier .env à la racine du projet (voir .env.example), par ex. :
    CAMERA_INDEX=1
    LEFT_HANDED=false
"""
import os
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

# Approche B : YOLOv8 entraîné sur HaGRID + vos images (scripts/prepare_hagrid.py, train_yolo.py)
YOLO_MODEL_PATH = MODELS_DIR / "maestro_yolo.pt"
YOLO_DATASET_DIR = DATA_DIR / "yolo"
YOLO_IMGSZ = 416            # taille d'entrée (compromis vitesse / précision)
YOLO_CONF = 0.35            # confiance minimale d'une détection
YOLO_EVERY = 1              # lancer YOLO toutes les N images (2 = deux fois plus rapide)
# Correspondance classes HaGRID -> gestes Maestro. Les autres classes HaGRID (call, ok, rock...)
# deviennent « none » : YOLO apprend à reconnaître une main sans commande (moins de fausses activations).
HAGRID_MAP = {
    "palm": "palm", "stop": "palm",
    "like": "thumbs_up", "dislike": "thumbs_down",
    "peace": "two_fingers",
    "fist": "fist",
    "no_gesture": "none",
}
HAGRID_NONE_CLASSES = ["call", "ok", "rock", "mute", "four", "three", "one", "two_up"]

# Classifieur utilisé par défaut par main.py : rules | ml | yolo
CLASSIFIER = "rules"

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

# Activation vocale (« Maestro » / « Maestro stop »)
VOICE_ENABLED = True
WAKE_WORDS = ["maestro"]
STOP_WORDS = ["stop"]
REQUIRE_WAKE_FOR_STOP = True   # il faut dire « Maestro stop », pas seulement « stop »
STOP_WINDOW_S = 4.0            # « stop » accepté jusqu'à 4 s après « Maestro » (phrase coupée en deux)
VOICE_USE_GRAMMAR = True       # limite la reconnaissance à ces mots (plus fiable)
AUTO_SLEEP_S = 60              # veille auto sans main visible (0 = jamais)
STOP_PAUSES_MUSIC = True       # « Maestro stop » (ou touche M) met aussi la musique en pause
VOSK_MODEL_DIR = MODELS_DIR / "vosk-model-small-fr-0.22"
VOSK_MODEL_URL = "https://alphacephei.com/vosk/models/vosk-model-small-fr-0.22.zip"

# Lecteur : local (fichiers du dossier music/) | media (Spotify GRATUIT par les touches média
# de Windows) | spotify (API Spotify, compte Premium obligatoire)
PLAYER = "local"
MEDIA_APP_EXE = "Spotify.exe"   # programme dont on lit le titre en mode media
ONLINE_ARTWORK = True           # cherche la vraie pochette en ligne (catalogue iTunes) si besoin
SHOW_NOW_PLAYING_S = 4.0        # affiche la fiche + pochette à chaque nouveau morceau (0 = non)
VOLUME_STEP = 0.1
SUPPORTED_AUDIO = (".mp3", ".ogg", ".wav")

# Les 5 gestes et leur commande
GESTURES = ["palm", "thumbs_up", "thumbs_down", "two_fingers", "index_left"]
NONE = "none"
FIST = "fist"                      # poing fermé : valide un choix dans la playlist
LABELS = [NONE] + GESTURES + [FIST]

GESTURE_NAMES_FR = {
    "palm": "Paume ouverte",
    "thumbs_up": "Pouce levé",
    "thumbs_down": "Pouce baissé",
    "two_fingers": "Deux doigts",
    "index_left": "Index à gauche",
    FIST: "Poing fermé",
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

# --- Rôles des deux mains --------------------------------------------------
# Rôles inversés : main DROITE = volume au pincement | main GAUCHE = morceaux
# (gestes, swipe, avance/recul). Remettre False pour revenir à gauche = volume.
LEFT_HANDED = True
SWAP_HANDEDNESS = False     # à activer si MediaPipe inverse gauche et droite sur votre caméra

# --- Volume au pincement (main gauche) -------------------------------------
PINCH_MIN = 0.15            # écart pouce-index (÷ taille de la main) -> volume 0 %
PINCH_MAX = 1.10            # écart -> volume 100 %
PINCH_ENGAGE_FRAMES = 5     # pose stable avant de prendre la main sur le volume
PINCH_SMOOTHING = 0.35      # lissage exponentiel (0 = figé, 1 = brut)

# --- Swipe (trajectoire de la main droite) ---------------------------------
SWIPE_WINDOW_S = 0.45       # durée maximale du mouvement
SWIPE_MIN_DIST = 0.20       # déplacement minimal (fraction de la largeur/hauteur de l'image)
SWIPE_DIRECTIONALITY = 2.0  # le mouvement doit être 2x plus horizontal que vertical (ou l'inverse)
SWIPE_COOLDOWN_S = 0.7
STILL_SPEED = 0.6           # au-delà (image/s), la main bouge : pas de geste statique

# --- Avancer / reculer : pincer et glisser (main droite) -------------------
SCRUB_PINCH_ON = 0.30       # écart pouce-index sous lequel on « attrape » la barre
SCRUB_PINCH_OFF = 0.50      # écart au-dessus duquel on la « relâche » (hystérésis)
SCRUB_ENGAGE_FRAMES = 4
SCRUB_SECONDS_PER_WIDTH = 60.0  # glisser sur toute la largeur de l'image = 60 s

# --- Playlist en carrousel et fiche du morceau -----------------------------
CAROUSEL_TIMEOUT_S = 8.0
INFO_DURATION_S = 5.0

# --- Baisse automatique du volume quand on parle à Maestro ------------------
DUCK_ENABLED = True
DUCK_LEVEL = 0.25           # la musique descend à 25 % de son volume
DUCK_HOLD_S = 3.0           # ... et y reste 3 s après le dernier « Maestro » entendu
DUCK_FADE_S = 0.35          # durée du fondu (baisse et remontée)

# --- Interface ---------------------------------------------------------------
DISPLAY_HEIGHT = 720        # l'image de la caméra est agrandie à cette hauteur avant l'affichage
WINDOW_SCREEN_FRACTION = 0.9  # la fenêtre occupe 90 % de l'écran au démarrage
START_FULLSCREEN = False    # True : démarre en plein écran (touche F pour basculer)
SHOW_VISUALIZER = False     # onde sonore animée (désactivée)
VISUALIZER_BANDS = 40       # barres de l'onde sonore
VISUALIZER_FPS = 20         # résolution temporelle de l'analyse audio
ANIMATION_S = 0.9           # durée de l'animation d'un geste validé


# --- Spotify (PLAYER=spotify ou python main.py --player spotify) -----------
# Identifiants de VOTRE application : https://developer.spotify.com/dashboard  (à mettre dans .env)
SPOTIFY_CLIENT_ID = ""
SPOTIFY_CLIENT_SECRET = ""
SPOTIFY_REDIRECT_URI = "http://127.0.0.1:8888/callback"   # à recopier tel quel dans le dashboard
SPOTIFY_PLAYLIST = ""       # nom, lien ou ID d'une de VOS playlists (vide = la première)
SPOTIFY_DEVICE = ""         # nom de l'appareil (vide = celui qui joue / le premier trouvé)
SPOTIFY_CACHE = MODELS_DIR / ".spotify_token"   # jeton de connexion (ne pas partager)

# --- Fichier .env ------------------------------------------------------------
ENV_FILE = ROOT / ".env"


def _parse_env(path: Path) -> dict[str, str]:
    """Lit un fichier .env simple : CLE=valeur, lignes # ignorées, guillemets facultatifs."""
    values = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.split(" #", 1)[0].strip().strip('"').strip("'")
        values[key.strip().upper()] = value
    return values


def _convert(raw: str, default):
    """Convertit le texte du .env dans le type de la valeur par défaut."""
    if isinstance(default, bool):
        if raw.lower() in ("1", "true", "vrai", "oui", "yes", "on"):
            return True
        if raw.lower() in ("0", "false", "faux", "non", "no", "off"):
            return False
        raise ValueError("attendu true/false")
    if isinstance(default, int):
        return int(raw)
    if isinstance(default, float):
        return float(raw)
    if isinstance(default, Path):
        p = Path(raw).expanduser()
        return p if p.is_absolute() else ROOT / p
    if isinstance(default, list):
        return [x.strip() for x in raw.split(",") if x.strip()]
    return raw


_FIXED = {"ROOT", "ENV_FILE", "GESTURES", "NONE", "FIST", "LABELS"}   # structure du projet : non modifiable


def load_env(path: Path = ENV_FILE) -> dict:
    """Applique le .env, puis les variables d'environnement MAESTRO_<NOM> (prioritaires)."""
    g = globals()
    settings = {k for k, v in g.items() if k.isupper() and k not in _FIXED
                and isinstance(v, (bool, int, float, str, Path, list))}
    found = {k.removeprefix("MAESTRO_"): v for k, v in _parse_env(path).items()}
    found.update({k[8:]: v for k, v in os.environ.items() if k.startswith("MAESTRO_")})
    applied = {}
    for key, raw in found.items():
        if key not in settings:
            print(f"[Maestro] .env : paramètre inconnu « {key} » (ignoré)")
            continue
        try:
            g[key] = applied[key] = _convert(raw, g[key])
        except ValueError as exc:
            print(f"[Maestro] .env : valeur invalide {key}={raw!r} ({exc}) -> {g[key]!r} gardé")
    return applied

load_env()
