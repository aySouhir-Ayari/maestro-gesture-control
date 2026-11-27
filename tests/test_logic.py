"""Tests de la logique (sans webcam) : python -m pytest tests"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from maestro import config  # noqa: E402
from maestro.features import normalized_vector  # noqa: E402
from maestro.gestures import RuleClassifier  # noqa: E402
from maestro.hand_tracker import Hand, select_main_hand  # noqa: E402
from maestro.player import MusicPlayer  # noqa: E402
from maestro.stabilizer import GestureStabilizer  # noqa: E402
from synthetic import POSES, make_hand  # noqa: E402


@pytest.mark.parametrize("label", list(POSES))
@pytest.mark.parametrize("scale", [0.6, 1.0, 1.3])
def test_rules_recognize_each_gesture(label, scale):
    lm = make_hand(**POSES[label], scale=scale)
    assert RuleClassifier().predict(lm).label == label


def test_features_invariant_to_position_and_scale():
    a = normalized_vector(make_hand(**POSES["palm"]))
    b = normalized_vector(make_hand(**POSES["palm"], center=(0.3, 0.6), scale=0.5))
    assert np.allclose(a, b, atol=1e-4)


def test_main_hand_is_the_largest():
    small = Hand(make_hand(scale=0.4), "Left", 0.9)
    big = Hand(make_hand(scale=1.0), "Right", 0.9)
    assert select_main_hand([small, big]) is big
    assert select_main_hand([]) is None


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def run(stab, clock, labels, dt=0.04):
    fired = []
    for lab in labels:
        clock.t += dt
        out = stab.update(lab)
        if out:
            fired.append(out)
    return fired


def test_stabilizer_needs_stable_frames():
    clock = FakeClock()
    stab = GestureStabilizer(stable_frames=8, cooldown=1.0, clock=clock)
    # geste instable (alterne) : rien ne se déclenche
    assert run(stab, clock, ["palm", "two_fingers"] * 10) == []
    # geste stable : une seule commande, même maintenu longtemps
    assert run(stab, clock, ["palm"] * 60) == ["palm"]


def test_stabilizer_rearms_after_release():
    clock = FakeClock()
    stab = GestureStabilizer(stable_frames=5, cooldown=0.5, clock=clock)
    fired = run(stab, clock, ["palm"] * 10 + ["none"] * 20 + ["palm"] * 10)
    assert fired == ["palm", "palm"]


def test_volume_repeats_while_held():
    clock = FakeClock()
    stab = GestureStabilizer(stable_frames=5, cooldown=0.5, repeat_interval=0.5, clock=clock)
    fired = run(stab, clock, ["thumbs_up"] * 80)  # ~3,2 s
    assert fired.count("thumbs_up") >= 5


def test_player_commands_without_audio(tmp_path):
    p = MusicPlayer(music_dir=tmp_path)  # dossier vide -> mode simulation
    p.execute("play_pause")
    assert p.playing
    p.execute("volume_up")
    assert p.volume == pytest.approx(0.6)
    for _ in range(10):
        p.execute("volume_down")
    assert p.volume == 0.0
    p.execute("play_pause")
    assert not p.playing


def test_every_gesture_has_a_command():
    assert set(config.COMMANDS) == set(config.GESTURES)


# --- activation vocale ---------------------------------------------------
from maestro.activation import ActivationController  # noqa: E402
from maestro.voice import ACTIVATE, DEACTIVATE, VoiceCommandParser  # noqa: E402


@pytest.mark.parametrize("text,expected", [
    ("maestro", ACTIVATE),
    ("mestro", ACTIVATE),             # prononciation approximative
    ("maestro stop", DEACTIVATE),
    ("maestro, stop", DEACTIVATE),
    ("stop", None),                   # « stop » seul ne suffit pas
    ("[unk]", None),
    ("", None),
])
def test_voice_parser(text, expected):
    assert VoiceCommandParser().parse(text) == expected


def test_stop_after_split_sentence():
    clock = FakeClock()
    parser = VoiceCommandParser(clock=clock)
    assert parser.parse("maestro") == ACTIVATE
    clock.t += 1.5
    assert parser.parse("stop") == DEACTIVATE      # « Maestro… stop » coupé en deux
    clock.t += 1.0
    assert parser.parse("stop") is None            # déjà consommé
    assert parser.parse("maestro") == ACTIVATE
    clock.t += 10
    assert parser.parse("stop") is None            # trop tard


def test_activation_cycle_and_auto_sleep():
    clock = FakeClock()
    act = ActivationController(start_active=False, auto_sleep_s=10, clock=clock)
    assert not act.active
    assert act.handle(ACTIVATE) and act.active
    clock.t += 5
    act.update(hand_present=True)
    clock.t += 9
    assert not act.update(hand_present=False) and act.active      # 9 s sans main
    clock.t += 2
    assert act.update(hand_present=False) and not act.active      # > 10 s -> veille
    act.toggle()
    assert act.active
    assert act.handle(DEACTIVATE) and not act.active


def test_stabilizer_reset():
    clock = FakeClock()
    stab = GestureStabilizer(stable_frames=5, clock=clock)
    run(stab, clock, ["palm"] * 3)
    stab.reset()
    assert stab.progress == 0


def test_player_pause_is_idempotent(tmp_path):
    p = MusicPlayer(music_dir=tmp_path)
    p.pause()
    assert not p.playing          # déjà en pause : rien ne change
    p.execute("play_pause")
    p.pause()
    assert not p.playing
    p.pause()
    assert not p.playing


# --- nouvelles fonctionnalités : mouvements, pincement, rôles, playlist -----
from maestro.controller import CAROUSEL, NORMAL, MaestroController, assign_roles  # noqa: E402
from maestro.controls import PinchVolume, SeekScrubber  # noqa: E402
from maestro.features import pinch_ratio  # noqa: E402
from maestro.motion import LEFT, RIGHT, UP, MotionTracker  # noqa: E402
from synthetic import pinch_hand  # noqa: E402


def move(tracker, clock, start, end, steps=10, dt=0.03):
    out = None
    for k in range(steps + 1):
        clock.t += dt
        p = np.array(start) + (np.array(end) - np.array(start)) * k / steps
        out = tracker.update(p) or out
    return out


def test_swipe_directions():
    clock = FakeClock()
    m = MotionTracker(clock=clock)
    assert move(m, clock, (0.7, 0.5), (0.3, 0.52)) == LEFT
    clock.t += 1
    assert move(m, clock, (0.3, 0.5), (0.7, 0.48)) == RIGHT
    clock.t += 1
    assert move(m, clock, (0.5, 0.8), (0.52, 0.4)) == UP


def test_slow_or_small_motion_is_not_a_swipe():
    clock = FakeClock()
    m = MotionTracker(clock=clock)
    assert move(m, clock, (0.5, 0.5), (0.58, 0.5)) is None              # trop petit
    clock.t += 1
    assert move(m, clock, (0.3, 0.5), (0.7, 0.5), steps=60, dt=0.03) is None  # trop lent (1,8 s)
    clock.t += 1
    assert move(m, clock, (0.3, 0.3), (0.6, 0.6)) is None               # diagonale


def test_pinch_volume_follows_finger_gap():
    pv = PinchVolume(engage_frames=1, smoothing=1.0)
    assert pv.update(pinch_hand(0.05)) == pytest.approx(0.0, abs=0.01)
    assert pv.update(pinch_hand(1.2)) == pytest.approx(1.0, abs=0.01)
    mid = pv.update(pinch_hand(0.6))
    assert 0.3 < mid < 0.7
    assert pv.update(make_hand(**POSES["palm"])) is None   # main ouverte : volume figé


def test_fist_does_not_mute():
    pv = PinchVolume(engage_frames=1)
    assert pv.update(make_hand(**POSES["fist"])) is None


def test_scrubber_drag_and_release():
    sc = SeekScrubber(engage_frames=2, seconds_per_width=60)
    assert pinch_ratio(pinch_hand(0.0)) < 0.3
    for _ in range(3):
        assert sc.update(pinch_hand(0.0, center=(0.4, 0.7)), position=10, duration=100) is None
    assert sc.active
    sc.update(pinch_hand(0.0, center=(0.6, 0.7)), position=10, duration=100)   # +0,2 largeur
    assert sc.target == pytest.approx(22, abs=0.5)
    released = sc.update(pinch_hand(1.0, center=(0.6, 0.7)), position=10, duration=100)
    assert released == pytest.approx(22, abs=0.5) and not sc.active


def _hand(label, side, center=(0.5, 0.75), scale=1.0):
    return Hand(make_hand(**POSES[label], center=center, scale=scale), side, 0.95)


def test_assign_roles():
    left, right = _hand("palm", "Left", (0.25, 0.75)), _hand("palm", "Right", (0.75, 0.75))
    assert assign_roles([left, right]) == (left, right)
    assert assign_roles([left, right], left_handed=True) == (right, left)
    assert assign_roles([right]) == (None, right)
    # MediaPipe donne le même côté aux deux mains : on tranche par la position
    a, b = _hand("palm", "Right", (0.2, 0.75)), _hand("palm", "Right", (0.8, 0.75))
    assert assign_roles([b, a]) == (a, b)


class FakePlayer:
    def __init__(self, n=3):
        self.tracks = [f"t{i}" for i in range(n)]
        self.index, self.volume, self.position, self.duration = 0, 0.5, 10.0, 100.0
        self.track_name, self.calls = "t0", []

    def title_of(self, i):
        return self.tracks[i]

    def set_volume(self, v):
        self.volume = v

    def execute(self, cmd):
        self.calls.append(cmd)
        return cmd

    def play_index(self, i):
        self.index, self.track_name = i, self.tracks[i]
        self.calls.append(f"play_index:{i}")

    def seek(self, s):
        self.calls.append(f"seek:{s:.0f}")


def feed(brain, clock, hands_fn, frames, dt=0.03):
    state = None
    for k in range(frames):
        clock.t += dt
        state = brain.step(hands_fn(k))
    return state


def test_carousel_swipe_and_fist_selects_track():
    clock, player = FakeClock(), FakePlayer()
    brain = MaestroController(RuleClassifier(), player, left_handed=False, clock=clock)
    swipe_left = lambda k: [_hand("palm", "Right", (0.75 - 0.04 * k, 0.75))]
    still_palm = lambda k: [_hand("palm", "Right", (0.35, 0.75))]
    st = feed(brain, clock, swipe_left, 11)
    assert st.mode == CAROUSEL and st.carousel_index == 0      # 1er swipe : ouvre la playlist
    feed(brain, clock, still_palm, 30)                         # main immobile après le swipe
    assert player.calls == []                                  # ... ne déclenche pas pause
    clock.t += 1
    st = feed(brain, clock, swipe_left, 11)
    assert st.carousel_index == 1                              # swipe gauche : morceau suivant
    st = feed(brain, clock, lambda k: [_hand("fist", "Right", (0.4, 0.75))], 15)
    assert "play_index:1" in player.calls and st.mode == NORMAL


def test_swipe_up_shows_info_and_static_gestures_still_work():
    clock, player = FakeClock(), FakePlayer()
    brain = MaestroController(RuleClassifier(), player, left_handed=False, clock=clock)
    st = feed(brain, clock, lambda k: [_hand("palm", "Right", (0.5, 0.95 - 0.05 * k))], 11)
    assert st.info_visible
    feed(brain, clock, lambda k: [_hand("none", "Right")], 5)
    feed(brain, clock, lambda k: [_hand("two_fingers", "Right")], 15)
    assert player.calls == ["next_track"]


def test_left_hand_pinch_sets_volume_right_hand_unaffected():
    clock, player = FakeClock(), FakePlayer()
    brain = MaestroController(RuleClassifier(), player, left_handed=False, clock=clock)
    hands = lambda k: [Hand(pinch_hand(1.2, center=(0.25, 0.75)), "Left", 0.9),
                       _hand("none", "Right", (0.75, 0.75))]
    st = feed(brain, clock, hands, 20)
    assert st.volume_active and player.volume > 0.9
    assert player.calls == []


def test_player_info_and_simulated_position(tmp_path):
    clock = FakeClock()
    p = MusicPlayer(music_dir=tmp_path, clock=clock)
    info = p.info()
    assert info["count"] == 0 and info["artist"] == "Artiste inconnu"
    p.execute("play_pause")
    clock.t += 3
    assert p.position == pytest.approx(3)
    p.pause()
    clock.t += 5
    assert p.position == pytest.approx(3)       # en pause, la position ne bouge pas


def test_inverted_roles_with_left_handed():
    clock, player = FakeClock(), FakePlayer()
    brain = MaestroController(RuleClassifier(), player, left_handed=True, clock=clock)
    hands = lambda k: [Hand(pinch_hand(1.2, center=(0.75, 0.75)), "Right", 0.9),
                       _hand("none", "Left", (0.25, 0.75))]
    st = feed(brain, clock, hands, 20)
    assert st.volume_active and st.volume_hand.handedness == "Right"


def test_actions_are_reported_for_animations():
    clock, player = FakeClock(), FakePlayer()
    brain = MaestroController(RuleClassifier(), player, left_handed=False, clock=clock)
    feed(brain, clock, lambda k: [_hand("none", "Right")], 3)
    actions = []
    for k in range(15):
        clock.t += 0.03
        actions += brain.step([_hand("two_fingers", "Right", (0.6, 0.7))]).actions
    assert len(actions) == 1 and actions[0][0] == "next_track"
    assert actions[0][1] is not None


# --- baisse du volume quand on parle ----------------------------------------
def test_duck_fades_down_then_back_up(tmp_path):
    clock = FakeClock()
    p = MusicPlayer(music_dir=tmp_path, clock=clock)
    p.tick()
    p.duck(seconds=2.0)
    for _ in range(20):
        clock.t += 0.05
        p.tick()
    assert p.ducked and p._gain == pytest.approx(config.DUCK_LEVEL)
    assert p.volume == 0.5                       # le réglage utilisateur n'est pas modifié
    for _ in range(40):
        clock.t += 0.05
        p.tick()
    assert not p.ducked and p._gain == 1.0


def test_wake_word_detected_in_partial_sentence():
    parser = VoiceCommandParser()
    assert parser.mentions_wake("maestro")
    assert parser.mentions_wake("mestro sto")
    assert not parser.mentions_wake("bonjour")


def test_spectrum_shape_and_range():
    from maestro.visualizer import compute_spectrum
    sr = 8000
    t = np.arange(sr * 2) / sr
    x = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    spec = compute_spectrum(x, sr, bands=16, fps=10)
    assert spec.shape == (20, 16)
    assert spec.min() >= 0 and spec.max() <= 1
    assert spec[:, spec.mean(axis=0).argmax()].mean() > 0.8   # la bande du 440 Hz domine


# --- Évaluation, approche B (YOLO) et affichage --------------------------------

@pytest.mark.parametrize("label", list(POSES))
def test_rules_work_on_normalized_vectors(label):
    """train_classifier/evaluate appliquent les règles aux vecteurs du CSV : même résultat."""
    from maestro.evaluation import landmarks_from_vector
    lm = make_hand(**POSES[label])
    rules = RuleClassifier()
    assert rules.predict(landmarks_from_vector(normalized_vector(lm))).label == rules.predict(lm).label


def test_markdown_table_and_metrics():
    from maestro.evaluation import markdown_table, metrics
    m = metrics(["palm", "fist", "palm"], ["palm", "palm", "palm"], ["palm", "fist"])
    assert m["accuracy"] == pytest.approx(2 / 3)
    md = markdown_table([{"m": "A", "v": 0.5}, {"m": "B"}], [("m", "Méthode", ""), ("v", "Val", "{:.2f}")])
    assert md.splitlines() == ["| Méthode | Val |", "|---|---|", "| A | 0.50 |", "| B | - |"]


def test_yolo_detection_matched_to_mediapipe_hand():
    from maestro.yolo_detector import Detection, YoloClassifier, landmark_box, match
    lm = make_hand(**POSES["palm"])
    far = Detection("fist", 0.9, (0.0, 0.0, 0.05, 0.05))
    near = Detection("thumbs_up", 0.6, landmark_box(lm))
    assert match([far, near], lm) is near
    assert match([far], lm) is None
    clf = YoloClassifier.__new__(YoloClassifier)   # sans charger de modèle
    clf.detections = [far, near]
    assert clf.predict(lm).label == "thumbs_up"
    clf.detections = []
    assert clf.predict(lm).label == config.NONE


def test_hagrid_mapping_and_yolo_format():
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import prepare_hagrid as ph
    assert ph.map_hagrid("like") == "thumbs_up" and ph.map_hagrid("stop") == "palm"
    assert ph.map_hagrid("call") == config.NONE and ph.map_hagrid("peace_inverted") is None
    cls, cx, cy, w, h = ph.yolo_line("fist", 0.2, 0.4, 0.6, 0.8).split()
    assert int(cls) == config.LABELS.index("fist")
    assert (float(cx), float(cy), float(w), float(h)) == pytest.approx((0.4, 0.6, 0.4, 0.4))


def test_collect_box_contains_hand():
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from collect_data import HEADER, hand_box
    from maestro.evaluation import FEATURE_COLS
    lm = make_hand(**POSES["palm"])
    x0, y0, x1, y1 = hand_box(lm)
    assert x0 <= lm[:, 0].min() and x1 >= lm[:, 0].max() and y0 <= lm[:, 1].min() and y1 >= lm[:, 1].max()
    assert HEADER[-63:] == FEATURE_COLS


def test_canvas_upscaled_to_display_height():
    from maestro.display import to_canvas
    out = to_canvas(np.zeros((480, 640, 3), np.uint8), height=720)
    assert out.shape == (720, 960, 3)


def test_env_file_overrides_settings(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("# commentaire\nCAMERA_INDEX=2   # webcam USB\nLEFT_HANDED=false\nDUCK_LEVEL=oops\n"
                   'CLASSIFIER="ml"\n', encoding="utf-8")
    for key in ("CAMERA_INDEX", "LEFT_HANDED", "DUCK_LEVEL", "CLASSIFIER"):
        monkeypatch.setattr(config, key, getattr(config, key))   # restauré après le test
    monkeypatch.setenv("MAESTRO_CAMERA_INDEX", "3")              # la variable d'environnement gagne
    applied = config.load_env(env)
    assert config.CAMERA_INDEX == 3 and config.LEFT_HANDED is False and config.CLASSIFIER == "ml"
    assert "DUCK_LEVEL" not in applied and isinstance(config.DUCK_LEVEL, float)


# --- Spotify (client simulé : aucun réseau) -------------------------------------

class FakeSpotify:
    def __init__(self):
        self.calls = []

    def _get(self, url, **kw):
        if url.endswith("/items"):
            items = [{"item": {"type": "track", "uri": f"spotify:track:{i}", "name": f"Titre {i}",
                               "artists": [{"name": "Artiste"}], "duration_ms": 180000,
                               "album": {"name": "Album", "images": []}}} for i in range(3)]
            items.append({"item": {"type": "episode", "uri": "spotify:episode:x"}})   # ignoré
            return {"items": items, "next": None}
        return {"name": "Ma playlist", "uri": "spotify:playlist:" + "a" * 22}

    def devices(self):
        return {"devices": [{"id": "pc", "name": "MSI", "is_active": True, "volume_percent": 40}]}

    def current_playback(self):
        return None

    def __getattr__(self, name):            # start_playback, pause_playback, seek_track, volume...
        return lambda *a, **k: self.calls.append((name, a, k))


def make_spotify(monkeypatch):
    from maestro import spotify_player as spm
    monkeypatch.setattr(config, "SPOTIFY_PLAYLIST", "a" * 22)
    fake = FakeSpotify()
    return spm.SpotifyPlayer(sp=fake, start_thread=False), fake


def run_jobs(player):
    while not player._jobs.empty():
        fn, a, k = player._jobs.get()
        player._call(fn, *a, **k)


def test_spotify_playlist_and_commands(monkeypatch):
    player, fake = make_spotify(monkeypatch)
    assert len(player.tracks) == 3 and player.volume == 0.4
    assert player.title_of(1) == "Artiste - Titre 1" and player.duration == 180
    player.play_index(2)
    player.next_track()                     # boucle : retour au 1er morceau
    player.seek(30)
    player.play_pause()                     # pause
    run_jobs(player)
    names = [c[0] for c in fake.calls]
    assert names == ["start_playback", "start_playback", "seek_track", "pause_playback"]
    assert fake.calls[1][2]["offset"] == {"position": 0} and fake.calls[1][2]["device_id"] == "pc"
    assert fake.calls[2][1] == (30000,)
    assert player.info()["format"] == "Spotify" and not player.playing


def test_spotify_volume_is_grouped_and_ducked(monkeypatch):
    player, fake = make_spotify(monkeypatch)
    t = [100.0]
    player.clock = lambda: t[0]
    for v in (0.5, 0.55, 0.6):              # pincement : une valeur par image
        player.set_volume(v)
        player._flush_volume()
    assert [c[1] for c in fake.calls if c[0] == "volume"] == [(50,)]   # puis regroupé...
    t[0] += 0.5
    player._flush_volume()
    assert [c[1] for c in fake.calls if c[0] == "volume"] == [(50,), (60,)]   # ...dernière valeur
    player.duck()
    t[0] += 1
    player._flush_volume()
    assert fake.calls[-1][1] == (15,)       # 60 % x 0,25
    t[0] += 10
    player.tick()
    player._flush_volume()
    assert fake.calls[-1][1] == (60,)


def test_playlist_id_parsing():
    from maestro.spotify_player import playlist_id_from
    pid = "37i9dQZF1DXcBWIGoYBM5M"
    assert playlist_id_from(f"https://open.spotify.com/playlist/{pid}?si=abc") == pid
    assert playlist_id_from(f"spotify:playlist:{pid}") == pid
    assert playlist_id_from("Ma playlist chill") is None


# --- Lecteur « media » (Spotify gratuit, touches média simulées) ----------------

class FakeVolume:
    def __init__(self, v=0.5):
        self.v = v

    def get(self):
        return self.v

    def set(self, v):
        self.v = v


def make_media(titles, volume_api=None):
    from maestro.media_player import MediaKeysPlayer
    keys, t = [], [0.0]
    p = MediaKeysPlayer(press=keys.append, read_titles=lambda: titles, volume_api=volume_api,
                        clock=lambda: t[0], session=None, artwork_loader=lambda a, b: None)
    return p, keys, t


def test_media_player_reads_spotify_title_and_sends_keys():
    from maestro import media_player as mp
    p, keys, _ = make_media(["Daft Punk - One More Time"])
    assert p.playing and p.title == "One More Time" and p.artist == "Daft Punk"
    p.next_track()
    p.play_pause()
    assert keys == [mp.VK_NEXT, mp.VK_PLAY_PAUSE] and not p.playing
    p2, _, _ = make_media(["Spotify Free"])
    assert not p2.playing and p2.app_found and p2.track_name.startswith("Spotify")


def test_media_volume_with_pycaw_and_with_keys():
    from maestro import media_player as mp
    vol = FakeVolume(0.4)
    p, keys, t = make_media(["A - B"], volume_api=vol)
    p.set_volume(0.7)
    assert vol.v == 0.7 and keys == []
    p.duck()
    assert vol.v == pytest.approx(0.7 * config.DUCK_LEVEL, abs=0.01)
    t[0] += 10
    p.tick()
    assert vol.v == 0.7
    p, keys, t = make_media(["A - B"])                 # sans pycaw : touches volume (2 %)
    p.set_volume(0.6)
    assert keys == [mp.VK_VOLUME_UP] * 5


def test_swipe_skips_track_when_player_has_no_playlist():
    from maestro.controller import MaestroController
    p, keys, _ = make_media(["A - B"])
    brain = MaestroController(RuleClassifier(), p, left_handed=False)
    from maestro.controller import FrameState
    st = FrameState()
    brain._on_swipe("left", st, 0.0)
    brain._on_swipe("right", st, 0.0)
    from maestro import media_player as mp
    assert keys == [mp.VK_NEXT, mp.VK_PREV] and brain.mode == "normal"


def test_help_lines_follow_player_and_hands():
    from maestro.ui import help_lines
    media, _, _ = make_media(["A - B"])
    text = " ".join(e for _, e in help_lines(media, True))
    assert "MAIN DROITE = VOLUME" in text and "playlist" not in text and "avancer" not in text
    local = " ".join(e for _, e in help_lines(MusicPlayer(), False))
    assert "MAIN GAUCHE = VOLUME" in local and "playlist" in local and "avancer" in local



# --- Vraies pochettes ---------------------------------------------------------------

def _png(color):
    import cv2
    img = np.zeros((64, 64, 3), np.uint8)
    img[:] = color
    return cv2.imencode(".png", img)[1].tobytes()


def _wait_cover(player, tries=200):
    import time as _t
    for _ in range(tries):
        c = player.cover_of(0)
        if c is not None:
            return c
        _t.sleep(0.01)
    return None


def test_media_cover_comes_from_windows_session():
    from types import SimpleNamespace
    import time as _t
    from maestro.media_player import MediaKeysPlayer
    sess = SimpleNamespace(available=True, title="One More Time", artist="Daft Punk", album="Discovery",
                           playing=True, position=42.0, duration=320.0, thumbnail=_png((0, 0, 255)),
                           thumb_title="One More Time", updated=_t.monotonic() + 100)
    itunes_calls = []
    p = MediaKeysPlayer(press=lambda k: None, read_titles=lambda: [], volume_api=None, session=sess,
                        artwork_loader=lambda a, t: itunes_calls.append(t))
    assert p.title == "One More Time" and p.position == 42.0 and p.duration == 320.0
    cover = _wait_cover(p)
    assert cover is not None and cover.shape == (300, 220, 3)
    assert tuple(cover[100, 110]) == (0, 0, 255)     # l'image de Spotify, pas une pochette générée
    assert itunes_calls == []


def test_media_cover_falls_back_to_online_catalogue():
    from maestro.media_player import MediaKeysPlayer
    asked = []

    def fake_itunes(artist, title):
        asked.append((artist, title))
        return _png((0, 255, 0))

    p = MediaKeysPlayer(press=lambda k: None, read_titles=lambda: ["Daft Punk - One More Time"],
                        volume_api=None, session=None, artwork_loader=fake_itunes)
    cover = _wait_cover(p)
    assert asked == [("Daft Punk", "One More Time")] and tuple(cover[100, 110]) == (0, 255, 0)


def test_local_player_uses_real_artwork_only_for_named_files(tmp_path, monkeypatch):
    from maestro import artwork
    (tmp_path / "Daft Punk - One More Time.wav").write_bytes(b"")
    (tmp_path / "01 - Maestro Allegro.wav").write_bytes(b"")
    monkeypatch.setattr(artwork, "itunes_artwork", lambda a, t: _png((255, 0, 0)) if a == "Daft Punk" else None)
    p = MusicPlayer(music_dir=tmp_path)
    names = [t.stem for t in p.tracks]
    i_named, i_generic = names.index("Daft Punk - One More Time"), names.index("01 - Maestro Allegro")
    import time as _t
    for _ in range(200):
        if p.cover_of(i_named) is not None and p._artwork.ready(str(p.tracks[i_generic])):
            break
        p.cover_of(i_generic)
        _t.sleep(0.01)
    assert tuple(p.cover_of(i_named)[100, 110]) == (255, 0, 0)
    assert p.cover_of(i_generic) is None             # pas d'artiste : pochette générée


def test_clean_title_for_search():
    from maestro.artwork import _clean
    assert _clean("Blinding Lights (feat. X) - Remastered 2011") == "Blinding Lights"
