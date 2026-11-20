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
