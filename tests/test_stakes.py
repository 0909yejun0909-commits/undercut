import numpy as np
import pandas as pd

from tests.helpers import make_laps
from undercut import stakes


def results(rows):
    return pd.DataFrame(rows, columns=["DriverNumber", "FinishPosition", "Status"])


def attempt(chaser=2, leader=1):
    return pd.DataFrame([dict(race_id="r", lap=10, chaser=chaser, leader=leader)])


def laps_with_position(pos):
    laps = make_laps({1: dict(offset=0, laps=15, base=90.0), 2: dict(offset=2, laps=15, base=90.0)})
    laps.loc[(laps.DriverNumber == 2) & (laps.LapNumber == 9), "Position"] = pos
    return {"r": laps}


def test_points_follow_the_top_ten_scale_and_unclassified_scores_zero():
    assert [stakes.points(p) for p in (1, 2, 10, 11)] == [25, 18, 1, 0]
    assert stakes.points(np.nan) == 0


def test_chaser_ahead_at_flag_with_points_swing_and_places_gained():
    res = {"r": results([(1, 5, "Finished"), (2, 4, "Finished")])}
    d = stakes.add_stakes(attempt(), laps_with_position(6.0), res).iloc[0]
    assert d.ahead_at_flag == 1 and d.pair_points == 12 - 10 and d.places_gained == 2
    assert d.position_band == "P4-6" and d.front == 1 and d.points_fight


def test_disqualified_car_is_behind_and_scores_nothing():
    res = {"r": results([(1, 3, "Disqualified"), (2, 8, "Finished")])}
    d = stakes.add_stakes(attempt(), laps_with_position(12.0), res).iloc[0]
    assert d.ahead_at_flag == 1 and d.pair_points == 4
    assert d.front == 0 and not d.points_fight


def test_unclassified_chaser_is_never_ahead():
    res = {"r": results([(1, 15, "Lapped"), (2, np.nan, "Did not start")])}
    assert stakes.add_stakes(attempt(), laps_with_position(14.0), res).iloc[0].ahead_at_flag == 0


def test_small_circuits_are_pooled():
    c = pd.Series(["Bahrain", "Monaco", "Bahrain"])
    assert stakes.pool_circuits(c, {"Bahrain"}).tolist() == ["Bahrain", "other", "Bahrain"]


def test_shuffle_keeps_each_race_on_one_circuit_and_the_circuit_mix():
    d = pd.DataFrame(dict(race_id=["a", "a", "b", "c", "c", "c"], circuit=["X", "X", "Y", "Z", "Z", "Z"]))
    s = stakes.shuffle_circuits(d, np.random.default_rng(1))
    assert (s.groupby("race_id").circuit.nunique() == 1).all()
    assert sorted(s.groupby("race_id").circuit.first()) == ["X", "Y", "Z"]
