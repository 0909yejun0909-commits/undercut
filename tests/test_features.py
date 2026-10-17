import numpy as np
import pytest

from tests.helpers import make_laps
from tests.test_attempts import INFO, PAIR
from undercut import attempts, features

CAR3 = dict(offset=30.0, laps=20, base=91.0, fresh=88.0, pits={8: "HARD"}, slow={8: 11.0, 9: 9.0})
LAPS_WITH_CAR3 = make_laps({**PAIR(1.0), 3: CAR3})
ATTEMPT_ROW = attempts.find_attempts(make_laps(PAIR(1.0)), INFO)


def test_fresh_tyre_gain_uses_other_cars_only():
    assert features.fresh_tyre_gain(LAPS_WITH_CAR3, around_lap=10, exclude={1, 2}) == pytest.approx(3.0)


def test_fresh_tyre_gain_ignores_stops_outside_window():
    assert np.isnan(features.fresh_tyre_gain(LAPS_WITH_CAR3, around_lap=19, exclude={1, 2}, window=10))


def test_pit_loss_is_in_plus_out_lap_over_normal_pace():
    assert features.pit_loss(LAPS_WITH_CAR3, exclude={1, 2}) == pytest.approx(20.0)


def test_deg_rate_recovers_slope():
    laps = make_laps({3: dict(offset=0.0, laps=15, base=90.0, deg=0.1, compound="HARD")})
    assert features.deg_rate(laps, exclude=set()) == pytest.approx(0.1, abs=1e-6)


def test_add_features_margin_rule_and_categories():
    f = features.add_features(ATTEMPT_ROW.assign(delay=5.0, gap_s=1.0), {"2024_05": LAPS_WITH_CAR3})
    r = f.iloc[0]
    assert r.gain_s == pytest.approx(3.0) and r.pit_loss_s == pytest.approx(20.0)
    assert r.margin_s == pytest.approx(3.0 * 5 - 1.0) and r.rule_pred == 1.0
    assert (r.delay_cat, r.compound_change, r.tyre_age_diff, r.y2026) == ("4-5", "harder", 0, 0)


def test_rule_predicts_failure_when_gap_beats_gain():
    r = features.add_features(ATTEMPT_ROW.assign(gap_s=2.9), {"2024_05": LAPS_WITH_CAR3}).iloc[0]
    assert r.margin_s == pytest.approx(0.1) and r.rule_pred == 1.0
    r = features.add_features(ATTEMPT_ROW.assign(gap_s=3.5), {"2024_05": LAPS_WITH_CAR3}).iloc[0]
    assert r.margin_s == pytest.approx(-0.5) and r.rule_pred == 0.0


def test_mechanism_columns():
    r = features.add_features(ATTEMPT_ROW, {"2024_05": LAPS_WITH_CAR3}).iloc[0]
    assert r.mech_inlap_loss == pytest.approx(11.0)
    assert r.mech_outlap_gain == pytest.approx(4.0)  # leader in-lap 101 − chaser out-lap 97
    assert r.mech_traffic == 0


def test_traffic_when_another_car_is_just_ahead_after_the_out_lap():
    laps = make_laps({**PAIR(1.0), 3: dict(offset=18.0, laps=20, base=90.0)})  # car 3 ends lap 11 at 1008, chaser at 1009
    a = attempts.find_attempts(laps, INFO)
    r = features.add_features(a[(a.chaser == 2) & (a.leader == 1)], {"2024_05": laps}).iloc[0]
    assert r.mech_traffic == 1


def test_no_other_stops_gives_nan_gain_and_rule():
    f = features.add_features(ATTEMPT_ROW, {"2024_05": make_laps(PAIR(1.0))})
    assert np.isnan(f.gain_s.iloc[0]) and np.isnan(f.margin_s.iloc[0]) and np.isnan(f.rule_pred.iloc[0])
