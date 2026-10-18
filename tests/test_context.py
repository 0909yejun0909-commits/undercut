import numpy as np
import pandas as pd
import pytest

from tests.helpers import make_laps
from undercut import context

ATT = pd.Series(dict(race_id="r", lap=10, chaser=2, leader=1))


def test_rejoin_traffic_counts_cars_one_pit_stop_behind():
    # end of lap 9: leader 810, chaser 812, car 3 is 21 s behind the chaser and 23 s behind the leader
    laps = make_laps({1: dict(offset=0, laps=15, base=90.0), 2: dict(offset=2, laps=15, base=90.0),
                      3: dict(offset=23, laps=15, base=90.0), 4: dict(offset=60, laps=15, base=90.0)})
    c = context.attempt_context(laps, ATT, pit_loss_s=20.0)
    assert c["chaser_traffic"] == 1 and c["leader_traffic"] == 0


def test_closing_rate_is_gap_change_per_lap_over_three_laps():
    laps = make_laps({1: dict(offset=0, laps=15, base=90.0), 2: dict(offset=2, laps=15, base=89.9)})
    assert context.attempt_context(laps, ATT, 20.0)["closing_s_per_lap"] == pytest.approx(0.1)


def test_leader_fade_is_the_slope_of_its_last_five_laps():
    laps = make_laps({1: dict(offset=0, laps=15, base=90.0, deg=0.2), 2: dict(offset=2, laps=15, base=90.0)})
    assert context.attempt_context(laps, ATT, 20.0)["leader_fade_s_per_lap"] == pytest.approx(0.2)


def test_leader_blocked_when_within_a_second_of_the_car_ahead():
    free = make_laps({1: dict(offset=0, laps=15, base=90.0), 2: dict(offset=2, laps=15, base=90.0),
                      3: dict(offset=-3, laps=15, base=90.0)})
    stuck = make_laps({1: dict(offset=0, laps=15, base=90.0), 2: dict(offset=2, laps=15, base=90.0),
                       3: dict(offset=-0.6, laps=15, base=90.0)})
    assert context.attempt_context(free, ATT, 20.0)["leader_blocked"] == 0
    assert context.attempt_context(stuck, ATT, 20.0)["leader_blocked"] == 1


def test_first_stop_and_position():
    laps = make_laps({1: dict(offset=0, laps=15, base=90.0), 2: dict(offset=2, laps=15, base=90.0, pits={4: "HARD"})})
    laps.loc[(laps.DriverNumber == 2) & (laps.LapNumber == 9), "Position"] = 7.0
    c = context.attempt_context(laps, ATT, 20.0)
    assert c["first_stop"] == 0 and c["position"] == 7.0


def test_missing_laps_give_nan_not_an_error():
    laps = make_laps({1: dict(offset=0, laps=15, base=90.0), 2: dict(offset=2, laps=15, base=89.9)},
                     green_off={(2, 6)})
    c = context.attempt_context(laps, ATT, 20.0)
    assert np.isnan(c["closing_s_per_lap"])


def test_track_temp_is_the_median_reading_over_the_attempt():
    w = pd.DataFrame({"t_s": [0, 100, 200, 300, 400], "TrackTemp": [20.0, 30.0, 32.0, 34.0, 50.0]})
    assert context.track_temp(w, 90, 310) == pytest.approx(32.0)
    assert context.track_temp(w, 410, 420) == pytest.approx(50.0)  # no reading inside: nearest one


def test_street_circuits():
    assert context.is_street("Monaco Grand Prix") and not context.is_street("British Grand Prix")
