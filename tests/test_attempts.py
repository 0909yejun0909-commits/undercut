import numpy as np
import pytest

from tests.helpers import make_laps
from undercut import attempts

INFO = {"Year": 2024, "Round": 5, "race_id": "2024_05", "wet": False}


def PAIR(offset2, leader_pit=11, chaser_penalty=None, leader_laps=20, third=None):
    """Car 1 leads; car 2 runs offset2 s behind and pits on lap 10 (in-lap +11 s, out-lap +9 s); fresh tyres are
    2 s a lap faster. Car 1 ends lap 13 at 1186 (900 + 101 + 97 + 88); car 2 at 1184 + offset2."""
    cars = {
        1: dict(offset=0.0, laps=leader_laps, base=90.0, fresh=88.0, pits={leader_pit: "HARD"},
                slow={leader_pit: 11.0, leader_pit + 1: 9.0}),
        2: dict(offset=offset2, laps=20, base=90.0, fresh=88.0, pits={10: "HARD"}, slow={10: 11.0, 11: 9.0}),
    }
    if chaser_penalty:
        cars[2] = dict(offset=offset2, laps=20, base=90.0, penalty={chaser_penalty})
    if third:
        cars[3] = dict(laps=20, base=90.0, **third)
    return cars


def test_undercut_that_works_has_gap_delay_and_success():
    a = attempts.find_attempts(make_laps(PAIR(1.0)), INFO)
    r = a.iloc[0]
    assert len(a) == 1 and (r.chaser, r.leader, r.lap) == (2, 1, 10)
    assert r.gap_s == pytest.approx(1.0) and r.delay == 1 and r.check_lap == 13 and r.success == 1
    assert (r.chaser_old, r.chaser_new, r.chaser_age, r.leader_age) == ("MEDIUM", "HARD", 9, 9)
    assert r.responded and (r.race_id, r.year, r.rnd, r.chaser_code, r.leader_code) == ("2024_05", 2024, 5, "D2", "D1")


def test_undercut_fails_when_gap_beats_tyre_gain():
    assert attempts.find_attempts(make_laps(PAIR(2.5)), INFO).success.tolist() == [0]


def test_gap_above_window_is_not_an_attempt():
    assert attempts.find_attempts(make_laps(PAIR(3.5)), INFO).empty


def test_gap_window_is_a_parameter():
    assert len(attempts.find_attempts(make_laps(PAIR(3.5)), INFO, window_s=5.0)) == 1


def test_leader_pitting_same_lap_is_not_an_attempt():
    assert attempts.find_attempts(make_laps(PAIR(1.0, leader_pit=10)), INFO).empty


def test_no_stop_within_five_laps_is_no_response():
    r = attempts.find_attempts(make_laps(PAIR(1.0, leader_pit=16)), INFO).iloc[0]
    assert not r.responded and np.isnan(r.success) and r.chaser_ahead_end == 1


def test_penalty_drive_through_is_not_a_stop():
    laps = make_laps(PAIR(1.0, chaser_penalty=10))
    assert attempts.stops(laps).query("DriverNumber == 2").empty
    assert attempts.find_attempts(laps, INFO).empty


def test_safety_car_in_window_excludes_attempt():
    assert attempts.find_attempts(make_laps(PAIR(1.0), green_off={(1, 12), (2, 12)}), INFO).empty


def test_relaxed_green_allows_safety_car_after_stop_window():
    laps = make_laps(PAIR(1.0), green_off={(1, 13), (2, 13)})
    assert attempts.find_attempts(laps, INFO).empty
    assert len(attempts.find_attempts(laps, INFO, require_green=False)) == 1


def test_missing_lap_row_in_window_excludes_attempt():
    assert attempts.find_attempts(make_laps(PAIR(1.0), drop={(1, 12)}), INFO).empty


def test_retirement_before_check_lap_excludes_attempt():
    assert attempts.find_attempts(make_laps(PAIR(1.0, leader_laps=12)), INFO).empty


def test_leader_is_the_car_directly_ahead():
    a = attempts.find_attempts(make_laps(PAIR(1.0, third={"offset": -0.5})), INFO)  # car 3 is 0.5 s ahead of car 1
    assert a[a.responded].leader.tolist() == [1]


def test_check_offset_two_checks_one_lap_later():
    assert attempts.find_attempts(make_laps(PAIR(1.0)), INFO, check_offset=2).check_lap.tolist() == [14]


def test_wet_race_returns_empty_frame_with_columns():
    a = attempts.find_attempts(make_laps(PAIR(1.0)), {**INFO, "wet": True})
    assert a.empty and list(a.columns) == attempts.COLUMNS
