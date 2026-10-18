import numpy as np
import pandas as pd
import pytest

from tests.helpers import make_laps
from undercut import cars


def test_team_key_merges_renamed_teams():
    assert {cars.team_key(n) for n in ("AlphaTauri", "RB", "Racing Bulls")} == {"Racing Bulls"}
    assert {cars.team_key(n) for n in ("Alfa Romeo", "Kick Sauber", "Audi")} == {"Sauber/Audi"}
    assert cars.team_key("Haas F1 Team") == "Haas" and cars.team_key("McLaren") == "McLaren"


def test_race_pace_is_median_clean_lap_and_skips_short_races():
    laps = make_laps({1: dict(offset=0, laps=20, base=90.0, pits={10: "HARD"}, slow={5: 30.0}),
                      2: dict(offset=1, laps=20, base=91.0),
                      3: dict(offset=2, laps=5, base=89.0)})
    p = cars.race_pace(laps, min_laps=10)
    assert p[1] == pytest.approx(90.0) and p[2] == pytest.approx(91.0)
    assert 3 not in p.index


def test_car_deg_is_that_cars_own_slope():
    laps = make_laps({1: dict(offset=0, laps=20, base=90.0, deg=0.10), 2: dict(offset=1, laps=20, base=90.0, deg=0.02)})
    d = cars.car_deg(laps)
    assert d[1] == pytest.approx(0.10) and d[2] == pytest.approx(0.02)


def test_stop_losses_are_relative_to_the_race_median():
    laps = make_laps({1: dict(offset=0, laps=20, base=90.0, pits={8: "HARD"}, slow={8: 5.0, 9: 15.0}),
                      2: dict(offset=1, laps=20, base=90.0, pits={9: "HARD"}, slow={9: 6.0, 10: 16.0}),
                      3: dict(offset=2, laps=20, base=90.0, pits={10: "HARD"}, slow={10: 7.0, 11: 17.0})})
    s = cars.stop_losses(laps).set_index("DriverNumber")
    assert s.loss_rel.to_dict() == pytest.approx({1: -2.0, 2: 0.0, 3: 2.0})


def test_car_table_is_relative_to_the_race_median():
    laps = make_laps({1: dict(offset=0, laps=20, base=90.0, deg=0.02), 2: dict(offset=1, laps=20, base=90.5, deg=0.05),
                      3: dict(offset=2, laps=20, base=91.0, deg=0.08)})
    t = cars.car_table(laps).set_index("DriverNumber")
    assert t.pace_rel.to_dict() == pytest.approx({1: -0.5 - 0.03 * 11, 2: 0.0, 3: 0.5 + 0.03 * 11})
    assert t.deg_rel.to_dict() == pytest.approx({1: -0.03, 2: 0.0, 3: 0.03})


def test_season_median_leaves_the_race_out():
    stops = pd.DataFrame({"race_id": ["2024_01", "2024_02", "2024_03", "2024_01", "2023_05"],
                          "team": ["A", "A", "A", "B", "A"], "loss_rel": [1.0, 3.0, 5.0, -1.0, 99.0]})
    assert cars.season_median(stops, "A", "2024_02", "loss_rel") == pytest.approx(3.0)  # median of 1 and 5
    assert cars.season_median(stops, "A", "2024_01", "loss_rel") == pytest.approx(4.0)  # 2023 never used
    assert np.isnan(cars.season_median(stops, "B", "2024_01", "loss_rel"))


def test_add_car_features_signs_favour_the_chaser():
    att = pd.DataFrame({"race_id": ["2024_02"], "chaser": [1], "leader": [2]})
    teams = pd.DataFrame({"race_id": ["2024_02"] * 2, "DriverNumber": [1, 2], "TeamName": ["McLaren", "Haas F1 Team"]})
    car_tbl = pd.DataFrame({"race_id": ["2024_01", "2024_01", "2024_02", "2024_02"], "team": ["McLaren", "Haas"] * 2,
                            "pace_rel": [-0.3, 0.2, 9.0, 9.0], "deg_rel": [-0.01, 0.02, 9.0, 9.0]})
    stops = pd.DataFrame({"race_id": ["2024_01", "2024_01"], "team": ["McLaren", "Haas"], "loss_rel": [-0.5, 1.0]})
    f = cars.add_car_features(att, teams, car_tbl, stops).iloc[0]
    assert (f.chaser_team, f.leader_team) == ("McLaren", "Haas")
    assert f.pace_adv_s == pytest.approx(0.5)   # chaser's car 0.5 s a lap faster in the season's other races
    assert f.deg_adv_s == pytest.approx(0.03)   # leader's tyres wear 0.03 s/lap faster
    assert f.crew_adv_s == pytest.approx(1.5)   # leader's crew loses 1.5 s more per stop


def test_pre_stop_pace_uses_only_laps_before_the_attempt():
    att = pd.DataFrame({"race_id": ["r"], "chaser": [1], "leader": [2], "lap": [12]})
    # chaser 0.4 s a lap faster up to lap 10, then 5 s a lap slower: only laps 2..10 (before lap n-1 = 11) count
    laps = make_laps({1: dict(offset=0, laps=20, base=90.0, slow={k: 5.4 for k in range(11, 21)}),
                      2: dict(offset=1, laps=20, base=90.4)})
    assert cars.pre_stop_pace_adv(att, {"r": laps}, min_laps=5).iloc[0] == pytest.approx(0.4)
    assert np.isnan(cars.pre_stop_pace_adv(att, {"r": laps}, min_laps=10).iloc[0])
