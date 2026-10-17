import pandas as pd

from undercut import load

td = lambda s: pd.to_timedelta(s, unit="s")
FRAME = pd.DataFrame({
    "Driver": ["VER"] * 3, "DriverNumber": ["1"] * 3, "LapNumber": [1.0, 2.0, 3.0], "Position": [1.0, 1.0, 2.0],
    "Time": [td(90), td(181), td(300)], "LapTime": [td(90), td(91), td(119)],
    "PitInTime": [pd.NaT, td(180), pd.NaT], "PitOutTime": [pd.NaT, pd.NaT, td(205)],
    "TrackStatus": ["1", "1", "1"], "Compound": ["MEDIUM", "MEDIUM", "HARD"],
    "TyreLife": [1.0, 2.0, 1.0], "Stint": [1.0, 1.0, 2.0],
})
DRY_WEATHER = pd.DataFrame({"Time": [td(100), td(200)], "Rainfall": [False, False]})


def test_laps_table_converts_times_and_flags():
    t = load.laps_table(FRAME)
    assert t.end_s.tolist()[:2] == [90.0, 181.0]
    assert t.lap_time_s.tolist() == [90.0, 91.0, 119.0]
    assert t.pit_in.tolist() == [False, True, False]
    assert t.pit_out.tolist() == [False, False, True]
    assert t.DriverNumber.dtype.kind == "i" and t.LapNumber.dtype.kind == "i"


def test_only_track_status_1_is_green():
    t = load.laps_table(FRAME.assign(TrackStatus=["1", "12", "4"]))
    assert t.green.tolist() == [True, False, False]


def test_rows_without_lap_end_time_are_dropped():
    assert len(load.laps_table(FRAME.assign(Time=[FRAME.Time[0], pd.NaT, FRAME.Time[2]]))) == 2


def test_inter_tyre_makes_race_wet():
    t = load.laps_table(FRAME.assign(Compound=["MEDIUM", "INTERMEDIATE", "INTERMEDIATE"]))
    assert load.is_wet(t, DRY_WEATHER)


def test_rain_during_race_makes_race_wet():
    t = load.laps_table(FRAME)
    assert not load.is_wet(t, DRY_WEATHER)
    assert load.is_wet(t, DRY_WEATHER.assign(Rainfall=[False, True]))


def test_rain_after_the_race_is_not_wet():
    t = load.laps_table(FRAME)
    assert not load.is_wet(t, pd.DataFrame({"Time": [td(400)], "Rainfall": [True]}))
