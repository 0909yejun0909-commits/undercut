"""FastF1 race sessions → one tidy lap table per race. Reads the Battery Battles cache in offline mode only: the
FastF1 archive has returned HTTP 401 since about 2026-10-15 (FastF1 issue #978), and no step here touches the
network. Times are session-relative seconds (`end_s` = lap-end crossing time), which is all lap-end timing needs."""
import json
import logging
import os

import fastf1
import pandas as pd

logging.getLogger("fastf1").setLevel(logging.ERROR)
CACHE = os.environ.get("UNDERCUT_FASTF1_CACHE", os.path.expanduser("~/Projects/battery-battles/data/cache"))
LAPS = "data/laps"
CALENDAR = {2023: 22, 2024: 24, 2025: 24, 2026: 17}
WET = {"INTERMEDIATE", "WET"}


def get_race(year, rnd):
    fastf1.Cache.enable_cache(CACHE)
    fastf1.Cache.offline_mode(True)
    s = fastf1.get_session(year, rnd, "R")
    s.load(laps=True, telemetry=False, weather=True, messages=False)
    return s


def laps_table(laps):
    k = laps.dropna(subset=["Time", "LapNumber"])
    out = pd.DataFrame({
        "Driver": k["Driver"].to_numpy(),
        "DriverNumber": k["DriverNumber"].astype(int).to_numpy(),
        "LapNumber": k["LapNumber"].astype(int).to_numpy(),
        "Position": k["Position"].astype(float).to_numpy(),
        "end_s": k["Time"].dt.total_seconds().to_numpy(),
        "lap_time_s": k["LapTime"].dt.total_seconds().to_numpy(),
        "pit_in": k["PitInTime"].notna().to_numpy(),
        "pit_out": k["PitOutTime"].notna().to_numpy(),
        "green": k["TrackStatus"].astype(str).eq("1").to_numpy(),
        "Compound": k["Compound"].astype(str).to_numpy(),
        "TyreLife": k["TyreLife"].astype(float).to_numpy(),
        "Stint": k["Stint"].astype(float).to_numpy(),
    })
    return out.sort_values(["DriverNumber", "LapNumber"]).reset_index(drop=True)


def results_table(results):
    return pd.DataFrame({
        "DriverNumber": results["DriverNumber"].astype(int).to_numpy(),
        "Driver": results["Abbreviation"].to_numpy(),
        "GridPosition": results["GridPosition"].astype(float).to_numpy(),
        "FinishPosition": results["Position"].astype(float).to_numpy(),
        "Status": results["Status"].astype(str).to_numpy(),
    })


def is_wet(laps_tbl, weather):
    if laps_tbl["Compound"].isin(WET).any():
        return True
    t = weather["Time"].dt.total_seconds()
    start = (laps_tbl["end_s"] - laps_tbl["lap_time_s"].fillna(0)).min()
    during = (t >= start) & (t <= laps_tbl["end_s"].max())
    return bool(weather.loc[during, "Rainfall"].astype(bool).any())


def race_id(year, rnd):
    return f"{year}_{rnd:02d}"


def load_race_files(year, rnd, root=LAPS):
    stem = f"{root}/{race_id(year, rnd)}"
    return {"laps": pd.read_parquet(f"{stem}_laps.parquet"), "results": pd.read_parquet(f"{stem}_results.parquet"),
            "info": json.load(open(f"{stem}_info.json"))}
