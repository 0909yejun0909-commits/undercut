"""Cached FastF1 weather → data/weather.parquet (race_id, t_s, TrackTemp, AirTemp). Offline only."""
import glob
import json
import logging

import fastf1
import pandas as pd

from undercut import load

logging.getLogger("fastf1").setLevel(logging.ERROR)


def build(root=load.LAPS):
    fastf1.Cache.enable_cache(load.CACHE)
    fastf1.Cache.offline_mode(True)
    rows = []
    for f in sorted(glob.glob(f"{root}/*_info.json")):
        info = json.load(open(f))
        s = fastf1.get_session(info["Year"], info["Round"], "R")
        s.load(laps=False, telemetry=False, weather=True, messages=False)
        w = s.weather_data
        rows.append(pd.DataFrame({"race_id": info["race_id"], "t_s": w["Time"].dt.total_seconds().to_numpy(),
                                  "TrackTemp": w["TrackTemp"].astype(float).to_numpy(),
                                  "AirTemp": w["AirTemp"].astype(float).to_numpy()}))
    out = pd.concat(rows, ignore_index=True)
    out.to_parquet("data/weather.parquet")
    print(f"{out.race_id.nunique()} races, {len(out)} readings")


if __name__ == "__main__":
    build()
