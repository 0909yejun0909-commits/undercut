"""Cached FastF1 race results → data/teams.parquet (race_id, DriverNumber, TeamName). Offline only."""
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
        s.load(laps=False, telemetry=False, weather=False, messages=False)
        r = s.results
        rows.append(pd.DataFrame({"race_id": info["race_id"], "DriverNumber": r["DriverNumber"].astype(int).to_numpy(),
                                  "TeamName": r["TeamName"].astype(str).to_numpy()}))
    out = pd.concat(rows, ignore_index=True)
    out.to_parquet("data/teams.parquet")
    print(f"{out.race_id.nunique()} races, {len(out)} rows")


if __name__ == "__main__":
    build()
