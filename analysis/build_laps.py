"""Cached FastF1 races → data/laps/{year}_{rnd}_{laps,results}.parquet + _info.json. Offline only."""
import json
import os

from undercut import load


def build(root=load.LAPS):
    os.makedirs(root, exist_ok=True)
    written = 0
    for year, n in load.CALENDAR.items():
        for rnd in range(1, n + 1):
            stem = f"{root}/{load.race_id(year, rnd)}"
            if all(os.path.exists(f"{stem}{x}") for x in ("_laps.parquet", "_results.parquet", "_info.json")):
                written += 1
                continue
            try:
                s = load.get_race(year, rnd)
                laps = load.laps_table(s.laps)
                info = {"Year": year, "Round": rnd, "Event": s.event["EventName"], "race_id": load.race_id(year, rnd),
                        "wet": load.is_wet(laps, s.weather_data), "total_laps": int(laps.LapNumber.max())}
            except Exception as e:  # a race missing from the cache; report and keep going
                print(f"FAILED {year} {rnd}: {e}", flush=True)
                continue
            laps.to_parquet(f"{stem}_laps.parquet")
            load.results_table(s.results).to_parquet(f"{stem}_results.parquet")
            json.dump(info, open(f"{stem}_info.json", "w"), indent=1)
            written += 1
            print(f"{year} R{rnd} {info['Event']}: {len(laps)} laps, wet={info['wet']}", flush=True)
    print(f"{written} races")


if __name__ == "__main__":
    build()
