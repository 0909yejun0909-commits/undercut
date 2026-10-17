"""data/laps → data/attempts_{variant}.parquet (attempts + features) and results/attempt_counts.csv."""
import glob
import json
import os

import pandas as pd

from undercut import attempts, features, load

VARIANTS = {
    "main": {},
    "gap2": {"window_s": 2.0},
    "gap5": {"window_s": 5.0},
    "check2": {"check_offset": 2},
    "relaxed": {"require_green": False},
}


def races(root=load.LAPS):
    for f in sorted(glob.glob(f"{root}/*_info.json")):
        info = json.load(open(f))
        yield info, pd.read_parquet(f.replace("_info.json", "_laps.parquet"))


def build():
    os.makedirs("results", exist_ok=True)
    data = list(races())
    laps_by_race = {info["race_id"]: laps for info, laps in data}
    counts = []
    for name, kw in VARIANTS.items():
        a = pd.concat([x for info, laps in data if not (x := attempts.find_attempts(laps, info, **kw)).empty], ignore_index=True)
        f = features.add_features(a, laps_by_race)
        f.to_parquet(f"data/attempts_{name}.parquet")
        for year, g in f.groupby("year"):
            r = g[g.responded]
            counts.append(dict(variant=name, year=year, n_responded=len(r), n_no_response=int((~g.responded).sum()),
                               success_rate=r.success.mean()))
        print(f"{name}: {int(f.responded.sum())} responded, {int((~f.responded).sum())} no response", flush=True)
    pd.DataFrame(counts).to_csv("results/attempt_counts.csv", index=False)


if __name__ == "__main__":
    build()
