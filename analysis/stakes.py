"""Exploratory (added after pre-registration): what an undercut attempt is worth at the flag, and does it work
differently at the front of the field? 2023-25, main sample."""
import glob

import numpy as np
import pandas as pd

from analysis.rq1 import YEARS, rate_ci
from analysis.rq5 import write
from undercut import models, stakes


def load():
    laps, res = {}, {}
    for f in sorted(glob.glob("data/laps/*_laps.parquet")):
        rid = f.split("/")[-1][:7]
        laps[rid] = pd.read_parquet(f)
        res[rid] = pd.read_parquet(f.replace("_laps.parquet", "_results.parquet"))
    d = models.prepare(pd.read_parquet("data/attempts_main.parquet"))
    return stakes.add_stakes(d, laps, res)


def mean_ci(g, col, n=2000):
    boot = models.bootstrap_by_race(g, lambda s: s[col].mean(), n=n)
    lo, hi = np.nanpercentile(boot, [2.5, 97.5])
    return dict(n=len(g), mean=g[col].mean(), median=g[col].median(), ci_lo=lo, ci_hi=hi)


def at_flag(d):
    rows = []
    for ok, g in d.groupby("success"):
        rows.append(dict(success=int(ok), measure="ahead_at_flag", **mean_ci(g, "ahead_at_flag")))
        rows.append(dict(success=int(ok), measure="places_gained", **mean_ci(g, "places_gained")))
        rows.append(dict(success=int(ok), measure="pair_points_points_fight",
                         **mean_ci(g[g.points_fight], "pair_points")))
    pf = d[d.points_fight]
    stat = lambda s: s[s.success == 1].pair_points.mean() - s[s.success == 0].pair_points.mean()
    lo, hi = np.nanpercentile(models.bootstrap_by_race(pf, stat), [2.5, 97.5])
    rows.append(dict(success=-1, measure="pair_points_points_fight_diff", n=len(pf), mean=stat(pf), median=np.nan,
                     ci_lo=lo, ci_hi=hi))
    return pd.DataFrame(rows)


def by_position(d):
    rows = [dict(band=b, **rate_ci(g), median_gap_s=g.gap_s.median(), next_lap=(g.delay_cat == "1").mean())
            for b, g in d.groupby("position_band", observed=True)]
    for f, g in d.groupby("front"):
        rows.append(dict(band="top six" if f else "P7+", **rate_ci(g), median_gap_s=g.gap_s.median(),
                         next_lap=(g.delay_cat == "1").mean()))
    return pd.DataFrame(rows)


def main():
    d = load()
    old = d[d.year.isin(YEARS)]
    print(f"{len(old)} attempts 2023-25; missing position {old.position.isna().sum()}")
    write(at_flag(old), "results/stakes_flag.csv")
    write(by_position(old), "results/stakes_position.csv")
    write(models.odds_table(models.fit_logit(old, "success ~ gap_s + delay_cat + front")),
          "results/stakes_front.csv")


if __name__ == "__main__":
    main()
