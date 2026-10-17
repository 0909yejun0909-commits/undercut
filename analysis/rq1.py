"""RQ1: how often does the undercut work, and what decides it? 2023-25 main sample."""
import numpy as np
import pandas as pd

from analysis.build_attempts import VARIANTS
from undercut import models

YEARS = [2023, 2024, 2025]
GAP_BINS = [0, 1, 2, 3.0001]
GAP_LABELS = ["0-1 s", "1-2 s", "2-3 s"]


def load(variant="main", years=YEARS):
    a = pd.read_parquet(f"data/attempts_{variant}.parquet")
    return a[a.year.isin(years)]


def rate_ci(g, n=2000):
    boot = models.bootstrap_by_race(g, lambda s: s.success.mean(), n=n)
    lo, hi = np.nanpercentile(boot, [2.5, 97.5])
    return dict(n=len(g), rate=g.success.mean(), ci_lo=lo, ci_hi=hi)


def rates(d):
    rows = [dict(group="overall", level="all", **rate_ci(d))]
    for lvl, g in d.groupby("delay_cat", observed=True):
        rows.append(dict(group="delay", level=lvl, **rate_ci(g)))
    for lvl, g in d.groupby(pd.cut(d.gap_s, GAP_BINS, right=False, labels=GAP_LABELS), observed=True):
        rows.append(dict(group="gap", level=str(lvl), **rate_ci(g)))
    for lvl, g in d.groupby("compound_change", observed=True):
        rows.append(dict(group="compound_change", level=lvl, **rate_ci(g)))
    return pd.DataFrame(rows)


def odds_with_scaled(res):
    """Adds degradation per 0.1 s/lap (a realistic step; per 1 s/lap the odds ratio is astronomically large)."""
    t = models.odds_table(res)
    r = t[t.term == "deg_s_per_lap"].iloc[0]
    extra = dict(term="deg_per_0.1s_per_lap", odds_ratio=r.odds_ratio ** 0.1, ci_lo=r.ci_lo ** 0.1,
                 ci_hi=r.ci_hi ** 0.1, p=r.p)
    return pd.concat([t, pd.DataFrame([extra])], ignore_index=True)


def lookup(res, d):
    med = {c: d[c].median() for c in ("tyre_age_diff", "pit_loss_s", "deg_s_per_lap")}
    grid = pd.DataFrame([dict(gap_s=g, delay_cat=k, compound_change="same", **med)
                         for k in models.DELAY_LEVELS for g in np.arange(0.5, 3.01, 0.5)])
    grid["delay_cat"] = pd.Categorical(grid.delay_cat, categories=models.DELAY_LEVELS)
    grid["compound_change"] = pd.Categorical(grid.compound_change, categories=models.CHANGE_LEVELS)
    grid["p"] = np.asarray(res.predict(grid))
    return grid[["gap_s", "delay_cat", "p"]]


def sensitivity():
    rows = []
    for v in VARIANTS:
        d = models.prepare(load(v))
        t = models.odds_table(models.fit_logit(d)).set_index("term")
        rows.append(dict(variant=v, n=len(d), rate=d.success.mean(), or_gap=t.at["gap_s", "odds_ratio"],
                         or_gap_lo=t.at["gap_s", "ci_lo"], or_gap_hi=t.at["gap_s", "ci_hi"],
                         or_delay45=t.at["delay_cat[T.4-5]", "odds_ratio"]))
    return pd.DataFrame(rows)


def main():
    raw = load()
    d = models.prepare(raw)
    print(f"RQ1 sample: {len(d)} attempts ({d.attrs['dropped']} dropped for missing features), "
          f"{d.race_id.nunique()} races", flush=True)
    rates(d).to_csv("results/rq1_rates.csv", index=False)
    res = models.fit_logit(d)
    odds_with_scaled(res).to_csv("results/rq1_odds.csv", index=False)
    lookup(res, d).to_csv("results/rq1_lookup.csv", index=False)
    sensitivity().to_csv("results/rq1_sensitivity.csv", index=False)
    mech = d.groupby("success")[["mech_swing", "mech_stop_diff", "mech_tyre_phase", "mech_traffic", "gap_s",
                                  "gain_s", "margin_s"]].mean()
    mech.insert(0, "n", d.groupby("success").size())
    mech.reset_index().to_csv("results/mechanism.csv", index=False)
    nr = raw[~raw.responded]
    pd.DataFrame([dict(n=len(nr), races=nr.race_id.nunique(), n_responded=len(d))]).to_csv(
        "results/no_response.csv", index=False)
    print(models.odds_table(res).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
