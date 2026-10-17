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
    """Adds degradation per 0.01 s/lap of tyre age (about the spread between circuits; per 1 s/lap the odds ratio is
    astronomically large because no circuit comes close to 1 s/lap)."""
    t = models.odds_table(res)
    r = t[t.term == "deg_s_per_lap"].iloc[0]
    extra = dict(term="deg_per_0.01s_per_lap", odds_ratio=r.odds_ratio ** 0.01, ci_lo=r.ci_lo ** 0.01,
                 ci_hi=r.ci_hi ** 0.01, p=r.p)
    return pd.concat([t, pd.DataFrame([extra])], ignore_index=True)


def lookup(res, d):
    med = {c: d[c].median() for c in ("tyre_age_diff", "pit_loss_s", "deg_s_per_lap")}
    common = d.compound_change.mode()[0]  # hold the compound choice at the most common one
    grid = pd.DataFrame([dict(gap_s=g, delay_cat=k, compound_change=common, **med)
                         for k in models.DELAY_LEVELS for g in np.arange(0.5, 3.01, 0.5)])
    grid["delay_cat"] = pd.Categorical(grid.delay_cat, categories=models.DELAY_LEVELS)
    grid["compound_change"] = pd.Categorical(grid.compound_change, categories=models.CHANGE_LEVELS)
    grid["p"] = np.asarray(res.predict(grid))
    return grid[["gap_s", "delay_cat", "compound_change", "p"]]


MECH = ["mech_swing", "mech_stop_diff", "mech_tyre_phase"]


def mechanism_table(d):
    """Means (which add up: swing = stop_diff + tyre_phase) and medians by outcome, all over the same rows: the
    attempts where both stops' time loss could be measured."""
    k = d.dropna(subset=MECH)
    cols = MECH + ["mech_traffic", "gap_s", "gain_s", "margin_s"]
    t = k.groupby("success")[cols].mean()
    for c in MECH:
        t[f"median_{c}"] = k.groupby("success")[c].median()
    t.insert(0, "n", k.groupby("success").size())
    return t.reset_index()


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
    mechanism_table(d).to_csv("results/mechanism.csv", index=False)
    nr = raw[~raw.responded]
    pd.DataFrame([dict(n=len(nr), races=nr.race_id.nunique(), n_responded=len(d))]).to_csv(
        "results/no_response.csv", index=False)
    print(models.odds_table(res).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
