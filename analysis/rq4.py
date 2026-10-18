"""RQ4 (exploratory, added after the pre-registration): which teams and cars close the gap? 2023-25 main sample."""
import glob
import json

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

from analysis.rq1 import YEARS, rate_ci
from undercut import cars, models

NOTE = "# exploratory: added after pre-registration, 2023-25 n={n}\n"
TRAITS = ["pace_adv_s", "deg_adv_s", "crew_adv_s"]
FORMULA = "success ~ gap_s + delay_cat + " + " + ".join(TRAITS)
# odds ratios per a readable step: 0.1 s/lap of pace, 0.01 s/lap of tyre wear, 1 s of stop loss
STEP = {"pace_adv_s": 0.1, "deg_adv_s": 0.01, "crew_adv_s": 1.0}


def write(df, path, n):
    with open(path, "w") as f:
        f.write(NOTE.format(n=n))
        df.to_csv(f, index=False)


def load_all():
    """Per-car tables for every dry race, tagged with its team."""
    car_tbl, stops, laps_by_race = [], [], {}
    for f in sorted(glob.glob("data/laps/*_info.json")):
        info = json.load(open(f))
        if info["wet"]:
            continue
        rid = info["race_id"]
        laps = pd.read_parquet(f.replace("_info.json", "_laps.parquet"))
        laps_by_race[rid] = laps
        car_tbl.append(cars.car_table(laps).assign(race_id=rid))
        stops.append(cars.stop_losses(laps).assign(race_id=rid))
    teams = pd.read_parquet("data/teams.parquet")
    tm = {(r.race_id, r.DriverNumber): cars.team_key(r.TeamName) for r in teams.itertuples(index=False)}
    out = []
    for t in (pd.concat(car_tbl, ignore_index=True), pd.concat(stops, ignore_index=True)):
        t["team"] = [tm.get(k) for k in zip(t.race_id, t.DriverNumber)]
        out.append(t)
    return teams, out[0], out[1], laps_by_race


def by_team(d):
    rows = []
    for role, col in (("chaser", "chaser_team"), ("leader", "leader_team")):
        for team, g in d.groupby(col):
            rows.append(dict(role=role, team=team, **rate_ci(g), mean_swing_s=g.mech_swing.mean(),
                             median_swing_s=g.mech_swing.median()))
    return pd.DataFrame(rows).sort_values(["role", "rate"], ascending=[True, False])


def splits(d):
    """Success rate when each trait favours the chaser (> 0) and when it does not."""
    rows = []
    for t in TRAITS:
        for side, g in (("favours chaser", d[d[t] > 0]), ("favours leader", d[d[t] <= 0])):
            rows.append(dict(trait=t, side=side, **rate_ci(g)))
    return pd.DataFrame(rows)


def scaled(t):
    t = t.copy()
    for term, step in STEP.items():
        i = t.term == term
        t.loc[i, ["odds_ratio", "ci_lo", "ci_hi"]] = t.loc[i, ["odds_ratio", "ci_lo", "ci_hi"]] ** step
        t.loc[i, "term"] = f"{term} per {step:g}"
    return t


def swing_ols(d):
    res = smf.ols("mech_swing ~ gap_s + delay_cat + " + " + ".join(TRAITS), data=d).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(d.race_id)[0]})
    ci = res.conf_int()
    return pd.DataFrame({"term": res.params.index, "coef_s": res.params.to_numpy(), "ci_lo": ci[0].to_numpy(),
                         "ci_hi": ci[1].to_numpy(), "p": res.pvalues.to_numpy()})


def team_tests(d):
    """Likelihood-ratio test: does adding the chaser's (or leader's) team improve a gap + delay model?"""
    base = models.fit_logit(d, "success ~ gap_s + delay_cat", cluster=False)
    rows = []
    for col in ("chaser_team", "leader_team"):
        res = models.fit_logit(d, f"success ~ gap_s + delay_cat + C({col})", cluster=False)
        lr, df = 2 * (res.llf - base.llf), res.df_model - base.df_model
        rows.append(dict(added=col, lr_stat=lr, df=df, p=stats.chi2.sf(lr, df)))
    return pd.DataFrame(rows)


def main():
    teams, car_tbl, stops, laps_by_race = load_all()
    a = pd.read_parquet("data/attempts_main.parquet")
    d = models.prepare(a[a.year.isin(YEARS)])
    d = cars.add_car_features(d, teams, car_tbl, stops)
    full = d.dropna(subset=TRAITS)
    n = len(full)
    print(f"{len(d)} attempts, {n} with all three traits; dropped {len(d) - n}")
    write(by_team(d), "results/rq4_teams.csv", len(d))
    write(splits(full), "results/rq4_splits.csv", n)
    write(scaled(models.odds_table(models.fit_logit(full, FORMULA))), "results/rq4_odds.csv", n)
    write(swing_ols(full.dropna(subset=["mech_swing"])), "results/rq4_swing.csv", n)
    write(team_tests(d), "results/rq4_team_test.csv", len(d))
    d["pre_pace_adv_s"] = cars.pre_stop_pace_adv(d, laps_by_race)
    pre = d.dropna(subset=["pre_pace_adv_s"])
    t = models.odds_table(models.fit_logit(pre, "success ~ gap_s + delay_cat + pre_pace_adv_s"))
    i = t.term == "pre_pace_adv_s"
    t.loc[i, ["odds_ratio", "ci_lo", "ci_hi"]] = t.loc[i, ["odds_ratio", "ci_lo", "ci_hi"]] ** 0.1
    t.loc[i, "term"] = "pre_pace_adv_s per 0.1"
    write(t, "results/rq4_prepace.csv", len(pre))
    rows = [dict(side=side, **rate_ci(g)) for side, g in (("chaser faster", pre[pre.pre_pace_adv_s > 0]),
                                                           ("chaser not faster", pre[pre.pre_pace_adv_s <= 0]))]
    write(pd.DataFrame(rows), "results/rq4_prepace_split.csv", len(pre))
    # check on the crew measure: does a slower crew (season-level) go with a costlier stop in the attempt itself?
    k = full.dropna(subset=["mech_stop_diff"])
    write(pd.DataFrame([dict(n=len(k), corr_crew_vs_stop_diff=k.crew_adv_s.corr(k.mech_stop_diff),
                             corr_pace_vs_gap=full.pace_adv_s.corr(full.gap_s))]),
          "results/rq4_checks.csv", n)


if __name__ == "__main__":
    main()
