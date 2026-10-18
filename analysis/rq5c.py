"""RQ5c (exploratory): do factors we had not considered improve prediction? Each factor is added on its own to the
gap-only (before the stop) and gap + delay (by response) models. Bar fixed before scoring: the held-out-season
Brier improvement's 95% CI must exclude zero AND the 2026 Brier must not get worse."""
import glob
import json

import pandas as pd

from analysis.rq2 import load
from analysis.rq5 import write
from undercut import cars, context, models, predict

FACTORS = ["track_temp_c", "chaser_traffic", "leader_traffic", "closing_s_per_lap", "leader_fade_s_per_lap",
           "leader_blocked", "position", "street", "teammates", "first_stop"]


def inputs():
    laps, events = {}, {}
    for f in sorted(glob.glob("data/laps/*_info.json")):
        info = json.load(open(f))
        laps[info["race_id"]] = pd.read_parquet(f.replace("_info.json", "_laps.parquet"))
        events[info["race_id"]] = info["Event"]
    w = pd.read_parquet("data/weather.parquet")
    weather = {k: g for k, g in w.groupby("race_id")}
    t = pd.read_parquet("data/teams.parquet")
    team_of = {(r.race_id, r.DriverNumber): cars.team_key(r.TeamName) for r in t.itertuples(index=False)}
    return laps, weather, events, team_of


def screen(d, factor):
    k = d.dropna(subset=[factor])
    old = k[k.year < 2026]
    t = models.odds_table(models.fit_logit(old, f"{predict.GAP_DELAY} + {factor}")).set_index("term").loc[factor]
    row = dict(factor=factor, n=len(old), n_2026=int((k.year == 2026).sum()), odds_ratio=t.odds_ratio,
               ci_lo=t.ci_lo, ci_hi=t.ci_hi, p=t.p)
    for label, base in (("pre", predict.GAP_ONLY), ("post", predict.GAP_DELAY)):
        m = {"base": models._glm(base), "plus": models._glm(f"{base} + {factor}")}
        loso, t26 = models.loso(k, m), models.season_out(k, m, [2023, 2024, 2025], [2026])
        diff, lo, hi = models.bootstrap_metric_diff(loso, "plus", "base", "brier")
        d26 = models.metrics(t26.y, t26.plus)["brier"] - models.metrics(t26.y, t26.base)["brier"]
        row.update({f"{label}_diff": diff, f"{label}_lo": lo, f"{label}_hi": hi, f"{label}_diff_2026": d26,
                    f"{label}_passes": bool(hi < 0 and d26 <= 0)})
    return row


def main():
    d = context.add_context(load(), *inputs())
    print(d[FACTORS].describe().T[["count", "mean", "min", "max"]].round(2))
    write(pd.DataFrame([screen(d, f) for f in FACTORS]), "results/rq5c_screen.csv")


if __name__ == "__main__":
    main()
