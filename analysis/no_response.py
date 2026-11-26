"""Exploratory, added after review (reviewer comment 1): what happened in the no-response cases, where the chaser
pitted but the leader did not stop within five laps? Each case is classed by what the leader did next, and its
outcome is checked the same way as a standard attempt (chaser ahead at the end of the lap after the leader's
out-lap) or, if the leader never stopped again, at the last lap both cars completed. Then the main gap result is
re-checked on a wider set that adds the late responses."""
import json

import numpy as np
import pandas as pd

from undercut import attempts, models

YEARS = [2023, 2024, 2025]


def classify(row, laps):
    E = laps.pivot(index="LapNumber", columns="DriverNumber", values="end_s")
    st = attempts.stops(laps)
    lead = sorted(st[st.DriverNumber == row.leader].lap)
    chase = sorted(st[st.DriverNumber == row.chaser].lap)
    n = row.lap
    later = [m for m in lead if m > n + attempts.MAX_DELAY]
    both = E[[row.chaser, row.leader]].dropna()
    out = dict(early_stop=n <= 3)
    if later:
        m = later[0]
        k = m + 2
        out.update(kind="late_response", delay=m - n,
                   chaser_extra_stop=any(n < c < m for c in chase))
        out["ahead"] = float(E.at[k, row.chaser] < E.at[k, row.leader]) if k in both.index else np.nan
        G = laps.pivot(index="LapNumber", columns="DriverNumber", values="green").eq(True)
        out["all_green"] = all(G.at[j, d] if j in G.index else False
                               for j in range(n - 1, k + 1) for d in (row.chaser, row.leader))
    else:
        out.update(kind="never_stopped", delay=np.nan, chaser_extra_stop=any(c > n for c in chase))
        last = both.index.max() if len(both) else None
        out["ahead"] = float(both.at[last, row.chaser] < both.at[last, row.leader]) if last is not None else np.nan
        out["reached_flag"] = last is not None and last >= E.index.max() - 1
    return out


def main():
    a = pd.read_parquet("data/attempts_main.parquet")
    nr = a[~a.responded].copy()
    laps = {r: pd.read_parquet(f"data/laps/{r}_laps.parquet") for r in nr.race_id.unique()}
    info = pd.DataFrame([classify(r, laps[r.race_id]) for r in nr.itertuples()], index=nr.index)
    nr = nr.drop(columns="delay").join(info)
    nr.to_parquet("data/no_response_classified.parquet")

    rows = []
    for (y, kind), g in nr.assign(era=np.where(nr.year < 2026, "2023-25", "2026")).groupby(["era", "kind"]):
        rows.append(dict(era=y, kind=kind, n=len(g), early_stop=int(g.early_stop.sum()),
                         chaser_extra_stop=int(g.chaser_extra_stop.sum()), ahead_rate=g.ahead.mean(),
                         median_delay=g.delay.median(), median_gap=g.gap_s.median()))
    summary = pd.DataFrame(rows)
    summary.to_csv("results/no_response_kinds.csv", index=False)
    print(summary.to_string())

    # Wider set: standard attempts plus late responses where the chaser made no extra stop in between (a true
    # one-for-one exchange of stops) and every lap green up to the check lap, 2023-25.
    std = models.prepare(a[a.year.isin(YEARS)])
    late = nr[(nr.kind == "late_response") & ~nr.chaser_extra_stop & ~nr.early_stop & nr.year.isin(YEARS)
              & nr.ahead.notna()].copy()
    print(f"late, clean one-for-one: {len(late)}; of those all-green to the check lap: {int(late.all_green.sum())}, "
          f"rate {late[late.all_green].ahead.mean():.3f} vs {late.ahead.mean():.3f}")
    late = late[late.all_green.astype(bool)]  # same green-flag rule as standard attempts
    late["success"] = late.ahead
    late["delay_bin"] = pd.cut(late.delay, [5, 10, 100], labels=["6-10", "11+"])
    by_delay = [dict(set="standard", delay=str(k), n=len(g), rate=g.success.mean())
                for k, g in std.groupby("delay_cat", observed=True)]
    by_delay += [dict(set="late", delay=str(k), n=len(g), rate=g.success.mean())
                 for k, g in late.groupby("delay_bin", observed=True)]
    pd.DataFrame(by_delay).to_csv("results/no_response_by_delay.csv", index=False)
    print(pd.DataFrame(by_delay).to_string())

    wide = pd.concat([std.assign(late=0), late.assign(late=1)], ignore_index=True)
    wide["success"] = wide.success.astype(float)
    fits = []
    for name, df, f in [("standard, gap only", std, "success ~ gap_s"),
                        ("standard + late, gap only", wide, "success ~ gap_s"),
                        ("standard + late, gap + late", wide, "success ~ gap_s + late"),
                        ("late only, gap only", late, "success ~ gap_s")]:
        t = models.odds_table(models.fit_logit(df, f))
        t = t[t.term != "Intercept"].assign(model=name, n=len(df))
        fits.append(t)
    fits = pd.concat(fits, ignore_index=True)
    fits.to_csv("results/no_response_odds.csv", index=False)
    print(fits.to_string())

    # Every stop from within 3 s, as a team sees it before stopping: standard attempts, clean late responses and
    # cases where the leader never stopped again (outcome at the last lap both cars completed), 2023-25.
    never = nr[(nr.kind == "never_stopped") & nr.year.isin(YEARS) & nr.ahead.notna()].assign(success=lambda x: x.ahead)
    stops = pd.concat([std.assign(kind="responded"), late.assign(kind="late_response"),
                       never.assign(kind="never_stopped")], ignore_index=True)[["race_id", "gap_s", "success", "kind"]]
    stops["success"] = stops.success.astype(float)
    stops["gap_band"] = pd.cut(stops.gap_s, [0, 1, 2, 3.0001], right=False, labels=["0-1 s", "1-2 s", "2-3 s"])
    rows = [dict(group="all", level="all", n=len(stops), rate=stops.success.mean())]
    rows += [dict(group="kind", level=k, n=len(g), rate=g.success.mean()) for k, g in stops.groupby("kind")]
    rows += [dict(group="gap", level=str(k), n=len(g), rate=g.success.mean())
             for k, g in stops.groupby("gap_band", observed=True)]
    t = models.odds_table(models.fit_logit(stops, "success ~ gap_s"))
    r = t[t.term == "gap_s"].iloc[0]
    rows.append(dict(group="gap_odds_ratio", level="per 1 s", n=len(stops), rate=r.odds_ratio, ci_lo=r.ci_lo,
                     ci_hi=r.ci_hi, p=r.p))
    pd.DataFrame(rows).to_csv("results/no_response_all_stops.csv", index=False)
    print(pd.DataFrame(rows).to_string())


if __name__ == "__main__":
    main()
