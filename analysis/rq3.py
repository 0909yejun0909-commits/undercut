"""RQ3 (exploratory, small 2026 sample): did the 2026 rules change the undercut?"""
import numpy as np
import pandas as pd

from undercut import models

NOTE = "# exploratory: 2026 R1-17 sample n={n}\n"


def write(df, path, n):
    with open(path, "w") as f:
        f.write(NOTE.format(n=n))
        df.to_csv(f, index=False)


def main():
    d = models.prepare(pd.read_parquet("data/attempts_main.parquet"))
    d["is2026"] = (d.year == 2026).astype(int)
    old, new = d[d.year < 2026], d[d.year == 2026]
    diff = lambda s: s[s.is2026 == 1].success.mean() - s[s.is2026 == 0].success.mean()
    boot = models.bootstrap_by_race(d, diff)
    lo, hi = np.nanpercentile(boot, [2.5, 97.5])
    write(pd.DataFrame([dict(rate_2026=new.success.mean(), n_2026=len(new), races_2026=new.race_id.nunique(),
                             rate_2023_25=old.success.mean(), n_2023_25=len(old), diff=diff(d), ci_lo=lo, ci_hi=hi)]),
          "results/rq3_rate.csv", len(new))
    write(models.odds_table(models.fit_logit(d, models.FORMULA + " + y2026")), "results/rq3_odds.csv", len(new))
    p = models.season_out(d, {"logit": models.MODELS["logit"]}, [2023, 2024, 2025], [2026])
    write(pd.DataFrame([dict(mean_predicted=p.logit.mean(), observed=p.y.mean(), n=len(p))]),
          "results/rq3_calibration.csv", len(new))
    # same-rule comparison of the levers: delay and gap in 2026 vs before
    write(d.groupby(["is2026", "delay_cat"], observed=True).success.agg(["size", "mean"]).reset_index(),
          "results/rq3_by_delay.csv", len(new))


if __name__ == "__main__":
    main()
