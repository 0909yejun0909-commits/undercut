"""Exploratory, added after review (reviewer comment 4): how well does the race-level fresh-tyre estimate behind the
rule of thumb (gain per lap x response delay) match the time each chaser actually gained on the leader outside the
stops (the tyre phase of results/mechanism.csv)? 2023-25 standard attempts with both stops measured."""
import numpy as np
import pandas as pd

from undercut import models


def main():
    d = models.prepare(pd.read_parquet("data/attempts_main.parquet"))
    d = d[d.year < 2026].dropna(subset=["mech_tyre_phase"])
    est, real = d.gain_s * d.delay, d.mech_tyre_phase.astype(float)
    miss = real - est
    rows = [dict(group="all", n=len(d), corr=np.corrcoef(est, real)[0, 1], est_median=est.median(),
                 real_median=real.median(), miss_median=miss.median(), miss_q25=miss.quantile(0.25),
                 miss_q75=miss.quantile(0.75), share_miss_over_2s=(miss.abs() > 2).mean())]
    for k, g in d.assign(miss=miss).groupby("compound_change", observed=True):
        rows.append(dict(group=f"compound_{k}", n=len(g), miss_median=g.miss.median(),
                         share_miss_over_2s=(g.miss.abs() > 2).mean()))
    out = pd.DataFrame(rows)
    out.to_csv("results/tyre_gain_check.csv", index=False)
    print(out.to_string())


if __name__ == "__main__":
    main()
