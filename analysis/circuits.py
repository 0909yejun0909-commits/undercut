"""Exploratory (added after pre-registration): does the undercut work differently at some circuits, beyond the gap
and the response delay, and does knowing the circuit help prediction? 2023-25, main sample."""
import glob
import json

import numpy as np
import pandas as pd
from scipy import stats

from analysis.rq1 import YEARS, rate_ci
from analysis.rq2 import load
from analysis.rq5 import write
from undercut import models, predict, stakes

N_PERM = 1000


def with_circuits(d):
    events = {(i := json.load(open(f)))["race_id"]: i["Event"] for f in glob.glob("data/laps/*_info.json")}
    return d.assign(circuit=d.race_id.map(events).str.replace(" Grand Prix", ""))


def by_circuit(d):
    rows = [dict(circuit=c, races=g.race_id.nunique(), **rate_ci(g), median_gap_s=g.gap_s.median(),
                 median_deg=g.deg_s_per_lap.median(), median_pit_loss_s=g.pit_loss_s.median(),
                 median_gain_s=g.gain_s.median()) for c, g in d.groupby("circuit")]
    return pd.DataFrame(rows).sort_values("rate")


def lr_stat(d, keep):
    d = d.assign(circ=stakes.pool_circuits(d.circuit, keep))
    base = models.fit_logit(d, predict.GAP_DELAY, cluster=False)
    res = models.fit_logit(d, f"{predict.GAP_DELAY} + C(circ)", cluster=False)
    return 2 * (res.llf - base.llf), res.df_model - base.df_model


def circuit_glm(keep):
    """gap + delay + circuit; a circuit the training seasons never saw (or pooled) is predicted as 'other'."""
    def fit(train):
        tr = train.assign(circ=stakes.pool_circuits(train.circuit, keep))
        res = models.fit_logit(tr, f"{predict.GAP_DELAY} + C(circ)", cluster=False)
        seen = set(tr.circ)
        return lambda test: np.asarray(res.predict(test.assign(
            circ=[c if c in seen else "other" for c in stakes.pool_circuits(test.circuit, keep)])), dtype=float)
    return fit


def tests(d, rates):
    keep = set(rates.circuit[rates.n >= stakes.MIN_CIRCUIT])
    lr, df = lr_stat(d, keep)
    rng = np.random.default_rng(0)
    null = []
    for _ in range(N_PERM):
        try:
            null.append(lr_stat(stakes.shuffle_circuits(d, rng), keep)[0])
        except ValueError:  # a shuffle can leave a pooled level empty or rank-deficient
            continue
    m = {"gap_delay": models._glm(predict.GAP_DELAY), "plus_circuit": circuit_glm(keep)}
    loso = models.loso(d, m)
    diff, lo, hi = models.bootstrap_metric_diff(loso, "plus_circuit", "gap_delay", "brier")
    rows = [dict(test="likelihood ratio", stat=lr, df=df, p=stats.chi2.sf(lr, df), n=len(d), circuits=len(keep)),
            dict(test="permutation (races reshuffled)", stat=lr, df=df, p=np.mean(np.array(null) >= lr),
                 n=len(null), circuits=len(keep)),
            dict(test="held-out Brier, gap + delay", stat=models.metrics(loso.y, loso.gap_delay)["brier"]),
            dict(test="held-out Brier, + circuit", stat=models.metrics(loso.y, loso.plus_circuit)["brier"]),
            dict(test="held-out Brier difference", stat=diff, ci_lo=lo, ci_hi=hi)]
    for col in ("median_gap_s", "median_deg", "median_pit_loss_s", "median_gain_s"):
        r, p = stats.spearmanr(rates[col], rates.rate)
        rows.append(dict(test=f"circuit rate vs {col} (Spearman)", stat=r, p=p, n=len(rates)))
    return pd.DataFrame(rows)


def main():
    d = with_circuits(load())
    old = d[d.year.isin(YEARS)]
    rates = by_circuit(old)
    write(rates, "results/circuit_rates.csv")
    write(tests(old, rates), "results/circuit_tests.csv")


if __name__ == "__main__":
    main()
