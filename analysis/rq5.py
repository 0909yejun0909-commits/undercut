"""RQ5 (exploratory, added after pre-registration): can an undercut be predicted before the chaser stops?
Leave-one-season-out over 2023-25, then train 2023-25 → test 2026, as in RQ2."""
import numpy as np
import pandas as pd

from analysis.rq2 import load
from undercut import models, predict

NOTE = "# exploratory: added after pre-registration\n"
PAIRS = [("gap_only", "base_rate"), ("prestop", "gap_only"), ("prestop", "base_rate"), ("gap_delay", "after_response"),
         ("gap_delay", "base_rate"), ("gap_delay", "gap_only")]


def write(df, path):
    with open(path, "w") as f:
        f.write(NOTE)
        df.to_csv(f, index=False)


def scores(pred, label):
    return [dict(held_out=label, model=m, n=len(p := pred.dropna(subset=[m])), **models.metrics(p.y, p[m]))
            for m in predict.PREDICT_MODELS]


def calibration(pred, col, bins=(0, 0.3, 0.5, 0.7, 1.0001)):
    """Predicted vs observed success, in bands of the predicted chance."""
    g = pred.dropna(subset=[col]).groupby(pd.cut(pred[col], bins, right=False), observed=True)
    return pd.DataFrame({"band": [str(b) for b in g.groups], "n": g.size().to_numpy(),
                         "mean_predicted": g[col].mean().to_numpy(), "observed": g.y.mean().to_numpy()})


def main():
    d = load()
    old = d[d.year < 2026]
    loso = models.loso(d, predict.PREDICT_MODELS)
    test26 = models.season_out(d, predict.PREDICT_MODELS, [2023, 2024, 2025], [2026])
    rows = [r for y, g in loso.groupby("year") for r in scores(g, str(y))] + scores(loso, "pooled") + scores(test26, "2026")
    write(pd.DataFrame(rows), "results/rq5_scores.csv")
    diffs = [dict(split=split, a=a, b=b, metric="brier", **dict(zip(["diff", "ci_lo", "ci_hi"],
                                                                 models.bootstrap_metric_diff(pred, a, b, "brier"))))
             for split, pred in (("loso", loso), ("2026", test26)) for a, b in PAIRS]
    write(pd.DataFrame(diffs), "results/rq5_diffs.csv")
    write(pd.concat([calibration(loso, m).assign(model=m) for m in ("gap_only", "prestop", "gap_delay")]),
          "results/rq5_calibration.csv")
    write(pd.concat([models.odds_table(models.fit_logit(old, f)).assign(model=m) for m, f in
                     (("gap_only", predict.GAP_ONLY), ("gap_delay", predict.GAP_DELAY), ("prestop", predict.PRESTOP))]),
          "results/rq5_odds.csv")


if __name__ == "__main__":
    main()
