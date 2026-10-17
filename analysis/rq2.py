"""RQ2: does a model beat the rule of thumb? Leave-one-season-out over 2023-25, then train 2023-25 → test 2026."""
import pandas as pd

from undercut import models

PAIRS = [("logit", "rule_cal"), ("logit", "base_rate"), ("rule_cal", "base_rate"), ("rule", "base_rate")]
METRICS = ["brier", "accuracy"]


def load():
    return models.prepare(pd.read_parquet("data/attempts_main.parquet"))


def scores(pred, label):
    rows = []
    for name in models.MODELS:
        p = pred.dropna(subset=[name])
        rows.append(dict(held_out=label, model=name, n=len(p), **models.metrics(p.y, p[name])))
    return rows


def main():
    d = load()
    loso = models.loso(d, models.MODELS)
    test26 = models.season_out(d, models.MODELS, [2023, 2024, 2025], [2026])
    rows = [r for y, g in loso.groupby("year") for r in scores(g, str(y))] + scores(loso, "pooled")
    pd.DataFrame(rows).to_csv("results/rq2_loso.csv", index=False)
    pd.DataFrame(scores(test26, "2026")).to_csv("results/rq2_2026.csv", index=False)
    diffs = [dict(split=split, a=a, b=b, metric=m, **dict(zip(["diff", "ci_lo", "ci_hi"],
                                                              models.bootstrap_metric_diff(pred, a, b, m))))
             for split, pred in (("loso", loso), ("2026", test26)) for a, b in PAIRS for m in METRICS]
    pd.DataFrame(diffs).to_csv("results/rq2_diffs.csv", index=False)
    loso.to_csv("results/rq2_predictions_loso.csv", index=False)
    test26.to_csv("results/rq2_predictions_2026.csv", index=False)


if __name__ == "__main__":
    main()
