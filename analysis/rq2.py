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


def rule_summary(d, n_boot=2000):
    """How the raw rule does on 2023-25, and where the calibrated rule puts a 50/50 undercut, with a race-bootstrap
    95% CI. `breakeven_no_outlap_s` counts only full fresh-tyre laps (gain × (delay − 1)), leaving out the cold
    out-lap: a second reasonable reading of the rule, which shows how much the break-even depends on the estimator."""
    import numpy as np
    d = d.assign(margin_no_outlap=d.gain_s * (d.delay - 1) - d.gap_s)
    ci = lambda col: np.nanpercentile(models.bootstrap_by_race(d, lambda s: models.breakeven(s, col), n=n_boot),
                                      [2.5, 97.5])
    res = models.fit_logit(d, "success ~ margin_s")
    t = models.odds_table(res).set_index("term")
    yes, no = d[d.rule_pred == 1], d[d.rule_pred == 0]
    return pd.DataFrame([dict(n=len(d), accuracy=(d.rule_pred == d.success).mean(), n_rule_yes=len(yes),
                              success_rule_yes=yes.success.mean(), n_rule_no=len(no), success_rule_no=no.success.mean(),
                              or_margin=t.at["margin_s", "odds_ratio"], or_margin_lo=t.at["margin_s", "ci_lo"],
                              or_margin_hi=t.at["margin_s", "ci_hi"],
                              breakeven_margin_s=models.breakeven(d), breakeven_lo=ci("margin_s")[0],
                              breakeven_hi=ci("margin_s")[1], breakeven_no_outlap_s=models.breakeven(d, "margin_no_outlap"),
                              breakeven_no_outlap_lo=ci("margin_no_outlap")[0],
                              breakeven_no_outlap_hi=ci("margin_no_outlap")[1])])


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
    rule_summary(d[d.year < 2026]).to_csv("results/rq2_rule.csv", index=False)
    loso.to_csv("results/rq2_predictions_loso.csv", index=False)
    test26.to_csv("results/rq2_predictions_2026.csv", index=False)


if __name__ == "__main__":
    main()
