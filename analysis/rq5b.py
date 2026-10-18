"""RQ5b (exploratory): can a better model beat gap-only (before the stop) and gap + delay (by response)? Candidates
fixed before scoring: those two, each trained on the wider attempt set, and ridge logistic regressions with every
factor, trained on the standard or the wider set. Same splits as RQ2; test rows are always standard attempts."""
import glob
import json
import os

import pandas as pd

from analysis.rq2 import load
from analysis.rq5 import calibration, write
from undercut import attempts, features, models, predict

WIDE = "data/attempts_wide.parquet"


def total_laps():
    return {(i := json.load(open(f)))["race_id"]: i["total_laps"] for f in glob.glob("data/laps/*_info.json")}


def build_wide():
    """Gap up to 5 s and yellow-flag laps allowed: every attempt the standard set has, plus more."""
    if not os.path.exists(WIDE):
        data = [(json.load(open(f)), pd.read_parquet(f.replace("_info.json", "_laps.parquet")))
                for f in sorted(glob.glob("data/laps/*_info.json"))]
        a = pd.concat([x for info, laps in data
                       if not (x := attempts.find_attempts(laps, info, window_s=5.0, require_green=False)).empty],
                      ignore_index=True)
        features.add_features(a, {info["race_id"]: laps for info, laps in data}).to_parquet(WIDE)
    return models.prepare(pd.read_parquet(WIDE))


def candidates(w):
    return {
        "base_rate": models.MODELS["base_rate"],
        "gap_only": models._glm(predict.GAP_ONLY),
        "gap_only_wide": predict.wide(models._glm(predict.GAP_ONLY), w),
        "ridge_pre": predict.ridge(predict.PRE_RIDGE),
        "ridge_pre_wide": predict.wide(predict.ridge(predict.PRE_RIDGE), w),
        "gap_delay": models._glm(predict.GAP_DELAY),
        "gap_delay_wide": predict.wide(models._glm(predict.GAP_DELAY), w),
        "ridge_post": predict.ridge(predict.POST_RIDGE),
        "ridge_post_wide": predict.wide(predict.ridge(predict.POST_RIDGE), w),
    }


def main():
    tl = total_laps()
    d, w = predict.add_extra(load(), tl), predict.add_extra(build_wide(), tl)
    print(f"standard {len(d[d.year < 2026])} / wide {len(w[w.year < 2026])} attempts in 2023-25")
    m = candidates(w)
    loso = models.loso(d, m)
    test26 = models.season_out(d, m, [2023, 2024, 2025], [2026])
    rows = [dict(held_out=lbl, model=k, n=len(p := pred.dropna(subset=[k])), **models.metrics(p.y, p[k]))
            for lbl, pred in [(str(y), g) for y, g in loso.groupby("year")] + [("pooled", loso), ("2026", test26)]
            for k in m]
    write(pd.DataFrame(rows), "results/rq5b_scores.csv")
    pairs = [(k, "gap_only") for k in ("gap_only_wide", "ridge_pre", "ridge_pre_wide")] + \
            [(k, "gap_delay") for k in ("gap_delay_wide", "ridge_post", "ridge_post_wide")]
    diffs = [dict(split=s, a=a, b=b, **dict(zip(["diff", "ci_lo", "ci_hi"], models.bootstrap_metric_diff(pred, a, b, "brier"))))
             for s, pred in (("loso", loso), ("2026", test26)) for a, b in pairs]
    write(pd.DataFrame(diffs), "results/rq5b_diffs.csv")
    write(pd.concat([calibration(loso, k).assign(model=k) for k in m if k != "base_rate"]), "results/rq5b_calibration.csv")
    loso.to_csv("results/rq5b_predictions_loso.csv", index=False)


if __name__ == "__main__":
    main()
