import numpy as np
import pandas as pd
import pytest

from undercut import models


def _sim(n=400, seed=1):
    rng = np.random.default_rng(seed)
    year = rng.choice([2023, 2024, 2025], n)
    d = pd.DataFrame({
        "race_id": [f"{y}_{r:02d}" for y, r in zip(year, rng.integers(1, 21, n))], "year": year, "responded": True,
        "gap_s": rng.uniform(0, 3, n), "delay_cat": rng.choice(models.DELAY_LEVELS, n),
        "tyre_age_diff": rng.normal(0, 5, n), "compound_change": rng.choice(models.CHANGE_LEVELS, n),
        "pit_loss_s": rng.normal(20, 2, n), "deg_s_per_lap": rng.normal(0.05, 0.03, n),
    })
    d["margin_s"] = 1.5 * d.delay_cat.map({"1": 1, "2": 2, "3": 3, "4-5": 4.5}) - d.gap_s
    d["rule_pred"] = (d.margin_s > 0).astype(float)
    eta = 1.0 - 1.0 * d.gap_s + d.delay_cat.map({"1": 0, "2": 0.5, "3": 1.0, "4-5": 2.0})
    d["success"] = (rng.uniform(size=n) < 1 / (1 + np.exp(-eta))).astype(float)
    return d


SIM = _sim()


def test_metrics_values():
    m = models.metrics([1, 0], [0.8, 0.4])
    assert m["brier"] == pytest.approx(0.1) and m["accuracy"] == 1.0
    assert m["logloss"] == pytest.approx(-(np.log(0.8) + np.log(0.6)) / 2)


def test_prepare_fixes_category_levels_and_counts_drops():
    p = models.prepare(SIM.assign(margin_s=[np.nan] + [1.0] * (len(SIM) - 1)))
    assert list(p.delay_cat.cat.categories) == models.DELAY_LEVELS
    assert list(p.compound_change.cat.categories) == models.CHANGE_LEVELS
    assert p.attrs["dropped"] == 1 and len(p) == len(SIM) - 1


def test_prepare_keeps_only_responded_attempts():
    p = models.prepare(SIM.assign(responded=[False] * 10 + [True] * (len(SIM) - 10)))
    assert len(p) == len(SIM) - 10


def test_logit_recovers_gap_effect_sign():
    res = models.fit_logit(models.prepare(SIM))
    t = models.odds_table(res).set_index("term")
    assert t.loc["gap_s", "odds_ratio"] < 1 and t.loc["gap_s", "ci_hi"] < 1
    assert list(t.columns) == ["odds_ratio", "ci_lo", "ci_hi", "p"]


def test_loso_never_trains_on_the_held_out_season():
    seen = []
    spy = {"spy": lambda tr: (seen.append(set(tr.year)), lambda te: np.zeros(len(te)))[1]}
    out = models.loso(models.prepare(SIM), spy)
    assert seen == [{2024, 2025}, {2023, 2025}, {2023, 2024}]
    assert len(out) == len(SIM) and list(out.columns[:3]) == ["race_id", "year", "y"]


def test_season_out_survives_rank_deficient_fold():
    d = models.prepare(SIM)
    d = d[~((d.year == 2023) & (d.delay_cat == "4-5"))]
    out = models.season_out(d, models.MODELS, train_years=[2023], test_years=[2025])
    assert out["logit"].isna().all() and out["base_rate"].notna().all()


def test_base_rate_and_rule_models():
    d = models.prepare(SIM)
    assert np.allclose(models.MODELS["base_rate"](d)(d.head(3)), d.success.mean())
    assert np.array_equal(models.MODELS["rule"](d)(d.head(3)), d.rule_pred.head(3).to_numpy())


def test_bootstrap_diff_of_identical_models_is_zero():
    pred = models.loso(models.prepare(SIM), {k: models.MODELS["base_rate"] for k in ("a", "b")})
    assert models.bootstrap_metric_diff(pred, "a", "b", "brier", n=200) == (0.0, 0.0, 0.0)


def test_bootstrap_diff_favours_the_better_model():
    pred = models.loso(models.prepare(SIM), {"logit": models.MODELS["logit"], "base": models.MODELS["base_rate"]})
    diff, lo, hi = models.bootstrap_metric_diff(pred, "logit", "base", "brier", n=300)
    assert diff < 0 and hi < 0
