import numpy as np
import pandas as pd
import pytest

from tests.test_models import _sim
from undercut import models, predict

SIM = models.prepare(_sim(n=600, seed=4))


def test_prestop_models_never_see_the_response_delay():
    for f in (predict.PRESTOP, predict.GAP_ONLY):
        assert "delay" not in f
    assert set(predict.PREDICT_MODELS) == {"base_rate", "gap_only", "prestop", "gap_delay", "after_response"}


def test_chance_gives_before_stop_and_each_response():
    c = predict.chance(predict.fit(SIM), gap_s=1.0)
    assert set(c) == {"before_stop", "1", "2", "3", "4-5"}
    assert all(0 < v < 1 for v in c.values())


def test_chance_falls_as_the_gap_grows():
    fits = predict.fit(SIM)
    near, far = predict.chance(fits, 0.5), predict.chance(fits, 2.5)
    assert all(near[k] > far[k] for k in near)


def test_chance_uses_the_gap_only_and_gap_delay_fits():
    fits = predict.fit(SIM)
    assert set(fits) == {"gap_only", "gap_delay"}
    assert fits["gap_only"].model.formula == predict.GAP_ONLY
    assert fits["gap_delay"].model.formula == predict.GAP_DELAY



def _signal(n=600, seed=5, strength=1.5):
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({"year": rng.choice([2023, 2024, 2025], n), "x": rng.normal(size=n), "z": rng.normal(size=n)})
    d["race_id"] = [f"{y}_{i % 20:02d}" for y, i in zip(d.year, range(n))]
    d["success"] = (rng.uniform(size=n) < 1 / (1 + np.exp(-strength * d.x))).astype(float)
    return d


def test_ridge_with_huge_penalty_predicts_the_base_rate():
    d = _signal()
    p = predict.ridge("success ~ x + z", alphas=[1e4])(d)(d.head(5))
    assert np.allclose(p, d.success.mean(), atol=0.01)


def test_ridge_with_tiny_penalty_matches_plain_logistic():
    d = _signal()
    p = predict.ridge("success ~ x + z", alphas=[1e-8])(d)(d.head(5))
    q = models._glm("success ~ x + z")(d)(d.head(5))
    assert np.allclose(p, q, atol=1e-3)


def test_ridge_picks_more_shrinkage_for_pure_noise():
    noise, signal = _signal(strength=0.0, n=300), _signal(strength=2.0, n=300)
    grid = [0.001, 0.01, 0.1, 1.0]
    assert predict.choose_alpha(noise, "success ~ x + z", grid) > predict.choose_alpha(signal, "success ~ x + z", grid)


def test_wide_training_uses_only_the_training_seasons():
    seen = []
    spy = lambda tr: (seen.append(set(tr.year)), lambda te: np.zeros(len(te)))[1]
    wide = pd.DataFrame({"year": [2023, 2024, 2025, 2026]})
    predict.wide(spy, wide)(pd.DataFrame({"year": [2024, 2025]}))
    assert seen == [{2024, 2025}]


def test_add_extra_features():
    d = pd.DataFrame({"race_id": ["a", "b"], "lap": [10, 30], "chaser_old": ["SOFT", "HARD"]})
    e = predict.add_extra(d, {"a": 50, "b": 60})
    assert e.progress.tolist() == pytest.approx([0.2, 0.5]) and e.off_soft.tolist() == [1, 0]
