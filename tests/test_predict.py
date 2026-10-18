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
