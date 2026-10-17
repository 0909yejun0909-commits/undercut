"""Models for 'does the undercut work?', scoring, and season splits. Everything here was fixed (git tag `prereg`)
before any model saw a real attempt; later changes are logged in notes/decisions.md."""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

DELAY_LEVELS = ["1", "2", "3", "4-5"]
CHANGE_LEVELS = ["harder", "same", "softer"]
FORMULA = "success ~ gap_s + delay_cat + tyre_age_diff + compound_change + pit_loss_s + deg_s_per_lap"
MODEL_COLS = ["success", "gap_s", "delay_cat", "tyre_age_diff", "compound_change", "pit_loss_s", "deg_s_per_lap",
              "margin_s", "rule_pred"]


def prepare(df):
    """Responded attempts with every model column present; categories get fixed levels so every split has the
    same design columns. The number of responded rows dropped for missing values is in .attrs['dropped']."""
    d = df[df.responded.astype(bool)]
    out = d.dropna(subset=MODEL_COLS).copy()
    out["delay_cat"] = pd.Categorical(out.delay_cat, categories=DELAY_LEVELS)
    out["compound_change"] = pd.Categorical(out.compound_change, categories=CHANGE_LEVELS)
    out["success"] = out.success.astype(float)
    out.attrs["dropped"] = len(d) - len(out)
    return out.reset_index(drop=True)


def fit_logit(df, formula=FORMULA, cluster=True):
    """Binomial GLM; refuses a rank-deficient design (e.g. a training fold with no '4-5' responses), whose
    coefficients would otherwise be arbitrary. With cluster=True, standard errors are clustered by race."""
    model = smf.glm(formula, data=df, family=sm.families.Binomial())
    if np.linalg.matrix_rank(model.exog) < model.exog.shape[1]:
        raise ValueError(f"rank-deficient design for {formula}")
    if cluster:
        return model.fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(df.race_id)[0]})
    return model.fit()


def odds_table(res):
    ci = res.conf_int()
    return pd.DataFrame({"term": res.params.index, "odds_ratio": np.exp(res.params.to_numpy()),
                         "ci_lo": np.exp(ci[0].to_numpy()), "ci_hi": np.exp(ci[1].to_numpy()),
                         "p": res.pvalues.to_numpy()})


def _base_rate(train):
    rate = train.success.mean()
    return lambda test: np.full(len(test), rate)


def _rule(train):
    return lambda test: test.rule_pred.to_numpy(dtype=float)


def _glm(formula):
    def fit(train):
        res = fit_logit(train, formula, cluster=False)
        return lambda test: np.asarray(res.predict(test), dtype=float)
    return fit


MODELS = {"base_rate": _base_rate, "rule": _rule, "rule_cal": _glm("success ~ margin_s"), "logit": _glm(FORMULA)}


def breakeven(df, margin_col="margin_s"):
    """The rule margin at which a logistic fit of success on that margin gives 50%: −intercept / slope."""
    res = fit_logit(df, f"success ~ {margin_col}", cluster=False)
    return float(-res.params["Intercept"] / res.params[margin_col])


def metrics(y, p):
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    q = np.clip(p, 1e-6, 1 - 1e-6)
    return {"brier": float(np.mean((p - y) ** 2)),
            "logloss": float(-np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))),
            "accuracy": float(np.mean((p >= 0.5) == (y == 1)))}


def season_out(df, models, train_years, test_years):
    train, test = df[df.year.isin(train_years)], df[df.year.isin(test_years)]
    out = pd.DataFrame({"race_id": test.race_id.to_numpy(), "year": test.year.to_numpy(),
                        "y": test.success.to_numpy(dtype=float)})
    for name, fit in models.items():
        try:
            out[name] = fit(train)(test)
        except ValueError:  # rank-deficient training fold: no prediction for this model in this split
            out[name] = np.nan
    return out


def loso(df, models, seasons=(2023, 2024, 2025)):
    return pd.concat([season_out(df, models, [s for s in seasons if s != held], [held]) for held in seasons],
                     ignore_index=True)


def bootstrap_by_race(df, fit, n=2000, seed=0, race_col="race_id"):
    rng = np.random.default_rng(seed)
    races = df[race_col].unique()
    groups = {k: g for k, g in df.groupby(race_col)}
    out = []
    for _ in range(n):
        pick = rng.choice(races, size=len(races), replace=True)
        sample = pd.concat([groups[k].assign(**{race_col: f"{k}#{i}"}) for i, k in enumerate(pick)], ignore_index=True)
        try:
            out.append(fit(sample))
        except Exception:  # a resample can be degenerate (e.g. perfect separation); count it as missing
            out.append(np.nan)
    return np.array(out, dtype=float)


def bootstrap_metric_diff(pred, a, b, metric, n=2000, seed=0):
    """metric(a) − metric(b) on the rows where both predicted, with a 95% CI from resampling races."""
    d = pred.dropna(subset=[a, b])
    stat = lambda s: metrics(s.y, s[a])[metric] - metrics(s.y, s[b])[metric]
    boot = bootstrap_by_race(d, stat, n=n, seed=seed)
    lo, hi = np.nanpercentile(boot, [2.5, 97.5])
    return float(stat(d)), float(lo), float(hi)
