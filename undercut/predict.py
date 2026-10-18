"""Predicting an undercut (exploratory, added after the pre-registration; see notes/decisions.md). The RQ1/RQ2 model
uses the leader's response delay, which is known only once the leader pits, so it cannot be used to decide whether
to stop. RQ5 compares pre-stop models on held-out seasons; the gap alone predicted best, and adding tyre ages,
the tyre choice and track features made held-out predictions worse. `chance` therefore uses the gap-only model
before the stop and the gap + delay model for each possible response, the best of each kind in rq5_scores.csv.
All chances assume the leader responds within five laps, as every attempt in the data did."""
import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm

from undercut import models

GAP_ONLY = "success ~ gap_s"
GAP_DELAY = "success ~ gap_s + delay_cat"
PRESTOP = "success ~ gap_s + tyre_age_diff + compound_change + pit_loss_s + deg_s_per_lap"
EXTRA = " + gain_s + progress + off_soft"
PRE_RIDGE = PRESTOP + EXTRA
POST_RIDGE = PRE_RIDGE + " + delay_cat"
PREDICT_MODELS = {"base_rate": models.MODELS["base_rate"], "gap_only": models._glm(GAP_ONLY),
                  "prestop": models._glm(PRESTOP), "gap_delay": models._glm(GAP_DELAY),
                  "after_response": models.MODELS["logit"]}


def fit(df):
    return {"gap_only": models.fit_logit(df, GAP_ONLY, cluster=False),
            "gap_delay": models.fit_logit(df, GAP_DELAY, cluster=False)}


def chance(fits, gap_s):
    """{'before_stop': p, '1': p, '2': p, '3': p, '4-5': p}: the chance before the stop, then the chance if the leader
    responds that many laps after the chaser's stop."""
    grid = pd.DataFrame({"gap_s": gap_s, "delay_cat": pd.Categorical(models.DELAY_LEVELS, categories=models.DELAY_LEVELS)})
    out = {"before_stop": float(np.asarray(fits["gap_only"].predict(grid.iloc[:1]))[0])}
    out.update(zip(models.DELAY_LEVELS, map(float, np.asarray(fits["gap_delay"].predict(grid)))))
    return out


def add_extra(df, total_laps):
    """progress = attempt lap / race length; off_soft = 1 if the chaser is coming off the soft compound."""
    return df.assign(progress=df.lap / df.race_id.map(total_laps), off_soft=(df.chaser_old == "SOFT").astype(int))


def _ridge_fit(train, formula, alpha):
    """Logistic regression with an L2 penalty on standardised columns; the intercept is not penalised."""
    y, X = patsy.dmatrices(formula, train, return_type="dataframe")
    mu, sd = X.mean(), X.std(ddof=0).replace(0, 1)
    mu["Intercept"], sd["Intercept"] = 0.0, 1.0
    pen = np.where(X.columns == "Intercept", 0.0, alpha)
    res = sm.GLM(y, (X - mu) / sd, family=sm.families.Binomial()).fit_regularized(alpha=pen, L1_wt=0.0)
    beta = np.asarray(res.params)

    def predict(test):
        Xt = patsy.build_design_matrices([X.design_info], test, return_type="dataframe")[0]
        return 1 / (1 + np.exp(-((Xt - mu) / sd).to_numpy() @ beta))
    return predict


def choose_alpha(train, formula, alphas):
    """The penalty with the lowest Brier score over leave-one-season-out folds inside `train` (ties go to the
    larger penalty). Only the training seasons are ever seen."""
    if train.year.nunique() > 1:
        key = train.year
    else:  # one season: three folds of whole races instead
        key = train.race_id.map({r: i % 3 for i, r in enumerate(train.race_id.unique())})
    best = None
    for a in sorted(alphas, reverse=True):
        p, y = [], []
        for k in key.unique():
            tr, te = train[key != k], train[key == k]
            p.append(_ridge_fit(tr, formula, a)(te)); y.append(te.success.to_numpy())
        b = models.metrics(np.concatenate(y), np.concatenate(p))["brier"]
        if best is None or b < best[0] - 1e-12:
            best = (b, a)
    return best[1]


def ridge(formula, alphas=(0.003, 0.01, 0.03, 0.1, 0.3, 1.0)):
    """A model in the models.MODELS form: train → predict function, with the penalty chosen inside train."""
    def fit(train):
        a = alphas[0] if len(alphas) == 1 else choose_alpha(train, formula, alphas)
        return _ridge_fit(train, formula, a)
    return fit


def wide(fit, wide_df):
    """Trains `fit` on the wider attempt set (same training seasons) instead of the standard attempts."""
    return lambda train: fit(wide_df[wide_df.year.isin(set(train.year))])
