"""Predicting an undercut (exploratory, added after the pre-registration; see notes/decisions.md). The RQ1/RQ2 model
uses the leader's response delay, which is known only once the leader pits, so it cannot be used to decide whether
to stop. RQ5 compares pre-stop models on held-out seasons; the gap alone predicted best, and adding tyre ages,
the tyre choice and track features made held-out predictions worse. `chance` therefore uses the gap-only model
before the stop and the gap + delay model for each possible response, the best of each kind in rq5_scores.csv.
All chances assume the leader responds within five laps, as every attempt in the data did."""
import numpy as np
import pandas as pd

from undercut import models

GAP_ONLY = "success ~ gap_s"
GAP_DELAY = "success ~ gap_s + delay_cat"
PRESTOP = "success ~ gap_s + tyre_age_diff + compound_change + pit_loss_s + deg_s_per_lap"
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
