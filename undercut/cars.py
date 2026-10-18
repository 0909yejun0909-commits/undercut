"""Car and team traits for the exploratory 'which cars close the gap?' analysis (added after the pre-registration;
see notes/decisions.md). Each trait is a chaser-versus-leader difference, signed so that positive favours the
chaser. Every trait is the team's median over the OTHER races of the same season (each race measured relative to
that race's field), so it describes the car and crew without seeing the attempt itself: a car that wins an
undercut then runs in clean air, which would flatter a pace measure taken from the same race."""
import numpy as np
import pandas as pd

from undercut.attempts import stops
from undercut.features import _lap_times, _stop_loss, deg_rate

TEAM_KEYS = {"AlphaTauri": "Racing Bulls", "RB": "Racing Bulls", "Alfa Romeo": "Sauber/Audi",
             "Kick Sauber": "Sauber/Audi", "Audi": "Sauber/Audi", "Haas F1 Team": "Haas"}


def team_key(name):
    """One name per team lineage across the renames of 2023–26."""
    return TEAM_KEYS.get(name, name)


def _clean(laps):
    return laps[laps.green & ~laps.pit_in & ~laps.pit_out & (laps.LapNumber > 1)].dropna(subset=["lap_time_s"])


def race_pace(laps, min_laps=10):
    """Each car's median clean lap (green, not lap 1, not an in- or out-lap); cars with fewer clean laps are left out."""
    g = _clean(laps).groupby("DriverNumber").lap_time_s
    n = g.size()
    return g.median()[n >= min_laps]


def car_deg(laps):
    """Each car's own tyre-wear slope: deg_rate run on that car's stints alone."""
    return pd.Series({d: deg_rate(g, exclude=set()) for d, g in laps.groupby("DriverNumber")}, dtype=float)


def stop_losses(laps):
    """Every green stop's time loss, and the same minus the race's median stop loss (loss_rel), which takes out the
    track's pit-lane length."""
    lt = _lap_times(laps)
    out = pd.DataFrame([(d, p, _stop_loss(lt[d], p)) for d, p in stops(laps).itertuples(index=False)],
                       columns=["DriverNumber", "lap", "loss"]).dropna(subset=["loss"])
    out["loss_rel"] = out.loss - out.loss.median()
    return out.reset_index(drop=True)


def car_table(laps):
    """Each car's race pace and tyre-wear slope minus the race's median car (negative pace_rel = faster car)."""
    pace, deg = race_pace(laps), car_deg(laps)
    out = pd.DataFrame({"pace_rel": pace - pace.median(), "deg_rel": deg - deg.median()})
    return out.rename_axis("DriverNumber").reset_index()


def season_median(table, team, race_id, col):
    """Median of `col` for `team` over the other races of the same season; NaN if it has none."""
    s = table[(table.team == team) & (table.race_id.str[:4] == race_id[:4]) & (table.race_id != race_id)]
    return float(s[col].median()) if s[col].notna().any() else np.nan


def add_car_features(attempts, teams, car_tbl, stops_tbl):
    """chaser_team, leader_team and three season-level advantages, positive = favours the chaser: pace_adv_s
    (leader's team's relative pace minus chaser's, s/lap), deg_adv_s (same for the tyre-wear slope) and crew_adv_s
    (leader's team's relative stop loss minus chaser's)."""
    tm = {(r.race_id, r.DriverNumber): team_key(r.TeamName) for r in teams.itertuples(index=False)}
    out = []
    for r in attempts.itertuples(index=False):
        ct, lt = tm.get((r.race_id, r.chaser)), tm.get((r.race_id, r.leader))
        adv = lambda tbl, col: season_median(tbl, lt, r.race_id, col) - season_median(tbl, ct, r.race_id, col)
        out.append(dict(chaser_team=ct, leader_team=lt, pace_adv_s=adv(car_tbl, "pace_rel"),
                        deg_adv_s=adv(car_tbl, "deg_rel"), crew_adv_s=adv(stops_tbl, "loss_rel")))
    return attempts.reset_index(drop=True).join(pd.DataFrame(out, columns=[
        "chaser_team", "leader_team", "pace_adv_s", "deg_adv_s", "crew_adv_s"]))


def pre_stop_pace_adv(attempts, laps_by_race, min_laps=5):
    """On-the-day pace before the attempt: leader's median clean lap minus chaser's, over laps before lap n−1 only,
    so it cannot see what happened after the stop (positive = chaser faster). Laps spent stuck behind the leader cap
    the chaser's pace, which pulls this towards zero."""
    out = []
    for r in attempts.itertuples(index=False):
        laps = laps_by_race[r.race_id]
        pace = race_pace(laps[laps.LapNumber < r.lap - 1], min_laps=min_laps)
        out.append(pace.get(r.leader, np.nan) - pace.get(r.chaser, np.nan))
    return pd.Series(out, index=attempts.index, dtype=float)
