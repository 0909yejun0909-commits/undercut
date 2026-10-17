"""Race-level features for each attempt, measured on the other cars only (never the chaser or the leader), so the
rule of thumb and the models never see the attempt's own outcome. The mech_* columns describe what happened during
the attempt itself and are for the mechanism table only, never a model input."""
import numpy as np
import pandas as pd

from undercut.attempts import stops

ORDER = {"SOFT": 0, "MEDIUM": 1, "HARD": 2}
TRAFFIC_S = 1.5


def _lap_times(laps):
    return {d: g.set_index("LapNumber") for d, g in laps.groupby("DriverNumber")}


def _green_times(g, ks):
    """Lap times on laps ks, or None if any is missing or not green."""
    if not all(k in g.index and g.at[k, "green"] and pd.notna(g.at[k, "lap_time_s"]) for k in ks):
        return None
    return g.loc[list(ks), "lap_time_s"].to_numpy()


def _other_stops(laps, exclude):
    st = stops(laps)
    return st[~st.DriverNumber.isin(exclude)]


def fresh_tyre_gain(laps, around_lap, exclude, window=10):
    """Median over other cars' stops near `around_lap`: mean of the two laps before the in-lap minus mean of the two
    laps after the out-lap. Positive = fresh tyres are faster (fuel burn over those laps is included)."""
    lt, out = _lap_times(laps), []
    for d, p in _other_stops(laps, exclude).itertuples(index=False):
        if abs(p - around_lap) > window:
            continue
        t = _green_times(lt[d], range(p - 2, p + 4))
        if t is not None:
            out.append(t[:2].mean() - t[4:].mean())
    return float(np.median(out)) if out else np.nan


def _stop_loss(g, p):
    """In-lap + out-lap minus twice the car's median normal lap in p−3..p+4; NaN if either lap is not green."""
    pair = _green_times(g, [p, p + 1])
    near = [k for k in range(p - 3, p + 5) if k not in (p, p + 1) and k in g.index and g.at[k, "green"]
            and not (g.at[k, "pit_in"] or g.at[k, "pit_out"]) and pd.notna(g.at[k, "lap_time_s"])]
    if pair is None or not near:
        return np.nan
    return float(pair.sum() - 2 * g.loc[near, "lap_time_s"].median())


def pit_loss(laps, exclude):
    """Median over other cars' green stops of the stop's time loss over normal pace."""
    lt = _lap_times(laps)
    out = [x for d, p in _other_stops(laps, exclude).itertuples(index=False) if pd.notna(x := _stop_loss(lt[d], p))]
    return float(np.median(out)) if out else np.nan


def deg_rate(laps, exclude, min_laps=6):
    """Median over other cars' stints of the OLS slope of lap time on tyre age (s per lap of tyre age), using green
    laps other than lap 1, in-laps and out-laps. Net of fuel burn, which makes later laps faster."""
    k = laps[~laps.DriverNumber.isin(exclude) & laps.green & ~laps.pit_in & ~laps.pit_out & (laps.LapNumber > 1)]
    k = k.dropna(subset=["lap_time_s", "TyreLife"])
    out = [np.polyfit(g.TyreLife, g.lap_time_s, 1)[0] for _, g in k.groupby(["DriverNumber", "Stint"])
           if len(g) >= min_laps and g.TyreLife.nunique() > 1]
    return float(np.median(out)) if out else np.nan


def _change(old, new):
    o, n = ORDER.get(old), ORDER.get(new)
    if o is None or n is None:
        return np.nan
    return "harder" if n > o else "softer" if n < o else "same"


def _delay_cat(d):
    return np.nan if pd.isna(d) else "4-5" if d >= 4 else str(int(d))


def _mechanism(laps, r):
    lt = laps.pivot(index="LapNumber", columns="DriverNumber", values="lap_time_s")
    E = laps.pivot(index="LapNumber", columns="DriverNumber", values="end_s")
    n, b, a = r.lap, r.chaser, r.leader
    inlap = lt.at[n, b] - lt.at[n, a]
    outlap = lt.at[n + 1, b] - lt.at[n + 1, a] if n + 1 in lt.index else np.nan
    row = E.loc[n + 1].drop([a, b]).dropna() if n + 1 in E.index else pd.Series(dtype=float)
    ahead = E.at[n + 1, b] - row
    # swing = time the chaser gained on the leader from lap n−1 to the check lap; stop_diff = how much more the
    # leader's stop cost than the chaser's; the rest is the tyre phase (fresh vs old laps, plus any traffic)
    k = int(r.check_lap) if pd.notna(r.check_lap) else None
    swing = r.gap_s - (E.at[k, b] - E.at[k, a]) if k in E.index else np.nan
    g = _lap_times(laps)
    stop_diff = _stop_loss(g[a], int(r.leader_pit_lap)) - _stop_loss(g[b], n) if r.responded else np.nan
    return inlap, -outlap, int(((ahead > 0) & (ahead <= TRAFFIC_S)).any()), swing, stop_diff


def add_features(attempts, laps_by_race):
    out = []
    for r in attempts.itertuples(index=False):
        laps, ex = laps_by_race[r.race_id], {r.chaser, r.leader}
        gain = fresh_tyre_gain(laps, r.lap, ex)
        inlap, outlap, traffic, swing, stop_diff = _mechanism(laps, r)
        out.append(dict(gain_s=gain, pit_loss_s=pit_loss(laps, ex), deg_s_per_lap=deg_rate(laps, ex),
                        mech_inlap_loss=inlap, mech_outlap_gain=outlap, mech_traffic=traffic, mech_swing=swing,
                        mech_stop_diff=stop_diff, mech_tyre_phase=swing - stop_diff))
    f = attempts.reset_index(drop=True).join(pd.DataFrame(out, columns=[
        "gain_s", "pit_loss_s", "deg_s_per_lap", "mech_inlap_loss", "mech_outlap_gain", "mech_traffic", "mech_swing",
        "mech_stop_diff", "mech_tyre_phase"]))
    f["margin_s"] = f.gain_s * f.delay - f.gap_s
    f["rule_pred"] = np.where(f.margin_s.isna(), np.nan, (f.margin_s > 0).astype(float))
    f["tyre_age_diff"] = f.leader_age - f.chaser_age
    f["compound_change"] = [_change(o, n) for o, n in zip(f.chaser_old, f.chaser_new)]
    f["delay_cat"] = f.delay.map(_delay_cat)
    f["y2026"] = (f.year == 2026).astype(int)
    return f
