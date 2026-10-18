"""Race-situation factors for the exploratory 'more variables' screen (added after the pre-registration; see
notes/decisions.md). Everything here is known before the chaser stops: it uses lap n−1 and earlier, except the
track temperature, which is read over the attempt (it changes slowly, and a team knows it at the time)."""
import numpy as np
import pandas as pd

BAND_S = 2.0  # a car within ±2 s of where a car would rejoin counts as rejoin traffic
BLOCKED_S = 1.0
STREET = {"Monaco Grand Prix", "Singapore Grand Prix", "Azerbaijan Grand Prix", "Saudi Arabian Grand Prix",
          "Las Vegas Grand Prix"}


def is_street(event):
    return event in STREET


def track_temp(w, t0, t1):
    """Median track temperature over [t0, t1] (session seconds), or the reading nearest the middle if none."""
    inside = w[(w.t_s >= t0) & (w.t_s <= t1)]
    if len(inside):
        return float(inside.TrackTemp.median())
    return float(w.TrackTemp.iloc[(w.t_s - (t0 + t1) / 2).abs().argmin()])


def _clean(laps, car, ks):
    g = laps[(laps.DriverNumber == car) & laps.LapNumber.isin(ks)]
    return g[g.green & ~g.pit_in & ~g.pit_out].dropna(subset=["lap_time_s"])


def attempt_context(laps, r, pit_loss_s):
    n, c, l = int(r.lap), r.chaser, r.leader
    k = n - 1
    E = laps.pivot(index="LapNumber", columns="DriverNumber", values="end_s")
    others = E.loc[k].drop([c, l], errors="ignore").dropna()
    behind = lambda car: others - E.at[k, car]
    traffic = lambda car: int(behind(car).between(pit_loss_s - BAND_S, pit_loss_s + BAND_S).sum())
    ahead = E.at[k, l] - others
    ahead = ahead[ahead > 0]

    window = range(k - 3, k + 1)
    if all(len(_clean(laps, car, window)) == 4 for car in (c, l)):
        gap = lambda j: E.at[j, c] - E.at[j, l]
        closing = (gap(k - 3) - gap(k)) / 3
    else:
        closing = np.nan
    lf = _clean(laps, l, range(n - 5, n))
    fade = float(np.polyfit(lf.LapNumber, lf.lap_time_s, 1)[0]) if len(lf) >= 4 else np.nan
    me = laps[(laps.DriverNumber == c) & (laps.LapNumber == n)]
    pos = laps[(laps.DriverNumber == c) & (laps.LapNumber == k)].Position
    return dict(chaser_traffic=traffic(c), leader_traffic=traffic(l), closing_s_per_lap=closing,
                leader_fade_s_per_lap=fade, leader_blocked=int(len(ahead) > 0 and ahead.min() < BLOCKED_S),
                first_stop=int(me.Stint.iloc[0] == 1) if len(me) else np.nan,
                position=float(pos.iloc[0]) if len(pos) else np.nan)


def add_context(df, laps_by_race, weather, events, team_of):
    """Adds every factor above, plus track_temp_c, street (0/1) and teammates (0/1). weather: {race_id: frame with
    t_s, TrackTemp}; events: {race_id: event name}; team_of: {(race_id, DriverNumber): team}."""
    rows = []
    for r in df.itertuples(index=False):
        laps = laps_by_race[r.race_id]
        x = attempt_context(laps, r, r.pit_loss_s)
        e = laps[laps.DriverNumber == r.chaser].set_index("LapNumber").end_s
        x["track_temp_c"] = track_temp(weather[r.race_id], e.get(r.lap - 1, np.nan), e.get(r.lap + 2, np.nan))
        x["street"] = int(is_street(events[r.race_id]))
        x["teammates"] = int(team_of.get((r.race_id, r.chaser)) == team_of.get((r.race_id, r.leader)))
        rows.append(x)
    return df.reset_index(drop=True).join(pd.DataFrame(rows))


def add_hardness(df, comp):
    """Pirelli C-numbers (higher = softer) for the chaser's old and new tyres and the leader's old tyre, from the
    race's nomination in `comp` (reference/compounds.csv); step_c = new − old (negative = going harder);
    race_soft_c = the softest compound that race."""
    c = comp.set_index("race_id")
    num = lambda rid, label: float(c.at[rid, label.lower()]) if label in ("SOFT", "MEDIUM", "HARD") else np.nan
    out = pd.DataFrame({"old_c": [num(r, x) for r, x in zip(df.race_id, df.chaser_old)],
                        "new_c": [num(r, x) for r, x in zip(df.race_id, df.chaser_new)],
                        "leader_old_c": [num(r, x) for r, x in zip(df.race_id, df.leader_old)],
                        "race_soft_c": [float(c.at[r, "soft"]) for r in df.race_id]})
    out["step_c"] = out.new_c - out.old_c
    return df.reset_index(drop=True).join(out)
