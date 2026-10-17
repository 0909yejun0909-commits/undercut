"""Undercut attempts from lap-end crossing times. On the lap before the chaser's stop (n−1) the chaser crosses the
line within `window_s` of the car directly ahead (the leader). The chaser stops on lap n; the leader stops on lap m
within `max_delay` laps. The attempt works if the chaser is ahead at the end of lap m + 1 + check_offset (with
check_offset = 1: the lap after the leader's out-lap). Ordering cars by end_s on the same lap number means lapped
cars never count as the car ahead."""
import numpy as np
import pandas as pd

from undercut.load import WET

WINDOW_S = 3.0
MAX_DELAY = 5
CHECK_OFFSET = 1
COLUMNS = ["race_id", "year", "rnd", "chaser", "leader", "chaser_code", "leader_code", "lap", "gap_s", "responded",
           "leader_pit_lap", "delay", "check_lap", "success", "chaser_ahead_end", "chaser_age", "leader_age",
           "chaser_old", "chaser_new", "leader_old"]


def stops(laps):
    """Real tyre changes: a pit-in lap followed by lower TyreLife or a different compound. A drive-through or
    stop-go penalty keeps the tyres, so it is not a stop."""
    out = []
    for d, g in laps.sort_values("LapNumber").groupby("DriverNumber"):
        nxt = g.shift(-1)
        real = g.pit_in & (nxt.LapNumber == g.LapNumber + 1) & ((nxt.TyreLife < g.TyreLife) | (nxt.Compound != g.Compound))
        out += [(d, int(k)) for k in g.LapNumber[real]]
    return pd.DataFrame(out, columns=["DriverNumber", "lap"])


def find_attempts(laps, info, window_s=WINDOW_S, max_delay=MAX_DELAY, check_offset=CHECK_OFFSET, require_green=True):
    if info.get("wet") or laps.empty:
        return pd.DataFrame(columns=COLUMNS)
    piv = lambda col: laps.pivot(index="LapNumber", columns="DriverNumber", values=col)
    E, C, A = piv("end_s"), piv("Compound"), piv("TyreLife")
    G = piv("green").reindex_like(E).eq(True)  # missing laps are not green
    P = laps.assign(p=laps.pit_in | laps.pit_out).pivot(index="LapNumber", columns="DriverNumber", values="p")
    P = P.reindex_like(E).eq(True)
    code = laps.groupby("DriverNumber").Driver.first()
    st = stops(laps)
    stop_laps = st.groupby("DriverNumber").lap.apply(sorted).to_dict()

    def present(d, ks):
        return all(k in E.index and pd.notna(E.at[k, d]) for k in ks)

    rows = []
    for b, n in st.itertuples(index=False):
        if n - 1 not in E.index:
            continue
        prev = E.loc[n - 1].dropna().sort_values()
        if b not in prev.index or prev.index.get_loc(b) == 0:
            continue
        a = prev.index[prev.index.get_loc(b) - 1]
        gap = prev[b] - prev[a]
        if not 0 < gap <= window_s:
            continue
        if not present(a, [n - 1, n]) or P.at[n - 1, a] or P.at[n, a] or P.at[n - 1, b]:
            continue
        resp = [m for m in stop_laps.get(a, []) if n + 1 <= m <= n + max_delay]
        m = resp[0] if resp else None
        check = m + 1 + check_offset if resp else n + max_delay + 1
        green_to = m + 1 if resp and not require_green else check
        window = range(n - 1, check + 1)
        if not (present(a, window) and present(b, window)):
            continue
        if any(C.at[k, d] in WET for k in window for d in (a, b)):
            continue
        if not all(G.at[k, d] for k in range(n - 1, green_to + 1) for d in (a, b)):
            continue
        ahead = float(E.at[check, b] < E.at[check, a])
        rows.append(dict(race_id=info["race_id"], year=info["Year"], rnd=info["Round"], chaser=int(b), leader=int(a),
                         chaser_code=code[b], leader_code=code[a], lap=int(n), gap_s=float(gap), responded=bool(resp),
                         leader_pit_lap=float(m) if resp else np.nan, delay=float(m - n) if resp else np.nan,
                         check_lap=float(check), success=ahead if resp else np.nan,
                         chaser_ahead_end=np.nan if resp else ahead, chaser_age=A.at[n - 1, b], leader_age=A.at[n - 1, a],
                         chaser_old=C.at[n - 1, b], chaser_new=C.at[n + 1, b], leader_old=C.at[n - 1, a]))
    return pd.DataFrame(rows, columns=COLUMNS)
