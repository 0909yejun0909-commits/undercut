"""What an undercut attempt is worth at the flag, and where on the grid and at which circuit it happens. Exploratory:
added after pre-registration (notes/decisions.md)."""
import numpy as np
import pandas as pd

POINTS = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}  # no fastest-lap point
POS_BINS = [0, 3, 6, 10, 20]
POS_LABELS = ["P1-3", "P4-6", "P7-10", "P11+"]
FRONT = 6  # "front-runner": running in the top six before the stop
MIN_CIRCUIT = 4  # circuits with fewer 2023-25 attempts are pooled as "other" in the circuit tests


def finish(results):
    """{DriverNumber: finishing position}; a disqualified car counts as unclassified (NaN)."""
    r = results[results.Status != "Disqualified"]
    return dict(zip(r.DriverNumber, r.FinishPosition))


def points(pos):
    return POINTS.get(int(pos), 0) if pd.notna(pos) else 0


def running_position(laps, driver, lap):
    p = laps[(laps.DriverNumber == driver) & (laps.LapNumber == lap)].Position
    return float(p.iloc[0]) if len(p) else np.nan


def add_stakes(df, laps_by_race, results_by_race):
    """Adds the chaser's position on lap n−1, both cars' finishing positions, whether the chaser finished ahead,
    the places it gained from lap n−1 to the flag, and the points swing within the pair (chaser − leader).
    An unclassified car finishes behind any classified one and scores 0."""
    rows = []
    for r in df.itertuples(index=False):
        fin = finish(results_by_race[r.race_id])
        cf, lf = fin.get(r.chaser, np.nan), fin.get(r.leader, np.nan)
        pos = running_position(laps_by_race[r.race_id], r.chaser, r.lap - 1)
        rows.append(dict(position=pos, chaser_finish=cf, leader_finish=lf,
                         ahead_at_flag=float(pd.notna(cf) and (pd.isna(lf) or cf < lf)),
                         places_gained=pos - cf, pair_points=points(cf) - points(lf)))
    out = df.reset_index(drop=True).join(pd.DataFrame(rows))
    out["position_band"] = pd.cut(out.position, POS_BINS, labels=POS_LABELS)
    out["front"] = (out.position <= FRONT).astype(float)
    out["points_fight"] = out.position <= 11  # the car ahead was in the top ten
    return out


def pool_circuits(circuit, keep):
    """Circuits not in `keep` become 'other'."""
    return pd.Series(np.where(pd.Series(circuit).isin(keep), circuit, "other"), index=getattr(circuit, "index", None))


def shuffle_circuits(df, rng):
    """Reassigns circuits across races at random, keeping each race's attempts together (a permutation null)."""
    rc = df.groupby("race_id").circuit.first()
    swap = dict(zip(rc.index, rng.permutation(rc.to_numpy())))
    return df.assign(circuit=df.race_id.map(swap))
