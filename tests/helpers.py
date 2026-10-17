import numpy as np
import pandas as pd


def make_laps(cars, green_off=(), drop=()):
    """cars: {driver: dict(offset, laps, base, fresh, pits={lap: new_compound}, compound, slow={lap: s}, deg=0,
    penalty={lap})}. Lap time = base up to and including the first stop lap, fresh from its out-lap on, plus
    deg × TyreLife and any slow-lap seconds. TyreLife restarts at 1 on each out-lap; a penalty lap has pit_in but
    keeps its tyres. end_s = offset + cumulative lap time."""
    rows = []
    for d, c in cars.items():
        pits, slow, penalty = c.get("pits", {}), c.get("slow", {}), c.get("penalty", set())
        first = min(pits) if pits else np.inf
        compound, life, stint, t = c.get("compound", "MEDIUM"), 0, 1, c["offset"]
        for k in range(1, c["laps"] + 1):
            pit_out = k - 1 in pits
            if pit_out:
                compound, life, stint = pits[k - 1], 0, stint + 1
            life += 1
            lt = (c["base"] if k <= first else c.get("fresh", c["base"])) + c.get("deg", 0.0) * life + slow.get(k, 0.0)
            t += lt
            rows.append(dict(Driver=f"D{d}", DriverNumber=d, LapNumber=k, Position=np.nan, end_s=t, lap_time_s=lt,
                             pit_in=k in pits or k in penalty, pit_out=pit_out, green=(d, k) not in green_off,
                             Compound=compound, TyreLife=float(life), Stint=float(stint)))
    out = pd.DataFrame(rows)
    keep = ~pd.Series([(r.DriverNumber, r.LapNumber) in set(drop) for r in out.itertuples()], index=out.index)
    return out[keep].reset_index(drop=True)
