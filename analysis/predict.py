"""Chance that an undercut works, from the models fitted on 2023-25 (see undercut/predict.py for why only the gap
and the response are used). Example:
    uv run python -m analysis.predict --gap 1.2"""
import argparse

from analysis.rq2 import load
from undercut import predict

NAMES = {"1": "the next lap", "2": "2 laps later", "3": "3 laps later", "4-5": "4-5 laps later"}


def main(argv=None):
    a = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    a.add_argument("--gap", type=float, required=True, help="gap to the car ahead before the stop, seconds")
    x = a.parse_args(argv)
    d = load()
    c = predict.chance(predict.fit(d[d.year < 2026]), x.gap)
    print(f"Gap {x.gap:.1f} s, assuming the car ahead responds within five laps")
    print(f"  before the stop: {c['before_stop']:.0%} chance the undercut works")
    for k, name in NAMES.items():
        print(f"  if the car ahead pits {name}: {c[k]:.0%}")


if __name__ == "__main__":
    main()
