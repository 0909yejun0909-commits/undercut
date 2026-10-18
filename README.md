# When does the undercut work? Pit-stop battles in Formula 1, 2023–2026

Code and results for a paper written for the Wharton Sports Analytics Journal.

The **undercut** is when a car stuck behind a rival pits first and uses fresh tyres to come out ahead after the
rival's own stop. This repo finds every green-flag undercut attempt in the dry Grands Prix from 2023 to round 17 of
2026 and asks:

1. **RQ1.** How often does the undercut work, and what decides it?
2. **RQ2.** Does a statistical model predict it better than the commentators' rule of thumb (fresh-tyre gain per lap
   × laps before the leader responds > gap)?
3. **RQ3 (exploratory).** Did the 2026 rules change it?

**Headline:**
- In 2023–25, half of 154 attempts worked.
- The gap and the leader's response delay decide most of them.
- The rule of thumb is about 1 s too optimistic. Once it's calibrated, it predicts held-out seasons as well as the
  full model.

## Data

Lap timing comes from [FastF1](https://github.com/theOehrly/Fast-F1), read **offline** from a local cache. By
default the cache is `~/Projects/battery-battles/data/cache`; set `UNDERCUT_FASTF1_CACHE` to use another one.

Since about October 2026, the FastF1 timing archive returns HTTP 401 without an F1 TV token (FastF1 issue #978), so
a cold cache can't be filled from the archive. The cache used here holds 2023 R1–22, 2024 R1–24, 2025 R1–24 and
2026 R1–17.

Derived data are written to `data/`, which is not committed.

## Setup

```bash
uv sync
uv run pytest -q
```

## Pipeline

Run these steps in order. Each one is `uv run python -m analysis.<step>`.

| Step | What it does |
|---|---|
| `build_laps` | Cached races → one lap table per race (`data/laps/`), with wet flags |
| `build_teams` | Each driver's team in every race, from the cached results → `data/teams.parquet` |
| `build_attempts` | Finds undercut attempts and adds features from the other cars. Writes five variants (main, gap2, gap5, check2, relaxed) to `data/attempts_*.parquet`, plus `results/attempt_counts.csv` |
| `rq1` | Success rates, logistic regression, lookup table, sensitivity checks, where the time comes from → `results/rq1_*.csv`, `mechanism.csv`, `no_response.csv` |
| `rq2` | Rule of thumb vs. model: leave one season out, then 2026 → `results/rq2_*.csv` |
| `rq3` | 2026 vs. 2023–25 (exploratory) → `results/rq3_*.csv` |
| `rq4` | Which teams and cars close the gap (exploratory, added after pre-registration) → `results/rq4_*.csv` |
| `figures` | Paper figures → `figures/*.pdf` |

The paper is in `paper/main.tex` (build it with `tectonic paper/main.tex`).

## Pre-registration

Definitions, features, models and metrics were fixed and tagged **`prereg`** before any model saw a real attempt.
Every later change is listed in `notes/decisions.md`. `notes/spotcheck.md` records a hand check of 10 random
attempts.
