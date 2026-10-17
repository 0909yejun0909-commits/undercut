# Decisions

Fixed before any model saw a real attempt (git tag `prereg`). Any later change to these is logged below under
"Post-hoc changes" and reported in the paper.

## Data
- Races: 2023 R1–22, 2024 R1–24, 2025 R1–24, 2026 R1–17 Grands Prix (no sprints), read from the Battery Battles
  FastF1 cache in offline mode. Wet races (any intermediate/wet tyre or rainfall during the race) are excluded.

## Attempt (undercut/attempts.py)
- On lap n−1 the chaser crosses the line within 3.0 s of the car directly ahead on the same lap (the leader).
- The chaser makes a real tyre change on lap n (pit-in followed by lower tyre age or a new compound; penalties
  are not stops). Neither car is in or out of the pits on lap n−1, and the leader is not in the pits on lap n.
- The leader makes a real stop on lap m in n+1..n+5. Otherwise the case is "no response" (reported, not modelled).
- Success: the chaser crosses the end of lap m+2 (the lap after the leader's out-lap) ahead of the leader.
- Every lap n−1..m+2 exists for both cars, is green for both, and has no wet tyre.
- Sensitivity variants: 2 s and 5 s windows; check one lap later (m+3); green required only on n−1..m+1.

## Features (undercut/features.py), from other cars in the same race only
- gain_s: median over other cars' stops within 10 laps of n of mean(laps p−2, p−1) − mean(laps p+2, p+3).
- pit_loss_s: median over other cars' green stops of in-lap + out-lap − 2 × median normal lap in p−3..p+4.
- deg_s_per_lap: median over other cars' stints (≥ 6 green laps, no lap 1, no in/out laps) of the OLS slope of lap
  time on tyre age. Net of fuel burn.
- Rule of thumb: works if gain_s × delay − gap_s > 0.

## Models (undercut/models.py)
- Main: logistic regression `success ~ gap_s + delay_cat + tyre_age_diff + compound_change + pit_loss_s +
  deg_s_per_lap`, delay_cat ∈ {1, 2, 3, 4-5}, compound_change ∈ {harder, same, softer}; race-clustered SEs.
- Comparison models: base_rate (training success rate), rule (0/1), rule_cal (logit on the rule's margin), logit.
- Metrics: Brier score, log loss (not meaningful for the 0/1 rule), accuracy at p ≥ 0.5.
- RQ1 on 2023–25. RQ2: leave-one-season-out over 2023–25, then train 2023–25 → test 2026. Differences get 95% CIs
  from 2,000 race-level bootstrap resamples. RQ3 (exploratory): 2026 vs 2023–25 success rate, a 2026 term in the
  model, and calibration of the 2023–25 model on 2026.

## Post-hoc changes
(none yet)
