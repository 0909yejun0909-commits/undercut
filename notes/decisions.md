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
- 2026-10-18, attempts.py: fixed column dtypes so that concatenating with empty (wet) races keeps `responded`
  boolean. This is a bug fix; the definitions are unchanged.
- 2026-10-18, features.py (descriptive only, never a model input): added mech_swing (gap on lap n−1 minus gap at
  the check lap), mech_stop_diff (leader's stop loss minus chaser's) and mech_tyre_phase (= swing − stop_diff). The
  spot-check showed that in-lap vs out-lap times depend on each circuit's pit-lane layout, so the planned
  in-lap/out-lap columns don't separate the stop from the tyre effect.
- 2026-10-18, rq1: the no-response group is reported as a count only. "Chaser ahead before the leader stops" is 0%
  by construction, since the chaser has just lost about 20 s in the pits.
- 2026-10-18, reporting: the degradation odds ratio is also shown per 0.01 s/lap of tyre age (the observed range is
  about −0.08 to 0.07 s/lap). Same fit, rescaled.
- 2026-10-18, rq2: added results/rq2_rule.csv (the raw rule's hit rates and the calibrated rule's break-even margin,
  in-sample on 2023–25). This is descriptive; the RQ2 comparison is unchanged.
- 2026-10-18, after the final review (none of these changes the attempt definition, the models or the metrics):
  rq2_rule.csv adds a race-bootstrap CI for the break-even margin, plus the break-even with the out-lap left out
  (gain × (delay − 1)). The mechanism table averages all parts over the same rows (attempts where both stops' loss
  is measurable) and adds medians. The lookup chart holds the compound choice at the most common level ("harder")
  instead of "same".
- 2026-10-18, NEW exploratory RQ4 (asked for by the author after the paper was drafted; not pre-registered):
  which teams and cars close the gap? undercut/cars.py + analysis/rq4.py, results/rq4_*.csv. Team names come from
  the cached FastF1 results (analysis/build_teams.py → data/teams.parquet), with renamed teams merged
  (AlphaTauri/RB/Racing Bulls; Alfa Romeo/Kick Sauber/Audi). Car traits are chaser-vs-leader differences, each the
  team's median over the OTHER dry races of the same season, measured relative to each race's field: race pace,
  own tyre-wear slope, and stop loss (pit crew). A first version took pace from the attempt's own race; it showed a
  strong pace effect, but that measure sees the outcome (a successful chaser then runs in clean air), so it was
  replaced before anything was reported. A second pace measure uses only that race's laps before lap n−1. Team
  effects are tested by likelihood ratio against a gap + delay model. Six tests in all, none pre-specified.
- 2026-10-18, NEW exploratory RQ5 (asked for by the author; not pre-registered): can an undercut be predicted before
  the chaser stops? undercut/predict.py, analysis/rq5.py → results/rq5_*.csv, analysis/predict.py (command-line
  tool). Compared on leave-one-season-out and 2026 as in RQ2: base rate, gap only, the pre-registered features
  minus the response delay ("prestop"), gap + delay, and the full model. Gap only was the best pre-stop model; the
  prestop model was worse (overfits, overconfident in calibration). Gap + delay beat the full model. The tool uses
  gap only before the stop and gap + delay for each response. The choice was made by looking at the
  leave-one-season-out scores, so those scores slightly flatter the chosen models; 2026 is the cleaner check.
  A soft-tyre look (chaser coming off softs: 75% of 24 worked, OR 3.1 after gap + delay) was computed ad hoc
  and not saved or used.
- 2026-10-18, RQ5b (exploratory; the author asked for a better model): candidates fixed before scoring were gap
  only and gap + delay each trained on a wider attempt set (gap up to 5 s, yellow-flag laps allowed: 252 vs 154
  attempts in 2023–25), and ridge logistic regressions with every factor (plus fresh-tyre gain, race progress and
  coming off softs), penalty chosen by leave-one-season-out inside the training seasons only. analysis/rq5b.py →
  results/rq5b_*.csv. None beat the simple models; ridge was worse, wide training tied. A second round, added
  after seeing the first, tried a two-step model (predict the leader's response from gap and leader tyre age,
  then success from gap + delay): 0.210 vs 0.209 for gap only, so not kept or saved. The tool is unchanged.
- 2026-10-18, RQ5c (exploratory; the author asked about variables not yet considered): ten pre-stop factors
  (undercut/context.py; weather via analysis/build_weather.py → data/weather.parquet), each added alone to gap
  only and to gap + delay. analysis/rq5c.py → results/rq5c_screen.csv. Bar fixed before scoring: held-out Brier
  improvement with CI excluding 0, and no worse on 2026. None passed. In-sample, closing rate had OR 3.1 per s/lap
  (p = 0.03) and street circuit 0.47 (p = 0.10); with 20 tests about one p < 0.05 is expected by chance.
- 2026-10-18, tyre hardness (the author asked for the actual Pirelli compounds): reference/compounds.csv holds the
  hard/medium/soft C-numbers for all 87 races, parsed from each race's English Wikipedia article
  (analysis/build_compounds.py; the sentence and URL are kept as evidence). 16 races were checked against
  Pirelli press releases or news sites (reference/compounds_checks.csv; all agree, enforced by a test). The parser
  was written before its tests (tests/test_compounds.py added after). Five hardness factors (new tyre, old tyre,
  step, leader's old tyre, race's softest) were screened with the RQ5c bar → results/rq5c_tyres.csv: none passed,
  and none had an in-sample effect. The ad hoc "off softs" hint does not survive: the old tyre's real hardness
  has OR 1.00.
- 2026-10-18, what an undercut is worth, running position and circuits (exploratory; the author asked "what about
  points and position?" and "what about circuits?", after quick scratch looks at both). undercut/stakes.py,
  analysis/stakes.py → results/stakes_*.csv, analysis/circuits.py → results/circuit_*.csv. Finishing positions come
  from the cached official classification; a disqualified car is unclassified (two failed 2023–25 attempts: RUS
  2024_14, HAM 2025_02; the scratch version had counted them as P19–20). Points use 25…1 with no fastest-lap point.
  The points swing is within the pair, for attempts where the leader ran in the top ten on lap n−1. The position
  groups (P1–3, P4–6, P7–10, P11+) and the top-six indicator were chosen after the scratch look, so the top-six
  odds ratio (0.41, p = 0.06) is a hint. Circuit = event name (2023–25 only, where event and circuit agree; 2026's
  "Spanish" GP is Madrid). Circuits with < 4 attempts are pooled as "other" for the tests; held-out predictions
  map an unseen circuit to "other". The circuit improves the in-sample fit (LR p = 0.008, permutation p = 0.015)
  but not held-out prediction (Brier 0.200 → 0.232, CI of the difference includes 0).
