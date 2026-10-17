# When Does the Undercut Work? — Design

Date: 2026-10-17. Status: draft, waiting for the author's approval.

## Intent

The undercut is F1's best-known strategy move. A car stuck behind another pits first, then uses its fresh tyres to make up the gap while the car ahead is still on old tyres. Commentators treat it as a known quantity ("the undercut is worth about two seconds here"), but nobody publishes how often it actually works or what decides it.

The paper asks one fan-facing question: **when does the undercut work?** It answers with public timing data from 2023–2026.

The author chose this topic on 2026-10-17, over three other ideas (rain masters, driver vs. car, the safety-car lottery). It was already flagged as the paper after Battery Battles.

**Why this topic fits the Wharton Sports Analytics Journal (WSAJ):**
- The Fall 2025 WSAJ F1 paper lists pit strategy, tyre strategy and degradation as future work.
- No WSAJ paper has studied pit strategy.
- It uses the pattern recent accepted papers share: one sharp question, a benchmark to beat, large public data, standard methods, out-of-sample testing, and practical takeaways (see `notes/past_papers_analysis.md` in the fantasy repo).

## Research questions

- **RQ1 (main): how often does the undercut work, and what decides it?** Estimate the success rate and how it depends on:
  - the gap before the stop;
  - how many laps the car ahead waits before it responds;
  - tyre age and compound for both cars;
  - the circuit's pit-lane time loss and tyre wear.
- **RQ2 (benchmark): does a model beat the rule of thumb?** Compare three ways of predicting whether an attempt works, on held-out seasons:
  - the rule of thumb (defined below);
  - the overall success rate;
  - the RQ1 logistic model.
- **RQ3 (exploratory): did the 2026 rules change the undercut?** 2026 brought new cars, new tyres and changed energy use. Compare the 2026 success rate and model fit with 2023–25. This sample is small, so the paper labels RQ3 as exploratory.

## Definitions

**Undercut attempt.** Car B (the chaser) and car A (the leader) form an attempt when all of these hold:
1. On the lap before B's stop (lap n−1), B is directly behind A in race position, and B crosses the line within **3.0 s** of A. Sensitivity checks use 2 s and 5 s.
2. B pits on lap n. A did not pit on lap n−1 or n.
3. A pits within the next **5 laps**. If A doesn't respond within 5 laps, the case goes into a separate "no response" group, which is reported but not modelled.
4. Every lap from n−1 to the check lap (below) is green: no safety car, virtual safety car or red flag.
5. The race is dry, and neither car is on intermediate or wet tyres.
6. Neither car retires before the check lap.

**Success.** B is ahead of A at the end of the lap after A's out-lap, once both stops are done. A sensitivity check moves this one lap later.

**Rule of thumb (the benchmark).** The undercut is predicted to work if the fresh-tyre gain beats the gap:

> fresh-tyre gain per lap × laps before A responds > gap before the stop

- The fresh-tyre gain per lap is the median lap-time difference between fresh and old tyres. It's measured from other cars in the same race, never from A or B, so the benchmark doesn't use the outcome.
- This is the commentators' version, written down so it can be tested. The builder settles the exact estimator and states it in the paper.

**Leakage rule.** A feature may only use information from the attempt's own cars if it's known before the check lap. The response delay counts, because the paper asks how success depends on how long the leader waits. The out-lap time measured on A or B does not count: it goes into a mechanism table, not the prediction model.

## Data

- **Laps:** cached FastF1 data for 2023 R1–22, 2024 R1–24, 2025 R1–24 and 2026 R1–17, in `~/Projects/battery-battles/data/cache` (9.4 GB).
  - Tested on 2026-10-17: it loads in FastF1 **offline mode**.
  - It includes `Stint`, `TyreLife`, `Compound`, `PitInTime`, `PitOutTime`, `Position`, the lap-end `Time`, and `TrackStatus`.
- **Gaps:** measured from lap-end crossing times. Battery Battles validated this method: lap-end swaps have 0.978 recall against X/Y position data. The method is also immune to the ±0.5 s X/Y jitter.
- **Wet races:** `info.json` has a `wet` flag, built by the Battery Battles download.
- **The FastF1 data lock:** since about 2026-10-15 the FastF1 archive returns 401 errors (FastF1 issue #978). So:
  - Only cached races can be used.
  - Rounds 18 and later of 2026, and anything before 2023, need either a fix from FastF1 or **OpenF1** (`/v1/laps`, `/v1/pit`, `/v1/stints`, `/v1/position`), which still answered on 2026-10-17.
  - The paper does not depend on either source.

**Size, from a rough throwaway count on 2026-10-17** (strict 3 s window, all laps green; not in the repo):

| | Attempts | Success rate |
|---|---|---|
| 2023 | 42 | 0.50 |
| 2024 | 50 | 0.52 |
| 2025 | 45 | 0.44 |
| 2026 (R1–17) | 19 | 0.32 |
| **Total** | **156** | |

By how long A waited before pitting:

| A waited | Attempts | Success rate |
|---|---|---|
| 1 lap | 90 | 0.34 |
| 2 laps | 35 | 0.49 |
| 3 laps | 18 | 0.72 |
| 4 laps | 13 | 0.92 |

About 150–300 attempts is a modest sample (the 5 s window and the relaxed green rule add more). This constrains the methods: few predictors, honest confidence intervals, no deep learning.

## Methods

- **RQ1:** logistic regression of success on:
  - the gap;
  - the response delay (as a categorical variable);
  - the tyre-age difference;
  - the compound change (for example, B going from medium to hard);
  - the circuit's pit loss and its tyre-wear level, both measured from other cars.

  Standard errors are clustered by race. Report odds ratios with confidence intervals, plus a plain-language table such as "with a 1.5 s gap and a response next lap, the undercut works X% of the time."
- **RQ2:** leave-one-season-out over 2023–25: fit on two seasons, test on the third. Then fit on 2023–25 and test on 2026 as a test under new rules. Metrics: Brier score, log loss and accuracy, with bootstrap confidence intervals (resampling races) for the differences between models.
  - Optional: LightGBM as a check that the simple model isn't missing much. The Battery Battles finding, where the simple model beat LightGBM, is a likely outcome here too.
- **RQ3:** compare the 2026 success rate with 2023–25 (with a bootstrap CI), add a 2026 term to the RQ1 model, and check how well the 2023–25 model predicts 2026.
- **Mechanism table (descriptive):** for successful vs. failed attempts, show the out-lap gain, the in-lap loss, and whether B rejoined in traffic.

## Paper

- WSAJ format: LaTeX, double-spaced, line numbers, APA citations, abstract under 250 words.
- Sections: IMRaD, with separate Limitations and Reproducibility subsections and a GitHub link.
- Target length: about 4,800 words and about 8 references.
- Cite the Fall 2025 WSAJ F1 paper's future-work request, and any published pit-strategy work found during the literature check.
- Practical takeaway for fans and teams: a lookup chart of success probability by gap and response delay.
- The author writes the Discussion's interpretations and approves the research questions as worded (see the user-profile memory).

## Goals, non-goals, anti-goals

**Goals:**
- A clean, reproducible estimate of undercut success.
- A tested benchmark comparison.
- A 2026 comparison.

**Non-goals:**
- The overcut, as its own question. It goes in Future Work, with a short note on what happens in the "no response" cases.
- Safety-car stops.
- Sprints, which have no mandatory stop.
- Building a full race-strategy simulator.

**Anti-goals** (these would make the paper fail even if the code runs):
- Leaking outcome information into the prediction features.
- Presenting RQ3 as a firm finding when it rests on about 20 attempts.
- Methods above the level of the recent high-school-authored WSAJ papers.

## Constraints and schedule

- **Battery Battles comes first** until its Nov 16 submission, including the fallback for the FastF1 lock before Oct 26. No undercut work may slow it down.
- **The fantasy paper is also aimed at the spring issue.** Whether one author can put two papers in one issue isn't settled. My call: build the undercut paper after Nov 16, and choose its target issue once the spring deadline and rules are checked.
- Build effort: about 4–6 weeks of sessions.
  - Data and attempt detection: about 1 week, reusing `battles/f1data.py` and `battles/lapend.py`.
  - Models: about 1 week.
  - Paper: about 2 weeks.

## Repo

- A new repo at `~/Projects/undercut`, created after this spec is approved. It reads the Battery Battles cache by path and doesn't copy the 9.4 GB.
- It copies (doesn't import) the few loaders it needs, so the frozen Battery Battles repo stays untouched.
- It needs `uv`, pytest, and the same tools as Battery Battles.

## Calls I made (the author can override any of these)

- The 3 s attempt window, the 5-lap response limit, and checking success one lap after A's out-lap.
- Dry races and green-flag laps only.
- The rule of thumb as the benchmark. I found no betting or official pit-strategy benchmark that's public.
- Logistic regression as the main model, with LightGBM optional.
- Building after Nov 16, with the target issue decided later.
- A separate repo, with Battery Battles left untouched.

## Left to the builder

- The exact estimator for the fresh-tyre gain and the tyre-wear level.
- How to handle lapped traffic when B rejoins.
- The figure design (following the dataviz skill).
- The test layout.
