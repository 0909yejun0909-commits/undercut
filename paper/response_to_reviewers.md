---
title: "Response to Reviewers"
subtitle: "When Does the Undercut Work? (Yejun Kim)"
---

Thank you for the careful review. The question about the 314 excluded cases led to the most useful addition to the paper: following them up showed that the main result holds well beyond the five-lap window, and it turned up the 72 cases where an early stop almost never worked. Each comment is answered below, with the changes made. Section and page numbers refer to the revised manuscript. Sections 3.1–3.8 keep the numbers the reviewer used, and the new analysis is the new Section 3.9. All new results come from two scripts added to the public repository (`analysis/no_response.py` and `analysis/tyre_gain_check.py`) and are marked as exploratory, because they were added after peer review rather than planned in advance.

---

## Comment 1: Excluding attempts where the leader does not respond (Sections 2.2 and 3.1)

> *"...the author identifies 314 additional cases where the leader did not pit within that window... It would be helpful to explain more about these excluded cases or include an additional analysis showing whether the conclusions change when considering a wider range of pit strategies."*

I agree that this was the paper's largest gap. The new Section 3.9 follows up all 286 of these cases from 2023–25 (the other 28 are from 2026). They fall into two very different groups:

- **The leader never stopped again (72 cases).** These usually came late in the race (median lap 37), and the leader ran to the flag on its old tyres. The chaser finished ahead in only **2 of 72**. Here the chaser's stop was an extra stop that the leader did not need to answer, so these cases are not really undercuts.
- **The leader stopped later (214 cases)**, typically 12 laps after the chaser. I kept the 150 clean comparisons: every lap green, no second stop by the chaser in between, and a chaser stop after lap 3, matching the rules for the main attempts. In these the chaser came out ahead **69%** of the time (74% when the leader waited 6–10 laps, 64% when it waited 11 or more). With the safety-car cases put back in, the rate drops to 60%, because a safety car gives the leader a cheaper stop.

**Do the conclusions change? No.** Among the late responses, a larger gap still lowered the chaser's chances (odds ratio 0.58 per second, 95% CI 0.39–0.87). With these cases added to the main attempts (304 in total), the gap's odds ratio is 0.37 (CI 0.25–0.55), against 0.25 for the main attempts alone, and a late response is linked to higher odds of success than a response within five laps (odds ratio 1.85, CI 1.03–3.35). The main finding, that a small gap and a slow response help the chaser, therefore holds over a wider range of strategies, though the gap matters somewhat less when the leader waits a long time.

**Changes:** new Section 3.9 and a matching methods paragraph ("No-response cases") in Section 2.5. Sections 2.2 and 3.1 now point to Section 3.9; the Limitations, Conclusion and the list of post-hoc analyses in the pre-registration paragraph have been updated.

---

## Comment 2: Applying the results to actual race strategy (Sections 3.5 and 3.8)

> *"...the leader's response time isn't known when the chasing team makes its pitstop decision... I think this distinction could be emphasized more throughout the paper. It could be interesting to see more discussion how teams could actually use these results..."*

I agree, and the distinction is now made in four places:

1. **Introduction:** a new sentence says that most results describe an undercut *after* the leader has responded, which the chasing team cannot know when it decides, and points to RQ5 and Section 3.9 as the pit-wall view.
2. **RQ1 lookup chart (Section 3.2):** the paragraph now says that the chart answers a question asked after the leader has pitted, and points to the pre-stop results.
3. **Section 3.9:** a new number for the decision as a team actually faces it. Across **all 376 stops** from within 3 s in 2023–25, whether or not the leader responded, the chaser ended up ahead 48% of the time: **64% from within 1 s, 47% from 1–2 s and 29% from 2–3 s**.
4. **Discussion:** a new paragraph on how a team could use the results. Before the stop, the gap-only model is the right guide (70% at 1.0 s, 53% at 1.5 s, 36% at 2.0 s, if the leader responds within five laps). The no-response analysis adds a practical warning: against a rival that can run to the flag, an early stop almost never got the chaser ahead (2 of 72 cases). Taken together, against a rival that still had to stop the odds were about even or better from within 1.5 s, whereas from 2 s or more success relied on the rival waiting. The paper also notes that teams, who know how long their rival's tyres can last, can tell these situations apart better than public data can.

---

## Comment 3: Limited sample size for 2026 (Section 3.6)

> *"...it might be worthwhile to emphasize that these results are preliminary and could be revisited once more race data becomes available."*

Agreed. The 2026 results are now called preliminary in the abstract ("a preliminary and inconclusive drop"), at the end of the first paragraph of Section 3.6, in the Discussion, in the Limitations and in the Conclusion. Each of these says that they come from 22 attempts in the first 17 rounds of new rules and should be revisited once the full 2026 season and later seasons are available. The abstract stays under the 250-word limit.

---

## Comment 4: Tire performance (Sections 2.3 and 4)

> *"...tire performance can vary depending on the driver, compound, and race conditions. A little more discussion [of how] these differences might affect the prediction could be useful."*

I added both a measurement and a discussion.

- **How big is the problem?** The new check (end of Section 3.3) compares the rule's race-average estimate of the fresh-tyre gain with the time each chaser actually gained on track outside the stops. The two are clearly related (correlation 0.62), but for a single attempt the estimate was **off by more than 2 s in a third of cases (34%)**. That is more than the typical starting gap of 1.6 s. In the median attempt the chaser gained 0.6 s less than estimated. Some of this miss is traffic and the out-lap rather than the tyres, which the paper notes. The miss did not clearly depend on the compound change.
- **Section 2.3** now says that the gain is one number for the whole race, which knows nothing about the driver, the car, the compound or how the track changes, so a given chaser can gain more or less than it.
- **Discussion:** a new passage discusses where the variation may come from: drivers who are gentle on tyres, cars that warm them up quickly, a compound that suits the track, and a track that rubbers in and changes temperature. None of the measures I could test with public data (a car's season-long tyre wear, the compound change, Pirelli hardness, track temperature) explained it, so for an outsider it behaves like noise. That helps explain why a probability works better than a yes/no call. A team's own tyre data could narrow it.
- **Limitations** now say that the race-level features describe the race, not the specific tyres, drivers or cars involved.

---


Yejun Kim
