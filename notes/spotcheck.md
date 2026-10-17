# Spot-check of 10 random main attempts (seed 0), 2026-10-18

For each: lap times, pit flags, compounds and tyre ages for laps n−2..check lap, plus chaser−leader time at the line
on lap n−1 and at the check lap (negative = chaser ahead). Checked by eye: chaser stopped first, gap matches,
order at the check lap matches `success`.

| Race | Chaser / leader | Gap | Stops (chaser, leader) | Check-lap gap | success | Verdict |
|---|---|---|---|---|---|---|
| 2024_03 | MAG / BOT | 1.96 | L7, L8 | −26.25 | 1 | OK. Leader's stop was slow (126.6 s in-lap), so it was decided in the pits |
| 2025_21 | BEA / HAD | 1.29 | L17, L18 | −3.55 | 1 | OK |
| 2024_05 | HUL / BOT | 0.60 | L8, L9 | −1.26 | 1 | OK |
| 2024_03 | TSU / PER | 2.14 | L9, L14 | +1.72 | 0 | OK. Delay of 5, low other-car gain (0.13 s) |
| 2025_04 | LAW / ALO | 1.70 | L14, L16 | +1.30 | 0 | OK. ALO went medium to medium (TyreLife reset) |
| 2025_04 | DOO / TSU | 0.89 | L9, L11 | −4.66 | 1 | OK |
| 2023_01 | ALB / OCO | 2.54 | L11, L12 | −0.80 | 1 | OK. ALB went soft to new soft |
| 2026_06 | RUS / HAD | 0.50 | L31, L32 | −8.34 | 1 | OK |
| 2024_24 | PIA / ALO | 1.97 | L32, L37 | +3.18 | 0 | OK. Hard to new hard |
| 2024_17 | PER / PIA | 2.06 | L13, L15 | +0.80 | 0 | OK |

All 10 are OK. Note: where the stop's time loss shows up (in-lap vs out-lap) depends on the circuit's pit-lane
layout. The success check uses the order at the end of the lap after the leader's out-lap, so this doesn't affect
it.
