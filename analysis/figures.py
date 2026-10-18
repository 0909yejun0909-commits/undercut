"""Paper figures from results/*.csv. Palette from the dataviz skill's validated reference instance: an ordinal
one-hue blue ramp for the ordered response-delay levels, categorical slots in fixed order otherwise. Every series
is direct-labelled or in a legend, and markers differ, so the figures also read in black and white."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

INK, INK2, GRID = "#1f1f1e", "#5f5e5a", "#e4e3df"
DELAY = {"1": "#86b6ef", "2": "#3987e5", "3": "#1c5cab", "4-5": "#0d366b"}
MARK = {"1": "o", "2": "s", "3": "^", "4-5": "D"}
SLOT1, SLOT2 = "#2a78d6", "#eb6834"
MODEL_NAMES = {"base_rate": "Base rate", "rule": "Rule of thumb (yes/no)", "rule_cal": "Rule of thumb, calibrated",
               "logit": "Logistic model"}
DELAY_NAMES = {"1": "next lap", "2": "2 laps", "3": "3 laps", "4-5": "4-5 laps"}
LEGEND_DELAY = {"1": "the next lap", "2": "2 laps later", "3": "3 laps later", "4-5": "4-5 laps later"}
NUDGE = {"1": -5, "2": 4, "3": 4, "4-5": -4}

plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.axisbelow": True, "legend.frameon": False})


def read(name):
    return pd.read_csv(f"results/{name}.csv", comment="#", dtype={"delay_cat": str, "level": str})


def fig1_lookup():
    d = read("rq1_lookup")
    fig, ax = plt.subplots(figsize=(6, 3.8))
    for k, g in d.groupby("delay_cat", sort=False):
        ax.plot(g.gap_s, g.p, color=DELAY[k], lw=2, marker=MARK[k], ms=6, label=f"Leader stops {LEGEND_DELAY[k]}")
        ax.annotate(DELAY_NAMES[k], (g.gap_s.iloc[-1], g.p.iloc[-1]), xytext=(7, NUDGE[k]), textcoords="offset points",
                    va="center", color=INK, fontsize=9)
    ax.axhline(0.5, color=INK2, lw=0.8, ls=":")
    ax.set(xlabel="Gap to the car ahead before the stop (s)", ylabel="Chance the undercut works", ylim=(0, 1),
           xlim=(0.4, 3.45))
    ax.legend(loc="lower left", fontsize=8)
    return fig


def fig2_rates():
    d = read("rq1_rates")
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.2), sharey=True)
    for ax, group, title, names in ((axes[0], "delay", "Leader responds", DELAY_NAMES),
                                    (axes[1], "gap", "Gap before the stop", None)):
        g = d[d.group == group].reset_index(drop=True)
        x = range(len(g))
        ax.errorbar(x, g.rate, yerr=[g.rate - g.ci_lo, g.ci_hi - g.rate], fmt="o", color=SLOT1, ms=7, lw=1.5,
                    capsize=3)
        ax.set_xticks(list(x), [f"{names[v] if names else v}\nn={n}" for v, n in zip(g.level, g.n)])
        ax.set_xlim(-0.5, len(g) - 0.5)
        ax.set_title(title, fontsize=10, color=INK)
        ax.axhline(0.5, color=INK2, lw=0.8, ls=":")
    axes[0].set(ylabel="Share of undercuts that worked", ylim=(0, 1))
    return fig


def fig3_models():
    loso, t26 = read("rq2_loso"), read("rq2_2026")
    loso = loso[loso.held_out == "pooled"].set_index("model")
    t26 = t26.set_index("model")
    order = ["base_rate", "rule", "rule_cal", "logit"]
    fig, ax = plt.subplots(figsize=(6, 2.8))
    y = range(len(order))
    ax.plot(loso.loc[order, "brier"], y, "o", color=SLOT1, ms=8, label="Held-out seasons 2023-25")
    ax.plot(t26.loc[order, "brier"], y, "s", color=SLOT2, ms=8, label="2026 (trained on 2023-25)")
    ax.set_yticks(list(y), [MODEL_NAMES[m] for m in order])
    ax.invert_yaxis()
    ax.set_xlabel("Brier score (lower is better)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=8)
    return fig


def fig4_mechanism():
    m = read("mechanism").set_index("success")
    fig, ax = plt.subplots(figsize=(6, 2.8))
    rows = [(1.0, "Undercut worked"), (0.0, "Undercut failed")]
    for i, (s, label) in enumerate(rows):
        stop, tyre = m.at[s, "mech_stop_diff"], m.at[s, "mech_tyre_phase"]
        ax.barh(i - 0.18, stop, height=0.34, color=SLOT2, label="Pit stops (leader's loss minus chaser's)" if i == 0 else None)
        ax.barh(i + 0.18, tyre, height=0.34, color=SLOT1, label="Tyre phase (fresh vs old laps)" if i == 0 else None)
        for v, yy in ((stop, i - 0.18), (tyre, i + 0.18)):
            ax.annotate(f"{v:+.1f} s", (v, yy), xytext=(4 if v >= 0 else -4, 0), textcoords="offset points",
                        ha="left" if v >= 0 else "right", va="center", color=INK, fontsize=8)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks([0, 1], [r[1] for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel("Mean time gained by the chaser (s)")
    ax.set_xlim(-3, 6)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=8)
    return fig


def fig5_2026():
    d = read("rq3_by_delay")
    fig, ax = plt.subplots(figsize=(6, 3))
    w = 0.36
    for j, (flag, label, color) in enumerate(((0, "2023-25", SLOT1), (1, "2026 (exploratory)", SLOT2))):
        g = d[d.is2026 == flag].set_index("delay_cat").reindex(list(DELAY))
        x = [i + (j - 0.5) * w for i in range(len(DELAY))]
        ax.bar(x, g["mean"], width=w - 0.04, color=color, label=label)
        for xi, (n, v) in zip(x, zip(g["size"], g["mean"])):
            ax.annotate(f"n={int(n)}", (xi, v), xytext=(0, 3), textcoords="offset points", ha="center",
                        color=INK2, fontsize=7.5)
    ax.set_xticks(range(len(DELAY)), [DELAY_NAMES[k] for k in DELAY])
    ax.set(xlabel="Leader responds", ylabel="Share that worked", ylim=(0, 1.12))
    ax.legend(loc="upper left", fontsize=8)
    return fig


def fig6_teams():
    d = read("rq4_teams")
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.6))
    for ax, role, title, color, mark in ((axes[0], "chaser", "Team making the undercut", SLOT1, "o"),
                                         (axes[1], "leader", "Team being undercut", SLOT2, "s")):
        g = d[d.role == role].sort_values("rate").reset_index(drop=True)
        y = range(len(g))
        ax.errorbar(g.rate, y, xerr=[g.rate - g.ci_lo, g.ci_hi - g.rate], fmt=mark, color=color, ms=6, lw=1.2,
                    capsize=2)
        ax.set_yticks(list(y), [f"{t} ({n})" for t, n in zip(g.team, g.n)], fontsize=8)
        ax.axvline(0.5, color=INK2, lw=0.8, ls=":")
        ax.set(xlim=(0, 1), xlabel="Share that worked")
        ax.set_title(title, fontsize=10, color=INK)
    return fig


def main():
    os.makedirs("figures", exist_ok=True)
    for name, f in (("fig1_lookup", fig1_lookup), ("fig2_rates", fig2_rates), ("fig3_models", fig3_models),
                    ("fig4_mechanism", fig4_mechanism), ("fig5_2026", fig5_2026), ("fig6_teams", fig6_teams)):
        fig = f()
        fig.tight_layout()
        fig.savefig(f"figures/{name}.pdf")
        fig.savefig(f"figures/{name}.png", dpi=110)
        plt.close(fig)
        print(name)


if __name__ == "__main__":
    main()
