"""Figures for the writeup (print-sized for A4), saved as PNG and PDF in results/figures/.

1. fig1_battlefields: how often the submission (P04460) wins each battlefield against
   the expected baseline field, with its troop counts.
2. fig2_arms_race: each leader's P(top 20%) before and after a counterattack round was
   built against it. Values come from results/robustness_selection.csv as committed at
   each stage (selection run, 2,000 contests per scenario), read with `git show`.

Run from the project root:
    .venv/bin/python analysis/figures.py
"""

import io
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
PICK = "P04460"

# Reference palette (dataviz skill, light mode): series-1 blue and its ordinal ramp,
# text inks, recessive grid.
BLUE = "#2a78d6"
BLUE_LIGHT = "#86b6ef"
GRAY = "#8a8984"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"

# Stage commits: the pool before counterattack, then after rounds 1, 2, 3.
STAGES = {"before counterattack": "e047c0f", "after round 1": "f3eccee", "after round 2": "778aa3e", "after round 3": "1c93372"}
LEADERS = [  # (pool id, label, stage where it led untargeted, stage after it was targeted)
    ("P01229", "Original leader P01229\n(Riddler round 2 entry)", "before counterattack", "after round 1"),
    ("P04443", "Round-1 leader P04443\n(counter strategy)", "after round 1", "after round 2"),
    ("P04460", "Round-2 leader P04460\n(our submission)", "after round 2", "after round 3"),
]


def style(ax) -> None:
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_SECONDARY, labelsize=8, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def save(fig, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIGURES / f"{name}.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def figure_battlefields() -> None:
    table = pd.read_csv(RESULTS / f"pick_analysis_{PICK}_battlefields.csv")
    fortresses = table["troops"] >= 20
    fig, ax = plt.subplots(figsize=(6.3, 2.6))
    colors = [BLUE if f else BLUE_LIGHT for f in fortresses]
    bars = ax.bar(table["battlefield"], table["p_win"], width=0.62, color=colors, edgecolor="white", linewidth=1.5)
    for bar, p in zip(bars, table["p_win"]):
        ax.text(bar.get_x() + bar.get_width() / 2, p + 0.02, f"{p:.0%}", ha="center", va="bottom", fontsize=7.5, color=INK_SECONDARY)
    ax.set_xticks(table["battlefield"])
    ax.set_xticklabels([f"{b}\n({t})" for b, t in zip(table["battlefield"], table["troops"])])
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("Battlefield (troops placed)", fontsize=8, color=INK_SECONDARY)
    style(ax)
    ax.set_title(f"How often {PICK} [1, 5, 5, 3, 2, 5, 28, 6, 36, 9] wins each battlefield",
                 loc="left", fontsize=9.5, color=INK, pad=14)
    ax.text(0, 1.03, "Against the expected baseline field. Dark bars: the two 'fortress' battlefields (7 and 9).",
            transform=ax.transAxes, fontsize=7.5, color=INK_SECONDARY)
    save(fig, "fig1_battlefields")


def stage_table(stage: str) -> pd.DataFrame:
    csv_text = subprocess.run(
        ["git", "show", f"{STAGES[stage]}:results/robustness_selection.csv"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout
    return pd.read_csv(io.StringIO(csv_text), keep_default_na=False).set_index("label")


def arms_race_values() -> pd.DataFrame:
    tables = {stage: stage_table(stage) for stage in STAGES}
    rows = []
    for pool_id, label, before, after in LEADERS:
        for measure, column in (("pooled", "p_top20"), ("sophisticated", "p_top20_sophisticated_heavy")):
            rows.append({"id": pool_id, "label": label, "measure": measure,
                         "before": tables[before].loc[pool_id, column], "after": tables[after].loc[pool_id, column]})
    return pd.DataFrame(rows)


def figure_arms_race() -> None:
    values = arms_race_values()
    panels = (("pooled", "All four realistic scenarios (score)"),
              ("sophisticated", "Sophisticated-heavy field (15% counters)"))
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.5), sharey=True)
    labels = [label for _, label, _, _ in LEADERS]
    for ax, (measure, title) in zip(axes, panels):
        data = values[values["measure"] == measure].set_index("label").loc[labels]
        y = range(len(labels))[::-1]
        for yi, (_, row) in zip(y, data.iterrows()):
            ax.plot([row["before"], row["after"]], [yi, yi], color=GRID, linewidth=2, zorder=1)
            ax.scatter(row["before"], yi, s=46, color="white", edgecolor=GRAY, linewidth=1.5, zorder=2)
            ax.scatter(row["after"], yi, s=46, color=BLUE, edgecolor="white", linewidth=1.5, zorder=3)
            ax.text(row["after"] - 0.008, yi - 0.28, f"{row['after']:.1%}", ha="right", va="top", fontsize=7.5, color=INK_SECONDARY)
        ax.set_yticks(list(y))
        ax.set_yticklabels(labels, fontsize=7.5)
        ax.set_xlim(0.69, 1.01)
        ax.set_ylim(-0.7, len(labels) - 0.4)
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        ax.set_title(title, fontsize=8.5, color=INK, loc="left")
        style(ax)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color=GRID, linewidth=0.6)
    axes[0].set_xlabel("P(finishing in the top 20%)", fontsize=8, color=INK_SECONDARY)
    axes[1].set_xlabel("P(finishing in the top 20%)", fontsize=8, color=INK_SECONDARY)
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", markerfacecolor="white", markeredgecolor=GRAY, markersize=6, label="Before: leading, not yet targeted"),
        plt.Line2D([], [], marker="o", linestyle="", markerfacecolor=BLUE, markeredgecolor="white", markersize=7, label="After a counterattack round aimed at it"),
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 1.08), ncol=2, frameon=False, fontsize=7.5, labelcolor=INK_SECONDARY)
    fig.suptitle("Each counterattack round targets the current leaders: who holds up?", x=0.0, y=1.16, ha="left", fontsize=9.5, color=INK)
    save(fig, "fig2_arms_race")
    values.to_csv(FIGURES / "fig2_arms_race_values.csv", index=False)


def main() -> None:
    figure_battlefields()
    figure_arms_race()
    print(f"Saved figures to {FIGURES}")


if __name__ == "__main__":
    main()
