"""
Generate business-friendly charts for the stakeholder presentation.
Outputs PNGs to docs/deck_assets/ for embedding into the slide deck.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "src" / "data" / "raw" / "P1_Geico_Quote_Data.xlsx"
OUT = ROOT / "docs" / "deck_assets"
OUT.mkdir(parents=True, exist_ok=True)

NAVY = "#0B2545"
TEAL = "#2A9D8F"
AMBER = "#E9C46A"
RUST = "#E76F51"
GREY = "#8D99AE"

COLS = ['ID', 'ZIP', 'TYPE_Q', 'STAT_Q', 'AGE_PH', 'SEX', 'MARR_STAT',
        'HOME_STAT', 'DRIVE_XP', 'EDU_PH', 'DRIVE_RISK', 'AGE_CAR', 'TYPE_CAR',
        'OWN_CAR', 'AVG_MILE_DAILY', 'ANNUAL_MILE', 'NUM_CAR']


def load():
    raw = pd.read_excel(RAW)
    raw.columns = COLS
    raw["accept"] = (raw["STAT_Q"] == "Accepted").astype(int)
    return raw


def _style(ax, title):
    ax.set_title(title, fontsize=15, fontweight="bold", color=NAVY, pad=14)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=NAVY, labelsize=11)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(NAVY)


def bar_rate(raw, col, order, labelmap, fname, title):
    g = raw.groupby(col)["accept"].agg(["mean", "count"]).reindex(order)
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    bars = ax.bar([labelmap.get(o, o) for o in order], g["mean"] * 100,
                  color=TEAL, edgecolor="white", width=0.62)
    ax.axhline(raw["accept"].mean() * 100, color=RUST, ls="--", lw=2,
               label=f"Portfolio avg = {raw['accept'].mean()*100:.0f}%")
    for b, (m, n) in zip(bars, g[["mean", "count"]].values):
        ax.annotate(f"{m*100:.0f}%", (b.get_x() + b.get_width()/2, m*100),
                    ha="center", va="bottom", fontsize=11, fontweight="bold",
                    color=NAVY)
    ax.yaxis.set_major_formatter(PercentFormatter())
    ax.set_ylim(0, 100)
    ax.set_ylabel("Acceptance rate", color=NAVY, fontsize=11)
    ax.legend(frameon=False, fontsize=10, loc="upper right")
    _style(ax, title)
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=160)
    plt.close(fig)


def model_compare(fname="model_comparison.png"):
    # Verified test-set numbers from the pipeline run (held-out 20%).
    models = ["Always say\n'Accept'\n(no model)", "Simple 2-rule\n(analytics)",
              "Logistic\nRegression", "Random\nForest", "Gradient\nBoosting"]
    acc = [68.6, 78.3, 78.1, 73.0, 78.5]
    colors = [GREY, AMBER, TEAL, TEAL, NAVY]
    fig, ax = plt.subplots(figsize=(9.2, 4.9))
    bars = ax.bar(models, acc, color=colors, edgecolor="white", width=0.6)
    for b, v in zip(bars, acc):
        ax.annotate(f"{v:.1f}%", (b.get_x()+b.get_width()/2, v), ha="center",
                    va="bottom", fontsize=12, fontweight="bold", color=NAVY)
    ax.axhline(68.6, color=RUST, ls="--", lw=1.6, alpha=0.7)
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(PercentFormatter())
    ax.set_ylabel("Accuracy on held-out test data", color=NAVY, fontsize=11)
    _style(ax, "Accuracy: simple analytics already matches machine learning")
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=160)
    plt.close(fig)


def confusion(fname="confusion_gb.png"):
    # Gradient boosting confusion matrix from verified pipeline run.
    cm = np.array([[160, 154], [61, 625]])
    labels = ["Predicted\nDeny", "Predicted\nAccept"]
    rows = ["Actually\nDeny", "Actually\nAccept"]
    fig, ax = plt.subplots(figsize=(6.6, 5.2))
    im = ax.imshow(cm, cmap="Blues")
    notes = [["True Deny", "Type I error\n(false alarm)"],
             ["Type II error\n(missed accept)", "True Accept"]]
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i,j]}\n{notes[i][j]}", ha="center", va="center",
                    fontsize=12, fontweight="bold",
                    color="white" if cm[i, j] > 300 else NAVY)
    ax.set_xticks([0, 1], labels, fontsize=11, color=NAVY)
    ax.set_yticks([0, 1], rows, fontsize=11, color=NAVY)
    ax.set_title("Where the model is right and wrong\n(Gradient Boosting, 1,000 test quotes)",
                 fontsize=14, fontweight="bold", color=NAVY, pad=12)
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=160)
    plt.close(fig)


def main():
    raw = load()
    bar_rate(raw, "TYPE_Q", ["AutoOnly", "AutoCombo"],
             {"AutoOnly": "Auto only", "AutoCombo": "Auto + bundle"},
             "rate_quotetype.png", "Bundled quotes accept far more often")
    bar_rate(raw, "SEX", ["F", "M"], {"F": "Female", "M": "Male"},
             "rate_sex.png", "Acceptance differs sharply by applicant gender")
    bar_rate(raw, "DRIVE_RISK",
             ["Very Low", "Low", "Medium", "High", "Very High"], {},
             "rate_driverisk.png", "Driving-risk tier nudges acceptance")
    bar_rate(raw, "TYPE_CAR",
             ["Luxury", "Sedan", "SUV", "Minivan", "Sports"], {},
             "rate_cartype.png", "Vehicle type has a modest effect")
    model_compare()
    confusion()
    print("Charts written to", OUT)


if __name__ == "__main__":
    main()
