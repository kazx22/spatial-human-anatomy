"""
threshold_sensitivity.py — evaluates the weighted pseudo-gold framework across
a range of voting thresholds and produces publication-quality figures.

The threshold controls how much combined F1 weight a span must accumulate
across models before it enters the pseudo-gold set.  Lower thresholds admit
more spans (higher recall, lower precision); higher thresholds are more
conservative.

This script compares each threshold's pseudo-gold set against human gold to
show where precision and recall trade off and to confirm that the operating
threshold (1.5) sits at the F1 peak.

Outputs:
  figure/threshold_sensitivity_curve.png  — combined P/R/F1 vs threshold
  figure/sensitivity/threshold_<t>_bars.png — per-threshold P/R/F1 bar chart

Prerequisite: run `python -m src.candidate_gold` first so the sensitivity
JSONL files exist under data/gold/sensitivity/.

Run from the project root:
    python -m src.threshold_sensitivity
"""

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from seqeval.metrics import precision_score, recall_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_jsonl, group_by_row, span_to_bio

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "bc5cdr"
SENS_DIR = PROJECT_ROOT / "data" / "gold" / "sensitivity"
FIG_DIR = PROJECT_ROOT / "figure"

DOCS_FILE = DATA_DIR / "bc5cdr_train_docs.jsonl"
GOLD_FILE = DATA_DIR / "bc5cdr_train_entities.jsonl"

# Must match the sweep in candidate_gold.py
THRESHOLDS = [0.45, 0.70, 0.90, 1.20, 1.50, 1.80, 2.00]

# ------------------------------------------------------------------
# Colour palette — accessible, print-safe, matches graph.py
# ------------------------------------------------------------------
COL_PRECISION = "#1F6FB2"  # strong blue
COL_RECALL = "#D65A4A"  # coral red
COL_F1 = "#3D9B35"  # bright green
GRID_COL = "#CCCCCC"

plt.rcParams.update(
    {
        "font.size": 12,
        "axes.edgecolor": "#444444",
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.color": GRID_COL,
        "grid.linewidth": 0.6,
        "grid.alpha": 0.5,
    }
)


def build_gold_map(docs, gold_by_row):
    """
    Pre-build per-document (tokens, labels) pairs from human gold.

    Caching this avoids repeating span_to_bio for every threshold iteration.
    """
    gold_map = {}
    for doc in docs:
        row_id = doc["row_id"]
        text = doc["full_text"]
        entities = gold_by_row.get(row_id, [])
        tokens, labels = span_to_bio(text, entities)
        gold_map[row_id] = (tokens, labels)
    return gold_map


def evaluate_pseudo_gold(docs, gold_map, pseudo_by_row):
    """
    Score a pseudo-gold set against pre-built human gold BIO sequences.

    Documents where tokenisation doesn't match are skipped.  Returns
    (precision, recall, f1) as floats; returns (0, 0, 0) if no valid
    documents remain.
    """
    y_true, y_pred = [], []
    for doc in docs:
        row_id = doc["row_id"]
        text = doc["full_text"]
        gold_tokens, gold_labels = gold_map[row_id]
        pseudo_entities = pseudo_by_row.get(row_id, [])
        pred_tokens, pred_labels = span_to_bio(text, pseudo_entities)
        if pred_tokens != gold_tokens:
            continue
        if len(pred_labels) != len(gold_labels):
            continue
        y_true.append(gold_labels)
        y_pred.append(pred_labels)
    if not y_true:
        return 0.0, 0.0, 0.0
    return (
        precision_score(y_true, y_pred),
        recall_score(y_true, y_pred),
        f1_score(y_true, y_pred),
    )


def plot_combined_curve(thresholds, precisions, recalls, f1s, output_file):
    """
    Plot precision, recall, and F1 as a function of the voting threshold.

    Annotates the F1 peak with a vertical dashed line and a text label so
    the operating threshold choice is self-evident in the figure.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        thresholds,
        precisions,
        marker="o",
        markersize=7,
        linewidth=2.2,
        color=COL_PRECISION,
        label="Precision",
    )
    ax.plot(
        thresholds,
        recalls,
        marker="s",
        markersize=7,
        linewidth=2.2,
        color=COL_RECALL,
        label="Recall",
    )
    ax.plot(
        thresholds,
        f1s,
        marker="^",
        markersize=8,
        linewidth=2.2,
        color=COL_F1,
        label="F1-score",
    )

    peak_idx = int(np.argmax(f1s))
    ax.axvline(
        thresholds[peak_idx],
        color="#999999",
        linestyle="--",
        linewidth=1.2,
        alpha=0.7,
    )
    ax.annotate(
        f"Peak F1 = {f1s[peak_idx]:.3f}\n(threshold = {thresholds[peak_idx]})",
        xy=(thresholds[peak_idx], f1s[peak_idx]),
        xytext=(thresholds[peak_idx] + 0.18, f1s[peak_idx] - 0.18),
        fontsize=10,
        color="#333333",
        arrowprops=dict(arrowstyle="->", color="#666666", lw=1),
    )

    ax.set_xlabel("Weighted Voting Threshold", fontsize=13)
    ax.set_ylabel("Score", fontsize=13)
    ax.set_ylim(0, 1.05)
    ax.set_title("Pseudo-Gold Threshold Sensitivity vs Human Gold", fontsize=14, pad=14)
    ax.legend(frameon=True, edgecolor="#CCCCCC", fontsize=11, loc="center right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(f"  Saved combined curve -> {output_file}")


def plot_per_threshold_bars(threshold, p, r, f, output_file):
    """
    Bar chart showing precision, recall, and F1 at a single threshold.

    Produced for every threshold in the sweep so individual thresholds can
    be inspected separately from the combined curve.
    """
    metrics = ["Precision", "Recall", "F1-score"]
    values = [p, r, f]
    colors = [COL_PRECISION, COL_RECALL, COL_F1]
    x = np.arange(len(metrics))

    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.bar(x, values, width=0.55, color=colors, edgecolor="#333333", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=12)
    ax.set_ylim(0, 1.15)  # extra headroom so value labels don't hit the title
    ax.set_ylabel("Score", fontsize=13)
    ax.set_title(
        f"Pseudo-Gold vs Human Gold (threshold = {threshold})", fontsize=13, pad=18
    )
    for i, v in enumerate(values):
        ax.text(i, v + 0.025, f"{v:.4f}", ha="center", fontsize=11, color="#222222")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(f"  Saved bar chart (threshold {threshold}) -> {output_file}")


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(FIG_DIR / "sensitivity", exist_ok=True)

    print("Loading documents and human gold...")
    docs = load_jsonl(DOCS_FILE)
    gold_entities = load_jsonl(GOLD_FILE)
    gold_by_row = group_by_row(gold_entities)
    gold_map = build_gold_map(docs, gold_by_row)
    print(f"Loaded {len(docs)} documents.\n")

    precisions, recalls, f1s = [], [], []

    print("=" * 60)
    print("THRESHOLD SENSITIVITY EVALUATION")
    print("=" * 60)
    print(f"{'Threshold':>10}  {'Precision':>10}  {'Recall':>10}  {'F1':>10}")

    for t in THRESHOLDS:
        sens_file = SENS_DIR / f"weighted_candidate_gold_{t}.jsonl"
        if not sens_file.exists():
            print(f"[WARNING] Missing {sens_file} — run candidate_gold.py first")
            precisions.append(0.0)
            recalls.append(0.0)
            f1s.append(0.0)
            continue

        pseudo_entities = load_jsonl(sens_file)
        pseudo_by_row = group_by_row(pseudo_entities)
        p, r, f = evaluate_pseudo_gold(docs, gold_map, pseudo_by_row)

        precisions.append(p)
        recalls.append(r)
        f1s.append(f)
        print(f"{t:>10}  {p:>10.4f}  {r:>10.4f}  {f:>10.4f}")

        bar_file = FIG_DIR / "sensitivity" / f"threshold_{t}_bars.png"
        plot_per_threshold_bars(t, p, r, f, bar_file)

    curve_file = FIG_DIR / "threshold_sensitivity_curve.png"
    plot_combined_curve(THRESHOLDS, precisions, recalls, f1s, curve_file)

    print("\nDone. Figures saved under figure/ and figure/sensitivity/.")


if __name__ == "__main__":
    main()
