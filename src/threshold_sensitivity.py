"""
threshold_sensitivity.py
------------------------
Evaluates each weighted pseudo-gold threshold against human gold and
produces:
  (A) one combined sensitivity curve (precision, recall, F1 vs threshold)
  (B) one bar chart per threshold (P/R/F1 at that threshold)

Prerequisite: run `python -m src.candidate_gold` first so the sensitivity
files exist under data/gold/sensitivity/.

Run from PROJECT ROOT:
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
# Use the SAME pre-built human gold BIO file as bc5cdr_evaluation.py so the
# token alignment is identical across both scripts. Rebuilding BIO on the fly
# from raw entities produced different tokenisation and silently dropped the
# docs that mismatched, which inflated the F1 at every threshold.
GOLD_BIO_FILE = PROJECT_ROOT / "data" / "gold" / "bc5cdr_train_gold_bio.jsonl"

THRESHOLDS = [0.45, 0.70, 0.90, 1.20, 1.50, 1.80, 2.00]

# ------------------------------------------------------------------
# Professional styling (muted, journal-friendly palette)
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


# ------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------


def build_gold_map(gold_bio_records):
    # Use the pre-built BIO gold directly (same source as bc5cdr_evaluation.py).
    # Each record already has aligned tokens and bio_labels, so no re-tokenisation
    # happens here and the two scripts evaluate against an identical gold.
    gold_map = {}
    for record in gold_bio_records:
        gold_map[record["row_id"]] = (record["tokens"], record["bio_labels"])
    return gold_map


def evaluate_pseudo_gold(docs, gold_map, pseudo_by_row):
    y_true, y_pred = [], []

    total_docs = 0
    used_docs = 0
    skipped_missing_gold = 0
    skipped_token_mismatch = 0

    for doc in docs:
        total_docs += 1
        row_id = doc["row_id"]
        text = doc["full_text"]

        if row_id not in gold_map:
            skipped_missing_gold += 1
            continue

        gold_tokens, gold_labels = gold_map[row_id]
        pseudo_entities = pseudo_by_row.get(row_id, [])
        pred_tokens, pred_labels = span_to_bio(text, pseudo_entities)

        if pred_tokens != gold_tokens:
            skipped_token_mismatch += 1
            continue
        if len(pred_labels) != len(gold_labels):
            skipped_token_mismatch += 1
            continue

        y_true.append(gold_labels)
        y_pred.append(pred_labels)
        used_docs += 1

    # Surface skip counts so any silent doc-dropping is visible, exactly like
    # bc5cdr_evaluation.py reports. These should read 0 once the gold sources match.
    print(
        f"    [used {used_docs}/{total_docs}  "
        f"missing_gold {skipped_missing_gold}  "
        f"token_mismatch {skipped_token_mismatch}]"
    )

    if not y_true:
        return 0.0, 0.0, 0.0
    return (
        precision_score(y_true, y_pred),
        recall_score(y_true, y_pred),
        f1_score(y_true, y_pred),
    )


# ------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------


def plot_combined_curve(thresholds, precisions, recalls, f1s, output_file):
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

    # Mark the F1 peak
    peak_idx = int(np.argmax(f1s))
    ax.axvline(
        thresholds[peak_idx], color="#999999", linestyle="--", linewidth=1.2, alpha=0.7
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
    metrics = ["Precision", "Recall", "F1-score"]
    values = [p, r, f]
    colors = [COL_PRECISION, COL_RECALL, COL_F1]
    x = np.arange(len(metrics))

    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.bar(x, values, width=0.55, color=colors, edgecolor="#333333", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=12)
    ax.set_ylim(0, 1.15)  # headroom so value labels never collide with title
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


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(FIG_DIR / "sensitivity", exist_ok=True)

    print("Loading documents and human gold...")
    docs = load_jsonl(DOCS_FILE)
    gold_bio_records = load_jsonl(GOLD_BIO_FILE)
    gold_map = build_gold_map(gold_bio_records)
    print(f"Loaded {len(docs)} documents.\n")

    precisions, recalls, f1s = [], [], []

    print("=" * 60)
    print("THRESHOLD SENSITIVITY EVALUATION")
    print("=" * 60)
    print(f"{'Threshold':>10}  {'Precision':>10}  {'Recall':>10}  {'F1':>10}")

    for t in THRESHOLDS:
        sens_file = SENS_DIR / f"weighted_candidate_gold_{t}.jsonl"
        if not sens_file.exists():
            print(f"[WARNING] Missing {sens_file} - run candidate_gold.py first")
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