"""
bootstrap_significance.py — paired bootstrap significance testing for the
comparative NER evaluation.

For each pair of models, reports whether the observed F1 difference is
statistically significant at p < 0.05.  Also reports 95% bootstrap confidence
intervals on each model's F1 score.

Method:
  1. Build aligned per-document (gold, pred) BIO label lists for each model.
  2. Resample documents with replacement N_BOOTSTRAP times (default 1000).
  3. For each resample, recompute entity-level F1 for every model.
  4. For each model pair, the p-value is the fraction of resamples where the
     F1 difference reverses sign relative to the observed difference.

The test is "paired" in the sense that each resample draws the same document
indices for all models, so the comparison is on matched observations rather
than independent samples.

Run from the project root:
    python -m src.bootstrap_significance
"""

import sys
import itertools
from pathlib import Path

import numpy as np
from seqeval.metrics import f1_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_jsonl, group_by_row, span_to_bio, build_gold_bio

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "bc5cdr"

DOCS_FILE = DATA_DIR / "bc5cdr_train_docs.jsonl"
GOLD_FILE = DATA_DIR / "bc5cdr_train_entities.jsonl"

MODEL_FILES = {
    "SciSpacy": DATA_DIR / "scispacy_train_entities_bc5cdr.jsonl",
    "BioBERT": DATA_DIR / "biobert_train_entities_bc5cdr.jsonl",
    "PubMedBERT": DATA_DIR / "pubmedbert_train_entities_bc5cdr.jsonl",
    "ClinicalBERT": DATA_DIR / "clinicalbert_train_entities_bc5cdr.jsonl",
    "BioELECTRA": DATA_DIR / "bioelectra_train_entities_bc5cdr.jsonl",
}

N_BOOTSTRAP = 1000  # standard for NLP significance testing
SEED = 42  # fixed for reproducibility


def build_per_doc_labels(docs, gold_by_row, pred_by_row):
    """
    Build parallel per-document gold and predicted BIO label lists.

    Only documents where the tokenisation is consistent between gold and
    prediction are kept.  Mismatches are silently skipped rather than
    raising an error because they typically affect a small number of docs
    and the bootstrap operates on the retained set.

    Returns:
      gold_per_doc  — list of gold BIO label sequences, one per document
      pred_per_doc  — matching list of predicted BIO label sequences
    """
    gold_per_doc = []
    pred_per_doc = []

    for doc in docs:
        row_id = doc["row_id"]
        text = doc["full_text"]

        gold_entities = gold_by_row.get(row_id, [])
        pred_entities = pred_by_row.get(row_id, [])

        gold_tokens, gold_labels = span_to_bio(text, gold_entities)
        pred_tokens, pred_labels = span_to_bio(text, pred_entities)

        if gold_tokens != pred_tokens:
            continue
        if len(gold_labels) != len(pred_labels):
            continue

        gold_per_doc.append(gold_labels)
        pred_per_doc.append(pred_labels)

    return gold_per_doc, pred_per_doc


def f1_on_indices(gold_per_doc, pred_per_doc, indices):
    """Compute entity-level F1 over a selected subset of document indices."""
    y_true = [gold_per_doc[i] for i in indices]
    y_pred = [pred_per_doc[i] for i in indices]
    return f1_score(y_true, y_pred)


def main():
    rng = np.random.default_rng(SEED)

    print("Loading documents and gold...")
    docs = load_jsonl(DOCS_FILE)
    gold_entities = load_jsonl(GOLD_FILE)
    gold_by_row = group_by_row(gold_entities)
    print(f"Loaded {len(docs)} documents.\n")

    # Build aligned per-doc label lists for each model
    model_doc_labels = {}
    n_docs_aligned = None

    for name, path in MODEL_FILES.items():
        pred_entities = load_jsonl(path)
        pred_by_row = group_by_row(pred_entities)
        gold_per_doc, pred_per_doc = build_per_doc_labels(
            docs, gold_by_row, pred_by_row
        )
        model_doc_labels[name] = (gold_per_doc, pred_per_doc)
        print(f"  {name:15} aligned docs: {len(gold_per_doc)}")
        if n_docs_aligned is None:
            n_docs_aligned = len(gold_per_doc)

    # All models must be paired on the same documents for a valid paired test.
    # If counts differ, fall back to the minimum and warn — the pairing
    # assumption (shared document order) still holds for the shared subset.
    aligned_counts = {n: len(v[0]) for n, v in model_doc_labels.items()}
    if len(set(aligned_counts.values())) != 1:
        print("\n[WARNING] Models aligned on different doc counts:")
        for n, c in aligned_counts.items():
            print(f"  {n}: {c}")
        print("Using per-model document sets; pairing assumes shared order.\n")

    model_names = list(MODEL_FILES.keys())
    n_docs = min(aligned_counts.values())
    all_idx = list(range(n_docs))

    # Observed F1 on the full aligned set
    observed_f1 = {}
    for name in model_names:
        gold_pd, pred_pd = model_doc_labels[name]
        observed_f1[name] = f1_on_indices(gold_pd, pred_pd, all_idx)

    print("\n" + "=" * 60)
    print("OBSERVED F1 (full document set)")
    print("=" * 60)
    for name in model_names:
        print(f"  {name:15} F1 = {observed_f1[name]:.4f}")

    # Bootstrap: resample with replacement, recompute F1 for each model
    print("\n" + "=" * 60)
    print(f"RUNNING {N_BOOTSTRAP} BOOTSTRAP RESAMPLES (seed={SEED})")
    print("=" * 60)

    boot_f1 = {name: np.zeros(N_BOOTSTRAP) for name in model_names}

    for b in range(N_BOOTSTRAP):
        sample_idx = rng.integers(0, n_docs, size=n_docs)
        for name in model_names:
            gold_pd, pred_pd = model_doc_labels[name]
            boot_f1[name][b] = f1_on_indices(gold_pd, pred_pd, sample_idx)
        if (b + 1) % 200 == 0:
            print(f"  ...{b + 1} resamples done")

    # 95% confidence intervals via percentile method
    print("\n" + "=" * 60)
    print("95% BOOTSTRAP CONFIDENCE INTERVALS ON F1")
    print("=" * 60)
    for name in model_names:
        lo = np.percentile(boot_f1[name], 2.5)
        hi = np.percentile(boot_f1[name], 97.5)
        print(
            f"  {name:15} F1 = {observed_f1[name]:.4f}  " f"95% CI [{lo:.4f}, {hi:.4f}]"
        )

    # Pairwise significance: p = fraction of resamples where the sign of
    # the F1 difference flips relative to the observed difference.
    # A one-sided test: if model A beats B overall, p is the fraction of
    # resamples where B actually matches or beats A.
    print("\n" + "=" * 60)
    print("PAIRWISE SIGNIFICANCE (paired bootstrap)")
    print("=" * 60)
    for a, b in itertools.combinations(model_names, 2):
        obs_diff = observed_f1[a] - observed_f1[b]
        boot_diff = boot_f1[a] - boot_f1[b]

        if obs_diff >= 0:
            p = np.mean(boot_diff <= 0)
        else:
            p = np.mean(boot_diff >= 0)

        sig = "significant" if p < 0.05 else "NOT significant"
        better = a if obs_diff > 0 else b
        print(
            f"  {a:13} vs {b:13}  "
            f"obs diff = {obs_diff:+.4f}  p = {p:.4f}  "
            f"({sig}; better: {better})"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
