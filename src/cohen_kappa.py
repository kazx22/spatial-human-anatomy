"""
cohen_kappa.py — pairwise Cohen's kappa between all five NER models and
between each model and human gold.

Kappa is computed over flat token-level BIO label sequences rather than
entity spans.  This is intentional: span-level agreement would ignore
disagreements within an entity boundary (e.g. two models agreeing on the
B- tag but disagreeing on I- continuation), whereas token-level kappa
captures the full label distribution.

The Landis & Koch (1977) scale is used for interpretation:
  < 0.00  Poor
  0.00–0.20  Slight
  0.21–0.40  Fair
  0.41–0.60  Moderate
  0.61–0.80  Substantial
  0.81–1.00  Almost perfect

Run from the project root:
    python -m src.cohen_kappa
"""

import sys
import itertools
from pathlib import Path

import numpy as np
from sklearn.metrics import cohen_kappa_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_jsonl, group_by_row, span_to_bio

# ------------------------------------------------------------------
# File paths
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


def build_flat_labels(docs, pred_by_row):
    """
    Flatten all per-document BIO label sequences into a single list.

    The flat sequence is what cohen_kappa_score expects: one label per token
    across the entire corpus, in document order.  Both sequences being
    compared must be built from the same docs list to guarantee alignment.
    """
    flat = []
    for doc in docs:
        row_id = doc["row_id"]
        text = doc["full_text"]
        entities = pred_by_row.get(row_id, [])
        _, labels = span_to_bio(text, entities)
        flat.extend(labels)
    return flat


def interpret_kappa(k):
    """Landis & Koch (1977) verbal interpretation of a kappa value."""
    if k < 0:
        return "Poor (< 0)"
    elif k < 0.20:
        return "Slight"
    elif k < 0.40:
        return "Fair"
    elif k < 0.60:
        return "Moderate"
    elif k < 0.80:
        return "Substantial"
    else:
        return "Almost perfect"


def print_matrix(names, matrix):
    """Print the upper-triangle kappa values as a symmetric matrix."""
    col_w = 14
    header = " " * col_w + "".join(f"{n:>{col_w}}" for n in names)
    print(header)
    for i, name in enumerate(names):
        row = f"{name:<{col_w}}"
        for j in range(len(names)):
            if i == j:
                row += f"{'1.0000':>{col_w}}"
            elif j < i:
                # Symmetric: mirror the upper-triangle value
                row += f"{matrix[j][i]:>{col_w}.4f}"
            else:
                row += f"{matrix[i][j]:>{col_w}.4f}"
        print(row)


def main():
    print("Loading documents...")
    docs = load_jsonl(DOCS_FILE)
    print(f"Loaded {len(docs)} documents.\n")

    # Build flat label lists for every model and for human gold
    flat_labels = {}
    all_sources = dict(MODEL_FILES)
    all_sources["HumanGold"] = GOLD_FILE

    for name, path in all_sources.items():
        entities = load_jsonl(path)
        by_row = group_by_row(entities)
        flat = build_flat_labels(docs, by_row)
        flat_labels[name] = flat
        print(f"  {name:15} {len(flat)} tokens")

    # All sources must produce the same number of tokens — if not, the
    # flat sequences are misaligned and kappa is meaningless.
    lengths = {name: len(labels) for name, labels in flat_labels.items()}
    if len(set(lengths.values())) != 1:
        print("\n[ERROR] Token count mismatch:")
        for name, l in lengths.items():
            print(f"  {name}: {l}")
        raise ValueError("All sources must produce the same number of tokens.")

    model_names = list(MODEL_FILES.keys())
    n = len(model_names)
    matrix = [[None] * n for _ in range(n)]

    # 1. Pairwise inter-model kappa
    print("\n" + "=" * 60)
    print("PAIRWISE COHEN'S KAPPA — INTER-MODEL AGREEMENT")
    print("=" * 60)
    for i, j in itertools.combinations(range(n), 2):
        a, b = model_names[i], model_names[j]
        k = cohen_kappa_score(flat_labels[a], flat_labels[b])
        matrix[i][j] = k
        print(f"  {a:15} vs {b:15}  k = {k:.4f}  ({interpret_kappa(k)})")

    print("\nPairwise kappa matrix:")
    print_matrix(model_names, matrix)

    # 2. Each model vs human gold
    print("\n" + "=" * 60)
    print("MODEL vs HUMAN GOLD — COHEN'S KAPPA")
    print("=" * 60)
    gold = flat_labels["HumanGold"]
    vs_gold = []
    for name in model_names:
        k = cohen_kappa_score(flat_labels[name], gold)
        vs_gold.append(k)
        print(f"  {name:15} vs HumanGold  k = {k:.4f}  ({interpret_kappa(k)})")

    # 3. Summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    inter = [matrix[i][j] for i, j in itertools.combinations(range(n), 2)]
    print(f"  Inter-model kappa mean : {np.mean(inter):.4f}")
    print(f"  Inter-model kappa std  : {np.std(inter):.4f}")
    print(f"  Inter-model kappa min  : {np.min(inter):.4f}")
    print(f"  Inter-model kappa max  : {np.max(inter):.4f}")
    print(f"\n  Model-vs-gold mean     : {np.mean(vs_gold):.4f}")
    print(
        f"  Best vs gold           : {model_names[int(np.argmax(vs_gold))]} "
        f"(k = {np.max(vs_gold):.4f})"
    )


if __name__ == "__main__":
    main()
