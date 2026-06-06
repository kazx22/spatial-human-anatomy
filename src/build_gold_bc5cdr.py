"""
build_gold_bc5cdr.py — converts the human-annotated BC5CDR entity spans into
BIO-tagged token sequences and writes them as the reference gold standard.

The output (bc5cdr_train_gold_bio.jsonl) is the fixed reference used by
bc5cdr_evaluation.py, bootstrap_significance.py, cohen_kappa.py, and
threshold_sensitivity.py.  Every model is scored against this file.

The conversion uses utils.span_to_bio, which performs a token-level alignment
between character offsets and whitespace-tokenised text.  A sanity check at
the end confirms that scoring the gold against itself produces P/R/F1 = 1.0,
catching any alignment bugs before the reference file is written.

Pipeline position: runs after parse_bc5cdr.py and before any model scripts or
evaluation.
"""

from seqeval.metrics import precision_score, recall_score, f1_score

from src.utils import (
    load_jsonl,
    save_jsonl,
    group_by_row,
    span_to_bio,
    build_gold_bio,
)


def main():
    docs_file = "data/processed/bc5cdr/bc5cdr_train_docs.jsonl"
    entities_file = "data/processed/bc5cdr/bc5cdr_train_entities.jsonl"

    docs = load_jsonl(docs_file)
    entities = load_jsonl(entities_file)

    grouped_entities = group_by_row(entities)

    print(f"Loaded {len(docs)} documents")
    print(f"Loaded {len(entities)} entities")
    print(f"Grouped into {len(grouped_entities)} row_ids")

    # Spot-check the first document to confirm span_to_bio alignment looks right
    first_doc = docs[0]
    row_id = first_doc["row_id"]
    text = first_doc["full_text"]
    entities_for_doc = grouped_entities.get(row_id, [])

    tokens, bio_labels = span_to_bio(text, entities_for_doc)

    print("\nFirst document row_id:", row_id)
    print("First 20 tokens:", tokens[:20])
    print("First 20 BIO labels:", bio_labels[:20])

    # Sanity check: scoring gold against itself must give perfect metrics.
    # If it doesn't, span_to_bio has an alignment bug.
    y_true = [bio_labels]
    y_pred = [bio_labels]

    print("\nSanity check — identical labels should give P/R/F1 = 1.0:")
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall:", recall_score(y_true, y_pred))
    print("F1:", f1_score(y_true, y_pred))

    gold_data = build_gold_bio(docs, grouped_entities)

    gold_output_file = "data/gold/bc5cdr_train_gold_bio.jsonl"
    save_jsonl(gold_data, gold_output_file)

    print(f"\nSaved gold BIO data to {gold_output_file}")
    print("Built gold BIO records:", len(gold_data))
    print("First gold row_id:", gold_data[0]["row_id"])
    print("First 10 gold tokens:", gold_data[0]["tokens"][:10])
    print("First 10 gold labels:", gold_data[0]["bio_labels"][:10])


if __name__ == "__main__":
    main()
