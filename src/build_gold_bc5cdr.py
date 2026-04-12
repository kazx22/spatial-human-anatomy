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

    first_doc = docs[0]
    row_id = first_doc["row_id"]
    text = first_doc["full_text"]
    entities_for_doc = grouped_entities.get(row_id, [])

    tokens, bio_labels = span_to_bio(text, entities_for_doc)

    print("\nFirst document row_id:", row_id)
    print("First 20 tokens:", tokens[:20])
    print("First 20 BIO labels:", bio_labels[:20])

    y_true = [bio_labels]
    y_pred = [bio_labels]

    print("\nSanity check metrics with identical labels:")
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall:", recall_score(y_true, y_pred))
    print("F1:", f1_score(y_true, y_pred))

    gold_data = build_gold_bio(docs, grouped_entities)

    gold_output_file = "data/gold/bc5cdr_train_gold_bio.jsonl"
    save_jsonl(gold_data, gold_output_file)

    print(f"Saved gold BIO data to {gold_output_file}")

    print("\nBuilt gold BIO records:", len(gold_data))
    print("First gold row_id:", gold_data[0]["row_id"])
    print("First 10 gold tokens:", gold_data[0]["tokens"][:10])
    print("First 10 gold labels:", gold_data[0]["bio_labels"][:10])


if __name__ == "__main__":
    main()
