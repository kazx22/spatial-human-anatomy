import json
import time
from pathlib import Path
from transformers import pipeline
from src.biobert_chem_merger import merge_word

MODEL_NAME = "samrawal/bert-base-uncased_clinical-ner"

label_map = {
    "problem": "DISEASE",
    "treatment": "CHEMICAL",
    "test": "TEST",
}


def load_jsonl(file_path):
    records = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            records.append(json.loads(line))

    return records


def save_jsonl(records, output_file):
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def safe_text(text, max_chars=1500):
    return text[:max_chars]


def run_clinicalbert(docs, max_chars=1500):
    ner = pipeline(
        "ner",
        model=MODEL_NAME,
        aggregation_strategy="simple",
    )

    print("ClinicalBERT model max length:", ner.tokenizer.model_max_length)

    all_entities = []
    start_time = time.time()

    for i, doc_record in enumerate(docs):
        row_id = doc_record["row_id"]
        text = safe_text(doc_record["full_text"], max_chars=max_chars)

        results = ner(text)
        merged_results = merge_word(results)

        for ent in merged_results:
            raw_label = ent["entity_group"]
            mapped_label = label_map.get(raw_label)

            if mapped_label is None:
                continue

            entity = {
                "row_id": row_id,
                "text": ent["word"],
                "start_char": int(ent["start"]),
                "end_char": int(ent["end"]),
                "label": mapped_label,
                "confidence": float(ent["score"]),
            }
            all_entities.append(entity)

        if i % 25 == 0:
            print(f"Processed {i} documents...")

    total_time = time.time() - start_time
    avg_time = total_time / len(docs)

    print(f"Total time taken: {total_time:.2f} seconds")
    print(f"Average time per document: {avg_time:.4f} seconds")

    return all_entities


if __name__ == "__main__":
    input_file = Path("data/processed/bc5cdr/bc5cdr_train_docs.jsonl")
    output_file = Path("data/processed/bc5cdr/clinicalbert_train_entities_bc5cdr.jsonl")

    print("Loading BC5CDR train docs...")
    docs = load_jsonl(input_file)
    print(f"Loaded {len(docs)} documents")

    print(f"Running ClinicalBERT model: {MODEL_NAME}")
    entities = run_clinicalbert(docs, max_chars=1500)

    print(f"Predicted {len(entities)} entities")
    save_jsonl(entities, output_file)

    print(f"Saved ClinicalBERT entities to {output_file}")


# Total time taken: 75.12 seconds
# Average time per document: 0.1502 seconds
# Predicted 14479 entities
