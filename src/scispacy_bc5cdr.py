import json
import time
from pathlib import Path
import spacy

MODEL_NAME = "en_ner_bc5cdr_md"


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


def run_scispacy(docs):
    nlp = spacy.load(MODEL_NAME)
    all_entities = []

    start_time = time.time()

    for i, doc_record in enumerate(docs):
        row_id = doc_record["row_id"]
        text = doc_record["full_text"]

        doc = nlp(text)

        for ent in doc.ents:
            entity = {
                "row_id": row_id,
                "text": ent.text,
                "start_char": ent.start_char,
                "end_char": ent.end_char,
                "label": ent.label_,
                "confidence": 1.0,
            }
            all_entities.append(entity)

        if i % 50 == 0:
            print(f"Processed {i} documents...")

    end_time = time.time()
    total_time = end_time - start_time
    avg_time = total_time / len(docs)

    print(f"Total time taken: {total_time:.2f} seconds")
    print(f"Average time per document: {avg_time:.4f} seconds")

    return all_entities


if __name__ == "__main__":
    input_file = Path("data/processed/bc5cdr/bc5cdr_train_docs.jsonl")
    output_file = Path("data/processed/bc5cdr/scispacy_train_entities_bc5cdr.jsonl")

    print("Loading BC5CDR train docs...")
    docs = load_jsonl(input_file)
    print(f"Loaded {len(docs)} documents")

    print(f"Running SciSpacy model: {MODEL_NAME}")
    entities = run_scispacy(docs)

    print(f"Predicted {len(entities)} entities")
    save_jsonl(entities, output_file)

    print(f"Saved SciSpacy entities to {output_file}")


# Total time taken: 11.92 seconds
# Average time per document: 0.0238 seconds
# Predicted 8597 entities
