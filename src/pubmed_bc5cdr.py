import json
import time
from pathlib import Path
from transformers import pipeline

DISEASE_MODEL = (
    "sarahmiller137/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext-ft-ncbi-disease"
)
CHEMICAL_MODEL = "OpenMed/OpenMed-NER-ChemicalDetect-PubMed-335M"


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


def normalize_label(raw_label, fallback_label):
    raw = str(raw_label).upper()

    if "DISEASE" in raw:
        return "DISEASE"
    if "CHEM" in raw:
        return "CHEMICAL"

    return fallback_label


def safe_text(text, max_chars=1500):
    return text[:max_chars]


def predictions_to_entities(predictions, row_id, label_name):
    entities = []

    for pred in predictions:
        entity = {
            "row_id": row_id,
            "text": pred["word"],
            "start_char": int(pred["start"]),
            "end_char": int(pred["end"]),
            "label": normalize_label(pred.get("entity_group", label_name), label_name),
        }
        entities.append(entity)

    return entities


def run_pubmedbert(docs, max_chars=1500):
    disease_ner = pipeline(
        "token-classification",
        model=DISEASE_MODEL,
        aggregation_strategy="simple",
    )

    chemical_ner = pipeline(
        "token-classification",
        model=CHEMICAL_MODEL,
        aggregation_strategy="simple",
    )

    print("Disease model max length:", disease_ner.tokenizer.model_max_length)
    print("Chemical model max length:", chemical_ner.tokenizer.model_max_length)

    all_entities = []
    start_time = time.time()

    for i, doc_record in enumerate(docs):
        row_id = doc_record["row_id"]
        text = doc_record["full_text"]
        text = safe_text(text, max_chars=max_chars)

        disease_preds = disease_ner(text)
        chemical_preds = chemical_ner(text)

        disease_entities = predictions_to_entities(disease_preds, row_id, "DISEASE")
        chemical_entities = predictions_to_entities(chemical_preds, row_id, "CHEMICAL")

        all_entities.extend(disease_entities)
        all_entities.extend(chemical_entities)

        if i % 25 == 0:
            print(f"Processed {i} documents...")

    total_time = time.time() - start_time
    avg_time = total_time / len(docs)

    print(f"Total time taken: {total_time:.2f} seconds")
    print(f"Average time per document: {avg_time:.4f} seconds")

    return all_entities


if __name__ == "__main__":
    input_file = Path("data/processed/bc5cdr/bc5cdr_train_docs.jsonl")
    output_file = Path("data/processed/bc5cdr/pubmedbert_train_entities_bc5cdr.jsonl")

    print("Loading BC5CDR train docs...")
    docs = load_jsonl(input_file)
    print(f"Loaded {len(docs)} documents")

    print("Running PubMedBERT models...")
    entities = run_pubmedbert(docs, max_chars=1500)

    print(f"Predicted {len(entities)} entities")
    save_jsonl(entities, output_file)

    print(f"Saved PubMedBERT entities to {output_file}")


# Total time taken: 1353.70 seconds
# Average time per document: 2.7074 seconds
# Predicted 15807 entities
