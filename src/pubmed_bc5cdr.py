import json
import re
import time
from pathlib import Path
from transformers import pipeline

DISEASE_MODEL = (
    "sarahmiller137/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext-ft-ncbi-disease"
)
CHEMICAL_MODEL = "OpenMed/OpenMed-NER-ChemicalDetect-PubMed-335M"

MAX_TOKENS = 400
OVERLAP_SENTS = 1


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


def chunk_text(text, tokenizer, max_tokens=MAX_TOKENS, overlap_sents=OVERLAP_SENTS):

    raw_sentences = re.split(r"(?<=[.!?])\s+|\n+", text)

    sentence_offsets = []
    pos = 0
    for sent in raw_sentences:
        idx = pos
        sentence_offsets.append(idx)
        pos = idx + len(sent)

    chunks = []
    i = 0

    while i < len(raw_sentences):
        chunk_sents = []
        chunk_sent_offsets = []
        token_count = 0
        j = i

        while j < len(raw_sentences):
            sent = raw_sentences[j]
            sent_token_count = len(tokenizer.encode(sent, add_special_tokens=False))

            # +2 for [CLS] and [SEP]
            if token_count + sent_token_count + 2 > max_tokens and chunk_sents:
                break

            chunk_sents.append(sent)
            chunk_sent_offsets.append(sentence_offsets[j])
            token_count += sent_token_count
            j += 1

        chunk_str = " ".join(chunk_sents)
        char_offset = chunk_sent_offsets[0]
        chunks.append((chunk_str, char_offset))

        i = max(i + 1, j - overlap_sents)

    return chunks


def predictions_to_entities(predictions, row_id, label_name, char_offset=0):
    entities = []

    for pred in predictions:
        entity = {
            "row_id": row_id,
            "text": pred["word"],
            "start_char": int(pred["start"]) + char_offset,
            "end_char": int(pred["end"]) + char_offset,
            "label": normalize_label(pred.get("entity_group", label_name), label_name),
        }
        entities.append(entity)

    return entities


def deduplicate_entities(entities):
    """
    Remove duplicate entities that arise from overlapping chunks.
    Keeps the first occurrence of each (row_id, start_char, end_char, label) tuple.
    """
    seen = set()
    deduped = []

    for ent in entities:
        key = (ent["row_id"], ent["start_char"], ent["end_char"], ent["label"])

        if key not in seen:
            seen.add(key)
            deduped.append(ent)

    return deduped


def run_pubmedbert(docs):
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

        # --- disease model chunks ---
        for chunk, char_offset in chunk_text(text, disease_ner.tokenizer):
            disease_preds = disease_ner(chunk)
            all_entities.extend(
                predictions_to_entities(disease_preds, row_id, "DISEASE", char_offset)
            )

        # --- chemical model chunks ---
        for chunk, char_offset in chunk_text(text, chemical_ner.tokenizer):
            chemical_preds = chemical_ner(chunk)
            all_entities.extend(
                predictions_to_entities(chemical_preds, row_id, "CHEMICAL", char_offset)
            )

        if i % 25 == 0:
            print(f"Processed {i} documents...")

    all_entities = deduplicate_entities(all_entities)

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

    print("Running PubMedBERT models with chunking...")
    entities = run_pubmedbert(docs)

    print(f"Predicted {len(entities)} entities")
    save_jsonl(entities, output_file)

    print(f"Saved PubMedBERT entities to {output_file}")


# Total time taken: 1970.44 seconds
# Average time per document: 3.9409 seconds
# Predicted 18153 entities
