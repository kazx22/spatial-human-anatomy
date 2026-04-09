import json
import re
import time
from pathlib import Path
from transformers import pipeline

MODEL_NAME = "samrawal/bert-base-uncased_clinical-ner"

MAX_TOKENS = 400
OVERLAP_SENTS = 1
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


def deduplicate_entities(entities):

    seen = set()
    deduped = []

    for ent in entities:
        key = (ent["row_id"], ent["start_char"], ent["end_char"], ent["label"])

        if key not in seen:
            seen.add(key)
            deduped.append(ent)

    return deduped


def run_clinicalbert(docs):
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
        text = doc_record["full_text"]

        for chunk, char_offset in chunk_text(text, ner.tokenizer):
            results = ner(chunk)

            for ent in results:
                raw_label = ent["entity_group"]
                mapped_label = label_map.get(raw_label)

                if mapped_label is None:
                    continue

                entity = {
                    "row_id": row_id,
                    "text": ent["word"],
                    "start_char": int(ent["start"]) + char_offset,
                    "end_char": int(ent["end"]) + char_offset,
                    "label": mapped_label,
                    "confidence": float(ent["score"]),
                }
                all_entities.append(entity)

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
    output_file = Path("data/processed/bc5cdr/clinicalbert_train_entities_bc5cdr.jsonl")

    print("Loading BC5CDR train docs...")
    docs = load_jsonl(input_file)
    print(f"Loaded {len(docs)} documents")

    print(f"Running ClinicalBERT model: {MODEL_NAME} with chunking...")
    entities = run_clinicalbert(docs)

    print(f"Predicted {len(entities)} entities")
    save_jsonl(entities, output_file)

    print(f"Saved ClinicalBERT entities to {output_file}")


# Total time taken: 127.29 seconds
# Average time per document: 0.2546 seconds
# Predicted 18972 entities
