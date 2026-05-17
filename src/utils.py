import json
from collections import defaultdict


def load_jsonl(path: str):
    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            record = json.loads(line)
            records.append(record)

    return records


def save_jsonl(records, output_file: str):
    with open(output_file, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def group_by_row(entities):
    grouped = defaultdict(list)

    for entity in entities:
        row_id = entity["row_id"]
        grouped[row_id].append(entity)

    return grouped


def span_to_bio(text, entities):
    tokens = text.split()
    labels = ["O"] * len(tokens)

    for entity in entities:
        label = entity["label"]
        start = entity["start_char"]
        end = entity["end_char"]

        curr = 0
        first_token = True

        for i, token in enumerate(tokens):
            token_start = curr
            token_end = curr + len(token)

            if token_start >= end:
                break

            if token_end <= start:
                curr += len(token) + 1
                continue

            overlaps = token_start < end and token_end > start

            if overlaps:
                if first_token:
                    labels[i] = f"B-{label}"
                    first_token = False
                else:
                    labels[i] = f"I-{label}"

            curr += len(token) + 1

    return tokens, labels


def build_gold_bio(docs, grouped_entities):
    gold_data = []

    for doc in docs:
        row_id = doc["row_id"]
        text = doc["full_text"]
        entities = grouped_entities.get(row_id, [])

        tokens, bio_labels = span_to_bio(text, entities)

        record = {
            "row_id": row_id,
            "tokens": tokens,
            "bio_labels": bio_labels,
        }

        gold_data.append(record)

    return gold_data


def remove_TEST(input_file: str, output_file: str):
    with open(input_file, "r", encoding="utf-8") as infile, open(
        output_file, "w", encoding="utf-8"
    ) as outfile:

        for line in infile:
            entity = json.loads(line)

            if entity["label"] != "TEST":
                outfile.write(json.dumps(entity) + "\n")


def derive_weights(f1: dict) -> dict:
    total =  sum(f1.values())
    weights = {}

    for model, f1_score in f1.items():
        weights[model] = round(f1_score/total,4)
    
    return weights