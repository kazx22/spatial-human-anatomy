"""
utils.py

Shared helpers I use across the whole pipeline: reading/writing JSONL,
grouping entities by document, and the span-to-BIO conversion that the
evaluation depends on. I pulled these out into one place so the model
scripts and the evaluation all use the exact same logic instead of each
having its own copy.
"""

import json
from collections import defaultdict


def load_jsonl(path: str):
    # I store everything (docs, gold, predictions) as one JSON object per
    # line, so this just reads a file back into a list of dicts. Skipping
    # blank lines so a stray newline at the end of a file doesn't crash it.
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
    # Counterpart to load_jsonl. ensure_ascii=False so biomedical terms with
    # accents/Greek letters stay readable instead of being escaped.
    with open(output_file, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def group_by_row(entities):
    # Predictions come in as a flat list of entities across all documents.
    # For evaluation I need them organised per document, so this buckets
    # them by row_id (the document ID).
    grouped = defaultdict(list)

    for entity in entities:
        row_id = entity["row_id"]
        grouped[row_id].append(entity)

    return grouped


def span_to_bio(text, entities):
    """
    Convert character-offset entity spans into token-level BIO tags.

    This is the core of how I make everything comparable. Every model and
    the gold standard store entities as (start_char, end_char, label), but
    seqeval needs BIO tags per token. So I split the text into tokens, walk
    through them tracking each token's character span, and tag any token
    that overlaps an entity span: B- for the first token of the entity,
    I- for the rest, O for everything else.

    Note: I match on character-span OVERLAP rather than exact boundaries.
    This is deliberate. A predicted span and the gold span don't always
    break on the same whitespace, so requiring exact offset equality would
    throw away near-correct predictions. Overlap is the fairer call here.
    """
    tokens = text.split()
    labels = ["O"] * len(tokens)

    for entity in entities:
        label = entity["label"]
        start = entity["start_char"]
        end = entity["end_char"]

        curr = 0  # running character position of the current token
        first_token = True  # so I know whether to write B- or I-

        for i, token in enumerate(tokens):
            token_start = curr
            token_end = curr + len(token)

            # Tokens are in order, so once a token starts at or after the
            # entity end, no later token can overlap it. Stop early.
            if token_start >= end:
                break

            # This token finishes before the entity even starts: skip it.
            # +1 accounts for the single space I split on.
            if token_end <= start:
                curr += len(token) + 1
                continue

            # True overlap between this token and the entity span.
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
    # Run span_to_bio over every document to build the full BIO-tagged gold
    # set. Each record keeps the tokens alongside their labels so I can line
    # predictions up against them later.
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


def derive_weights(f1: dict) -> dict:
    # For the weighted pseudo-gold I weight each model by its F1 against
    # human gold. This normalises a dict of {model: f1} so the weights sum
    # to 1. I ended up using raw F1 scores directly as the weights in the
    # voting, but I keep this here as the normalised alternative.
    total = sum(f1.values())
    weights = {}

    for model, f1_score in f1.items():
        weights[model] = round(f1_score / total, 4)

    return weights
