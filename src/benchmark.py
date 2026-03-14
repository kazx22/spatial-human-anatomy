import json
from pathlib import Path
from collections import defaultdict
from seqeval.metrics import (
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
import pandas as pd


def load_jsonl(path: str):
    entities = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entity = json.loads(line)
            entities.append(entity)
    return entities


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


def evaluate_model(notes, gold_by_row, pred_by_row, model_name):
    y_true = []
    y_pred = []

    for row_id, text in notes.items():
        gold_entities = gold_by_row.get(row_id, [])
        pred_entities = pred_by_row.get(row_id, [])

        gold_tokens, gold_labels = span_to_bio(text, gold_entities)
        pred_tokens, pred_labels = span_to_bio(text, pred_entities)

        if gold_tokens != pred_tokens:
            print(f"[WARNING] Token mismatch at row_id {row_id} for {model_name}")
            continue

        y_true.append(gold_labels)
        y_pred.append(pred_labels)

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f"\n{model_name}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")


# C:\University\Projects\spatial-human-anatomy\data\gold\candidate_gold_entities.jsonl


def main():
    notes_path = Path("data/processed/cleaned_notes.csv")
    gold_path = Path("data/gold/candidate_gold_entities.jsonl")
    scispacy_path = Path("data/processed/ner/scispacy/entities_clean.jsonl")
    biobert_path = Path(
        "data/processed/ner/transformers/biobert/biobert_entities.jsonl"
    )
    clinicalbert_path = Path(
        "data/processed/ner/transformers/clinicalbert/clinicalbert_entities_clean.jsonl"
    )

    df = pd.read_csv(notes_path)

    notes = {}
    for row_id, text in enumerate(df["note_text"].fillna("").astype(str)):
        notes[row_id] = text

    gold_entities = load_jsonl(gold_path)
    scispacy_entities = load_jsonl(scispacy_path)
    biobert_entities = load_jsonl(biobert_path)
    clinicalbert_entities = load_jsonl(clinicalbert_path)

    gold_by_row = group_by_row(gold_entities)
    scispacy_by_row = group_by_row(scispacy_entities)
    biobert_by_row = group_by_row(biobert_entities)
    clinicalbert_by_row = group_by_row(clinicalbert_entities)

    evaluate_model(notes, gold_by_row, scispacy_by_row, "SciSpacy")
    evaluate_model(notes, gold_by_row, biobert_by_row, "BioBERT")
    evaluate_model(notes, gold_by_row, clinicalbert_by_row, "ClinicalBERT")


if __name__ == "__main__":
    main()
