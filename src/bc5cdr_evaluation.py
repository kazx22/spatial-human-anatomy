from pathlib import Path

from seqeval.metrics import (
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

from src.graph import (
    add_model_result,
    add_human_comparison_result,
    plot_all,
)
from src.utils import load_jsonl, group_by_row, span_to_bio


def print_per_label_summary(report_dict):
    print("\nPer-label performance:")

    for label in ["DISEASE", "CHEMICAL"]:
        if label in report_dict:
            print(
                f"{label:<10} "
                f"Precision: {report_dict[label]['precision']:.4f}  "
                f"Recall: {report_dict[label]['recall']:.4f}  "
                f"F1: {report_dict[label]['f1-score']:.4f}  "
                f"Support: {report_dict[label]['support']}"
            )


def evaluate_model(notes, gold_by_row, pred_by_row, model_name, runtime):
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
    print(classification_report(y_true, y_pred, digits=4))

    report_dict = classification_report(
        y_true,
        y_pred,
        digits=4,
        output_dict=True,
    )

    print_per_label_summary(report_dict)

    flat_y_true = [label for row in y_true for label in row]
    flat_y_pred = [label for row in y_pred for label in row]

    add_model_result(
        model_name,
        precision,
        recall,
        f1,
        report_dict,
        runtime,
        y_true=flat_y_true,
        y_pred=flat_y_pred,
    )


def evaluate_candidate_vs_human(notes, gold_bio_map, candidate_by_row, runtime=0.0):
    y_true = []
    y_pred = []

    for row_id, text in notes.items():
        gold_labels = gold_bio_map.get(row_id)
        candidate_entities = candidate_by_row.get(row_id, [])

        if gold_labels is None:
            print(f"[WARNING] Missing human gold BIO for row_id {row_id}")
            continue

        pred_tokens, pred_labels = span_to_bio(text, candidate_entities)
        gold_tokens = text.split()

        if len(gold_tokens) != len(gold_labels):
            print(f"[WARNING] Gold token/label mismatch at row_id {row_id}")
            continue

        if len(pred_tokens) != len(gold_labels):
            print(f"[WARNING] Prediction token/label mismatch at row_id {row_id}")
            continue

        y_true.append(gold_labels)
        y_pred.append(pred_labels)

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print("\nCandidate Gold vs Human Gold")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(classification_report(y_true, y_pred, digits=4))

    report_dict = classification_report(
        y_true,
        y_pred,
        digits=4,
        output_dict=True,
    )

    print_per_label_summary(report_dict)

    flat_y_true = [label for row in y_true for label in row]
    flat_y_pred = [label for row in y_pred for label in row]

    add_human_comparison_result(
        "Candidate Gold vs Human Gold",
        precision,
        recall,
        f1,
        report_dict,
        runtime,
        y_true=flat_y_true,
        y_pred=flat_y_pred,
    )


def main():
    docs_path = Path("data/processed/bc5cdr/bc5cdr_train_docs.jsonl")
    human_gold_bio_path = Path("data/gold/bc5cdr_train_gold_bio.jsonl")
    candidate_gold_path = Path("data/gold/candidate_gold_train_entities_bc5cdr.jsonl")

    scispacy_path = Path("data/processed/bc5cdr/scispacy_train_entities_bc5cdr.jsonl")
    clinicalbert_path = Path(
        "data/processed/bc5cdr/clinicalbert_train_entities_bc5cdr_clean.jsonl"
    )
    bioelectra_path = Path(
        "data/processed/bc5cdr/bioelectra_train_entities_bc5cdr.jsonl"
    )
    pubmedbert_path = Path(
        "data/processed/bc5cdr/pubmedbert_train_entities_bc5cdr.jsonl"
    )

    docs = load_jsonl(docs_path)

    notes = {}
    for doc in docs:
        notes[doc["row_id"]] = doc["full_text"]

    human_gold_bio = load_jsonl(human_gold_bio_path)
    candidate_gold_entities = load_jsonl(candidate_gold_path)

    scispacy_entities = load_jsonl(scispacy_path)
    pubmedbert_entities = load_jsonl(pubmedbert_path)
    clinicalbert_entities = load_jsonl(clinicalbert_path)
    bioelectra_entities = load_jsonl(bioelectra_path)

    gold_bio_map = {}
    for record in human_gold_bio:
        gold_bio_map[record["row_id"]] = record["bio_labels"]

    candidate_by_row = group_by_row(candidate_gold_entities)
    scispacy_by_row = group_by_row(scispacy_entities)
    pubmedbert_by_row = group_by_row(pubmedbert_entities)
    clinicalbert_by_row = group_by_row(clinicalbert_entities)
    bioelectra_by_row = group_by_row(bioelectra_entities)

    evaluate_candidate_vs_human(
        notes,
        gold_bio_map,
        candidate_by_row,
        runtime=0.0,
    )

    evaluate_model(
        notes,
        candidate_by_row,
        scispacy_by_row,
        "SciSpacy",
        runtime=0.0238,
    )

    evaluate_model(
        notes,
        candidate_by_row,
        pubmedbert_by_row,
        "PubMedBERT",
        runtime=2.7074,
    )

    evaluate_model(
        notes,
        candidate_by_row,
        clinicalbert_by_row,
        "ClinicalBERT",
        runtime=0.1502,
    )

    evaluate_model(
        notes,
        candidate_by_row,
        bioelectra_by_row,
        "BioELECTRA",
        runtime=0.1324,
    )

    plot_all()


if __name__ == "__main__":
    main()
