"""
bc5cdr_evaluation.py — central evaluation harness for the comparative NER study.

Runs three evaluation passes and writes figures via graph.py:

  1. Each model vs human gold — the primary comparison reported in the paper.
  2. Each model vs majority candidate gold — agreement with unweighted ensemble.
  3. Each model vs weighted candidate gold — agreement with the F1-weighted
     pseudo-gold built by candidate_gold.py.

Also evaluates the two pseudo-gold sets themselves against human gold, which
quantifies how closely the ensemble approximates expert annotation.

All five models are used off-the-shelf with zero fine-tuning.  BioBERT and
PubMedBERT are dual-model pipelines (separate disease + chemical checkpoints);
SciSpacy, ClinicalBERT, and BioELECTRA are single-model.

Pipeline position: runs after all five model scripts and candidate_gold.py.
"""

from pathlib import Path

from seqeval.metrics import (
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

from src.graph import (
    add_model_result_human,
    add_model_result_candidate,
    add_human_comparison_result,
    plot_all,
)

from src.utils import (
    load_jsonl,
    group_by_row,
    span_to_bio,
    build_gold_bio,
)

from src.entity_filtering import label_aware_filter


def print_per_label_summary(report_dict):
    """Print DISEASE and CHEMICAL scores from a seqeval classification_report dict."""
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


def evaluate_model_against_human_gold(
    notes,
    gold_bio_map,
    pred_by_row,
    model_name,
    runtime,
):
    """
    Score a model's predictions against human-annotated BC5CDR gold.

    Converts span predictions to BIO sequences on a per-document basis and
    aligns them with the pre-built gold BIO sequences.  Documents where
    tokens don't align (can happen if a model mutates whitespace) are skipped
    and counted as mismatches rather than silently dropped.

    Results are accumulated in graph.py's MODULE_RESULTS_HUMAN list so
    plot_all() can include them later.
    """
    y_true = []
    y_pred = []

    total_docs = 0
    used_docs = 0
    skipped_missing_gold = 0
    skipped_token_mismatch = 0

    for row_id, text in notes.items():
        total_docs += 1

        gold_record = gold_bio_map.get(row_id)
        pred_entities = pred_by_row.get(row_id, [])

        if gold_record is None:
            print(f"[WARNING] Missing human gold BIO for row_id {row_id}")
            skipped_missing_gold += 1
            continue

        gold_tokens = gold_record["tokens"]
        gold_labels = gold_record["bio_labels"]

        pred_tokens, pred_labels = span_to_bio(text, pred_entities)

        if pred_tokens != gold_tokens:
            print(f"[WARNING] Token mismatch at row_id {row_id} for {model_name}")
            skipped_token_mismatch += 1
            continue

        if len(pred_labels) != len(gold_labels):
            print(
                f"[WARNING] Label length mismatch at row_id {row_id} for {model_name}"
            )
            skipped_token_mismatch += 1
            continue

        y_true.append(gold_labels)
        y_pred.append(pred_labels)
        used_docs += 1

    print(f"\n{model_name} evaluation summary vs Human Gold")
    print(f"Total docs:               {total_docs}")
    print(f"Used docs:                {used_docs}")
    print(f"Skipped missing gold:     {skipped_missing_gold}")
    print(f"Skipped token mismatches: {skipped_token_mismatch}")

    if not y_true or not y_pred:
        print(f"[ERROR] No valid evaluation rows for {model_name}")
        return

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f"\n{model_name} vs Human Gold")
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

    # Flatten for confusion matrix storage in graph.py
    flat_y_true = [label for row in y_true for label in row]
    flat_y_pred = [label for row in y_pred for label in row]

    add_model_result_human(
        model_name,
        precision,
        recall,
        f1,
        report_dict,
        runtime,
        y_true=flat_y_true,
        y_pred=flat_y_pred,
    )


def evaluate_model_against_candidate_gold(
    notes,
    candidate_gold_bio_map,
    pred_by_row,
    model_name,
    runtime,
):
    """
    Score a model's predictions against the pseudo-gold (majority or weighted).

    Structurally identical to evaluate_model_against_human_gold; kept
    separate so the two comparison types can be accumulated into different
    lists in graph.py and plotted independently.
    """
    y_true = []
    y_pred = []

    total_docs = 0
    used_docs = 0
    skipped_missing_gold = 0
    skipped_token_mismatch = 0

    for row_id, text in notes.items():
        total_docs += 1

        gold_record = candidate_gold_bio_map.get(row_id)
        pred_entities = pred_by_row.get(row_id, [])

        if gold_record is None:
            print(f"[WARNING] Missing candidate gold BIO for row_id {row_id}")
            skipped_missing_gold += 1
            continue

        gold_tokens = gold_record["tokens"]
        gold_labels = gold_record["bio_labels"]

        pred_tokens, pred_labels = span_to_bio(text, pred_entities)

        if pred_tokens != gold_tokens:
            print(
                f"[WARNING] Token mismatch at row_id {row_id} "
                f"for {model_name} vs candidate gold"
            )
            skipped_token_mismatch += 1
            continue

        if len(pred_labels) != len(gold_labels):
            print(
                f"[WARNING] Label length mismatch at row_id {row_id} "
                f"for {model_name} vs candidate gold"
            )
            skipped_token_mismatch += 1
            continue

        y_true.append(gold_labels)
        y_pred.append(pred_labels)
        used_docs += 1

    print(f"\n{model_name} evaluation summary vs Candidate Gold")
    print(f"Total docs:               {total_docs}")
    print(f"Used docs:                {used_docs}")
    print(f"Skipped missing gold:     {skipped_missing_gold}")
    print(f"Skipped token mismatches: {skipped_token_mismatch}")

    if not y_true or not y_pred:
        print(f"[ERROR] No valid evaluation rows for {model_name} vs candidate gold")
        return

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f"\n{model_name} vs Candidate Gold")
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

    add_model_result_candidate(
        model_name,
        precision,
        recall,
        f1,
        report_dict,
        runtime,
        y_true=flat_y_true,
        y_pred=flat_y_pred,
    )


def evaluate_candidate_vs_human(
    notes,
    human_gold_bio_map,
    candidate_by_row,
    runtime=0.0,
    candidate_name="Candidate Gold vs Human Gold",
):
    """
    Evaluate a pseudo-gold set (majority or weighted) against human gold.

    This is not a model evaluation — it measures how well the ensemble
    approximates expert annotation.  The result appears in Figure 2 of the
    paper as a quality check on the pseudo-gold framework itself.
    """
    y_true = []
    y_pred = []

    total_docs = 0
    used_docs = 0
    skipped_missing_gold = 0
    skipped_token_mismatch = 0

    for row_id, text in notes.items():
        total_docs += 1

        gold_record = human_gold_bio_map.get(row_id)
        candidate_entities = candidate_by_row.get(row_id, [])

        if gold_record is None:
            print(f"[WARNING] Missing human gold BIO for row_id {row_id}")
            skipped_missing_gold += 1
            continue

        gold_tokens = gold_record["tokens"]
        gold_labels = gold_record["bio_labels"]

        pred_tokens, pred_labels = span_to_bio(text, candidate_entities)

        if pred_tokens != gold_tokens:
            print(f"[WARNING] Token mismatch at row_id {row_id} for {candidate_name}")
            skipped_token_mismatch += 1
            continue

        if len(pred_labels) != len(gold_labels):
            print(
                f"[WARNING] Label length mismatch at row_id {row_id} for {candidate_name}"
            )
            skipped_token_mismatch += 1
            continue

        y_true.append(gold_labels)
        y_pred.append(pred_labels)
        used_docs += 1

    print(f"\n{candidate_name} evaluation summary")
    print(f"Total docs:               {total_docs}")
    print(f"Used docs:                {used_docs}")
    print(f"Skipped missing gold:     {skipped_missing_gold}")
    print(f"Skipped token mismatches: {skipped_token_mismatch}")

    if not y_true or not y_pred:
        print(f"[ERROR] No valid evaluation rows for {candidate_name}")
        return

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f"\n{candidate_name}")
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
        candidate_name,
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
    weighted_candidate_gold_path = Path(
        "data/gold/weighted_candidate_gold_train_entities_bc5cdr.jsonl"
    )

    scispacy_path = Path("data/processed/bc5cdr/scispacy_train_entities_bc5cdr.jsonl")
    biobert_path = Path("data/processed/bc5cdr/biobert_train_entities_bc5cdr.jsonl")
    pubmedbert_path = Path(
        "data/processed/bc5cdr/pubmedbert_train_entities_bc5cdr.jsonl"
    )
    clinicalbert_path = Path(
        "data/processed/bc5cdr/clinicalbert_train_entities_bc5cdr.jsonl"
    )
    bioelectra_path = Path(
        "data/processed/bc5cdr/bioelectra_train_entities_bc5cdr.jsonl"
    )

    docs = load_jsonl(docs_path)

    # Build a plain row_id -> text lookup for BIO conversion
    notes = {}
    for doc in docs:
        notes[doc["row_id"]] = doc["full_text"]

    human_gold_bio = load_jsonl(human_gold_bio_path)

    candidate_gold_entities = load_jsonl(candidate_gold_path)
    weighted_candidate_gold_entities = load_jsonl(weighted_candidate_gold_path)

    scispacy_entities = load_jsonl(scispacy_path)
    biobert_entities = load_jsonl(biobert_path)
    pubmedbert_entities = load_jsonl(pubmedbert_path)
    clinicalbert_entities = load_jsonl(clinicalbert_path)
    bioelectra_entities = load_jsonl(bioelectra_path)

    # Label-aware confidence filtering applies separate thresholds for
    # DISEASE and CHEMICAL spans.  These filtered variants are evaluated
    # alongside the raw predictions so the paper can report the effect.
    biobert_filtered_entities = label_aware_filter(
        biobert_entities,
        disease_threshold=0.5,
        chemical_threshold=0.7,
    )

    pubmedbert_filtered_entities = label_aware_filter(
        pubmedbert_entities,
        disease_threshold=0.5,
        chemical_threshold=0.7,
    )

    clinicalbert_filtered_entities = label_aware_filter(
        clinicalbert_entities,
        disease_threshold=0.5,
        chemical_threshold=0.7,
    )

    bioelectra_filtered_entities = label_aware_filter(
        bioelectra_entities,
        disease_threshold=0.5,
        chemical_threshold=0.7,
    )

    human_gold_bio_map = {}
    for record in human_gold_bio:
        human_gold_bio_map[record["row_id"]] = record

    candidate_by_row = group_by_row(candidate_gold_entities)
    weighted_candidate_by_row = group_by_row(weighted_candidate_gold_entities)

    candidate_gold_bio = build_gold_bio(docs, candidate_by_row)
    weighted_candidate_gold_bio = build_gold_bio(docs, weighted_candidate_by_row)

    candidate_gold_bio_map = {}
    for record in candidate_gold_bio:
        candidate_gold_bio_map[record["row_id"]] = record

    weighted_candidate_gold_bio_map = {}
    for record in weighted_candidate_gold_bio:
        weighted_candidate_gold_bio_map[record["row_id"]] = record

    scispacy_by_row = group_by_row(scispacy_entities)
    biobert_by_row = group_by_row(biobert_entities)
    pubmedbert_by_row = group_by_row(pubmedbert_entities)
    clinicalbert_by_row = group_by_row(clinicalbert_entities)
    bioelectra_by_row = group_by_row(bioelectra_entities)

    biobert_filtered_by_row = group_by_row(biobert_filtered_entities)
    pubmedbert_filtered_by_row = group_by_row(pubmedbert_filtered_entities)
    clinicalbert_filtered_by_row = group_by_row(clinicalbert_filtered_entities)
    bioelectra_filtered_by_row = group_by_row(bioelectra_filtered_entities)

    # ------------------------------------------------------------------
    # Pass 1: pseudo-gold quality vs human gold
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("MAJORITY CANDIDATE GOLD VS HUMAN GOLD")
    print("=" * 60)

    evaluate_candidate_vs_human(
        notes,
        human_gold_bio_map,
        candidate_by_row,
        runtime=0.0,
        candidate_name="Majority Candidate Gold vs Human Gold",
    )

    print("\n" + "=" * 60)
    print("WEIGHTED CANDIDATE GOLD VS HUMAN GOLD")
    print("=" * 60)

    evaluate_candidate_vs_human(
        notes,
        human_gold_bio_map,
        weighted_candidate_by_row,
        runtime=0.0,
        candidate_name="Weighted Candidate Gold vs Human Gold",
    )

    # ------------------------------------------------------------------
    # Pass 2: models vs human gold (primary results table in the paper)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("MODELS VS HUMAN GOLD")
    print("=" * 60)

    # Runtimes are measured averages (seconds per document) from a single
    # run on an RTX 3070 Ti Mobile 8 GB.  They are stored here for the
    # runtime figure in graph.py rather than recomputed each evaluation.
    evaluate_model_against_human_gold(
        notes, human_gold_bio_map, scispacy_by_row, "SciSpacy", runtime=0.0238
    )
    evaluate_model_against_human_gold(
        notes, human_gold_bio_map, biobert_by_row, "BioBERT", runtime=0.4634
    )
    evaluate_model_against_human_gold(
        notes,
        human_gold_bio_map,
        biobert_filtered_by_row,
        "BioBERT + Label-Aware Filter",
        runtime=0.4634,
    )
    evaluate_model_against_human_gold(
        notes, human_gold_bio_map, pubmedbert_by_row, "PubMedBERT", runtime=3.4778
    )
    evaluate_model_against_human_gold(
        notes,
        human_gold_bio_map,
        pubmedbert_filtered_by_row,
        "PubMedBERT + Label-Aware Filter",
        runtime=3.4778,
    )
    evaluate_model_against_human_gold(
        notes, human_gold_bio_map, clinicalbert_by_row, "ClinicalBERT", runtime=0.2107
    )
    evaluate_model_against_human_gold(
        notes,
        human_gold_bio_map,
        clinicalbert_filtered_by_row,
        "ClinicalBERT + Label-Aware Filter",
        runtime=0.2107,
    )
    evaluate_model_against_human_gold(
        notes, human_gold_bio_map, bioelectra_by_row, "BioELECTRA", runtime=0.1267
    )
    evaluate_model_against_human_gold(
        notes,
        human_gold_bio_map,
        bioelectra_filtered_by_row,
        "BioELECTRA + Label-Aware Filter",
        runtime=0.1267,
    )

    # ------------------------------------------------------------------
    # Pass 3: models vs majority candidate gold
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("MODELS VS MAJORITY CANDIDATE GOLD")
    print("=" * 60)

    evaluate_model_against_candidate_gold(
        notes, candidate_gold_bio_map, scispacy_by_row, "SciSpacy vs Majority", 0.0238
    )
    evaluate_model_against_candidate_gold(
        notes, candidate_gold_bio_map, biobert_by_row, "BioBERT vs Majority", 0.4634
    )
    evaluate_model_against_candidate_gold(
        notes,
        candidate_gold_bio_map,
        biobert_filtered_by_row,
        "BioBERT + Label-Aware Filter vs Majority",
        0.4634,
    )
    evaluate_model_against_candidate_gold(
        notes,
        candidate_gold_bio_map,
        pubmedbert_by_row,
        "PubMedBERT vs Majority",
        3.4778,
    )
    evaluate_model_against_candidate_gold(
        notes,
        candidate_gold_bio_map,
        pubmedbert_filtered_by_row,
        "PubMedBERT + Label-Aware Filter vs Majority",
        3.4778,
    )
    evaluate_model_against_candidate_gold(
        notes,
        candidate_gold_bio_map,
        clinicalbert_by_row,
        "ClinicalBERT vs Majority",
        0.2107,
    )
    evaluate_model_against_candidate_gold(
        notes,
        candidate_gold_bio_map,
        clinicalbert_filtered_by_row,
        "ClinicalBERT + Label-Aware Filter vs Majority",
        0.2107,
    )
    evaluate_model_against_candidate_gold(
        notes,
        candidate_gold_bio_map,
        bioelectra_by_row,
        "BioELECTRA vs Majority",
        0.1267,
    )
    evaluate_model_against_candidate_gold(
        notes,
        candidate_gold_bio_map,
        bioelectra_filtered_by_row,
        "BioELECTRA + Label-Aware Filter vs Majority",
        0.1267,
    )

    # ------------------------------------------------------------------
    # Pass 4: models vs weighted candidate gold
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("MODELS VS WEIGHTED CANDIDATE GOLD")
    print("=" * 60)

    evaluate_model_against_candidate_gold(
        notes,
        weighted_candidate_gold_bio_map,
        scispacy_by_row,
        "SciSpacy vs Weighted",
        0.0238,
    )
    evaluate_model_against_candidate_gold(
        notes,
        weighted_candidate_gold_bio_map,
        biobert_by_row,
        "BioBERT vs Weighted",
        0.4634,
    )
    evaluate_model_against_candidate_gold(
        notes,
        weighted_candidate_gold_bio_map,
        biobert_filtered_by_row,
        "BioBERT + Label-Aware Filter vs Weighted",
        0.4634,
    )
    evaluate_model_against_candidate_gold(
        notes,
        weighted_candidate_gold_bio_map,
        pubmedbert_by_row,
        "PubMedBERT vs Weighted",
        3.4778,
    )
    evaluate_model_against_candidate_gold(
        notes,
        weighted_candidate_gold_bio_map,
        pubmedbert_filtered_by_row,
        "PubMedBERT + Label-Aware Filter vs Weighted",
        3.4778,
    )
    evaluate_model_against_candidate_gold(
        notes,
        weighted_candidate_gold_bio_map,
        clinicalbert_by_row,
        "ClinicalBERT vs Weighted",
        0.2107,
    )
    evaluate_model_against_candidate_gold(
        notes,
        weighted_candidate_gold_bio_map,
        clinicalbert_filtered_by_row,
        "ClinicalBERT + Label-Aware Filter vs Weighted",
        0.2107,
    )
    evaluate_model_against_candidate_gold(
        notes,
        weighted_candidate_gold_bio_map,
        bioelectra_by_row,
        "BioELECTRA vs Weighted",
        0.1267,
    )
    evaluate_model_against_candidate_gold(
        notes,
        weighted_candidate_gold_bio_map,
        bioelectra_filtered_by_row,
        "BioELECTRA + Label-Aware Filter vs Weighted",
        0.1267,
    )

    plot_all()


if __name__ == "__main__":
    main()
