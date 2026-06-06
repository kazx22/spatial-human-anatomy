"""
candidate_gold.py — builds the weighted pseudo-gold standard from the five
model prediction files.

Each model votes for every (row_id, start_char, end_char, label) span it
predicted.  A span is admitted to the pseudo-gold set only when the sum of
the F1 weights of its voters meets or exceeds min_weighted_score (default
1.5).  That threshold sits just above SciSpacy's weight alone (0.8959), so
no single model — however accurate — can unilaterally promote a span.

Pipeline position: runs after all five model scripts have written their
per-model JSONL files and before bc5cdr_evaluation.py or the analysis
scripts.
"""

from src.utils import remove_TEST
import json
from collections import defaultdict
from pathlib import Path

ALLOWED_LABELS = {"DISEASE", "CHEMICAL"}

# ------------------------------------------------------------------
# Model weights = overall F1 score against human gold on BC5CDR.
# These are the actual measured F1 scores, so the weighting scheme
# is fully reproducible and matches what the paper claims.
# ------------------------------------------------------------------
MODEL_WEIGHTS = {
    "scispacy": 0.8959,
    "biobert": 0.7259,
    "pubmedbert": 0.5677,
    "clinicalbert": 0.4213,
    "bioelectra": 0.5470,
}

# Maximum achievable weighted score (all five models agree).
# Used as an upper bound for the threshold sensitivity sweep.
MAX_WEIGHTED_SCORE = sum(MODEL_WEIGHTS.values())  # ~2.1647


def load_jsonl(file_path: str):
    """
    Read a model's prediction JSONL and return only DISEASE/CHEMICAL spans.

    I filter here rather than at voting time so the vote table never sees
    labels that have no BC5CDR equivalent (e.g. ClinicalBERT's TEST label,
    which is stripped upstream by remove_TEST before this is called).
    """
    entities = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entity = json.loads(line)
            if entity["label"] not in ALLOWED_LABELS:
                continue
            entities.append(entity)
    return entities


def entity_key(entity: dict):
    """
    Canonical key for a predicted span.

    Using character offsets rather than token indices keeps the key
    consistent across models that tokenise differently.
    """
    return (
        entity["row_id"],
        entity["start_char"],
        entity["end_char"],
        entity["label"],
    )


def add_votes(vote_table, entity_store, entities, model_name):
    """
    Record that model_name predicted each span in entities.

    vote_table maps each key to the list of models that predicted it.
    entity_store keeps the first-seen entity dict for that key so we can
    reconstruct the full span record when writing output — the actual text
    and offsets are identical across models for the same key, so first-seen
    is fine.
    """
    for entity in entities:
        key = entity_key(entity)
        vote_table[key].append(model_name)
        if key not in entity_store:
            entity_store[key] = entity


def build_candidate_gold(
    scif,
    biobertf,
    pubmedf,
    clinicf,
    bioelectraf,
    output_file,
    min_weighted_score=1.5,
    verbose=True,
):
    """
    Aggregate five model prediction files into a single pseudo-gold JSONL.

    For each unique span key, I compute:
      weighted_score  = sum of F1 weights of all models that predicted it
      agreement_score = fraction of models that predicted it (0–1)

    Only spans where weighted_score >= min_weighted_score are written to
    output.  The threshold default of 1.5 requires at least two models to
    agree, with the combined weight exceeding SciSpacy's solo score so even
    the strongest single model cannot carry a span into the gold set alone.

    Each output record carries the original span fields plus voters,
    vote_count, agreement_score, and weighted_score for downstream
    diagnostics and evaluation.
    """
    vote_table = defaultdict(list)
    entity_store = {}

    sci_entities = load_jsonl(scif)
    biobert_entities = load_jsonl(biobertf)
    pubmed_entities = load_jsonl(pubmedf)
    clinic_entities = load_jsonl(clinicf)
    bioelectra_entities = load_jsonl(bioelectraf)

    if verbose:
        print(f"SciSpacy:     {len(sci_entities)}")
        print(f"BioBERT:      {len(biobert_entities)}")
        print(f"PubMedBERT:   {len(pubmed_entities)}")
        print(f"ClinicalBERT: {len(clinic_entities)}")
        print(f"BioELECTRA:   {len(bioelectra_entities)}")

    add_votes(vote_table, entity_store, sci_entities, "scispacy")
    add_votes(vote_table, entity_store, biobert_entities, "biobert")
    add_votes(vote_table, entity_store, pubmed_entities, "pubmedbert")
    add_votes(vote_table, entity_store, clinic_entities, "clinicalbert")
    add_votes(vote_table, entity_store, bioelectra_entities, "bioelectra")

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    saved_count = 0
    vote_distribution = defaultdict(int)
    weighted_score_distribution = defaultdict(int)

    with open(output_path, "w", encoding="utf-8") as f:
        for key, voters in vote_table.items():
            vote_count = len(voters)
            vote_distribution[vote_count] += 1

            weighted_score = sum(MODEL_WEIGHTS[voter] for voter in voters)
            agreement_score = vote_count / len(MODEL_WEIGHTS)

            rounded_weighted_score = round(weighted_score, 2)
            weighted_score_distribution[rounded_weighted_score] += 1

            if weighted_score >= min_weighted_score:
                entity = dict(entity_store[key])
                entity["voters"] = voters
                entity["vote_count"] = vote_count
                entity["agreement_score"] = round(agreement_score, 4)
                entity["weighted_score"] = round(weighted_score, 4)
                f.write(json.dumps(entity, ensure_ascii=False) + "\n")
                saved_count += 1

    if verbose:
        print("\n--- Weighted Candidate Gold Summary ---")
        print(f"Total unique entities: {len(vote_table)}")
        print(f"Saved (weighted_score >= {min_weighted_score}): {saved_count}")
        print(f"Output file: {output_file}")

        print("\nVote count distribution:")
        for vote_count in sorted(vote_distribution):
            print(f"  {vote_count} votes: {vote_distribution[vote_count]}")

        print("\nWeighted score distribution:")
        for score in sorted(weighted_score_distribution):
            print(f"  {score}: {weighted_score_distribution[score]}")

    return saved_count


if __name__ == "__main__":
    # ClinicalBERT emits a TEST label that has no BC5CDR equivalent.
    # Strip it before the voting step so it never pollutes the vote table.
    remove_TEST(
        "data/processed/bc5cdr/clinicalbert_train_entities_bc5cdr.jsonl",
        "data/processed/bc5cdr/clinicalbert_train_entities_bc5cdr_clean.jsonl",
    )

    scif = "data/processed/bc5cdr/scispacy_train_entities_bc5cdr.jsonl"
    biobertf = "data/processed/bc5cdr/biobert_train_entities_bc5cdr.jsonl"
    pubmedf = "data/processed/bc5cdr/pubmedbert_train_entities_bc5cdr.jsonl"
    clinicf = "data/processed/bc5cdr/clinicalbert_train_entities_bc5cdr_clean.jsonl"
    bioelectraf = "data/processed/bc5cdr/bioelectra_train_entities_bc5cdr.jsonl"

    # Main pseudo-gold build at the operating threshold.
    # 1.5 sits just above SciSpacy's solo weight (0.8959), so even the
    # strongest single model cannot carry a span into the gold set alone.
    build_candidate_gold(
        scif,
        biobertf,
        pubmedf,
        clinicf,
        bioelectraf,
        "data/gold/weighted_candidate_gold_train_entities_bc5cdr.jsonl",
        min_weighted_score=1.5,
    )

    # ------------------------------------------------------------------
    # THRESHOLD SENSITIVITY SWEEP
    # Builds a pseudo-gold set at each threshold so its effect on the
    # number of retained entities can be analysed.  Max weighted score
    # is ~2.16 (all five models agree), so the sweep spans that range.
    # Evaluate each output against human gold with bc5cdr_evaluation.py.
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("THRESHOLD SENSITIVITY SWEEP")
    print(f"(max achievable weighted score = {MAX_WEIGHTED_SCORE:.4f})")
    print("=" * 60)

    sweep = [0.45, 0.70, 0.90, 1.20, 1.50, 1.80, 2.00]
    for threshold in sweep:
        out = f"data/gold/sensitivity/weighted_candidate_gold_{threshold}.jsonl"
        saved = build_candidate_gold(
            scif,
            biobertf,
            pubmedf,
            clinicf,
            bioelectraf,
            out,
            min_weighted_score=threshold,
            verbose=False,
        )
        print(f"  threshold {threshold:>4}:  {saved:>6} entities retained  ->  {out}")
