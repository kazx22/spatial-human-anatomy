from src.utils import remove_TEST, derive_weights
import json
from collections import defaultdict
from pathlib import Path

ALLOWED_LABELS = {"DISEASE", "CHEMICAL"}

F1_SCORES = {
    "scispacy": 0.8959,
    "biobert": 0.4525,
    "pubmedbert": 0.4014,
    "clinicalbert": 0.3507,
    "bioelectra": 0.4642,
}

MODEL_WEIGHTS = derive_weights(F1_SCORES)


def load_jsonl(file_path: str):
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
    return (
        entity["row_id"],
        entity["start_char"],
        entity["end_char"],
        entity["label"],
    )


def add_votes(vote_table, entity_store, entities, model_name):
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
    min_weighted_score=0.5,
):
    vote_table = defaultdict(list)
    entity_store = {}

    sci_entities = load_jsonl(scif)
    biobert_entities = load_jsonl(biobertf)
    pubmed_entities = load_jsonl(pubmedf)
    clinic_entities = load_jsonl(clinicf)
    bioelectra_entities = load_jsonl(bioelectraf)

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

    print("\n--- Weighted Candidate Gold Summary ---")
    print(f"Total unique entities: {len(vote_table)}")
    print(f"Saved (weighted_score >= {min_weighted_score}): {saved_count}")
    print(f"Output file: {output_file}")

    print("\nVote count distribution:")
    for vote_count in sorted(vote_distribution):
        print(f"{vote_count} votes: {vote_distribution[vote_count]}")

    print("\nWeighted score distribution:")
    for score in sorted(weighted_score_distribution):
        print(f"{score}: {weighted_score_distribution[score]}")


if __name__ == "__main__":
    remove_TEST(
        "data/processed/bc5cdr/clinicalbert_train_entities_bc5cdr.jsonl",
        "data/processed/bc5cdr/clinicalbert_train_entities_bc5cdr_clean.jsonl",
    )

    build_candidate_gold(
        "data/processed/bc5cdr/scispacy_train_entities_bc5cdr.jsonl",
        "data/processed/bc5cdr/biobert_train_entities_bc5cdr.jsonl",
        "data/processed/bc5cdr/pubmedbert_train_entities_bc5cdr.jsonl",
        "data/processed/bc5cdr/clinicalbert_train_entities_bc5cdr_clean.jsonl",
        "data/processed/bc5cdr/bioelectra_train_entities_bc5cdr.jsonl",
        "data/gold/weighted_candidate_gold_train_entities_bc5cdr.jsonl",
        min_weighted_score=0.5,
    )

    # for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
    #     build_candidate_gold(
    #         scif, biobertf, pubmedf, clinicf, bioelectraf,
    #         f"data/gold/sensitivity/weighted_candidate_gold_{threshold}.jsonl",
    #         min_weighted_score=threshold,
    #     )
