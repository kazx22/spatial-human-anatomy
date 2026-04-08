from src.utils import remove_TEST
import json
from collections import defaultdict
from pathlib import Path


ALLOWED_LABELS = {"DISEASE", "CHEMICAL"}


def load_jsonl(file_path: str):
    entities = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            entity = json.loads(line)

            # 🔒 Filter labels
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


def build_candidate_gold(
    scif,
    pubmedf,
    clinicf,
    bioelectraf,
    output_file,
    min_votes=3,
):
    vote_table = defaultdict(list)
    entity_store = {}

    sci_entities = load_jsonl(scif)
    pubmed_entities = load_jsonl(pubmedf)
    clinic_entities = load_jsonl(clinicf)
    bioelectra_entities = load_jsonl(bioelectraf)

    print(f"SciSpacy:   {len(sci_entities)}")
    print(f"PubMedBERT: {len(pubmed_entities)}")
    print(f"ClinicalBERT: {len(clinic_entities)}")
    print(f"BioELECTRA: {len(bioelectra_entities)}")

    for entity in sci_entities:
        key = entity_key(entity)
        vote_table[key].append("scispacy")
        if key not in entity_store:
            entity_store[key] = entity

    for entity in pubmed_entities:
        key = entity_key(entity)
        vote_table[key].append("pubmedbert")
        if key not in entity_store:
            entity_store[key] = entity

    for entity in clinic_entities:
        key = entity_key(entity)
        vote_table[key].append("clinicalbert")
        if key not in entity_store:
            entity_store[key] = entity

    for entity in bioelectra_entities:
        key = entity_key(entity)
        vote_table[key].append("bioelectra")
        if key not in entity_store:
            entity_store[key] = entity

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    saved_count = 0
    vote_distribution = defaultdict(int)

    with open(output_path, "w", encoding="utf-8") as f:
        for key, voters in vote_table.items():
            vote_distribution[len(voters)] += 1

            if len(voters) >= min_votes:
                entity = dict(entity_store[key])
                entity["voters"] = voters

                f.write(json.dumps(entity, ensure_ascii=False) + "\n")
                saved_count += 1

    print("\n--- Candidate Gold Summary ---")
    print(f"Total unique entities: {len(vote_table)}")
    print(f"Saved (>= {min_votes} votes): {saved_count}")

    print("\nVote distribution:")
    for k in sorted(vote_distribution):
        print(f"{k} votes: {vote_distribution[k]}")


if __name__ == "__main__":
    remove_TEST(
        "data/processed/bc5cdr/clinicalbert_train_entities_bc5cdr.jsonl",
        "data/processed/bc5cdr/clinicalbert_train_entities_bc5cdr_clean.jsonl",
    )

    build_candidate_gold(
        "data/processed/bc5cdr/scispacy_train_entities_bc5cdr.jsonl",
        "data/processed/bc5cdr/pubmedbert_train_entities_bc5cdr.jsonl",
        "data/processed/bc5cdr/clinicalbert_train_entities_bc5cdr_clean.jsonl",
        "data/processed/bc5cdr/bioelectra_train_entities_bc5cdr.jsonl",
        "data/gold/candidate_gold_train_entities_bc5cdr.jsonl",
        min_votes=3,
    )
