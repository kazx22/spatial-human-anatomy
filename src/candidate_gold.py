from src.utils import remove_TEST
import json
from collections import defaultdict
from pathlib import Path


# # if key not in vote_table:
# #     vote_table[key] = []

# # vote_table[key].append("scispacy")


# # ALLOWED_LABELS = {"DISEASE", "CHEMICAL"}


# def loadJsonl(file_path: str):
#     entities = []
#     with open(file_path, "r", encoding="utf-8") as f:
#         for line in f:
#             line = line.strip()
#             if not line:
#                 continue
#             entity = json.loads(line)
#             entities.append(entity)
#     return entities


# def entity_key(entity: dict):
#     return (
#         entity["row_id"],
#         entity["text"],
#         entity["start_char"],
#         entity["end_char"],
#         entity["label"],
#     )


# def build_gold(scif, bertf, clinicf, bioelectraf, output_file):
#     vote_table = defaultdict(list)
#     sci_entities = loadJsonl(scif)
#     bert_entities = loadJsonl(bertf)
#     clinic_entities = loadJsonl(clinicf)
#     bioelectra_entities = loadJsonl(bioelectraf)
#     for entity in sci_entities:
#         key = entity_key(entity)
#         vote_table[key].append("scispacy")
#     for entity in bert_entities:
#         key = entity_key(entity)
#         vote_table[key].append("bert")
#     for entity in clinic_entities:
#         key = entity_key(entity)
#         vote_table[key].append("clinicalbert")
#     for entity in bioelectra_entities:
#         key = entity_key(entity)
#         vote_table[key].append("bioelectra")
#     with open(output_file, "w", encoding="utf-8") as f:
#         for key, voters in vote_table.items():
#             if len(voters) >= 3:
#                 row_id, text, start_char, end_char, label = key
#                 entity = {
#                     "row_id": row_id,
#                     "text": text,
#                     "start_char": start_char,
#                     "end_char": end_char,
#                     "label": label,
#                     "voters": voters,
#                 }
#                 f.write(json.dumps(entity) + "\n")


# if __name__ == "__main__":
#     remove_TEST(
#         "data/processed/ner/scispacy/entities.jsonl",
#         "data/processed/ner/scispacy/entities_clean.jsonl",
#     )

#     remove_TEST(
#         "data/processed/ner/transformers/biobert/biobert_entities.jsonl",
#         "data/processed/ner/transformers/biobert/biobert_entities_clean.jsonl",
#     )

#     remove_TEST(
#         "data/processed/ner/transformers/bioelectra/bioelectra_entities.jsonl",
#         "data/processed/ner/transformers/bioelectra/bioelectra_entities_clean.jsonl",
#     )

#     remove_TEST(
#         "data/processed/ner/transformers/clinicalbert/clinicalbert_entities.jsonl",
#         "data/processed/ner/transformers/clinicalbert/clinicalbert_entities_clean.jsonl",
#     )

#     build_gold(
#         "data/processed/ner/scispacy/entities_clean.jsonl",
#         "data/processed/ner/transformers/biobert/biobert_entities_clean.jsonl",
#         "data/processed/ner/transformers/clinicalbert/clinicalbert_entities_clean.jsonl",
#         "data/processed/ner/transformers/bioelectra/bioelectra_entities_clean.jsonl",
#         "data/gold/candidate_gold_entities.jsonl",
#     )


# if key not in vote_table:
#     vote_table[key] = []

# vote_table[key].append("scispacy")


# ALLOWED_LABELS = {"DISEASE", "CHEMICAL"}


def load_jsonl(file_path: str):
    entities = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            entity = json.loads(line)
            entities.append(entity)

    return entities


def entity_key(entity: dict):
    return (
        entity["row_id"],
        entity["start_char"],
        entity["end_char"],
        entity["label"],
    )


def build_candidate_gold(scif, bertf, clinicf, bioelectraf, output_file, min_votes=3):
    vote_table = defaultdict(list)
    entity_store = {}

    sci_entities = load_jsonl(scif)
    bert_entities = load_jsonl(bertf)
    clinic_entities = load_jsonl(clinicf)
    bioelectra_entities = load_jsonl(bioelectraf)

    for entity in sci_entities:
        key = entity_key(entity)
        vote_table[key].append("scispacy")
        if key not in entity_store:
            entity_store[key] = entity

    for entity in bert_entities:
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

    with open(output_path, "w", encoding="utf-8") as f:
        for key, voters in vote_table.items():
            if len(voters) >= min_votes:
                entity = dict(entity_store[key])
                entity["voters"] = voters
                f.write(json.dumps(entity, ensure_ascii=False) + "\n")
                saved_count += 1

    print(f"Loaded {len(sci_entities)} entities from scispacy")
    print(f"Loaded {len(bert_entities)} entities from pubmedbert")
    print(f"Loaded {len(clinic_entities)} entities from clinicalbert")
    print(f"Loaded {len(bioelectra_entities)} entities from bioelectra")
    print(f"Saved {saved_count} candidate gold entities to {output_file}")


if __name__ == "__main__":
    remove_TEST(
        "data/processed/bc5cdr/clinicalbert_train_entities_bc5dr.jsonl",
        "data/processed/bc5cdr/clinicalbert_train_entities_bc5cdr_clean.jsonl",
    )

    build_candidate_gold(
        "data/processed/bc5cdr/scispacy_train_entities_bc5cdr.jsonl",
        "data/processed/bc5cdr/pubmedbert_train_entities_bc5dr.jsonl",
        "data/processed/bc5cdr/clinicalbert_train_entities_bc5cdr_clean.jsonl",
        "data/processed/bc5cdr/bioelectra_train_entities_bc5cdr.jsonl",
        "data/gold/candidate_gold_train_entities_bc5cdr.jsonl",
        min_votes=3,
    )
