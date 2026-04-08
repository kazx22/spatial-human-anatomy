from src.entity_normalize_pipeline import normalize_entities

normalize_entities(
    r"data\processed\ner\scispacy\entities.jsonl",
    r"data\processed\ner\scispacy\entities_normalized.jsonl",
)

print("Normalization complete.")
# normalize_entities(
#     "data/biobert_entities.jsonl", "data/biobert_entities_normalized.jsonl"
# )

# normalize_entities(
#     "data/clinicalbert_entities.jsonl", "data/clinicalbert_entities_normalized.jsonl"
# )
