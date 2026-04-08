from transformers import pipeline
from src.biobert_chem_merger import merge_word
from pathlib import Path
import json
import pandas as pd
import time

MODEL_NAME = "d4data/biomedical-ner-all"
df = pd.read_csv("data/processed/cleaned_notes.csv")

ner = pipeline("ner", model=MODEL_NAME, aggregation_strategy="simple")


label_map = {
    "Disease_disorder": "DISEASE",
    "Sign_symptom": "DISEASE",
    "Medication": "CHEMICAL",
    "Therapeutic_procedure": "CHEMICAL",
}

start_time = time.time()

BASE_DIR = Path("data/processed/ner/transformers/bioelectra")
BASE_DIR.mkdir(parents=True, exist_ok=True)

output_path = BASE_DIR / "bioelectra_entities.jsonl"
LIMIT = len(df)

with output_path.open("w", encoding="utf-8") as f:
    for i in range(min(LIMIT, len(df))):
        text = str(df["note_text"].iloc[i])

        results = ner(text)
        merged_results = merge_word(results)

        for ent in merged_results:
            mapped_label = label_map.get(ent["entity_group"])

            if mapped_label is None:
                continue

            entity_data = {
                "row_id": i,
                "text": ent["word"],
                "label": mapped_label,
                "start_char": ent["start"],
                "end_char": ent["end"],
                "confidence": float(ent["score"]),
            }
            f.write(json.dumps(entity_data, ensure_ascii=False) + "\n")

print("Saved NER entities for", min(LIMIT, len(df)), "notes to:", output_path)

with output_path.open("r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        print(line.strip())
        if i == 2:
            break

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Total time taken: {elapsed_time:.2f} seconds")

notes_processed = min(LIMIT, len(df))
avg_time = elapsed_time / notes_processed

print("Average time per note:", round(avg_time, 4), "seconds")
