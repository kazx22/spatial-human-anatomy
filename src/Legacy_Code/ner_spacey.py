import spacy
from pathlib import Path
import json
import pandas as pd
import time

# nlp = spacy.load("en_core_sci_md")
nlp = spacy.load("en_ner_bc5cdr_md")

# text = "Patient was diagnosed with diabetes and prescribed metformin."

df = pd.read_csv("data/processed/cleaned_notes.csv")
text = str(df["note_text"].iloc[0])

# print("Processing first note...\n")
# doc = nlp(text)

# for ent in doc.ents:
#     print(
#         "Text:",
#         ent.text,
#         "| Label:",
#         ent.label_,
#         "| Start Char:",
#         ent.start_char,
#         "| End Char:",
#         ent.end_char,
#     )
start_time = time.time()

BASE_DIR = Path("data/processed/ner/scispacy")
BASE_DIR.mkdir(parents=True, exist_ok=True)
output_path = BASE_DIR / "entities.jsonl"
LIMIT = len(df)
with output_path.open("w", encoding="utf-8") as f:
    for i in range(min(LIMIT, len(df))):
        text = str(df["note_text"].iloc[i])
        doc = nlp(text)

        for ent in doc.ents:
            entity_data = {
                "row_id": i,
                "text": ent.text,
                "label": ent.label_,
                "start_char": ent.start_char,
                "end_char": ent.end_char,
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
