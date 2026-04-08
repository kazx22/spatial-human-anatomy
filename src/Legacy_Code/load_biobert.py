from transformers import pipeline
from src.biobert_chem_merger import merge_word
from pathlib import Path
import json
import pandas as pd
import time

# !----------------------- Test Case -----------------------!

# MODEL_NAME = (
#     "sarahmiller137/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext-ft-ncbi-disease"
# )

# # sarahmiller137/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext-ft-ncbi-disease
# # Ishan0612/biobert-ner-disease-ncbi -major issues with model. Cant label any entity. Did not work with pipeline. Issues with tokenization and entity recognition.

# ner = pipeline(
#     task="token-classification", model=MODEL_NAME, aggregation_strategy="simple"
# )
# print("NER pipeline loaded successfully")

# # text = "The patient has hypertension and asthma and is taking metformin and aspirin."

# # text = """
# # The patient is a 65 year old male with a history of hypertension, type 2 diabetes, and asthma.
# # He reports chest discomfort and persistent headache for the past two days.
# # Current medications include metformin, aspirin, and atorvastatin.
# # Blood pressure remains elevated despite treatment with lisinopril.
# # """


# res_disease_raw = ner(text)

# print("Input text:")
# print(text)
# print("\nDetected entities:")

# res_disease = []
# for entity in res_disease_raw:
#     if entity["entity_group"] == "LABEL_1":
#         res_disease.append(
#             {
#                 "word": entity["word"],
#                 "entity_group": "DISEASE",
#                "score": float(entity["score"]),
#                 "start": entity["start"],
#                 "end": entity["end"],
#             }
#         )


# # for entity in res_disease:
# #     print(f"Text: {entity['word']}")
# #     print(f"Label: {entity['entity_group']}")
# #     print(f"Score: {entity['score']:.4f}")
# #     print(f"Start: {entity['start']}")
# #     print(f"End: {entity['end']}")
# #     print("-" * 30)


# MODEL_NAME = "OpenMed/OpenMed-NER-ChemicalDetect-PubMed-335M"
# ner = pipeline(
#     task="ner", model=MODEL_NAME, tokenizer=MODEL_NAME, aggregation_strategy="simple"
# )

# res_chem_raw = ner(text)

# res_chem = []
# for entity in res_chem_raw:
#     res_chem.append(
#         {
#             "word": entity["word"],
#             "entity_group": "CHEMICAL",
#             "score": entity["score"],
#             "start": entity["start"],
#             "end": entity["end"],
#         }
#     )


# # print("Detected entities:")
# # for entity in res_chem:
# #     print(f"Text: {entity['word']}")
# #     print(f"Label: {entity['entity_group']}")
# #     print(f"Score: {entity['score']:.4f}")
# #     print(f"Start: {entity['start']}")
# #     print(f"End: {entity['end']}")
# #     print("-" * 30)


# result = res_disease + res_chem


# for entity in result:
#     print(f"Text: {entity['word']}")
#     print(f"Label: {entity['entity_group']}")
#     print(f"Score: {entity['score']:.4f}")
#     print(f"Start: {entity['start']}")
#     print(f"End: {entity['end']}")
#     print("-" * 30)


# res_chem = merge_word(res_chem)

# result = res_disease + res_chem
# # result = sorted(result, key=lambda x: x["start"])

# print("Input text:")
# print(text)
# print("\nDetected entities:")

# for entity in result:
#     print(f"Text: {entity['word']}")
#     print(f"Label: {entity['entity_group']}")
#     print(f"Score: {entity['score']:.4f}")
#     print(f"Start: {entity['start']}")
#     print(f"End: {entity['end']}")
#     print("-" * 30)


# !------------- JSONL -------------!
from transformers import pipeline
from src.biobert_chem_merger import merge_word
from pathlib import Path
import json
import pandas as pd
import time

DISEASE_MODEL = (
    "sarahmiller137/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext-ft-ncbi-disease"
)
CHEM_MODEL = "OpenMed/OpenMed-NER-ChemicalDetect-PubMed-335M"


def safe_text(text, max_chars=1500):
    return str(text)[:max_chars]


disease_ner = pipeline(
    task="token-classification",
    model=DISEASE_MODEL,
    aggregation_strategy="simple",
)

chem_ner = pipeline(
    task="ner",
    model=CHEM_MODEL,
    tokenizer=CHEM_MODEL,
    aggregation_strategy="simple",
)

df = pd.read_csv("data/processed/cleaned_notes.csv")

BASE_DIR = Path("data/processed/ner/transformers/biobert")
BASE_DIR.mkdir(parents=True, exist_ok=True)

output_path = BASE_DIR / "biobert_entities.jsonl"

start_time = time.time()


texts = [safe_text(t) for t in df["note_text"]]


disease_outputs = disease_ner(texts, batch_size=8)
chem_outputs = chem_ner(texts, batch_size=8)

with output_path.open("w", encoding="utf-8") as f:

    for i in range(len(texts)):

        res_disease_raw = disease_outputs[i]
        res_chem_raw = chem_outputs[i]

        res_disease = []
        for entity in res_disease_raw:
            if entity["entity_group"] == "LABEL_1":
                res_disease.append(
                    {
                        "word": entity["word"],
                        "entity_group": "DISEASE",
                        "score": float(entity["score"]),
                        "start": int(entity["start"]),
                        "end": int(entity["end"]),
                    }
                )

        res_chem = []
        for entity in res_chem_raw:
            res_chem.append(
                {
                    "word": entity["word"],
                    "entity_group": "CHEMICAL",
                    "score": float(entity["score"]),
                    "start": int(entity["start"]),
                    "end": int(entity["end"]),
                }
            )

        res_chem = merge_word(res_chem)

        result = res_disease + res_chem
        result = sorted(result, key=lambda x: x["start"])

        for ent in result:
            entity_data = {
                "row_id": i,
                "text": ent["word"],
                "label": ent["entity_group"],
                "start_char": ent["start"],
                "end_char": ent["end"],
                "confidence": ent["score"],
            }

            f.write(json.dumps(entity_data, ensure_ascii=False) + "\n")

print("Saved NER entities to:", output_path)

end_time = time.time()
elapsed_time = end_time - start_time

print("Total time:", round(elapsed_time, 2), "seconds")
print("Average per note:", round(elapsed_time / len(texts), 4), "seconds")
# still having the issue of the model not recognizing entities. When there is more than one sentence, the model seems to struggle. It works better when there is only one sentence.
# I will need to look into this further and see if there are any solutions or workarounds.

# !--------Current Problems with the models--------!

# type as DISEASE

# This is a partial span error.
# The model should ideally detect type 2 diabetes, but instead only grabbed type.

# chest as DISEASE

# This is a false positive.
# It is over-predicting.

# lis, ##ino, ##pri, ##l -- fixed this with the merge_word function, but it is still not perfect. will do the hyphen  and space merge later.

# This is a subtoken split problem.
# The chemical model found the medication, but did not merge the pieces into one clean span.

# So the models are usable, but they need post-processing.
